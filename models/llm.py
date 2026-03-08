import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from models.tta import PEFTAdaINPatcher
from utils.losses import focal_loss

class PEFTTSLLM(nn.Module):
    def __init__(self, args, device, adapter_dir='./', use_load=False):
        """
        use_load : load trained the best adapter yes / no?
        adapter_dir : the directory of the best model saved
        """
        super().__init__()
        
        self.args = args
        self.device = device
        self.focal_loss = focal_loss(gamma=2, alpha=[1,3], device=device)

        self.hf_config = AutoConfig.from_pretrained(args.model_id)

        # hidden state 뽑기 좋은 AutoModel 사용
        backbone = AutoModel.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map=device
            )

        # 1) backbone freeze
        for p in backbone.parameters():
            p.requires_grad = False

        # 2) PEFT 붙이기 (PEFT에서 use_dora=True) :contentReference[oaicite:3]{index=3}
        peft_args = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.rank_dim,
            lora_alpha=args.peft_alpha,
            lora_dropout=args.peft_dropout,
            target_modules=list(args.target_modules),
            bias="none",
            use_dora=True if args.use_dora else False
            )
        
        self.backbone = get_peft_model(backbone, peft_args)

        if use_load: # load best model for inference
            self.backbone.load_adapter(f"{adapter_dir}/best_adapter", adapter_name="default", is_trainable=False)
            self.backbone.set_adapter("default")

        if args.adapt_mode:
            self.backbone_w_tta = PEFTAdaINPatcher(self.backbone, adapter_name="default")

        self.hidden_size = self.hf_config.hidden_size

        # [Summary 토큰 추가]
        if args.h_pool == "last":
            self.sum_tok = nn.Parameter(torch.empty(1, 1, self.hidden_size))
            nn.init.normal_(self.sum_tok, mean=0.0, std=0.02)
            self.tsvar_emb = nn.Embedding(args.te_n_vars, self.hidden_size)     # e.g. Time series (= Vital IDs)
        
        # Variable-wise aggregation module (self_attention)
        self.cls = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=args.agg_n_heads, dim_feedforward=2*self.hidden_size, batch_first=True)
        self.aggregator = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=args.agg_n_layers)

        # 4) downstream head
        if args.task == "classification":
            self.head = nn.Linear(self.hidden_size, args.num_labels)
        elif args.task == "regression":
            self.head = nn.Linear(self.hidden_size, args.num_labels)  # 보통 num_labels=1
        else:
            raise ValueError("task must be 'classification' or 'regression'")

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # last_hidden_state: (B, L, d_model)
        if self.args.h_pool == "last":
            # return last_hidden_state[:, -1, :] # summarize token 붙인 케이스

            # Mask가 아닌 맨 마지막 토큰만 사용하는 경우 
            if attention_mask is None:
                return last_hidden_state[:, -1, :]
            
            # 마지막 유효 토큰 위치로 gather
            lengths = attention_mask.long().sum(dim=1) - 1  # (B,)
            return last_hidden_state[torch.arange(last_hidden_state.size(0), device=last_hidden_state.device), lengths]
        
        elif self.args.h_pool == "mean":
            if attention_mask is None:
                return last_hidden_state.mean(dim=1)
            
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            raise ValueError("pooling must be 'last' or 'mean'")

    def forward(self, inputs_embeds: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, var_idx_ts: Optional[torch.tensor] = None, num_ts: int = 7) -> Dict[str, Any]:
        """
        inputs_embeds(이미 임베딩된 벡터 = aligned)를 input으로 넣어줌.
        transformers AutoModel은 inputs_embeds를 지원함. :contentReference[oaicite:4]{index=4}
        """

        # ## Add summarize tokens with variable embedding 
        # if self.args.h_pool == 'last':
        #     sum_tok = self.sum_tok.expand(inputs_embeds.shape[0], 1, self.hidden_size)      # (B*D, 1, d)
        #     sum_tok = sum_tok + self.tsvar_emb(var_idx_ts).unsqueeze(1)                     # (B*D, 1, d)
        #     inputs_embeds  = torch.cat([inputs_embeds, sum_tok], 1)                         # (B*D, L+1, d)

        #     sum_tok_mask = torch.ones(attention_mask.shape[0], 1, device=self.device, dtype=attention_mask.dtype)
        #     attention_mask = torch.cat([attention_mask, sum_tok_mask], 1)  # (B*D, L+1)
        
        ## LLM forward
        if not self.args.adapt_mode: # source train mode
            outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)

        else: # target adaptation mode
            outputs = self.backbone_w_tta(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
            
        h = outputs.last_hidden_state  # (B*D, L+1, d_model)

        pooled = self._pool(h, attention_mask)  # (B*D, d_model)

        ## Aggregation with self-attention
        pooled = pooled.reshape(-1, num_ts, self.hidden_size)        # (B, D, d_model)
        cls = self.cls.expand(pooled.shape[0], 1, self.hidden_size)  # (B, 1, d_model)
        pooled = torch.cat([cls, pooled], dim=1)                     # (B, 1+D, d_model)
        
        out = self.aggregator(pooled) # (B, 1+D, d_model)
        out = out[:,0,:]              # (B, d_model)
        logits = self.head(out)       # (B, C) or (B, 1)
        
        if self.args.task == "classification":
            # loss = F.cross_entropy(logits, labels.long())
            loss = self.focal_loss(logits, labels.long())
        else:
            # regression: labels shape (B,) or (B,1)
            loss = F.mse_loss(logits.squeeze(-1), labels.float().view(-1))

        return {"loss": loss, "logits": logits, "pooled": pooled}