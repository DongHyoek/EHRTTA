import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from tta import *

class PEFTTSLLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args

        hf_config = AutoConfig.from_pretrained(args.model_id)

        # hidden state 뽑기 좋은 AutoModel 사용
        backbone = AutoModel.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
            )

        # 1) backbone freeze
        for p in backbone.parameters():
            p.requires_grad = False

        # 2) DoRA 붙이기 (PEFT에서 use_dora=True) :contentReference[oaicite:3]{index=3}
        peft_args = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.dora_r,
            lora_alpha=args.dora_alpha,
            lora_dropout=args.dora_dropout,
            target_modules=list(args.target_modules),
            bias="none",
            use_dora=True
            )
        
        self.backbone = get_peft_model(backbone, peft_args)

        self.backbone_w_tta = PEFTAdaINPatcher(self.backbone, adapter_name="default")

        hidden_size = hf_config.hidden_size

        # 4) 다운스트림 head
        if args.task == "classification":
            self.head = nn.Linear(hidden_size, args.num_labels)
        elif args.task == "regression":
            self.head = nn.Linear(hidden_size, args.num_labels)  # 보통 num_labels=1
        else:
            raise ValueError("task must be 'classification' or 'regression'")

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # last_hidden_state: (B, L, H)
        if self.args.pooling == "last":
            if attention_mask is None:
                return last_hidden_state[:, -1, :]
            # 마지막 유효 토큰 위치로 gather
            lengths = attention_mask.long().sum(dim=1) - 1  # (B,)
            return last_hidden_state[torch.arange(last_hidden_state.size(0), device=last_hidden_state.device), lengths]
        
        elif self.args.pooling == "mean":
            if attention_mask is None:
                return last_hidden_state.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            raise ValueError("pooling must be 'last' or 'mean'")

    def forward(self, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        input_ids(이미 토크나이즈된 정수) OR inputs_embeds(이미 임베딩된 벡터) 둘 중 하나를 주입.
        transformers AutoModel은 inputs_embeds를 지원함. :contentReference[oaicite:4]{index=4}
        """

        outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        h = outputs.last_hidden_state  # (B, L, H)

        pooled = self._pool(h, attention_mask)  # (B, H)
        logits = self.head(pooled)              # (B, C) or (B, 1)

        if labels is not None:
            if self.args.task == "classification":
                loss = F.cross_entropy(logits, labels.long())
            else:
                # regression: labels shape (B,) or (B,1)
                loss = F.mse_loss(logits.squeeze(-1), labels.float().view(-1))

        return {"loss": loss, "logits": logits, "pooled": pooled}