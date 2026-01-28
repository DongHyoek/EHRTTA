# Here is the LLM modeling codes with PEFT LoRA
import torch
import torch.nn as nn
import transformers

from transformers import AutoConfig, AutoTokenizer, AutoTokenizer, AutoModelForCausalLM, LlamaModel
from peft import LoraConfig


class TemporalLLM(nn.Module):

    def __init__(self, args):
        super(TemporalLLM, self).__init__()

        # self.tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        self.model = AutoModelForCausalLM.from_pretrained(args.model_type)


    def forward(self, tokenized_vec):

        output = self.model(tokenized_vec)

        return output
    
@dataclass
class DownstreamConfig:
    model_id: str = "meta-llama/Meta-Llama-3.1-8B"  # base로 쓰는 걸 추천(인스트럭트도 가능)
    num_labels: int = 2
    task: str = "classification"  # "classification" or "regression"
    pooling: str = "last"         # "last" or "mean"
    dora_r: int = 8
    dora_alpha: int = 16
    dora_dropout: float = 0.0
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")  # attention에만
    torch_dtype: torch.dtype = torch.bfloat16


class LlamaForDownstream(nn.Module):
    def __init__(self, cfg: DownstreamConfig):
        super().__init__()
        self.cfg = cfg

        hf_config = AutoConfig.from_pretrained(cfg.model_id)

        # ✅ 생성용(CausalLM) 말고, hidden state 뽑기 좋은 LlamaModel 사용
        backbone = LlamaModel.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.torch_dtype,
            device_map="auto",
        )

        # 1) backbone freeze
        for p in backbone.parameters():
            p.requires_grad = False

        # 2) DoRA 붙이기 (PEFT에서 use_dora=True) :contentReference[oaicite:3]{index=3}
        peft_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg.dora_r,
            lora_alpha=cfg.dora_alpha,
            lora_dropout=cfg.dora_dropout,
            target_modules=list(cfg.target_modules),
            bias="none",
            use_dora=True,
        )
        self.backbone = get_peft_model(backbone, peft_cfg)

        # 3) A 출력(low-rank vector) 변형 삽입
        n = wrap_all_lora_A_modules_inplace(self.backbone, FeatureTransform())
        print(f"[INFO] Wrapped lora_A modules: {n}")

        hidden_size = hf_config.hidden_size

        # 4) 다운스트림 head
        if cfg.task == "classification":
            self.head = nn.Linear(hidden_size, cfg.num_labels)
        elif cfg.task == "regression":
            self.head = nn.Linear(hidden_size, cfg.num_labels)  # 보통 num_labels=1
        else:
            raise ValueError("task must be 'classification' or 'regression'")

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # last_hidden_state: (B, L, H)
        if self.cfg.pooling == "last":
            if attention_mask is None:
                return last_hidden_state[:, -1, :]
            # 마지막 유효 토큰 위치로 gather
            lengths = attention_mask.long().sum(dim=1) - 1  # (B,)
            return last_hidden_state[torch.arange(last_hidden_state.size(0), device=last_hidden_state.device), lengths]
        elif self.cfg.pooling == "mean":
            if attention_mask is None:
                return last_hidden_state.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            raise ValueError("pooling must be 'last' or 'mean'")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        ✅ input_ids(이미 토크나이즈된 정수) OR inputs_embeds(이미 임베딩된 벡터) 둘 중 하나를 주입.
        transformers LlamaModel은 inputs_embeds를 지원함. :contentReference[oaicite:4]{index=4}
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

        loss = None
        if labels is not None:
            if self.cfg.task == "classification":
                loss = F.cross_entropy(logits, labels.long())
            else:
                # regression: labels shape (B,) or (B,1)
                loss = F.mse_loss(logits.squeeze(-1), labels.float().view(-1))

        return {"loss": loss, "logits": logits, "pooled": pooled}

if __name__ == "__main__":
    import torch
    from transformers import pipeline

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    prompt = "Explain what a digital twin is in 3 bullet points."

    out = pipe(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id = pipe.tokenizer.eos_token_id
    )

    print(out[0]["generated_text"])