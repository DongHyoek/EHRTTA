import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType


# -------------------------
# 1) A(x) 출력에 적용할 커스텀 함수 예시
# -------------------------
class FeatureTransform(nn.Module):
    """
    DoRA/LoRA의 lora_A 출력 (low-rank vector)에 적용할 변환.
    예: L2 normalization
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, L, r)
        norm = z.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        return z / norm


class AWithTransform(nn.Module):
    """
    기존 lora_A(Linear)를 감싸서 A(x) -> transform(A(x)) 반환
    """
    def __init__(self, a_linear: nn.Module, transform: nn.Module):
        super().__init__()
        self.a_linear = a_linear
        self.transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.a_linear(x)
        z = self.transform(z)
        return z


def wrap_all_lora_A_modules_inplace(peft_model: nn.Module, transform: nn.Module) -> int:
    """
    PEFT가 삽입한 LoRA/DoRA 레이어들 중 lora_A 모듈을 찾아 wrapper로 교체.
    (PEFT 내부 구현이 버전별로 조금 달라서 '탐색 기반'으로 최대한 견고하게 작성)

    Returns:
        교체한 lora_A 개수
    """
    replaced = 0

    for module in peft_model.modules():
        # PEFT LoRA layer들은 보통 lora_A, lora_B 같은 attribute를 가짐(버전/타겟 레이어 타입별로 약간 다름)
        if hasattr(module, "lora_A"):
            lora_A = getattr(module, "lora_A")

            # lora_A가 adapter_name -> nn.Module 형태로 담긴 dict/ModuleDict인 경우가 흔함
            if isinstance(lora_A, (nn.ModuleDict, dict)):
                for adapter_name, a_mod in list(lora_A.items()):
                    # 이미 wrapper면 skip
                    if isinstance(a_mod, AWithTransform):
                        continue
                    lora_A[adapter_name] = AWithTransform(a_mod, transform)
                    replaced += 1

            # 일부 구현에서는 단일 모듈일 수도 있으니 방어적으로 처리
            elif isinstance(lora_A, nn.Module) and not isinstance(lora_A, AWithTransform):
                setattr(module, "lora_A", AWithTransform(lora_A, transform))
                replaced += 1

    return replaced

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

class PEFTTSLLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args

        hf_config = AutoConfig.from_pretrained(args.model_id)

        # hidden state 뽑기 좋은 AutoModel 사용
        backbone = AutoModel.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
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
            use_dora=True,
        )
        self.backbone = get_peft_model(backbone, peft_args)

        # 3) A 출력(low-rank vector) 변형 삽입
        n = wrap_all_lora_A_modules_inplace(self.backbone, FeatureTransform())
        print(f"[INFO] Wrapped lora_A modules: {n}")

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

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        ✅ input_ids(이미 토크나이즈된 정수) OR inputs_embeds(이미 임베딩된 벡터) 둘 중 하나를 주입.
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

        loss = None
        if labels is not None:
            if self.args.task == "classification":
                loss = F.cross_entropy(logits, labels.long())
            else:
                # regression: labels shape (B,) or (B,1)
                loss = F.mse_loss(logits.squeeze(-1), labels.float().view(-1))

        return {"loss": loss, "logits": logits, "pooled": pooled}