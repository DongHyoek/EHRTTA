from typing import Dict, List, Tuple, Optional, Literal, Any
import random
import torch
import torch.nn as nn

StatsMode = Literal["aggregate", "distribution"]
SelectMode = Literal["mean_of_dist", "random", "cycle"]

class RunningStats:
    """
    채널(r)별 mean/var를 누적 추정.
    z shape: (B, L, r)
    누적 축: B*L
    """
    def __init__(self, r: int, device: Optional[torch.device] = None):
        self.r = r
        self.count = 0
        self.sum = torch.zeros(r, device=device)
        self.sumsq = torch.zeros(r, device=device)

    @torch.no_grad()
    def update(self, z: torch.Tensor):
        z2 = z.detach().reshape(-1, z.size(-1))  # (B*L, r)
        self.sum += z2.sum(dim=0)
        self.sumsq += (z2 ** 2).sum(dim=0)
        self.count += z2.size(0)

    @torch.no_grad()
    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-12
        denom = max(self.count, 1)
        mean = self.sum / denom
        var = (self.sumsq / denom) - mean**2
        var = torch.clamp(var, min=eps)
        return mean, var


class StatsBank:
    """
    key(레이어 식별자) -> 통계 dict
    - aggregate: key -> {"mean": (r,), "var": (r,)}
    - distribution: key -> {"means": [ (r,), ... ], "vars": [ (r,), ... ]}
    """
    def __init__(self, mode: StatsMode, r: int, device: Optional[torch.device] = None):
        self.mode = mode
        self.r = r
        self.device = device
        self._agg: Dict[str, RunningStats] = {}
        self._dist: Dict[str, Dict[str, List[torch.Tensor]]] = {}

    def _ensure_key(self, key: str):
        if self.mode == "aggregate":
            if key not in self._agg:
                self._agg[key] = RunningStats(self.r, device=self.device)
        else:
            if key not in self._dist:
                self._dist[key] = {"means": [], "vars": []}

    @torch.no_grad()
    def update_from_batch(self, key: str, z: torch.Tensor):
        """
        z: (B,L,r) - lora_A 출력
        """
        self._ensure_key(key)

        if self.mode == "aggregate":
            self._agg[key].update(z)
            return

        # distribution: 배치별 mean/var 저장
        z2 = z.detach().reshape(-1, z.size(-1))  # (B*L, r)
        mean = z2.mean(dim=0)
        var = z2.var(dim=0, unbiased=False).clamp_min(1e-12)
        self._dist[key]["means"].append(mean)
        self._dist[key]["vars"].append(var)

    @torch.no_grad()
    def finalize(self) -> Dict[str, Any]:
        if self.mode == "aggregate":
            out: Dict[str, Dict[str, torch.Tensor]] = {}
            for k, rs in self._agg.items():
                mean, var = rs.finalize()
                out[k] = {"mean": mean, "var": var}
            return out
        else:
            return self._dist

    def to_payload(self) -> Dict[str, Any]:
        return {"mode": self.mode, "r": self.r, "data": self.finalize()}

    def save(self, path: str):
        torch.save(self.to_payload(), path)

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> Dict[str, Any]:
        return torch.load(path, map_location=map_location)
    
class StatsCollector(nn.Module):
    """
    lora_A 출력 z=A(x)을 StatsBank에 기록하고 z는 그대로 반환.
    """
    def __init__(self, a_linear: nn.Module, bank, key: str):
        super().__init__()
        self.a_linear = a_linear
        self.bank = bank
        self.key = key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.a_linear(x)
        # 방어: 일부 구현에서 (B,r)로 나올 수 있으므로 (B,1,r)로 맞춰 bank에 기록
        if z.dim() == 2:
            z_ = z.unsqueeze(1)     # (B,1,r)
        else:
            z_ = z                  # (B,L,r)
        self.bank.update_from_batch(self.key, z_)
        return z


class PEFTAdaIN(nn.Module):
    """
    z = A(x) (B,L,r)를 AdaIN처럼 변환:
      z_hat = (z - mu_c) / sigma_c * sigma_t + mu_t

    - mu_c, sigma_c: 현재 입력(content)에서 계산
      * instance_wise=True: 샘플별로 L축 통계 (B,1,r)
      * instance_wise=False: 배치 전체(B*L) 통계 (1,1,r)

    - mu_t, sigma_t: Dataset A에서 저장한 target stats
      * aggregate: key -> {"mean": (r,), "var": (r,)}
      * distribution: key -> {"means": [...], "vars": [...]}
        - selection:
          - mean_of_dist: means/vars 평균
          - random: 랜덤 선택
          - cycle: 순환 선택
    """
    def __init__(
        self,
        a_linear: nn.Module,
        stats_payload: Dict[str, Any],
        key: str,
        style_mode: StatsMode,
        selection: SelectMode = "mean_of_dist",
        seed: int = 0,
        eps: float = 1e-6,
        instance_wise: bool = True,
    ):
        super().__init__()
        self.a_linear = a_linear
        self.key = key
        self.eps = eps
        self.instance_wise = instance_wise

        # payload: {"mode": "...", "r": 8, "data": {...}}
        if stats_payload.get("mode") != style_mode:
            raise ValueError(f"payload mode({stats_payload.get('mode')}) != style_mode({style_mode})")

        self.style_mode = style_mode
        self.selection = selection
        self.data = stats_payload["data"]

        self._rng = random.Random(seed)
        self._cycle_idx = 0

    @torch.no_grad()
    def _get_target_stats(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.data[self.key]

        if self.style_mode == "aggregate":
            mu_t = entry["mean"].to(device)
            var_t = entry["var"].to(device).clamp_min(1e-12)
            return mu_t, var_t

        # distribution
        means: List[torch.Tensor] = entry["means"]
        vars_: List[torch.Tensor] = entry["vars"]
        if len(means) == 0:
            raise RuntimeError(f"[{self.key}] distribution stats empty")

        if self.selection == "mean_of_dist":
            mu_t = torch.stack([m.to(device) for m in means], dim=0).mean(dim=0)
            var_t = torch.stack([v.to(device) for v in vars_], dim=0).mean(dim=0).clamp_min(1e-12)
            return mu_t, var_t

        if self.selection == "random":
            i = self._rng.randrange(len(means))
            return means[i].to(device), vars_[i].to(device).clamp_min(1e-12)

        # cycle
        i = self._cycle_idx % len(means)
        self._cycle_idx += 1
        return means[i].to(device), vars_[i].to(device).clamp_min(1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.a_linear(x)  # (B,L,r) or (B,r)

        squeeze_back = False
        if z.dim() == 2:
            z = z.unsqueeze(1)  # (B,1,r)
            squeeze_back = True

        device = z.device
        mu_t, var_t = self._get_target_stats(device)
        sigma_t = torch.sqrt(var_t + self.eps).view(1, 1, -1)
        mu_t = mu_t.view(1, 1, -1)

        # content stats
        if self.instance_wise:
            # per-sample stats over L: (B,1,r)
            mu_c = z.mean(dim=1, keepdim=True)
            var_c = z.var(dim=1, keepdim=True, unbiased=False).clamp_min(1e-12)
        else:
            # global batch stats over (B*L): (1,1,r)
            z2 = z.reshape(-1, z.size(-1))
            mu_c = z2.mean(dim=0).view(1, 1, -1)
            var_c = z2.var(dim=0, unbiased=False).view(1, 1, -1).clamp_min(1e-12)

        sigma_c = torch.sqrt(var_c + self.eps)

        z_hat = (z - mu_c) / sigma_c * sigma_t + mu_t

        if squeeze_back:
            z_hat = z_hat.squeeze(1)  # (B,r)
        return z_hat

def wrap_lora_A_inplace(
    peft_model: nn.Module,
    wrapper_factory,
    adapter_name: str = "default",
) -> int:
    replaced = 0
    for module_path, module in peft_model.named_modules():
        if not hasattr(module, "lora_A"):
            continue

        lora_A = getattr(module, "lora_A")

        if isinstance(lora_A, nn.ModuleDict) and adapter_name in lora_A:
            a_mod = lora_A[adapter_name]
            if isinstance(a_mod, (StatsCollector, PEFTAdaIN)):
                continue
            key = f"{module_path}.lora_A[{adapter_name}]"
            lora_A[adapter_name] = wrapper_factory(a_mod, key)
            replaced += 1

        elif isinstance(lora_A, dict) and adapter_name in lora_A:
            a_mod = lora_A[adapter_name]
            if isinstance(a_mod, (StatsCollector, PEFTAdaIN)):
                continue
            key = f"{module_path}.lora_A[{adapter_name}]"
            lora_A[adapter_name] = wrapper_factory(a_mod, key)
            replaced += 1

    return replaced


@torch.no_grad()
def collect_loraA_stats(
    model: nn.Module,
    dataloaderA,
    r: int = 8,
    mode: StatsMode = "aggregate",
    adapter_name: str = "default",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    bank = StatsBank(mode=mode, r=r, device=device)

    def factory(a_mod, key):
        return StatsCollector(a_mod, bank, key)

    replaced = wrap_lora_A_inplace(model, factory, adapter_name=adapter_name)
    print(f"[INFO] Wrapped lora_A for stats collection: {replaced}")

    for batch in dataloaderA:
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(**batch)

    return bank.to_payload()


def apply_adain_transform_to_loraA_for_inference(
    model: nn.Module,
    stats_payload: Dict[str, Any],
    adapter_name: str = "default",
    selection: SelectMode = "mean_of_dist",
    seed: int = 0,
    instance_wise: bool = True,
) -> int:
    style_mode: StatsMode = stats_payload["mode"]

    def factory(a_mod, key):
        return PEFTAdaIN(
            a_linear=a_mod,
            stats_payload=stats_payload,
            key=key,
            style_mode=style_mode,
            selection=selection,
            seed=seed,
            instance_wise=instance_wise,
        )

    replaced = wrap_lora_A_inplace(model, factory, adapter_name=adapter_name)
    print(f"[INFO] Wrapped lora_A with AdaIN transform: {replaced}")
    return replaced