import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal, Any

import torch
import torch.nn as nn


# -------------------------
# (A) 통계 누적용 (aggregate 모드)
# -------------------------
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
        # z: (B,L,r)
        z2 = z.detach()
        z2 = z2.reshape(-1, z2.size(-1))  # (B*L, r)

        self.sum += z2.sum(dim=0)
        self.sumsq += (z2 ** 2).sum(dim=0)
        self.count += z2.size(0)

    @torch.no_grad()
    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        반환: (mean[r], var[r])  (population var)
        """
        eps = 1e-12
        denom = max(self.count, 1)
        mean = self.sum / denom
        var = (self.sumsq / denom) - mean**2
        var = torch.clamp(var, min=eps)
        return mean, var


# -------------------------
# (B) 레이어별로 통계 저장하는 Bank
# -------------------------
StatsMode = Literal["aggregate", "distribution"]

@dataclass
class LayerStatsAggregate:
    mean: torch.Tensor  # (r,)
    var: torch.Tensor   # (r,)

@dataclass
class LayerStatsDistribution:
    means: List[torch.Tensor]  # list of (r,)
    vars: List[torch.Tensor]   # list of (r,)

class StatsBank:
    """
    key(레이어 식별자) -> 통계
    - aggregate: 전체(A 데이터셋 전체)로 누적해 mean/var 1개
    - distribution: 미니배치마다 mean/var를 list로 저장 (분포처럼)
    """
    def __init__(self, mode: StatsMode, r: int, device: Optional[torch.device] = None):
        self.mode = mode
        self.r = r
        self.device = device
        self._agg: Dict[str, RunningStats] = {}
        self._dist: Dict[str, LayerStatsDistribution] = {}

    def _ensure_key(self, key: str):
        if self.mode == "aggregate":
            if key not in self._agg:
                self._agg[key] = RunningStats(self.r, device=self.device)
        else:
            if key not in self._dist:
                self._dist[key] = LayerStatsDistribution(means=[], vars=[])

    @torch.no_grad()
    def update_from_batch(self, key: str, z: torch.Tensor):
        """
        z: (B,L,r) - lora_A 출력
        """
        self._ensure_key(key)
        if self.mode == "aggregate":
            self._agg[key].update(z)
        else:
            # 배치 통계: B*L 축으로 채널별 mean/var
            z2 = z.detach().reshape(-1, z.size(-1))
            mean = z2.mean(dim=0)
            var = z2.var(dim=0, unbiased=False).clamp_min(1e-12)
            self._dist[key].means.append(mean)
            self._dist[key].vars.append(var)

    @torch.no_grad()
    def finalize(self) -> Dict[str, Any]:
        """
        mode에 따라 정리된 dict 반환.
        aggregate -> key -> {mean, var}
        distribution -> key -> {means(list), vars(list)}
        """
        if self.mode == "aggregate":
            out: Dict[str, LayerStatsAggregate] = {}
            for k, rs in self._agg.items():
                mean, var = rs.finalize()
                out[k] = LayerStatsAggregate(mean=mean, var=var)
            return out
        else:
            return self._dist

    def save(self, path: str):
        payload = {
            "mode": self.mode,
            "r": self.r,
            "data": self.finalize(),
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> Dict[str, Any]:
        payload = torch.load(path, map_location=map_location)
        return payload  # {"mode": ..., "r": ..., "data": ...}


# -------------------------
# (C) lora_A 래퍼들
# -------------------------
class LoraAStatCollector(nn.Module):
    """
    lora_A(Linear) 감싸서 z=A(x) 출력 통계를 StatsBank에 적재하고,
    출력은 그대로 반환(변형 없음).
    """
    def __init__(self, a_linear: nn.Module, bank: StatsBank, key: str):
        super().__init__()
        self.a_linear = a_linear
        self.bank = bank
        self.key = key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.a_linear(x)  # (B,L,r) 가 보통
        # 일부 구현에서 (B,r)로 나올 수도 있어 방어
        if z.dim() == 2:
            z_ = z.unsqueeze(1)  # (B,1,r)
        else:
            z_ = z
        self.bank.update_from_batch(self.key, z_)
        return z


class LoraAAdaIN(nn.Module):
    """
    lora_A 출력 z를 AdaIN처럼 변환:
      z_hat = (z - mu_c) / sigma_c * sigma_t + mu_t
    - mu_c, sigma_c: 현재 배치(또는 sample)에서 계산
    - mu_t, sigma_t: Dataset A에서 수집한 target stats
    """
    def __init__(
        self,
        a_linear: nn.Module,
        target_payload: Dict[str, Any],
        key: str,
        style_mode: Literal["aggregate", "distribution"] = "aggregate",
        selection: Literal["mean_of_dist", "random", "cycle"] = "mean_of_dist",
        seed: int = 0,
        eps: float = 1e-6,
        instance_wise: bool = True,  # AdaIN 느낌(샘플별)로 할지, 배치 전체로 할지
    ):
        super().__init__()
        self.a_linear = a_linear
        self.key = key
        self.style_mode = style_mode
        self.selection = selection
        self.eps = eps
        self.instance_wise = instance_wise

        self._rng = random.Random(seed)
        self._cycle_idx = 0

        # payload 구조:
        # target_payload = {"mode":..., "r":..., "data":...}
        self._data = target_payload["data"]
        self._payload_mode = target_payload["mode"]

        if self.style_mode != self._payload_mode:
            raise ValueError(
                f"style_mode({self.style_mode})와 payload mode({self._payload_mode})가 다릅니다."
            )

    @torch.no_grad()
    def _get_target_stats(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.style_mode == "aggregate":
            entry: LayerStatsAggregate = self._data[self.key]
            mu_t = entry.mean.to(device)
            var_t = entry.var.to(device)
            return mu_t, var_t

        # distribution
        entry: LayerStatsDistribution = self._data[self.key]
        means, vars_ = entry.means, entry.vars

        if len(means) == 0:
            raise RuntimeError(f"[{self.key}] distribution stats가 비어 있습니다.")

        if self.selection == "mean_of_dist":
            mu_t = torch.stack([m.to(device) for m in means], dim=0).mean(dim=0)
            var_t = torch.stack([v.to(device) for v in vars_], dim=0).mean(dim=0)
            return mu_t, var_t

        if self.selection == "random":
            i = self._rng.randrange(len(means))
            return means[i].to(device), vars_[i].to(device)

        # cycle
        i = self._cycle_idx % len(means)
        self._cycle_idx += 1
        return means[i].to(device), vars_[i].to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.a_linear(x)  # (B,L,r) 또는 (B,r)
        if z.dim() == 2:
            z = z.unsqueeze(1)  # (B,1,r)

        device = z.device
        mu_t, var_t = self._get_target_stats(device)
        sigma_t = torch.sqrt(var_t + self.eps)  # (r,)

        # content stats (현재 입력에서)
        if self.instance_wise:
            # 샘플별로 L축 평균/분산: (B,1,r)
            mu_c = z.mean(dim=1, keepdim=True)
            var_c = z.var(dim=1, keepdim=True, unbiased=False).clamp_min(1e-12)
        else:
            # 배치 전체(B*L) 기준 채널별 stats: (1,1,r)
            z2 = z.reshape(-1, z.size(-1))
            mu_c = z2.mean(dim=0).view(1, 1, -1)
            var_c = z2.var(dim=0, unbiased=False).view(1, 1, -1).clamp_min(1e-12)

        sigma_c = torch.sqrt(var_c + self.eps)

        # broadcast: (r,) -> (1,1,r)
        mu_t = mu_t.view(1, 1, -1)
        sigma_t = sigma_t.view(1, 1, -1)

        z_hat = (z - mu_c) / sigma_c * sigma_t + mu_t

        # 원래가 (B,r)였으면 다시 squeeze
        if z_hat.size(1) == 1 and x.dim() == 2:
            z_hat = z_hat.squeeze(1)
        return z_hat