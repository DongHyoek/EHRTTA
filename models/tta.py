from typing import Dict, List, Tuple, Optional, Literal, Any
import random
import torch
import torch.nn as nn

class RunningStats:
    """
    token-wise stats: z shape (B, L, r)
    mean/var 계산 축: B  -> 결과 (L, r)
    """
    def __init__(self, r: int, device: Optional[torch.device] = None):
        self.r = r
        self.device = device
        self.count = 0
        self.sum = None     # (L,r)로 첫 update 때 초기화
        self.sumsq = None   # (L,r)

    @torch.no_grad()
    def update(self, z: torch.Tensor):
        # z: (B,L,r)
        z2 = z.detach()
        L = z2.size(1)

        if self.sum is None:
            self.sum = torch.zeros(L, self.r, device=z2.device)
            self.sumsq = torch.zeros(L, self.r, device=z2.device)

        self.sum += z2.sum(dim=0)          # (L,r)
        self.sumsq += (z2 ** 2).sum(dim=0) # (L,r)
        self.count += z2.size(0)           # B 누적

    @torch.no_grad()
    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-12
        denom = max(self.count, 1)
        mean = self.sum / denom
        var = (self.sumsq / denom) - mean**2
        var = torch.clamp(var, min=eps)
        return mean, var  # (L,r), (L,r)
    
class StatsBank:
    """
    두 종류 존재 : 각 토큰별 r차원별 단일 스칼라값 , 각 토큰별 차원별 distribution의 분포 
    key : 각각의 PEFT layer  
    - aggregate: key -> {"mean": (L,r), "var": (L,r)}
    - distribution: key -> {"means": [ (L,r), ... ], "vars": [ (L,r), ... ]}
    """
    def __init__(self, mode: str, r: int, device: Optional[torch.device] = None):
        self.mode = mode
        self.r = r
        self.device = device
        self._agg = {}
        self._dist = {}

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

        # scalar values 사용
        if self.mode == "aggregate":
            self._agg[key].update(z)
            return

        else: # distribution: 배치별 mean/var 저장
            z2 = z.detach()  # (B, L, r)
            mean = z2.mean(dim=0)
            var = z2.var(dim=0, unbiased=False).clamp_min(1e-12)
            self._dist[key]["means"].append(mean) # (L, r)
            self._dist[key]["vars"].append(var)   # (L,r)

    @torch.no_grad()
    def finalize(self) -> Dict[str, Any]:
        if self.mode == "aggregate":
            out = {}

            for k, run_stats in self._agg.items():
                mean, var = run_stats.finalize()
                out[k] = {"mean": mean, "var": var}
            return out
        
        else:
            return self._dist

    def to_payload(self) -> Dict[str, Any]:
        return {"mode": self.mode, "r": self.r, "data": self.finalize()}

    def save(self, path: str):
        torch.save(self.to_payload(), path)

    @staticmethod # 클래스 안에 소속이 되어 있으나, 클래스 내부의 변수를 쓰거나 수정할 일이 없는 독립적인 함수를 나타낼 때 사용
    def load(path: str, map_location: str = "cpu") -> Dict[str, Any]:
        return torch.load(path, map_location=map_location)
    
class StatsCollector(nn.Module):
    """
    역할 : lora_A 출력 z=A(x)을 StatsBank에 기록하고 z는 그대로 반환.
    """
    def __init__(self, peft_layer: nn.Module, bank, key: str):
        super().__init__()
        self.peft_layer = peft_layer
        self.bank = bank
        self.key = key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.peft_layer(x)

        # 혹시라도 (B,r)로 나올 수 있으므로 (B,1,r)로 맞춰 bank에 기록
        if z.dim() == 2:
            z_ = z.unsqueeze(1)     # (B,1,r)

        else:
            z_ = z                  # (B,L,r)

        self.bank.update_from_batch(self.key, z_)

        return z

class AdaIN(nn.Module):
    """
    z = A(x) (B,L,r)를 AdaIN처럼 변환:
      z_hat = (z - mu_c) / sigma_c * sigma_t + mu_t

    - mu_c, sigma_c: 현재 입력(content)에서 계산

    - mu_t, sigma_t: Dataset A에서 저장한 target stats
      * aggregate: key -> {"mean": (r,), "var": (r,)}
      * distribution: key -> {"means": [...], "vars": [...]}
        - selection:
          - mean_of_dist: means/vars 평균
          - random: 랜덤 선택
          - cycle: 순환 선택
    """
    def __init__(self, peft_layer: nn.Module, stats_payload: Dict[str, Any], key: str, 
                 agg_mode: str, selection: str, seed: int = 0, eps: float = 1e-6):
        
        super().__init__()
        self.peft_layer = peft_layer
        self.key = key
        self.eps = eps

        # payload: {"mode": "...", "r": ..., "data": {...}}
        if stats_payload.get("mode") != agg_mode:
            raise ValueError(f"payload mode({stats_payload.get('mode')}) != agg_mode({agg_mode})")

        self.agg_mode = agg_mode
        self.selection = selection
        self.data = stats_payload["data"]

        self._rng = random.Random(seed)
        self._cycle_idx = 0

    @torch.no_grad()
    def _get_target_stats(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.data[self.key]

        # use stats scalar values 
        if self.agg_mode == "aggregate":
            mu_t = entry["mean"].to(device)
            var_t = entry["var"].to(device).clamp_min(1e-12)
            return mu_t, var_t
        
        else: # use stats distribution
            means = entry["means"]
            vars_ = entry["vars"]

            if len(means) == 0:
                raise RuntimeError(f"[{self.key}] distribution stats empty")

            if self.selection == "mean_of_dist":
                mu_t = torch.stack([m.to(device) for m in means], dim=0).mean(dim=0)
                var_t = torch.stack([v.to(device) for v in vars_], dim=0).mean(dim=0).clamp_min(1e-12)
                return mu_t, var_t

            # random
            if self.selection == "random":
                i = self._rng.randrange(len(means))
                return means[i].to(device), vars_[i].to(device).clamp_min(1e-12)

            # cycle
            i = self._cycle_idx % len(means)
            self._cycle_idx += 1
            return means[i].to(device), vars_[i].to(device).clamp_min(1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.peft_layer(x)  # (B,L,r) or (B,r)

        squeeze_back = False
        if z.dim() == 2:
            z = z.unsqueeze(1)  # (B,1,r)
            squeeze_back = True

        device = z.device
        mu_t, var_t = self._get_target_stats(device)
        sigma_t = torch.sqrt(var_t + self.eps).unsqueeze(0) # (1, L, r)
        mu_t = mu_t.unsqueeze(0) # (1, L, r)

        # content stats (1,L,r)
        mu_c = z.mean(dim=0, keepdim=True)
        var_c = z.var(dim=0, keepdim=True, unbiased=False).clamp_min(1e-12)

        sigma_c = torch.sqrt(var_c + self.eps)

        z_hat = (z - mu_c) / sigma_c * sigma_t + mu_t

        if squeeze_back:
            z_hat = z_hat.squeeze(1)  # (B,r)

        return z_hat

class PEFTAdaINPatcher:
    """
    PEFT 모델(self.backbone) 내부의 각 LoRA A projection(lora_A[adapter_name])을
    목적에 따라 기능들을 추가하는 클래스

    - collect():
        (학습이 끝난 모델을 고정한 상태에서) Source dataset을 다시 forward 하여,
        각 레이어의 lora_A 출력 z = A(x) (shape: B,L,r)에 대해
        'Batch 축'으로 token-position-wise 통계(mean/var)를 계산해 저장
        -> 레이어별로 (L,r) mean/var (또는 배치별 분포)를 담은 stats_payload(dict)를 반환

    - apply_adain():
        collect()로 얻은 stats_payload를 참조하는 AdaIN을 각 lora_A에 삽입
        이후 Target dataset(B)에서 forward 시마다 z=A(x)를
        content stats(현재 배치에서 계산)와 target stats(stats_payload)로 AdaIN 방식으로 변환한다.

    - restore():
        in-place 교체되기 전의 원래 lora_A 모듈을 복구
        (예: collect() 이후 stats 수집하는 부분을 제거하고 원상 복귀할 때 사용)
    """

    def __init__(self, peft_model: nn.Module, adapter_name: str = "default"):
        
        self.model = peft_model
        self.adapter_name = adapter_name

        # 원복용 저장소: key -> original a_mod
        self._original_lora_A: Dict[str, nn.Module] = {}

    def _iter_loraA(self):
        """
        backbone.named_modules()를 돌면서 lora_A[adapter_name]를 찾는다.
        yield 안 쓰고, 내부에서 바로 리스트로 모아 반환.
        """
        found = []
        name = self.adapter_name

        for module_path, module in self.model.named_modules():
            if not hasattr(module, "lora_A"):
                continue

            lora_A_mdc = getattr(module, "lora_A")

            if isinstance(lora_A_mdc, (nn.ModuleDict, dict)) and name in lora_A_mdc:
                key = f"{module_path}.lora_A[{name}]"
                found.append((key, lora_A_mdc, lora_A_mdc[name]))  # (identifier key, lora_A module dict, lora_A)

        return found

    def _attach_all(self, attach_cls, attach_kwargs: Dict[str, Any], skip_types: Tuple[type, ...] = ()):
        """
        모든 lora_A에 attach_cls로 목적에 맞게 AdaIN이나 StatsCollector를 붙임.
        attach_cls 생성자는 다음 형태를 기대:
          - StatsCollector(peft_layer, bank, key)
          - AdaIN(peft_layer, stats_payload, key, agg_mode, selection, seed, eps)
        """
        replaced = 0

        for key, container, lora_A in self._iter_loraA():
            if skip_types and isinstance(lora_A, skip_types):
                continue

            # 원본 저장(한 번만)
            if key not in self._original_lora_A:
                self._original_lora_A[key] = lora_A

            # attacher 생성
            attached_lora_a = attach_cls(peft_layer=lora_A, key=key, **attach_kwargs)
            container[self.adapter_name] = attached_lora_a
            replaced += 1

        return replaced

    def restore(self):
        """
        attached lora_A를 원래 lora_A로 되돌림.
        """
        restored = 0
        name = self.adapter_name

        for key, container, current in self._iter_loraA():
            if key in self._original_lora_A:
                container[name] = self._original_lora_A[key]
                restored += 1

        return restored

    @torch.no_grad()
    def collect(self, dataloader, r: int = 8, mode: str = "aggregate", device = None) -> Dict[str, Any]:
        """
        1) StatsCollector로 lora_A를 통과하고 나서 얻는 통계량들을 수집하도록 함.
        2) dataloader forward
        3) StatsBank payload 반환 (각 layer마다의 통계량들이 들어있는 값)

        이때, token-position-wise stats: 각 토큰 위치 l마다 r차원 평균/분산을 B축으로 계산하여 (L,r) 저장
        """
        self.model.eval()

        if device is None:
            device = next(self.model.parameters()).device

        bank = StatsBank(mode=mode, r=r, device=device)

        # StatsCollector -> attach_kwargs로 bank 전달
        replaced = self._attach_all(attach_cls=StatsCollector, 
                                    attach_kwargs={"bank": bank}, 
                                    skip_types=(StatsCollector, AdaIN))
        
        print(f"[INFO] attached lora_A for stats collection: {replaced}")

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = self.model(**batch)

        return bank.to_payload()

    def apply_adain(self, stats_payload: Dict[str, Any], *, selection: str = "mean_of_dist", 
                    seed: int = 0, eps: float = 1e-6, ) -> int:
        """
        stats_payload 기반으로 AdaIN attacher를 모든 lora_A에 적용.
        => Target Data에서 현재 들어온 미니 배치 단위에서의 통계량들을 계산하고 AdaIN style로 adaptation하는 구조
        """
        agg_mode = stats_payload["mode"]

        replaced = self._attach_all(
            attach_cls=AdaIN,
            attach_kwargs={"stats_payload": stats_payload, "agg_mode": agg_mode, 
                           "selection": selection, "seed": seed, "eps": eps}, # AdaIN이 이 인자를 받도록 해두면 좋음
            skip_types=(StatsCollector, AdaIN)
            )
        
        print(f"[INFO] attached lora_A with AdaIN transform: {replaced}")

        return replaced