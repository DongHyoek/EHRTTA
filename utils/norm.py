import torch
from typing import *


class TSScaler:
    """
    Feature-wise scaler for multivariate time series x: (B, D, L)

    - Source phase: fit() computes per-feature stats on source train set
    - Adaptation phase: transform_adapt() uses current mini-batch stats (or EMA) for scaling

    Missing values:
    - You can pass a boolean mask (B, D, L): True where observed
    - If mask is None, we infer observed = isfinite(x)
    - If a feature has zero observed values in a batch, we fallback to source (or running) stats
    """

    def __init__(self, args, eps: float = 1e-8, min_std: float = 1e-8):
        self.eps = eps
        self.min_std = min_std
        self.use_norm_ema = args.use_norm_ema
        self.norm_ema_alpha = args.norm_ema_alpha

        self.source_state = None
        self.running_state = None  # for EMA

    def _masked_mean_std(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Compute per-feature mean/std using only observed values.
        x, mask: (B, D, L) with mask boolean
        """
        B, D, L = x.shape

        # convert to (B, L, D) to reuse the same flattening logic (flatten B*L)
        x_bld = x.permute(0, 2, 1).contiguous()     # (B, L, D)
        m_bld = mask.permute(0, 2, 1).contiguous()  # (B, L, D)

        x_ = x_bld.reshape(-1, D)
        m_ = m_bld.reshape(-1, D)

        # counts per feature
        count = m_.sum(dim=0).to(x.dtype)  # (D,)

        # sum and mean
        # avoid NaNs: replace unobserved with 0 before summing
        x_obs = torch.where(m_, x_, torch.zeros_like(x_))
        s = x_obs.sum(dim=0)  # (D,)

        # mean: only valid where count>0
        mean = torch.zeros(D, device=x.device, dtype=x.dtype)
        nonzero = count > 0
        mean[nonzero] = s[nonzero] / count[nonzero]

        # variance
        # (x - mean)^2 only for observed entries
        diff = x_ - mean.unsqueeze(0)
        diff2 = torch.where(m_, diff * diff, torch.zeros_like(diff))
        var = torch.zeros(D, device=x.device, dtype=x.dtype)
        var[nonzero] = diff2.sum(dim=0)[nonzero] / count[nonzero]

        std = torch.sqrt(var + self.eps)
        std = torch.clamp(std, min=self.min_std)

        return {'mean' : mean, 'std' : std, 'count' : count}
    
    @staticmethod
    def _infer_mask(x: torch.Tensor) -> torch.Tensor:
        # observed where finite (not NaN/inf)
        return torch.isfinite(x)
    
    @torch.no_grad()
    def reset_source_accum(self):
        self._src_sum = None
        self._src_sumsq = None
        self._src_count = None

    @torch.no_grad()
    def update_source(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Accumulate per-feature sum/sumsq/count from a batch.
        x: (B, D, L)
        mask: (B, D, L) bool (True=observed)
        """
        if mask is None:
            mask = self._infer_mask(x)
        else:
            mask = mask.to(dtype=torch.bool, device=x.device)

        # flatten observed entries across B*L per feature
        # x,mask: (B,D,L) -> (B,L,D) -> (B*L, D)
        x_ = x.permute(0, 2, 1).reshape(-1, x.shape[1])      # (BL, D)
        m_ = mask.permute(0, 2, 1).reshape(-1, x.shape[1])   # (BL, D)

        x_obs = torch.where(m_, x_, torch.zeros_like(x_))    # unobs -> 0
        batch_sum = x_obs.sum(dim=0)                         # (D,)
        batch_sumsq = (x_obs * x_obs).sum(dim=0)             # (D,)
        batch_count = m_.sum(dim=0).to(x.dtype)              # (D,)

        if self._src_sum is None:
            self._src_sum = batch_sum.detach().clone()
            self._src_sumsq = batch_sumsq.detach().clone()
            self._src_count = batch_count.detach().clone()
        else:
            self._src_sum += batch_sum
            self._src_sumsq += batch_sumsq
            self._src_count += batch_count

    @torch.no_grad()
    def finalize_source(self):
        """
        Convert accumulated sum/sumsq/count into mean/std and set source_state.
        """
        assert self._src_sum is not None, "Call partial_fit_source() at least once."

        count = self._src_count
        nonzero = count > 0

        mean = torch.zeros_like(self._src_sum)
        mean[nonzero] = self._src_sum[nonzero] / count[nonzero]

        # var = E[x^2] - (E[x])^2
        ex2 = torch.zeros_like(self._src_sumsq)
        ex2[nonzero] = self._src_sumsq[nonzero] / count[nonzero]
        var = torch.clamp(ex2 - mean * mean, min=0.0)
        std = torch.sqrt(var + self.eps)
        std = torch.clamp(std, min=self.min_std)

        self.source_state = {"mean": mean, "std": std, "count": count}
        
        return self.source_state

    def transform_source(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Scale using source stats.
        """
        assert self.source_state is not None, "Call fit_source() first."
        
        if mask is None:
            mask = self._infer_mask(x)
        else:
            mask = mask.to(dtype=torch.bool, device=x.device)

        mean = self.source_state['mean'].view(1, -1, 1)  # (1, D, 1)
        std  = self.source_state['std'].view(1, -1, 1)   # (1, D, 1)

        # replace missing with mean before scaling -> becomes 0 after scaling (then we zero-fill anyway)
        x_filled = torch.where(mask, x, mean.expand_as(x))
        x_scaled = (x_filled - mean) / std

        # final: set missing positions to 0 (common for Transformer inputs)
        x_scaled = torch.where(mask, x_scaled, torch.zeros_like(x_scaled))
        return x_scaled

    @torch.no_grad()
    def transform_target(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """
        Online adaptation scaling:
        - compute batch stats from current mini-batch (observed-only)
        - if use_norm_ema: update running mean/std with EMA, and scale using running stats
        - else: scale using current batch stats

        Returns:
          x_scaled, used_state
        """
        # 만약 target data에서 특정 변수의 값들이 모두 결측인 경우 정규화는 건너 뜀
        
        if mask is None:
            mask = self._infer_mask(x)
        else:
            mask = mask.to(dtype=torch.bool, device=x.device)

        batch_state = self._masked_mean_std(x, mask)
        nonzero = batch_state['count'] > 0  # (D,)

        # Handle features with all-missing in this batch
        D = x.shape[1]  # x is (B, D, L)

        mean = torch.zeros(D, device=x.device, dtype=x.dtype)     # default mean=0
        std  = torch.ones(D,  device=x.device, dtype=x.dtype)     # default std=1 (no-op)
        mean[nonzero] = batch_state['mean'][nonzero]
        std[nonzero] = batch_state['std'][nonzero]

        used_state = {'mean' : mean , 'std' : std, 'count' : batch_state['count']}

        if self.use_norm_ema:
            
            # If running state is None
            if self.running_state is None:
                self.running_state = {'mean': mean.clone(), 'std':  std.clone(), 'count': batch_state['count'].clone()}

            a = self.norm_ema_alpha

            # update running_state dict safely (also clone at init; see note below)
            self.running_state['mean'][nonzero] = (1 - a) * self.running_state['mean'][nonzero] + a * used_state['mean'][nonzero]
            self.running_state['std'][nonzero]  = (1 - a) * self.running_state['std'][nonzero]  + a * used_state['std'][nonzero]

            mean2 = self.running_state['mean'].view(1, -1, 1)  # (1, D, 1)
            std2  = self.running_state['std'].view(1, -1, 1)   # (1, D, 1)
        else:
            mean2 = used_state['mean'].view(1, -1, 1)
            std2  = used_state['std'].view(1, -1, 1)

        x_filled = torch.where(mask, x, mean2.expand_as(x))
        x_scaled = (x_filled - mean2) / std2
        x_scaled = torch.where(mask, x_scaled, torch.zeros_like(x_scaled))

        return x_scaled, (self.running_state if self.use_norm_ema else used_state)

    def state_dict(self):
        def pack_state(st):
            if st is None:
                return None
            return {
                "mean": st["mean"].detach().cpu(),
                "std": st["std"].detach().cpu(),
                "count": st["count"].detach().cpu(),
            }

        return {
            "eps": self.eps,
            "min_std": self.min_std,
            "use_norm_ema": self.use_norm_ema,
            "norm_ema_alpha": self.norm_ema_alpha,
            "source_state": pack_state(self.source_state),
            "running_state": pack_state(self.running_state),
        }

    def load_state_dict(self, sd, device=None, dtype=None):
        self.eps = float(sd.get("eps", self.eps))
        self.min_std = float(sd.get("min_std", self.min_std))
        self.use_norm_ema = bool(sd.get("use_norm_ema", self.use_norm_ema))
        self.norm_ema_alpha = float(sd.get("norm_ema_alpha", self.norm_ema_alpha))

        def unpack_state(st):
            if st is None:
                return None
            mean = st["mean"]
            std = st["std"]
            count = st["count"]
            if device is not None:
                mean = mean.to(device)
                std = std.to(device)
                count = count.to(device)
            if dtype is not None:
                mean = mean.to(dtype)
                std = std.to(dtype)
                count = count.to(dtype)
            return {"mean": mean, "std": std, "count": count}

        self.source_state = unpack_state(sd.get("source_state", None))
        self.running_state = unpack_state(sd.get("running_state", None))