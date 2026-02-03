import torch
from typing import *


class TSScaler:
    """
    Feature-wise scaler for multivariate time series x: (B, L, D)

    - Source phase: fit() computes per-feature stats on source train set
    - Adaptation phase: transform_adapt() uses current mini-batch stats (or EMA) for scaling

    Missing values:
    - You can pass a boolean mask (B, L, D): True where observed
    - If mask is None, we infer observed = isfinite(x)
    - If a feature has zero observed values in a batch, we fallback to source (or running) stats
    """

    def __init__(self, args, eps: float = 1e-8, min_std: float = 1e-8):
        self.eps = eps
        self.min_std = min_std
        self.use_norm_ema = args.use_norm_ema
        self.norm_ema_alpha = args.norm_ema_alpha
        self.device = args.device

        self.source_state = None
        self.running_state = None  # for EMA

    @staticmethod
    def _infer_mask(x: torch.Tensor) -> torch.Tensor:
        # observed where finite (not NaN/inf)
        return torch.isfinite(x)

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

    def fit_source(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Fit source train stats.
        x: (B, L, D) - you can call this on concatenated batches too.
        """
        if self.device is not None:
            x = x.to(self.device)

        if mask is None:
            mask = self._infer_mask(x)
        else:
            mask = mask.to(dtype=torch.bool, device=x.device)

        state = self._masked_mean_std(x, mask)
        self.source_state = state

        # initialize running state if EMA is enabled
        if self.use_norm_ema:
            self.running_state = {'mean': state['mean'].clone(), 'std':  state['std'].clone(), 'count': state['count'].clone()}

    def transform_source(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Scale using source stats.
        """
        assert self.source_state is not None, "Call fit_source() first."
        
        if self.device is not None:
            x = x.to(self.device)
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
        
        if self.device is not None:
            x = x.to(self.device)
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