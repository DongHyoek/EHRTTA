import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import sys

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set, Any
from torch.nn.utils import weight_norm

import torch
import torch.nn as nn
from typing import Optional, Tuple


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "gelu"):

        super(FeedForward, self).__init__()

        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttnBlock(nn.Module):
    """
    Pre-LN Cross-Attention block:
      ts = ts + g * CrossAttn(LN(ts) as Q, LN(text) as K,V)
      ts = ts + FFN(LN(ts))

    Shapes (batch_first):
      ts   : (B, M, d)
      text : (B, N, d)
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None, dropout: float = 0.0, use_gating: bool = True, gate_init: float = 0.1):
        super(CrossAttnBlock, self).__init__()

        if d_ff is None:
            d_ff = 2 * d_model # e.g. 4096 * 2 = 8192

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)

        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.use_gating = use_gating
        if use_gating:
            # Scalar gate is simple and stable; you can switch to (d_model,) gate if you want.
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, ts: torch.Tensor, text: torch.Tensor, text_key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
          ts:  (B, M, d)
          text:(B, N, d)
          text_key_padding_mask: (B, N) bool, True for positions to mask (padding)
        Returns:
          ts_out: (B, M, d) / it changes to out : (B, N, d) 
          attn_weights: (B, M, N) if need_weights else None
        """
        q = self.ln_q(ts) # layer noramlization
        kv = self.ln_kv(text) # layer normalization

        # Cross-attn: Q from ts, K/V from text
        attn_out, attn_w = self.attn(query=q, key=kv, value=kv, key_padding_mask=text_key_padding_mask)  # masks K/V side need_weights=need_weights, average_attn_weights=False if need_weights else True,  # keep per-head if need_weights=False doesn't matter
        attn_out = self.attn_drop(attn_out)

        if self.use_gating:
            out = ts + self.gate * attn_out
        else:
            out = ts + attn_out

        # FFN
        out = out + self.ffn(self.ln_ffn(out))
        
        # 일단, Head별로 attention weight을 내보내도록 수정
        
        # # If need_weights and average_attn_weights=False, attn_w is (B, num_heads, M, N).
        # # You can average over heads here if you prefer.
        # if need_weights and attn_w is not None and attn_w.dim() == 4:
        #     attn_w = attn_w.mean(dim=1)  # -> (B, M, N)

        return out, attn_w if need_weights else None


class ModalityAlignment(nn.Module):
    """
    2-layer stack of CrossAttnBlock.
    Q = time-series tokens, K/V = text tokens.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None, 
                 dropout: float = 0.0, use_gating: bool = True, gate_init: float = 0.1):

        super(ModalityAlignment, self).__init__()

        self.block1 = CrossAttnBlock(d_model, n_heads, d_ff, dropout, use_gating, gate_init)
        self.block2 = CrossAttnBlock(d_model, n_heads, d_ff, dropout, use_gating, gate_init)

    def forward(self, ts: torch.Tensor, text: torch.Tensor, text_key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        text_key_padding_mask = (text_key_padding_mask == 0)  # bool tensor (B, N)

        ts, w1 = self.block1(ts, text, text_key_padding_mask=text_key_padding_mask, need_weights=need_weights)
        ts, w2 = self.block2(ts, text, text_key_padding_mask=text_key_padding_mask, need_weights=need_weights)

        # return last weights (or you can return both)
        return ts, w2 if need_weights else None


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    B, M, N, d = 32, 128, 256, 4096 # M : time series length, N : tokenized text length
    n_heads = 8

    ts = torch.randn(B, M, d)      # time-series tokens
    text = torch.randn(B, N, d)    # text tokens (e.g., from a text encoder)

    # Mask: True indicates padding positions in text (K/V) to ignore
    text_mask = torch.zeros(B, N, dtype=torch.bool)
    text_mask[:, -10:] = True  # pretend last 10 tokens are padding

    model = ModalityAlignment(d_model=d, n_heads=n_heads, dropout=0.1, use_gating=True)

    ts_enriched, attn_weights = model(ts, text, text_key_padding_mask=text_mask, need_weights=True)
    print(ts_enriched.shape)   # (B, M, d)
    print(attn_weights.shape)  # (B, M, N)