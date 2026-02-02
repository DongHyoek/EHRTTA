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

## Below codes are sourced by https://github.com/usail-hkust/ISTS-PLM

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TimeEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(TimeEmbedding, self).__init__()
        # Compute the positional encodings once in log space.

        self.periodic = nn.Linear(1, d_model-1)
        self.linear = nn.Linear(1, 1)
    
    def learn_time_embedding(self, tt):
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, time):
        return self.learn_time_embedding(time)
    
class VariableEmbedding(nn.Module):
    def __init__(self, n_var, d_model):
        super(VariableEmbedding, self).__init__()
        self.varible_emb = nn.Embedding(n_var, d_model)

    def forward(self, x):
        x = self.varible_emb(x.long())
        return x

class TaskEmbedding(nn.Module):
    def __init__(self, d_model, n_task=3):
        super(TaskEmbedding, self).__init__()
        self.varible_emb = nn.Embedding(n_task, d_model)

    def forward(self, x):
        x = self.varible_emb(x.long())
        return x


class ValueEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ValueEmbedding, self).__init__()

        self.projection = nn.Linear(c_in, d_model)

    def forward(self, x):
        return self.projection(x)
    

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding_ITS_Pooled(nn.Module):
    """
    입력:
      tt   : (B, D, L)  - 변수별 이벤트 시간 (패딩 포함)
      x    : (B, D, L)  - 변수별 값 (패딩 포함)
      mask : (B, D, L)  - 1=유효(관측), 0=pad

    출력:
      out  : (B, D, d_model)  - 변수별로 pooling된 표현
    """
    def __init__(self, d_model: int, n_var: int, device = None, dropout: float = 0.1, use_time: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_var = n_var
        self.use_time = use_time

        self.time_embedding = TimeEmbedding(d_model)
        # (value, mask) -> d_model (shared)
        self.value_embedding = ValueEmbedding(c_in=2, d_model=d_model)
        self.variable_embedding = VariableEmbedding(n_var=n_var, d_model=d_model)
        self.vars = torch.arange(n_var).to(device)
        self.device = device
        self.dropout = nn.Dropout(dropout)

    def forward(self, tt, x, mask):
        """
        tt, x, mask: (B, D, L)
        return: (B, D, d_model)
        """
        B, D, L = x.shape
        assert D == self.n_var, "The number of variables does not match with n_var"

        # (B, D, L, 1)
        tt1 = tt.unsqueeze(-1)
        x1 = x.unsqueeze(-1)
        m1 = mask.unsqueeze(-1)

        # value embedding: concat([x, mask]) -> (B, D, L, 2) -> (B, D, L, d_model)
        vx = torch.cat([x1, m1], dim=-1)
        value_emb = self.value_embedding(vx)

        if self.use_time:
            time_emb = self.time_embedding(tt1)                  # (B, D, L, d_model)
            sequence_emb = value_emb + (mask.unsqueeze(-1) * time_emb)  # pad는 mask=0이라 영향 제거
        else:
            sequence_emb = value_emb

        # 변수 ID embedding을 더해 변수별 의미 차이 주입
        vars_prompt = self.variable_embedding(self.vars.view(1, D)).unsqueeze(2) # (1, D, 1, d_model)
        sequence_emb = sequence_emb + vars_prompt                                          # (B, D, L, d_model)

        sequence_emb = self.dropout(sequence_emb)

        # (B, D, d_model)로 pooling
        out = masked_mean_pool_seq(sequence_emb, mask)                             # (B, D, d_model)
        
        return out

# ## Original Codes  
# class DataEmbedding_ITS_Ind_VarPrompt(nn.Module):
#     def __init__(self, c_in, d_model, n_var, device=None, dropout=0.1, use_te=True):
#         super(DataEmbedding_ITS_Ind_VarPrompt, self).__init__()

#         self.d_model = d_model
#         self.time_embedding = TimeEmbedding(d_model=d_model).to(device)
#         self.value_embedding = ValueEmbedding(c_in=2, d_model=d_model).to(device)
#         self.variable_embedding = VariableEmbedding(n_var=n_var, d_model=d_model).to(device)
#         self.vars = torch.arange(n_var).to(device)
#         self.device = device
#         self.use_te = use_te
        
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, tt, x, x_mark=None):
#         """ 
#         tt: (B, L, D)
#         x: (B, L, D) tensor containing the observed values.
#         x_mark: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.

#         불규칙 시계열이지만 일단 각 변수마다 최대 길이의 length인 L로 맞춰두고 이 중 관측된 값이 무엇인지에 대해서 확인할 수 있도록 x_mark를 둔 것임. 
#         """
#         B, L, D = x.shape
#         time_emb = self.time_embedding(tt.unsqueeze(dim=-1)) # # (B, L, D, d_model)
        
#         x = x.unsqueeze(dim=-1) # (B, L, D, 1)
#         x_mark = x_mark.unsqueeze(dim=-1) # (B, L, D, 1)
#         x_int = torch.cat([x, x_mark], dim=-1) # (B, L, D, 2)
#         value_emb = self.value_embedding(x_int) # (B, L, D, d_model)

#         # print(x_mark.shape, time_emb.shape, value_emb.shape)
#         if(self.use_te):
#             x = x_mark*time_emb + value_emb # (B, L, D, d_model)
#         else:
#             x = value_emb
        
#         vars_prompt = self.variable_embedding(self.vars.view(1, 1, -1).repeat(B, 1, 1))
#         # text_embedding 
#         x = torch.cat([vars_prompt, x], dim=1)
#         # print(x.shape)
#         x = x.permute(0, 2, 1, 3).reshape(B*D, L+1, self.d_model) # (B*D, L+1, d_model)
 
#         return self.dropout(x), vars_prompt

def masked_mean_pool_seq(h, mask, eps=1e-8):
    """
    h   : (B, D, L, d_model)
    mask: (B, D, L) with 1 valid, 0 pad
    return: (B, D, d_model)
    """
    m = mask.unsqueeze(-1)                      # (B, D, L, 1)
    summed = (h * m).sum(dim=2)                 # (B, D, d_model)
    denom = m.sum(dim=2).clamp_min(eps)         # (B, D, 1)
    return summed / denom
