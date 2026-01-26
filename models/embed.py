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

class TextGeneration(nn.Module):
    """
    build_patient_markdown_summary : To generate the text summary for lab & output event, it's inputs need to observation dataframe, static dataframe, concept dict dataframe, lab variable name list.
    """
    def __init__(self, time_col: str = "charttime", id_col: str = "stay_id", var_col: str = "full_var_name", val_col: str = "value", unit_col: str = "fixed_unit", 
                 normal_min : str = 'normal_min', normal_max : str = 'normal_max', round_ndigits: int = 2):
        # --- Input dataframe column names ---
        # charttime is already "minutes since admission" (int)
        self.time_col = time_col     # minutes since ICU admission (int)
        self.id_col = id_col
        self.var_col = var_col
        self.val_col = val_col
        self.unit_col = unit_col
        self.normal_min  = normal_min
        self.normal_max  = normal_max


        # --- Formatting ---
        self.round_ndigits = round_ndigits

    def build_patient_markdown_summary(self, observations: pd.DataFrame, demographics: pd.DataFrame, mapping_df : pd.DataFrame, labs_title: str = "Summary of Lab Results", 
                                       lab_var_list: Optional[list[str]] = None, output_var_names: Optional[Set[str]] = ['Urine output']) -> str:
        
        """
        observations: one patient's observation rows
        - must include: self.time_col (minutes), self.var_col, self.val_col
        demographics: {"Age":..., "Gender":..., "Weight":..., "Height":...}
        normal_ranges: {"WBC": (4,11), ...}
        output_var_names: variables to treat as "Output Events" section
        """

        output_var_names = output_var_names or set()

        df = observations.copy()
        df[self.val_col] = pd.to_numeric(df[self.val_col], errors="coerce")
        df[self.time_col] = pd.to_numeric(df[self.time_col], errors="coerce")  # ensure numeric minutes

        # ----------------------------
        # Header: demographics
        # ----------------------------
        md_lines = []
        md_lines.append("# Patient Demographics at ICU Admission\n")
        md_lines.append(f"- Age : {self.fmt_number(demographics[demographics['full_var_name'] == 'Age']['value'].values[0])}")
        md_lines.append(f"- Gender : {demographics[demographics['full_var_name'] == 'Sex']['value'].values[0]}")

        try:
            md_lines.append(f"- Weight : {self.fmt_number(demographics[demographics['full_var_name'] == 'Weight']['value'].values[0])}kg")
        except:
            md_lines.append(f"- Weight : Not observed")
        
        try:
            md_lines.append(f"- Height : {self.fmt_number(demographics[demographics['full_var_name'] == 'Height']['value'].values[0])}cm \n")
        except:
            md_lines.append(f"- Height : Not observed")

        md_lines.append("")

        # ----------------------------
        # Split labs vs outputs
        # ----------------------------
        is_output = df[self.var_col].isin(output_var_names)
        labs_df = df[~is_output].copy()
        output_df = df[is_output].copy()


        mapping_key = self.var_col  # "full_var_name"

        meta_cols = [self.unit_col, self.normal_min, self.normal_max]
        meta_cols = [c for c in meta_cols if c in mapping_df.columns]

        var_meta = (
            mapping_df.drop_duplicates(subset=[mapping_key])
                    .set_index(mapping_key)[meta_cols]
                    .to_dict(orient="index"))
        
        def _render_variable_section(title: str, section_df: pd.DataFrame, var_list: list[str]) -> list[str]:
            section_lines = [f"# {title}"]

            for var_name in var_list:
                # 1) metadata from MappingDF (unit + normal range)
                meta = var_meta.get(var_name, {})
                unit_str = meta.get(self.unit_col, "") if meta else ""
                ref_min = meta.get(self.normal_min, None) if meta else None
                ref_max = meta.get(self.normal_max, None) if meta else None

                # 2) patient rows for this var (may be empty)
                var_rows = section_df[section_df[self.var_col] == var_name]

                section_lines.append(f"- [{var_name}] ({unit_str})\n")
                section_lines.append(
                    f"\t- Normal Value Range : {self.fmt_number(ref_min, self.round_ndigits, 'normal_min')} to {self.fmt_number(ref_max, self.round_ndigits, 'normal_max')}"
                )

                # 3) if no observations -> keep unit/range, but stats/time are Not observed
                n_valid = int(var_rows[self.val_col].notna().sum()) if not var_rows.empty else 0
                if n_valid == 0:
                    section_lines.append("\t- (Not observed)")
                    continue

                # 4) has observations -> compute features
                feats = self.compute_variable_features(var_rows, self)
                section_lines.append(
                    f"\t- First/Last Obs : {self.fmt_number(feats['first_min'], 0)} / {self.fmt_number(feats['last_min'], 0)}, "
                    f"N : {self.fmt_number(feats['n_obs'], 0)}, Interval : {self.fmt_number(feats['interval_mean_min'], 0)}"
                )

                vs = feats["value_stats"]
                section_lines.append(
                    "\t- Statistics: "
                    f"[{self.fmt_number(vs['min'], self.round_ndigits)}, {self.fmt_number(vs['max'], self.round_ndigits)}, "
                    f"{self.fmt_number(vs['median'], self.round_ndigits)}, {self.fmt_number(vs['mean'], self.round_ndigits)}, "
                    f"{self.fmt_number(vs['std'], self.round_ndigits)}]"
                )
                section_lines.append("")

            return section_lines
    
        md_lines.append("## Time unit : minutes after admission, Obs : Observation Time, N : observation count, "
                        "Interval : mean interval between observations, Statistics : [min, max, median, mean, std]")
        md_lines += _render_variable_section(labs_title, labs_df, lab_var_list)
        md_lines.append("")  # spacer
        md_lines += _render_variable_section("Summary of Output Events", output_df, ['Urine output'])

        return "\n".join(md_lines).strip()

    def fmt_number(self, x: Any, ndigits: int = 3, type='measurement') -> str:
        """Format numbers; use 'Not observed' for None/NaN."""
        
        if x is None and type == 'measurement':
            return "Not observed"
        
        if pd.isna(x) and type == 'normal_max':
            return "inf"
        
        if pd.isna(x) and type == 'normal_min':
            return "-inf"

        try:
            if isinstance(x, float) and np.isnan(x):
                return "Not observed"
        except Exception:
            pass

        if isinstance(x, (int, np.integer)):
            return str(int(x))

        try:
            return f"{float(x):.{ndigits}f}"
        except Exception:
            return "Not observed"


    def nan_stats(self, x: np.ndarray) -> Dict[str, Optional[float]]:
        """Return min/max/mean/std on numeric array, ignoring NaNs."""
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if x.size == 0:
            return {"min": None, "max": None, 'median' : None ,"mean": None, "std": None}

        std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        return {
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            'median' : float(np.median(x)),
            "mean": float(np.mean(x)),
            "std": std,
        }
    
    def compute_variable_features(self, var_df: pd.DataFrame) -> Dict[str, Any]:

        """
        Compute summary features for ONE variable group.
        Assumption: self.time_col is already minutes (int) since admission.
        """
        df_sorted = var_df.sort_values(self.time_col).copy()

        times_min = pd.to_numeric(df_sorted[self.time_col], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(df_sorted[self.val_col], errors="coerce").to_numpy(dtype=float)

        # observation count (non-NaN values)
        n_obs = int(np.sum(~np.isnan(values)))

        # first/last time (minutes)
        first_min = float(times_min[0]) if times_min.size else None
        last_min = float(times_min[-1]) if times_min.size else None

        # interval mean (unique time 기준)
        unique_times = np.unique(times_min[~np.isnan(times_min)])
        if unique_times.size >= 2:
            interval_mean_min = float(np.mean(np.diff(unique_times)))
        else:
            interval_mean_min = None

        # value stats
        value_stats = self.nan_stats(values)

        # # slope stats: Δy / Δt (dt>0)
        # slopes = np.array([], dtype=float)
        # if times_min.size >= 2:
        #     dt = np.diff(times_min)
        #     dy = np.diff(values)
        #     mask = (~np.isnan(dt)) & (~np.isnan(dy)) & (dt > 0)
        #     if np.any(mask):
        #         slopes = dy[mask] / dt[mask]
        # slope_stats = self.nan_stats(slopes) if slopes.size else {"min": None, "max": None, "mean": None, "std": None}

        # # variability stats
        # if self.variability_mode == "abs_delta":
        #     variability_seq = np.abs(np.diff(values))
        #     variability_seq = variability_seq[~np.isnan(variability_seq)]
        # elif self.variability_mode == "rolling_std":
        #     rolling_std = (
        #         pd.Series(values)
        #         .rolling(self.rolling_window, min_periods=self.min_points_for_stats)
        #         .std()
        #         .to_numpy(dtype=float)
        #     )
        #     variability_seq = rolling_std[~np.isnan(rolling_std)]
        # else:
        #     raise ValueError("variability_mode must be 'abs_delta' or 'rolling_std'")

        # variability_stats = self.nan_stats(variability_seq) if variability_seq.size else {"min": None, "max": None, "mean": None, "std": None}

        return {
            "first_min": first_min,
            "last_min": last_min,
            "n_obs": n_obs,
            "interval_mean_min": interval_mean_min,
            "value_stats": value_stats
            }


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

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_ITS_Ind(nn.Module):
    def __init__(self, c_in, d_model, device=None, dropout=0.1, use_te=True):
        super(DataEmbedding_ITS_Ind, self).__init__()

        self.d_model = d_model
        self.time_embedding = TimeEmbedding(d_model=d_model).to(device)
        self.value_embedding = ValueEmbedding(c_in=2, d_model=d_model).to(device)
        self.device = device
        self.use_te = use_te
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tt, x, x_mark=None):
        """ 
        tt: (B, L, D)
        x: (B, L, D) tensor containing the observed values.
        x_mark: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        B, L, D = x.shape
        time_emb = self.time_embedding(tt.unsqueeze(dim=-1)) # # (B, L, D, d_model)
        # print(time_emb)
        x = x.unsqueeze(dim=-1) # (B, L, D, 1)
        x_mark = x_mark.unsqueeze(dim=-1) # (B, L, D, 1)
        x_int = torch.cat([x, x_mark], dim=-1) # (B, L, D, 2)
        value_emb = self.value_embedding(x_int) # (B, L, D, d_model)

        # print(x_mark.shape, time_emb.shape, value_emb.shape)
        if(self.use_te):
            x = x_mark*time_emb + value_emb # (B, L, D, d_model)
        else:
            x = value_emb
        
        x = x.permute(0, 2, 1, 3).reshape(B*D, L, self.d_model) # (B*D, L+1, d_model)
 
        return self.dropout(x)
    


class DataEmbedding_ITS_Ind_VarPrompt_indicator(nn.Module):
    def __init__(self, c_in, d_model, n_var, device=None, dropout=0.1):
        super(DataEmbedding_ITS_Ind_VarPrompt_indicator, self).__init__()

        self.d_model = d_model
        self.time_embedding = TimeEmbedding(d_model=d_model).to(device)
        self.value_embedding = ValueEmbedding(c_in=2, d_model=d_model).to(device)
        self.variable_embedding = VariableEmbedding(n_var=n_var, d_model=d_model).to(device)
        self.indicator_embedding = nn.Embedding(2, d_model).to(device)
        self.vars = torch.arange(n_var).to(device)
        self.device = device
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tt, x, x_mark=None):
        """ 
        tt: (B, L, D)
        x: (B, L, D) tensor containing the observed values.
        x_mark: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        B, L, D = x.shape
        time_emb = self.time_embedding(tt.unsqueeze(dim=-1)) # # (B, L, D, d_model)
        x_ones = torch.ones_like(x).to(self.device)
        # print(time_emb)
        x = x.unsqueeze(dim=-1) # (B, L, D, 1)
        x_mark = x_mark.unsqueeze(dim=-1) # (B, L, D, 1)
        x_int = torch.cat([x, x_mark], dim=-1) # (B, L, D, 2)
        value_emb = self.value_embedding(x_int) # (B, L, D, d_model)
        # print(x_mark.shape, time_emb.shape, value_emb.shape)
        x = x_mark*time_emb + value_emb + self.indicator_embedding(x_ones.int()) # (B, L, D, d_model)
        
        vars_tenosr = self.vars.view(1, 1, -1).repeat(B, 1, 1)
        vars_zeros = torch.zeros_like(vars_tenosr).to(self.device)
        test = self.indicator_embedding(vars_zeros.int())
        vars_prompt = self.variable_embedding(vars_tenosr) + self.indicator_embedding(vars_zeros.int())
        # text_embedding 
        x = torch.cat([vars_prompt, x], dim=1)
        # print(x.shape)
        x = x.permute(0, 2, 1, 3).reshape(B*D, L+1, self.d_model) # (B*D, L+1, d_model)
 
        return self.dropout(x), vars_prompt


class DataEmbedding_ITS_Ind_VarPrompt(nn.Module):
    def __init__(self, c_in, d_model, n_var, device=None, dropout=0.1, use_te=True):
        super(DataEmbedding_ITS_Ind_VarPrompt, self).__init__()

        self.d_model = d_model
        self.time_embedding = TimeEmbedding(d_model=d_model).to(device)
        self.value_embedding = ValueEmbedding(c_in=2, d_model=d_model).to(device)
        self.variable_embedding = VariableEmbedding(n_var=n_var, d_model=d_model).to(device)
        self.vars = torch.arange(n_var).to(device)
        self.device = device
        self.use_te = use_te
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tt, x, x_mark=None):
        """ 
        tt: (B, L, D)
        x: (B, L, D) tensor containing the observed values.
        x_mark: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        B, L, D = x.shape
        time_emb = self.time_embedding(tt.unsqueeze(dim=-1)) # # (B, L, D, d_model)
        # print(time_emb)
        x = x.unsqueeze(dim=-1) # (B, L, D, 1)
        x_mark = x_mark.unsqueeze(dim=-1) # (B, L, D, 1)
        x_int = torch.cat([x, x_mark], dim=-1) # (B, L, D, 2)
        value_emb = self.value_embedding(x_int) # (B, L, D, d_model)

        # print(x_mark.shape, time_emb.shape, value_emb.shape)
        if(self.use_te):
            x = x_mark*time_emb + value_emb # (B, L, D, d_model)
        else:
            x = value_emb
        
        vars_prompt = self.variable_embedding(self.vars.view(1, 1, -1).repeat(B, 1, 1))
        # text_embedding 
        x = torch.cat([vars_prompt, x], dim=1)
        # print(x.shape)
        x = x.permute(0, 2, 1, 3).reshape(B*D, L+1, self.d_model) # (B*D, L+1, d_model)
 
        return self.dropout(x), vars_prompt


# vector-based embedding
class DataEmbedding_ITS_Vector(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_ITS_Vector, self).__init__()

        self.time_embedding = TimeEmbedding(d_model=d_model)
        # self.variable_embedding = VariableEmbedding(n_var=n_var, d_model=d_model)
        self.value_embedding = ValueEmbedding(c_in=2*c_in, d_model=d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tt, x, x_mark=None, x_mark_agg=None):
        """ 
        tt: (B, L, 1)
        x: (B, L, D) tensor containing the observed values.
        x_mark: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        # print(tt.shape)
        time_emb = self.time_embedding(tt)
        x_int = torch.cat([x, x_mark], dim=-1)
        value_emb = self.value_embedding(x_int)

        # print(time_emb.shape, value_emb.shape)
        # if(x_mark == None):
        x = x_mark_agg*time_emb + value_emb

        return self.dropout(x)
    

class DataEmbedding_ITS_Set(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_ITS_Set, self).__init__()

        self.time_embedding = TimeEmbedding(d_model=d_model)
        self.variable_embedding = VariableEmbedding(n_var=c_in, d_model=d_model)
        self.value_embedding = ValueEmbedding(c_in=1, d_model=d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        
        time_emb = self.time_embedding(x[...,:1])
        var_emb = self.variable_embedding(x[...,1])
        value_emb = self.value_embedding(x[...,2:3])

        # print(time_emb.shape, var_emb.shape, value_emb.shape)
        x = time_emb + var_emb + value_emb
        # print(x_mark.shape, x.shape)
        x = x*x_mark

        return self.dropout(x)
    

class DataEmbedding_ITS_UNI(nn.Module):
    def __init__(self, c_in, d_model, n_var, device=None, dropout=0.1):
        super(DataEmbedding_ITS_UNI, self).__init__()

        self.d_model = d_model
        self.time_embedding = TimeEmbedding(d_model=d_model).to(device)
        self.value_embedding = ValueEmbedding(c_in=2, d_model=d_model).to(device)
        # self.variable_embedding = VariableEmbedding(n_var=n_var, d_model=d_model).to(device)
        self.task_embedding = TaskEmbedding(d_model=d_model).to(device)
        self.vars = torch.arange(n_var).to(device)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tt, x, x_mark=None):
        """ 
        tt: (B, L, D)
        x: (B, L, D) tensor containing the observed values.
        x_mark: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        B, L, D = x.shape
        time_emb = self.time_embedding(tt.unsqueeze(dim=-1)) # # (B, L, D, d_model)
        x = x.unsqueeze(dim=-1) # (B, L, D, 1)
        x_mark = x_mark.unsqueeze(dim=-1) # (B, L, D, 1)
        x_int = torch.cat([x, x_mark], dim=-1) # (B, L, D, 2)
        value_emb = self.value_embedding(x_int) # (B, L, D, d_model)
        
        # print(time_emb.shape, var_emb.shape, value_emb.shape)
        # if(x_mark == None):
        x = x_mark*time_emb + value_emb # (B, L, D, d_model)
        
        vars_prompt = self.variable_embedding(self.vars.view(1, 1, -1).repeat(B, 1, 1))
        # print(vars_prompt.shape)
        # text_embedding 
        x = torch.cat([vars_prompt, x], dim=1)
        # print(x.shape)

        x = x.permute(0, 2, 1, 3).reshape(B*D, L+1, self.d_model) # (B*D, L+1, d_model)
 
        return self.dropout(x)



class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class DataEmbedding_wo_time(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_time, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)