import pandas as pd
import numpy as np
import random 
import os 
import torch
import gc
import math
import json
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from typing import *

# 시계열 데이터 위한 mask 텐서 및 실제 관측된 값들만 기록되어 있는 텐서, text로 변환된 데이터,
# 그러면 시계열 형태로 reshape하는 과정 및 text 변환되어서 뿌려주는 형태로 가야함 

####### Time Series Modality #######
class ISTS_EHR_Dataset(Dataset):
    def __init__(self, args, df, outcome_df, ts_cols, splited_ids):

        super(ISTS_EHR_Dataset, self).__init__()
        
        self.args = args
        self.use_ts_trunc = args.use_ts_trunc
        self.max_length = args.max_length # (Option -> 만약 너무 긴 데이터가 들어왔을 때 처리하기 위한 상한선 설정)
        self.pids = splited_ids

        self.task_label = args.task_label
        self.pid = args.pid_col
        self.offset = args.time_col
        self.ts_cols = ts_cols

        self.D = len(self.ts_cols)

        # ts_df는 var_name 컬럼 기준이라고 가정(필요하면 full_var_name로 바꿔도 됨)
        self.ts_df = df[(df[self.args.var_col].isin(self.ts_cols)) & (df[self.pid].isin(self.pids))].copy()
        self.ts_df = self.ts_df.sort_values([self.pid, self.args.var_col, self.offset])

        outcomes = outcome_df[[self.pid, self.task_label]].sort_values(self.pid)
        outcomes = outcomes[outcomes[self.pid].isin(self.pids)]

        self.pid2y = dict(zip(outcomes[self.pid].tolist(), outcomes[self.task_label].tolist()))
        self.pid2ts = {pid: g for pid, g in self.ts_df.groupby(self.pid)}

    def __len__(self):
        return len(self.pids)
    
    def __getitem__(self, index):
        pid = self.pids[index]
        y = int(self.pid2y[pid])  

        group = self.pid2ts[pid]

        vars_list = []

        for col in self.ts_cols:
            group_value = group[group["var_name"] == col]
            
            if group_value.empty:
                t_list, x_list = [], []

            else:
                time_values = group_value[self.offset].to_numpy(dtype=np.float32)
                x_values = group_value[self.args.val_col].to_numpy(dtype=np.float32)

                # 최근 N개 제한(원하면)
                if self.use_ts_trunc and len(time_values) > self.max_length:
                    time_values = time_values[-self.max_length:]
                    x_values = x_values[-self.max_length:]

                t_list = time_values.tolist()
                x_list = x_values.tolist()

            vars_list.append({"time": t_list, "value": x_list})

        return {"pid": pid, "vars": vars_list, "y": y}

####### Text Modality #######
class TextBundle:
    """
    - pid로 observations/demographics를 찾아 TextGeneration에 넣고
    - 결과 markdown을 캐싱해서 반환
    - text generator : TextGeneration Class
    """
    def __init__(self, args, text_generator, dynamics_df: pd.DataFrame, static_df: pd.DataFrame, mapping_df: pd.DataFrame, 
                 text_var_col : List[str], lab_var_list: List[str], output_var_names=['Urine output']):
        
        self.args = args
        self.mapping_df = mapping_df
        self.text_gen = text_generator
        self.text_var_col = text_var_col
        self.lab_var_list = lab_var_list
        self.output_var_names = output_var_names

        merged_df = pd.merge(dynamics_df, mapping_df[[args.var_col, args.text_var_col]], on = args.var_col, how = 'left')

        # text용 관측치만 분리해 둠 (lab+output)
        # dynamics_df에 full_var_name 컬럼이 있어야 함
        self.text_df = merged_df[merged_df[self.text_var_col].isin(lab_var_list + self.output_var_names)].copy()
        self.text_df = self.text_df.sort_values([args.pid_col, args.time_col])

        # demographics도 pid로 묶을 수 있게 준비
        self.static_df = pd.merge(static_df, mapping_df[[args.var_col, args.text_var_col]], on = args.var_col, how = 'left')

        # pid -> df group cache (빨리 꺼내기)
        self.pid2obs = {pid: g for pid, g in self.text_df.groupby(args.pid_col)}
        self.pid2demo = {pid: g for pid, g in self.static_df.groupby(args.pid_col)}

        # pid -> markdown cache
        self.md_cache = {}

    def get_markdown(self, pid):
        
        if pid in self.md_cache:
            return self.md_cache[pid]

        obs = self.pid2obs.get(pid, pd.DataFrame(columns=[self.args.time_col, self.text_var_col, self.args.val_col]))
        demo = self.pid2demo.get(pid, pd.DataFrame(columns=[self.text_var_col, self.args.val_col]))

        markdown_data = self.text_gen.build_patient_markdown_summary(observations=obs, demographics=demo, mapping_df=self.mapping_df, 
                                                                     lab_var_list=self.lab_var_list, output_var_names=self.output_var_names)
        
        self.md_cache[pid] = markdown_data

        return markdown_data

class TextGeneration(nn.Module):
    """
    build_patient_markdown_summary : To generate the text summary for lab & output event, it's inputs need to observation dataframe, static dataframe, concept dict dataframe, lab variable name list.
    """
    def __init__(self, time_col: str = "charttime", id_col: str = "stay_id", text_var_col: str = "full_var_name", val_col: str = "value", unit_col: str = "fixed_unit", 
                 normal_min : str = 'normal_min', normal_max : str = 'normal_max', round_ndigits: int = 1):
        # --- Input dataframe column names ---
        # charttime is already "minutes since admission" (int)
        self.time_col = time_col     # minutes since ICU admission (int)
        self.id_col = id_col
        self.text_var_col = text_var_col
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
        - must include: self.time_col (minutes), self.text_var_col, self.val_col
        demographics: {"Age":..., "Gender":..., "Weight":..., "Height":...}
        normal_ranges: {"WBC": (4,11), ...}
        output_var_names: variables to treat as "Output Events" section
        """

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
        is_output = df[self.text_var_col].isin(output_var_names)
        labs_df = df[~is_output].copy()
        output_df = df[is_output].copy()


        mapping_key = self.text_var_col  # "full_var_name"

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
                var_rows = section_df[section_df[self.text_var_col] == var_name]

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
                feats = self.compute_variable_features(var_rows)
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

    def fmt_number(self, x: Any, ndigits: int = 1, type='measurement') -> str:
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

####### Mapping Meta Data #######
class MappingData:
    def __init__(self, args, df, all_var_cols, lab_cols, output_cols):
        self.args = args
        self.all_var_cols = all_var_cols
        self.lab_cols = lab_cols
        self.output_cols = output_cols
        self.metadata = df

    def _load_selected_vars(self):
        """
        min, max => 이상치 처리 시 사용
        cateogries => 해당 변수 category 분류 시 사용
        source_table => 어느 테이블에서 가져와야 하는지
        source_columns => 해당 테이블 어느 컬럼에서 가져와야 하는지
        source_itemid => 해당 테이블의 컬럼에서 어떤 id로 존재하는지
        """
        var_normal_min = []
        var_normal_max = []
        var_unit = []
        source_category = []
        source_table = []
        source_column = []
        source_itemid = []
        source_callback = []
        source_vars = [] # to map other lists
        for var in tqdm(self.all_var_cols):

            for i in range(len(self.metadata[var]['sources'][self.args.data_source])): # 여러 테이블에 분산되어 있는 경우 길이가 2가 넘을 수 있기 때문

                source_table.append(self.metadata[var]['sources'][self.args.data_source][i]['table'])

                try:
                    source_column.append(self.metadata[var]['sources'][self.args.data_source][i]['sub_var'])
                except:
                    source_column.append(self.metadata[var]['sources'][self.args.data_source][i]['val_var'])

                try:
                    if [223761, 224027] == self.metadata[var]['sources'][self.args.data_source][i]['ids']: # temp 관련 변수에서 skin temp는 제외 
                        source_itemid.append(223761)
                    
                    else:
                        source_itemid.append(self.metadata[var]['sources'][self.args.data_source][i]['ids'])
                except:
                    source_itemid.append(None)

                try:
                    source_callback.append(self.metadata[var]['sources'][self.args.data_source][i]['callback'])
                except:
                    source_callback.append(None)


                source_category.append(self.metadata[var]['category'])

                try:
                    var_unit.append(self.metadata[var]['unit'])
                except:
                    var_unit.append(None)

                try:
                    var_normal_min.append(self.metadata[var]['min'])
                except:
                    var_normal_min.append(None)
                
                try:
                    var_normal_max.append(self.metadata[var]['max'])
                except:
                    var_normal_max.append(None)

                source_vars.append(var)
            
        return var_normal_min, var_normal_max, var_unit, source_category, source_table, source_column, source_itemid, source_callback, source_vars

    def build_mapping_df(self):
        var_normal_min, var_normal_max, var_unit, source_category, source_table, source_column, source_itemid, source_callback, source_vars  = self._load_selected_vars()

        mapping_df = pd.DataFrame({
            'var_name' : source_vars,
            'normal_min' : var_normal_min,
            'normal_max' : var_normal_max,
            'category' : source_category,
            'table' : source_table,
            'column' : source_column,
            'itemid' : source_itemid,
            'unit' : var_unit,
            'method' : source_callback 
        })

        # FUll name 및 추가 
        full_name_dict = {'age' : 'Age', 'sex' : 'Sex', 'height' : 'Height', 'weight' : 'Weight', 'sbp' : 'Blood pressure (systolic)', 'dbp' : 'Blood pressure (diastolic)', 'hr' : 'Heart rate', 'map' : 'Mean arterial pressure', 'o2sat' : 'Oxygen saturation',
                        'resp' : 'Respiratory rate', 'temp' : 'Temperature', 'alb' : 'Albumin', 'alp' : 'Alkaline phosphatase', 'alt' : 'Alanine aminotransferase', 'ast' : 'Aspartate aminotransferase', 'be' : 'Base excess', 'bicar' : 'Bicarbonate',
                        'bili' : 'Bilirubin (total)', 'bili_dir' : 'Bilirubin (direct)', 'bnd' : 'Band form neutrophils', 'bun' : 'Blood urea nitrogen', 'ca' : 'Calcium', 'cai' : 'Calcium ionized', 'crea' : 'Creatinine', 'ck' : 'Creatinine kinase',
                        'ckmb' : 'Creatinine kinase MB', 'cl' : 'Chloride', 'pco2' : 'CO2 partial pressure', 'crp' : 'C-reactive protein', 'fgn' : 'Fibrinogen', 'glu' : 'Glucose', 'hgb' : 'Haemoglobin', 'inr_pt' : 'International normalised ratio (INR)',
                        'lact' : 'Lactate', 'lymph' : 'Lymphocytes', 'mch' : 'Mean cell haemoglobin', 'mchc' : 'Mean corpuscular haemoglobin concentration', 'mcv' : 'Mean corpuscular volume', 'methb' : 'Methaemoglobin', 'mg' : 'Magnesium', 
                        'neut' : 'Neutrophils', 'po2' : 'O2 partial pressure', 'ptt' : 'Partial thromboplastin time', 'ph' : 'pH of blood', 'phos' : 'Phosphate', 'plt' : 'Platelets', 'k' : 'Potassium', 'na' : 'Sodium', 'tnt' : 'Troponin T', 
                        'wbc' : 'White blood cells', 'fio2' : 'Fraction of inspired oxygen', 'urine' : 'Urine output'
                        }

        fixed_unit_dict = {'age' : 'Years', 'sex' : '', 'height' : 'cm', 'weight' : 'kg', 'sbp' : 'mmHg', 'dbp' : 'mmHg', 'hr' : 'beats/minute', 'map' : 'mmHg', 'o2sat' : '%',
                        'resp' : 'breaths/minute', 'temp' : '°C', 'alb' : 'g/dL', 'alp' : 'IU/L', 'alt' : 'IU/L', 'ast' : 'IU/L', 'be' : 'mmol/L', 'bicar' : 'mmol/L',
                        'bili' : 'mg/dL', 'bili_dir' : 'mg/dL', 'bnd' : '%', 'bun' : 'mg/dL', 'ca' : 'mg/dL', 'cai' : 'mmol/L', 'crea' : 'mg/dL', 'ck' : 'IU/L',
                        'ckmb' : 'ng/mL', 'cl' : 'mmol/L', 'pco2' : 'mmHg', 'crp' : 'mg/L', 'fgn' : 'mg/dL', 'glu' : 'mg/dL', 'hgb' : 'g/dL', 'inr_pt' : '',
                        'lact' : 'mmol/L', 'lymph' : '%', 'mch' : 'pg', 'mchc' : '%', 'mcv' : 'fL', 'methb' : '%', 'mg' : 'mg/dL', 
                        'neut' : '%', 'po2' : 'mmHg', 'ptt' : 'sec', 'ph' : '', 'phos' : 'mg/dL', 'plt' : '1,000 / μL', 'k' : 'mmol/L', 'na' : 'mmol/L', 'tnt' : 'ng/mL', 
                        'wbc' : '1,000 / μL', 'fio2' : '%', 'urine' : 'mL'
                        }

        mapping_df['full_var_name'] = mapping_df['var_name'].apply(lambda x : full_name_dict[x])
        mapping_df['fixed_unit'] = mapping_df['var_name'].apply(lambda x : fixed_unit_dict[x])
        mapping_df = mapping_df[['var_name', 'normal_min', 'normal_max', 'full_var_name', 'fixed_unit']] # only using variables
        
        return mapping_df, [full_name_dict[i] for i in self.lab_cols], [full_name_dict[i] for i in self.output_cols]


def make_collate_ists_with_text(D, text_bundle : TextBundle):

    def collate(batch):
        B = len(batch)
        pids = [sample["pid"] for sample in batch]
        y = torch.tensor([sample["y"] for sample in batch], dtype=torch.long)

        # ---- TS padding ----
        lengths = []
        for sample in batch:
            assert len(sample["vars"]) == D
            for d in range(D):
                lengths.append(len(sample["vars"][d]["time"]))
        Lmax = max(lengths) if lengths else 0

        tt = torch.zeros(B, D, Lmax, dtype=torch.float32)
        xx = torch.zeros(B, D, Lmax, dtype=torch.float32)
        mask = torch.zeros(B, D, Lmax, dtype=torch.float32)

        for b, sample in enumerate(batch):
            for d in range(D):
                time = torch.as_tensor(sample["vars"][d]["time"], dtype=torch.float32)
                value = torch.as_tensor(sample["vars"][d]["value"], dtype=torch.float32)
                Ld = time.numel()

                if Ld == 0: # if the variable is not observed -> all values are zero
                    continue

                tt[b, d, :Ld] = time         # (B, D, L)
                xx[b, d, :Ld] = value         # (B, D, L)
                mask[b, d, :Ld] = 1.0     # (B, D, L)

        # ---- Text (pid와 매칭) ----
        texts = [text_bundle.get_markdown(pid) for pid in pids]

        return tt, xx, mask, texts, y, pids
    
    return collate

def split_stay_ids_ehr(args, df):
    """
    df : outcome_df 
    
    """
    
    df = df.sort_values(args.pid_col).copy()
    stay_ids = df[args.pid_col].values
    labels   = df['los_reg'].values # doing startify split by using los output (= patient los)

    train_ids, temp_ids, train_lbls, temp_lbls = train_test_split(
        stay_ids, labels,
        train_size=args.train_ratio,
        stratify=labels,
        random_state=args.seed
    )

    valid_over_temp = args.val_ratio / (1 - args.train_ratio)

    valid_ids, test_ids, valid_lbls, test_lbls = train_test_split(
        temp_ids, temp_lbls,
        train_size=valid_over_temp,
        stratify=temp_lbls,
        random_state=args.seed
    )

    print(f"Train size: {len(train_ids)}, Valid size: {len(valid_ids)}, Test size: {len(test_ids)}")
    print(f"Train positive label: {sum(train_lbls)}, Valid Label positive label: {sum(valid_lbls)}, Test Label positive label : {sum(test_lbls)}")
    
    return train_ids, valid_ids, test_ids

def build_loaders(args):
    # 1) load data files
    dynamics_df = pd.read_csv(Path(args.data_path)/args.data_source/'dynamics_df.csv.gz', compression='gzip')
    static_df   = pd.read_csv(Path(args.data_path)/args.data_source/'static_df.csv.gz',   compression='gzip')
    outcome_df  = pd.read_csv(Path(args.data_path)/args.data_source/'outcome_df.csv.gz',  compression='gzip')
    
    with open(args.var_info_path, 'r') as f:
            metadata = json.load(f)

    # 2) Set variable groups 
    dg_cols = ["age", "sex", "height", "weight"] 
    ts_cols = ["dbp", "sbp", "map", "hr", "o2sat", "resp", "temp"]
    lab_cols = ["alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir","bnd", "bun", "ca", "cai", 
                "ck", "ckmb", "cl", "crea", "crp", "fgn", "fio2", "glu", "hgb", "inr_pt", "k", "lact",
                "lymph", "mch", "mchc", "mcv", "methb", "mg", "na", "neut", "pco2", "ph", "phos", "plt", 
                "po2", "ptt", "tnt", "wbc"]
    output_cols = ["urine"]
    all_var_cols = dg_cols + ts_cols + lab_cols + output_cols

    # 3) Genearte mapping dataframe 
    mappingdata = MappingData(args, df=metadata, all_var_cols=all_var_cols, lab_cols=lab_cols, output_cols=output_cols)
    mapping_df, lab_var_list, output_var_names = mappingdata.build_mapping_df()

    # 4) Generate TextGeneration + TextBundle 
    text_gen = TextGeneration(time_col=args.time_col, id_col=args.pid_col,
                              text_var_col="full_var_name", val_col="value", unit_col="fixed_unit",
                              normal_min="normal_min", normal_max="normal_max", round_ndigits=1)

    text_bundle = TextBundle(args, dynamics_df=dynamics_df,
                             static_df=static_df,
                             mapping_df=mapping_df,
                             text_var_col="full_var_name",
                             text_generator=text_gen,
                             lab_var_list=lab_var_list,
                             output_var_names=output_var_names)

    # 4) When using DataLoader it gets text, ts modalities
    collate_fn = make_collate_ists_with_text(len(ts_cols), text_bundle)
    
    # 5) Generate Timeseries Data and Final Loader
    if not args.adapt_mode:
        train_ids, valid_ids, test_ids = split_stay_ids_ehr(args, df=outcome_df) # get splited ids

        train_ds = ISTS_EHR_Dataset(args, dynamics_df, outcome_df, ts_cols, train_ids)
        valid_ds = ISTS_EHR_Dataset(args, dynamics_df, outcome_df, ts_cols, valid_ids)
        test_ds  = ISTS_EHR_Dataset(args, dynamics_df, outcome_df, ts_cols, test_ids)
        D = train_ds.D

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=0, collate_fn=collate_fn)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size,
                                                shuffle=False, num_workers=0, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                                shuffle=False, num_workers=0, collate_fn=collate_fn)
        
        return train_loader, valid_loader, test_loader

    else:
        _, _, test_ids = split_stay_ids_ehr(args, df=outcome_df) # get splited ids
        eval_ds = ISTS_EHR_Dataset(args, dynamics_df, outcome_df, ts_cols, test_ids)
        D = eval_ds.D

        eval_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                                shuffle=False, num_workers=0, collate_fn=collate_fn)
        
        return _, _, eval_loader