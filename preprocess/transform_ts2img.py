import os
import sys
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from pathlib import Path

from utils.ts2img import *

def build_example_lab_meta(marker_list) -> Dict[str, Dict]:
    """
    임상적으로 '나쁨' 방향은 반드시 변수별로 지정해야 함.
    평균/표준편차만으로는 절대 알 수 없음.
    """
    return {
        # Resp / Acid-base / Perfusion
        "ph": {"group": "Resp / Acid-base / Perfusion", "polarity": "both_bad", "marker": marker_list[0]},
        "bicar": {"group": "Resp / Acid-base / Perfusion", "polarity": "both_bad", "marker": marker_list[1]},
        "be": {"group": "Resp / Acid-base / Perfusion", "polarity": "both_bad", "marker": marker_list[2]},
        "pco2": {"group": "Resp / Acid-base / Perfusion", "polarity": "both_bad", "marker": marker_list[3]},
        "po2": {"group": "Resp / Acid-base / Perfusion", "polarity": "low_bad", "marker": marker_list[4]},
        "fio2": {"group": "Resp / Acid-base / Perfusion", "polarity": "high_bad", "marker": marker_list[5]},
        "methb": {"group": "Resp / Acid-base / Perfusion", "polarity": "high_bad", "marker": marker_list[6]},
        "lact": {"group": "Resp / Acid-base / Perfusion", "polarity": "high_bad", "marker": marker_list[7]},

        # Electrolytes / Minerals
        "na": {"group": "Electrolytes / Minerals", "polarity": "both_bad", "marker": marker_list[0]},
        "k": {"group": "Electrolytes / Minerals", "polarity": "both_bad", "marker": marker_list[1]},
        "cl": {"group": "Electrolytes / Minerals", "polarity": "both_bad", "marker": marker_list[2]},
        "ca": {"group": "Electrolytes / Minerals", "polarity": "both_bad", "marker": marker_list[3]},
        "cai": {"group": "Electrolytes / Minerals", "polarity": "both_bad", "marker": marker_list[4]},
        "mg": {"group": "Electrolytes / Minerals", "polarity": "both_bad", "marker": marker_list[5]},
        "phos": {"group": "Electrolytes / Minerals", "polarity": "both_bad", "marker": marker_list[6]},

        # Renal
        "bun": {"group": "Renal", "polarity": "high_bad", "marker": marker_list[0]},
        "crea": {"group": "Renal", "polarity": "high_bad", "marker": marker_list[1]},
        "urine": {"group": "Renal", "polarity": "low_bad", "marker": marker_list[2]},

        # Liver / Hepatobiliary
        "ast": {"group": "Liver / Hepatobiliary", "polarity": "high_bad", "marker": marker_list[0]},
        "alt": {"group": "Liver / Hepatobiliary", "polarity": "high_bad", "marker": marker_list[1]},
        "alp": {"group": "Liver / Hepatobiliary", "polarity": "high_bad", "marker": marker_list[2]},
        "bili": {"group": "Liver / Hepatobiliary", "polarity": "high_bad", "marker": marker_list[3]},
        "bili_dir": {"group": "Liver / Hepatobiliary", "polarity": "high_bad", "marker": marker_list[4]},
        "alb": {"group": "Liver / Hepatobiliary", "polarity": "low_bad", "marker": marker_list[5]},

        # Cardiac / Muscle Injury
        "ck": {"group": "Cardiac / Muscle Injury", "polarity": "high_bad", "marker": marker_list[0]},
        "ckmb": {"group": "Cardiac / Muscle Injury", "polarity": "high_bad", "marker": marker_list[1]},
        "tnt": {"group": "Cardiac / Muscle Injury", "polarity": "high_bad", "marker": marker_list[2]},

        # Hematology
        "hgb": {"group": "Hematology", "polarity": "both_bad", "marker": marker_list[0]},
        "plt": {"group": "Hematology", "polarity": "both_bad", "marker": marker_list[1]},
        "wbc": {"group": "Hematology", "polarity": "both_bad", "marker": marker_list[2]},
        "neut": {"group": "Hematology", "polarity": "both_bad", "marker": marker_list[3]},
        "lymph": {"group": "Hematology", "polarity": "both_bad", "marker": marker_list[4]},
        "bnd": {"group": "Hematology", "polarity": "both_bad", "marker": marker_list[5]},
        "mcv": {"group": "Hematology", "polarity": "both_bad", "marker": marker_list[6]},
        "mch": {"group": "Hematology", "polarity": "both_bad", "marker": marker_list[7]},
        "mchc": {"group": "Hematology", "polarity": "both_bad", "marker": marker_list[8]},

        # Coagulation
        "inr_pt": {"group": "Coagulation", "polarity": "high_bad", "marker": marker_list[0]},
        "ptt": {"group": "Coagulation", "polarity": "high_bad", "marker": marker_list[1]},
        "fgn": {"group": "Coagulation", "polarity": "both_bad", "marker": marker_list[2]},

        # Metabolic / Inflammation
        "glu": {"group": "Metabolic / Inflammation", "polarity": "both_bad", "marker": marker_list[0]},
        "crp": {"group": "Metabolic / Inflammation", "polarity": "high_bad", "marker": marker_list[1]},
    }

def make_resampled_df(df, lab_meta, all_cols, vital_cols, lab_cols, stay_id, 
                      time_limit_min=12*24, freq_min=5, aggfunc='last'):
    """
    df columns: ['stay_id', 'charttime', 'var_name', 'value']
    stay_id별로 5분 단위 resampling 후
    (time_steps, variables) matrix 반환
    
    time_limit_min:
        하루를 5분 간격 기준으로 고정하고 싶다면 1440분
        질문에서 말한 5*12*24 = 1440 과 동일
    """
    
    sub = df[(df['stay_id'] == stay_id) & (df['var_name'].isin(all_cols))].copy()

    # urine/hr 처리
    if 'urine' in sub['var_name'].values:
        df_urine = sub[sub['var_name']=='urine'].groupby(['stay_id', 'charttime'], as_index=False)['value'].sum()  
        df_urine = df_urine.sort_values(['stay_id', 'charttime'])

        df_urine['time_diff'] = df_urine.groupby('stay_id')['charttime'].diff().fillna(df_urine['charttime'])
        df_urine['time_diff_hours'] = df_urine['time_diff'] / 60 

        df_urine['value'] = df_urine['value']/df_urine['time_diff_hours']
        df_urine['var_name'] = 'urine'

        sub = pd.concat([sub[sub['var_name'] != 'urine'], df_urine[['stay_id','charttime','var_name','value']]])

    # 5분 bin 생성
    sub['time_bin'] = (sub['charttime'] // freq_min) + 1

    if aggfunc == 'last':
        # 같은 stay_id, var_name, time_bin 안의 값 평균
        agg = (
            sub.groupby(['time_bin', 'var_name'], as_index=False)['value']
            .last()
        )
    else:
        # 같은 stay_id, var_name, time_bin 안의 값 평균
        agg = (
            sub.groupby(['time_bin', 'var_name'], as_index=False)['value']
            .mean()
        )

    # 5분 단위 grid 생성
    full_time = np.arange(1, time_limit_min + 1)
    full_index = pd.MultiIndex.from_product(
        [full_time, all_cols],
        names=['time_bin', 'var_name']
    )

    full_df = pd.DataFrame(index=full_index).reset_index()
    merged = full_df.merge(agg, on=['time_bin', 'var_name'], how='left')

    # vital, residual split
    vital_df = merged[merged['var_name'].isin(vital_cols)]
    lab_add_df = merged[~merged['var_name'].isin(vital_cols)]
    lab_add_df['group'] = lab_add_df['var_name'].apply(lambda x : lab_meta[x]['group'])


    lab_times = df[(df['stay_id'] == stay_id) & (df['var_name'].isin(lab_cols))].copy()
    lab_times['time_bin'] = (lab_times['charttime'] // freq_min) + 1

    return vital_df, lab_add_df, lab_times['time_bin'].unique()

def set_fig_config():
    
    parser = argparse.ArgumentParser(description='EHRTTA')

    # data parameters
    parser.add_argument('--data_path', default='~/EHRTTA/data/miiv',
                        help='path where data is located')
    parser.add_argument('--freq_min', type=int, default=5,
                        help='the number of resampled time bin')
    parser.add_argument('--min_clip', type=float, default=-3,
                        help='the number of min z-value for cliping')
    parser.add_argument('--max_clip', type=float, default=3,
                        help='the number of max z-value for cliping')
    parser.add_argument("--scatter_ind", default=False, action='store_true',
                        help="option for drawing independent variables in one scatter plot")
    return parser

if __name__=="__main__":

    parser = set_fig_config()
    args = parser.parse_args()

    data_dir = Path(args.data_path)

    print(data_dir)
    
    if not os.path.exists(data_dir/'images'):
        os.makedirs(data_dir/'images')
    
    VITAL_GROUP_NAME = "Vital Signs"

    LAB_GROUPS = [
        "Resp / Acid-base / Perfusion",
        "Electrolytes / Minerals",
        "Renal",
        "Liver / Hepatobiliary",
        "Cardiac / Muscle Injury",
        "Hematology",
        "Coagulation",
        "Metabolic / Inflammation",
    ]

    ALL_GROUPS = [VITAL_GROUP_NAME] + LAB_GROUPS

    # marker pool
    DEFAULT_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*", "h", "8", "<", ">"]

    # 정상 범위 dictionary
    range_df = pd.read_excel('~/EHRTTA/data/normal_range.xlsx')
    STATS_DICT = {}
    for _, row in range_df[['var_name','mean','std']].iterrows():
        STATS_DICT[row['var_name']] = {'mean' : row['mean'], 'std' : row['std']}

    DEMO_COLS = ["age", "sex", "height", "weight"]

    LAB_COLS = ["alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir",
                    "bnd", "bun", "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", 
                    "fgn", "glu", "hgb", "inr_pt", "k", "lact",
                    "lymph", "mch", "mchc", "mcv", "methb", "mg", "na", "neut", 
                    "pco2", "ph", "phos", "plt", "po2", "ptt", "tnt", "wbc"]

    ADDITIONAL_COLS = ["fio2", "urine"]

    VITAL_COLS = ["dbp", "sbp", "map", "hr", "o2sat", "resp", "temp"]

    LAB_META = build_example_lab_meta(DEFAULT_MARKERS)

    # Data Load
    dynamics_df = pd.read_csv(data_dir/'dynamics_df.csv.gz', compression='gzip')
    static_df = pd.read_csv(data_dir/'static_df.csv.gz', compression='gzip')

    for id in tqdm(static_df['stay_id'].unique()):

        vital_df, lab_add_df, lab_obs_bins = make_resampled_df(
            df=dynamics_df,
            all_cols=VITAL_COLS + LAB_COLS + ADDITIONAL_COLS,
            vital_cols=VITAL_COLS,
            lab_cols=LAB_COLS,
            lab_meta=LAB_META,
            stay_id=id,
            time_limit_min=(60 / args.freq_min)*24,
            freq_min=args.freq_min,
            aggfunc='last'
        )

        create_icu_dashboard(
            vital_df = vital_df,
            lab_df = lab_add_df,
            lab_obs_bins = lab_obs_bins,
            vital_group_name=VITAL_GROUP_NAME,
            lab_groups=LAB_GROUPS,
            stats_dict=STATS_DICT,
            vital_vars=VITAL_COLS,
            lab_meta=LAB_META,
            save_path=f"{args.data_path}/images/{id}.png",
            start_time=0,
            end_time=(60 / args.freq_min)*24,
            fig_px=336,
            dpi=112,
            min_clip=args.min_clip,
            max_clip=args.max_clip,
            scatter_ind=args.scatter_ind,
        )