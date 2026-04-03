import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from typing import Dict, List, Optional, Tuple
from PIL import Image

import warnings
warnings.filterwarnings('ignore')


# =========================================================
# Standardization Functions
# =========================================================

def zscore_value(value: float, mean: float, std: float) -> float:
    if pd.isna(value):
        return np.nan
    
    if std is None or std == 0 or pd.isna(std):
        return 0.0
    
    return (value - mean) / std


def standardize_long_df(df: pd.DataFrame, stats_dict: Dict[str, Dict[str, float]], 
                        var_col: str = "var_name", value_col: str = "value", out_col: str = "zvalue") -> pd.DataFrame:
    """
    df: long format with [var_col, value_col]
    stats_dict[var] = {"mean": ..., "std": ...}
    """
    out = df.copy()

    def _row_z(row):
        var = row[var_col]
        val = row[value_col]
        
        if var not in stats_dict:
            return np.nan
        
        mean = stats_dict[var].get("mean", np.nan)
        std = stats_dict[var].get("std", np.nan)
        return zscore_value(val, mean, std)

    out[out_col] = out.apply(_row_z, axis=1)

    return out

# =========================================================
# Computation Lab risk scores 
# =========================================================

def compute_clinical_risk_from_z(z: float, polarity: str = "both_bad", mild_thr: float = 1.0, severe_thr: float = 2.0) -> int:
    """
    return:
        0 = low risk
        1 = medium risk
        2 = high risk

    polarity:
        - "high_bad" : 높은 값이 나쁨
        - "low_bad"  : 낮은 값이 나쁨
        - "both_bad" : 높거나 낮아도 나쁨
        - "neutral"  : 위험도 크기 반영 안 함
    """
    if pd.isna(z):
        return 0

    if polarity == "high_bad":
        score = z
    elif polarity == "low_bad":
        score = -z
    elif polarity == "both_bad":
        score = abs(z)
    else:
        score = 0.0

    if score > severe_thr:
        return 2
    elif score > mild_thr:
        return 1
    else:
        return 0

def value_color_from_z(z: float, low_thr: float = -1.0, high_thr: float = 1.0, cmap_name: str = 'coolwarm') -> str:
    """
    색상 의미:
      blue  = low
      lightgray = middle
      red   = high
    """
    cmap = plt.cm.get_cmap(cmap_name)

    if pd.isna(z):
        return "black"
    if z < low_thr:
        return cmap(0)   # blue
    elif z > high_thr:
        return cmap(1.0)   # red
    else:
        return cmap(0.5) # lightgray

def size_from_risk(risk_level: int) -> float:
    """
    임상적으로 나쁠수록 크게
    """
    size_map = {
        0: 16,   # small
        1: 36,   # medium
        2: 76,   # large
        }
    
    return size_map.get(risk_level, 16)

def marker_from_risk(risk_level: int) -> float:
    """
    임상적으로 나쁠수록 o -> △ -> X
    """
    size_map = {
        0: 'o',   # normal
        1: '^',   # mild
        2: 'X',   # severe
        }
    return size_map.get(risk_level, 'o')

# =========================================================
# Setting Grid and Panels
# =========================================================

def style_panel_as_grid( ax: plt.Axes, start_time: float, end_time: float, n_vertical_grid: int = 1, n_horizontal_grid: int = 1, 
                        show_xticklabels: bool = False, show_yticklabels: bool = False, min_clip: float = -3, max_clip: float = 3):
    """
    It only uses for draw all variables in subplot.
    """
    ax.set_xlim(start_time, end_time)
    ax.set_ylim(min_clip-0.5, max_clip+0.5)

    # # 바둑판형 grid
    # xgrid = np.linspace(start_time, end_time, n_vertical_grid)
    # ygrid = np.linspace(min_clip, max_clip, n_horizontal_grid)

    # ax.set_xticks(xgrid)
    # ax.set_yticks(ygrid)

    # # 5. 라벨 표시 여부 (이미지의 지저분한 숫자 제거)
    # if not show_xticklabels:
    #     ax.set_xticklabels([0, 50, 100, 150, 200, 250])
    # if not show_yticklabels:
    #     ax.set_yticklabels([-4, -3, -2, -1, 0, 1, 2, 3, 4])

    # ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.35)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, color='black', alpha=0.7)

    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

def assign_markers_to_labs(lab_names: List[str], marker_map: Optional[Dict[str, str]]) -> Dict[str, str]:
    """
    It only uses for draw variables distinctly.
    """

    out = {}

    for name in lab_names:
        out[name] = marker_map[name]

    return out

# =========================================================
# Create icu dashboard
# =========================================================

def create_icu_dashboard(
    vital_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    lab_obs_bins:List[int], 
    vital_group_name : str,
    lab_groups : List[str],
    stats_dict: Dict[str, Dict[str, float]],
    vital_vars: List[str],
    lab_meta: Dict[str, Dict],
    save_path: str = "icu_dashboard_336.png",
    time_col: str = "time_bin",
    value_col: str = "value",
    var_col: str = "var_name",
    group_col: str = "group",
    start_time: Optional[float] = 0,
    end_time: Optional[float] = 12*24,
    fig_px: int = 336,
    dpi: int = 112,
    min_clip: int=-3,
    max_clip: int=3,
    scatter_ind :bool = True,
):
    """
    vital_df:
      long format
      [time, variable, value]

    lab_df:
      long format
      [time, variable, value, group]

    lab_meta example:
      {
        "lactate": {"group": "Resp / Acid-base / Perfusion", "polarity": "high_bad", "marker": "o"},
        "pH":      {"group": "Resp / Acid-base / Perfusion", "polarity": "both_bad", "marker": "s"},
        ...
      }
    """

    # -----------------------------
    # 1. Standardization
    # -----------------------------
    vital_df = standardize_long_df(vital_df, stats_dict, var_col=var_col, value_col=value_col, out_col="zvalue")
    lab_df = standardize_long_df(lab_df, stats_dict, var_col=var_col, value_col=value_col, out_col="zvalue")

    # -----------------------------
    # 2. 1st subplot: Vital heatmap
    # -----------------------------

    heatmap_matrix = vital_df.pivot(index='time_bin', columns='var_name', values='zvalue')[vital_vars].values

    # figsize = (fig_px / dpi, fig_px / dpi)
    fig = plt.figure(figsize=(18,16))

    # panel이 최대한 꽉 차게
    gs = fig.add_gridspec(
        3, 3,
        left=0.04, right=0.993, top=0.985, bottom=0.03,
        wspace=0.18, hspace=0.25
    )

    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]

    ax0 = axes[0]

    for spine in ax0.spines.values():
        spine.set_linewidth(2)

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="black")

    # NaN 제외하고 clipping 범위 계산
    valid = heatmap_matrix[~np.isnan(heatmap_matrix)]
    if len(valid) == 0:
        print("All values are NaN.")
        ax0.set_facecolor('black')

    else:
        heatmap_cliped = np.clip(heatmap_matrix, min_clip, max_clip)

        # NaN을 masked array로 변환
        heatamp_masked_matrix = np.ma.masked_invalid(heatmap_cliped.T)

        ax0.imshow(
                heatamp_masked_matrix,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                interpolation="nearest",
                zorder=1
                )

        # draw lab observation bins as vertical line
        for i, b in enumerate(lab_obs_bins):
            ax0.axvline(
                x=b,
                color='green',
                linestyle='--',
                linewidth=0.8,
                alpha=0.6,
                zorder=2,
                label='Lab Observed time' if i == 0 else None
                )

    ax0.set_title(vital_group_name, pad=1.5, fontsize=16)
    ax0.set_xlim(start_time, end_time)

    ax0.set_yticks(np.arange(len(vital_vars)))
    ax0.set_yticklabels(vital_vars)
    ax0.tick_params(axis="y", length=0)

    # plt.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)

    # -----------------------------
    # 3. 2nd-9th subplots : lab scatter
    # -----------------------------

    if scatter_ind:
        # Version 1) Assign independent variable area in subplot 
        for idx, group_name in enumerate(lab_groups, start=1):
            ax = axes[idx]
            sub = lab_df[lab_df[group_col] == group_name].copy()
            ax.set_title(group_name, pad=1.5, fontsize=16)

            # 1. 이 그룹에 속한 변수 목록 고정
            group_labs = list(dict.fromkeys(sub[var_col].dropna().tolist()))
            group_labs = [x for x in group_labs if x in lab_meta]
            n_vars = len(group_labs)
            
            if n_vars == 0:
                continue

            # 2. Y축 범위를 변수 개수에 맞춰 설정 (각 변수당 1.0의 높이 할당)
            # 예: 변수가 3개면 ylim은 (0, 3)
            ax.set_ylim(-0.5, n_vars - 0.5)
            
            # Y축 눈금에 변수명 표시 (VLM이 텍스트-위치 매칭 학습 가능)
            ax.set_yticks(range(n_vars))
            ax.set_yticklabels(group_labs)

            #시간축(X축) 그리드 설정
            ax.set_xlim(start_time, end_time)
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)

            minor_ticks = []

            # 3. 시간축 그리드 및 변수별 베이스라인(z=0) 그리기
            for v_idx in range(n_vars):
                # 각 변수의 중앙선(z=0)을 흐리게 그려줌
                # 1. 배경 채우기 (홀수/짝수 행 구분으로 시각적 분리)
                if v_idx % 2 == 0:
                    ax.axhspan(v_idx - 0.5, v_idx + 0.5, color='gray', alpha=0.2, zorder=0)
                
                # 2. 레인 경계선 긋기 (Row 사이의 실선)
                ax.axhline(v_idx + 0.5 , color='black', linewidth=1, alpha=1, zorder=1)
                
                # 3. 중앙 기준선 (z=0 지점)
                ax.axhline(v_idx, color='gray', linestyle='--', linewidth=0.7, alpha=1, zorder=1)

                # 4. 각 minor tick을 구함. z=3일 때 0.3을 더했으므로, z=1, 2, 3 지점은 0.1, 0.2, 0.3 간격
                minor_ticks.extend([v_idx - 0.3, v_idx - 0.2, v_idx - 0.1, 
                                    v_idx + 0.1, v_idx + 0.2, v_idx + 0.3])

            ax.set_yticks(minor_ticks, minor=True)
            # ax.tick_params(axis='y', which='minor', length=2, width=0.5, color='gray')
            # ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5, color='black', alpha=0.2)

            # 4. 변수별 데이터 플로팅
            for v_idx, lab_name in enumerate(group_labs):
                lab_sub = sub[sub[var_col] == lab_name].copy()
                if lab_sub.empty:
                    continue

                meta = lab_meta.get(lab_name, {})
                polarity = meta.get("polarity", "both_bad")
                
                # Z-score 스케일링 (범위: -0.3 ~ +0.3로 압축하여 자기 레인 안에서만 움직이게 함)
                # z=3일 때 +0.3, z=-3일 때 -0.3가 되도록 변환
                z_values = np.clip(lab_sub["zvalue"].values, min_clip, max_clip)
                normalized_z = (z_values / max_clip) * 0.3 
                
                # 실제 그릴 Y 좌표 = 변수 인덱스(v_idx) + 정규화된 z값
                y_coords = v_idx + normalized_z

                colors = [value_color_from_z(z) for z in z_values]
                risks = [compute_clinical_risk_from_z(z, polarity=polarity) for z in z_values]
                sizes = [size_from_risk(r) for r in risks]
                markers = [marker_from_risk(r) for r in risks]

                unique_markers = set(markers) # 이 그룹에 포함된 마커 종류 (예: {'o', '^', 'X'})

                # 2. 마커 종류별로 데이터를 필터링하여 따로 그립니다.
                for m in unique_markers:
                    # 현재 마커(m)에 해당하는 인덱스만 추출
                    indices = [i for i, val in enumerate(markers) if val == m]
                    
                    if not indices:
                        continue
                        
                    # 해당 인덱스의 데이터만 필터링
                    x_subset = lab_sub[time_col].values[indices]
                    y_subset = np.array(y_coords)[indices]
                    s_subset = np.array(sizes)[indices]
                    c_subset = [colors[i] for i in indices]
                    
                    # 해당 마커 스타일로 scatter 호출
                    sc = ax.scatter(
                        x_subset,
                        y_subset,
                        marker=m,         # 리스트가 아닌 단일 마커 문자열 ('o' 등)
                        s=60,
                        c=c_subset,
                        edgecolors="black",
                        linewidths=1,
                        alpha=0.9,
                        clip_on=True     # 마커가 축 경계에서 잘리지 않도록 설정
                        )

            # 5. 스타일링: 박스 테두리 강조
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)

    ################################################

    else:
        ## Version 2) all variables in one subplot  
        for idx, group_name in enumerate(lab_groups, start=1):
            ax = axes[idx]
            sub = lab_df[lab_df[group_col] == group_name].copy()

            ax.set_title(group_name, pad=1.5)

            # 이 그룹에 속한 lab 목록
            group_labs = list(dict.fromkeys(sub[var_col].dropna().tolist()))
            group_labs = [x for x in group_labs if x in lab_meta]

            # # y축 범위는 z-score 기준으로 통일
            # ax.set_ylim(min_clip, max_clip)
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)

            style_panel_as_grid(ax=ax, start_time=start_time, end_time=end_time, n_vertical_grid=1, n_horizontal_grid=1, 
                                show_xticklabels=False, show_yticklabels=False, min_clip=min_clip, max_clip=max_clip)

            if len(group_labs) == 0:
                continue

            marker_map = assign_markers_to_labs(group_labs, marker_map={k: v.get("marker") for k, v in lab_meta.items() if "marker" in v})

            legend_handles = []

            for lab_name in group_labs:
                lab_sub = sub[sub[var_col] == lab_name].copy()
                if lab_sub.empty:
                    continue

                meta = lab_meta.get(lab_name, {})
                polarity = meta.get("polarity", "both_bad")
                marker = marker_map[lab_name]

                colors = []
                sizes = []

                for z in lab_sub["zvalue"].values:
                    color = value_color_from_z(z)
                    risk = compute_clinical_risk_from_z(z, polarity=polarity)
                    size = size_from_risk(risk)
                    colors.append(color)
                    sizes.append(size)

                ax.scatter(
                    lab_sub[time_col].values, 
                    np.clip(lab_sub["zvalue"].values, min_clip, max_clip), 
                    s=sizes, 
                    c=colors, 
                    marker=marker, 
                    edgecolors="black", 
                    linewidths=0.25, 
                    alpha=0.95,
                    clip_on=True,
                    )

                legend_handles.append(
                    Line2D([0], [0], marker=marker, color="none", markerfacecolor="purple", markeredgecolor="black",
                           markeredgewidth=1, label=lab_name)
                           )

            # panel 내부 legend
            if len(legend_handles) > 0:
                ax.legend(handles=legend_handles, loc="upper left", frameon=True, framealpha=0.8, borderpad=0.2, 
                          handletextpad=0.2, labelspacing=0.2, borderaxespad=0.2)
    
    # -----------------------------
    # 4. Clean up and Save
    # -----------------------------
    for ax in axes:
        ax.set_facecolor("white")

    # fig.savefig(save_path)
    plt.close(fig)
    fig.canvas.draw()

    # 2. 렌더링된 메모리 버퍼에서 RGBA 데이터 추출
    rgba_buffer = np.array(fig.canvas.renderer.buffer_rgba())

    # 3. PIL 이미지로 변환 및 리사이즈.
    img = Image.fromarray(rgba_buffer).convert("RGB")
    img.save(save_path)
