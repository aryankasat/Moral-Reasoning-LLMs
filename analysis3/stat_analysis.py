"""
stat_analysis.py — All statistical computations for Analysis 3 (Consistency).

Public API
----------
compute_within_model_sd(df)      -> pd.DataFrame   per-model SD + human baseline comparison
run_prompt_anova(df)             -> dict            KW tests per model + global summary
compute_icc_per_model(df)        -> pd.DataFrame   ICC(2,1) per model + scale correlation
compute_sample_agreement(df)     -> pd.DataFrame   exact/majority agreement + MAD per cell
compute_model_summary(df)        -> pd.DataFrame   merged per-model stats for plotting
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.stats import kruskal, ttest_1samp, spearmanr
import pingouin as pg


# ── Human baseline ────────────────────────────────────────────────────────
# Colby & Kohlberg (1987): adult moral reasoning stage SD ≈ 0.67
HUMAN_SD_BASELINE: float = 0.67


# ── 1. Within-model Stage SD ─────────────────────────────────────────────

def compute_within_model_sd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stage SD across all observations per model.

    Also runs a one-sample t-test comparing each model's SD to the human
    baseline (0.67 stage SD, Colby & Kohlberg 1987).

    Returns
    -------
    pd.DataFrame with columns:
        model_key, display_name, params_B, log_params, provider,
        n_obs, mean_stage, std_stage, min_stage, max_stage, stage_range
    """
    rows = []
    for key, grp in df.groupby("model_key"):
        stages = grp["kohlberg_stage"].astype(float)
        rows.append({
            "model_key":    key,
            "display_name": grp["display_name"].iloc[0],
            "params_B":     grp["params_B"].iloc[0],
            "log_params":   grp["log_params"].iloc[0],
            "provider":     grp["provider"].iloc[0],
            "n_obs":        len(stages),
            "mean_stage":   float(stages.mean()),
            "std_stage":    float(stages.std(ddof=1)),
            "min_stage":    int(stages.min()),
            "max_stage":    int(stages.max()),
            "stage_range":  int(stages.max() - stages.min()),
        })
    sd_df = (
        pd.DataFrame(rows)
        .sort_values("params_B")
        .reset_index(drop=True)
    )

    # One-sample t-test: are model SDs significantly different from human baseline?
    model_sds = sd_df["std_stage"].values
    t_stat, p_val = ttest_1samp(model_sds, popmean=HUMAN_SD_BASELINE)
    cohen_d = (float(np.mean(model_sds)) - HUMAN_SD_BASELINE) / float(np.std(model_sds, ddof=1))

    sd_df.attrs["ttest_vs_human"] = {
        "t": float(t_stat),
        "p": float(p_val),
        "df": len(model_sds) - 1,
        "mean_model_sd": float(np.mean(model_sds)),
        "human_baseline": HUMAN_SD_BASELINE,
        "cohen_d": float(cohen_d),
    }
    return sd_df


# ── 2. Prompt-type ANOVA (Kruskal-Wallis per model) ──────────────────────

def run_prompt_anova(df: pd.DataFrame) -> dict:
    """
    Per-model Kruskal-Wallis test: Stage ~ Prompt_Type.
    Effect size η² = (H − k + 1) / (n − k)  [based on Kwak & Kim, 2017].

    Also runs a global repeated-measures-style test across all models via
    pingouin mixed ANOVA (model as between-subject factor).

    Returns
    -------
    dict with keys:
        per_model  : pd.DataFrame  (model-level KW results)
        global_kw  : dict          (pooled KW across all data)
    """
    prompt_types = df["prompt_type"].dropna().unique()
    k = len(prompt_types)

    per_model_rows = []
    for key, grp in df.groupby("model_key"):
        groups = [
            grp.loc[grp["prompt_type"] == pt, "kohlberg_stage"].values
            for pt in prompt_types
            if (grp["prompt_type"] == pt).sum() > 0
        ]
        n = sum(len(g) for g in groups)

        if len(groups) < 2 or any(len(g) == 0 for g in groups):
            per_model_rows.append({
                "model_key":    key,
                "display_name": grp["display_name"].iloc[0],
                "params_B":     grp["params_B"].iloc[0],
                "provider":     grp["provider"].iloc[0],
                "H_stat":       np.nan,
                "p_value":      np.nan,
                "eta_sq":       np.nan,
                "significant":  False,
                "mean_ZS":      np.nan,
                "mean_COT":     np.nan,
                "mean_RP":      np.nan,
            })
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H, p = kruskal(*groups)

        # η² = (H − k + 1) / (n − k)
        eta_sq = (H - k + 1) / (n - k) if (n - k) > 0 else np.nan
        eta_sq = max(0.0, float(eta_sq))

        # Per-prompt means
        means = {pt: float(grp.loc[grp["prompt_type"] == pt, "kohlberg_stage"].mean())
                 for pt in ["ZERO_SHOT", "COT", "ROLEPLAY"]}

        per_model_rows.append({
            "model_key":    key,
            "display_name": grp["display_name"].iloc[0],
            "params_B":     grp["params_B"].iloc[0],
            "provider":     grp["provider"].iloc[0],
            "H_stat":       float(H),
            "p_value":      float(p),
            "eta_sq":       eta_sq,
            "significant":  bool(p < 0.05),
            "mean_ZS":      means.get("ZERO_SHOT", np.nan),
            "mean_COT":     means.get("COT", np.nan),
            "mean_RP":      means.get("ROLEPLAY", np.nan),
        })

    per_model_df = (
        pd.DataFrame(per_model_rows)
        .sort_values("params_B")
        .reset_index(drop=True)
    )

    # Global pooled KW
    global_groups = [
        df.loc[df["prompt_type"] == pt, "kohlberg_stage"].values
        for pt in ["ZERO_SHOT", "COT", "ROLEPLAY"]
        if (df["prompt_type"] == pt).sum() > 0
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H_global, p_global = kruskal(*global_groups)
    n_global = sum(len(g) for g in global_groups)
    eta_sq_global = max(0.0, (H_global - k + 1) / (n_global - k))

    return {
        "per_model": per_model_df,
        "global_kw": {
            "H": float(H_global),
            "p": float(p_global),
            "eta_sq": float(eta_sq_global),
            "n": n_global,
            "k": k,
        },
    }


# ── 3. Intraclass Correlation Coefficient (ICC) per model ────────────────

def compute_icc_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ICC(2,1) for each model.

    Measurement design (within each model):
      - "Subjects" = dilemmas (6 levels)
      - "Raters"   = prompt_types (3 levels): ZERO_SHOT, COT, ROLEPLAY
      - Rating     = mean kohlberg_stage for that dilemma×prompt_type
        (averaged over the 1 sample we have per cell)

    ICC(2,1): two-way random effects, absolute agreement, single measures.
    Interpretation: consistency of stage scores across dilemmas, treating
    prompt_type as the "rater" axis.

    Also computes Spearman correlation between params_B and ICC value.

    Returns
    -------
    pd.DataFrame with columns:
        model_key, display_name, params_B, log_params, provider,
        icc, icc_lo, icc_hi, icc_p, icc_interp
    """
    rows = []
    for key, grp in df.groupby("model_key"):
        # Pivot to (dilemma × prompt_type) matrix of mean stage
        pivot = (
            grp.groupby(["dilemma_type", "prompt_type"])["kohlberg_stage"]
            .mean()
            .unstack("prompt_type")
        )

        if pivot.shape[1] < 2 or pivot.shape[0] < 2:
            rows.append({
                "model_key":    key,
                "display_name": grp["display_name"].iloc[0],
                "params_B":     grp["params_B"].iloc[0],
                "log_params":   grp["log_params"].iloc[0],
                "provider":     grp["provider"].iloc[0],
                "icc":          np.nan,
                "icc_lo":       np.nan,
                "icc_hi":       np.nan,
                "icc_p":        np.nan,
                "icc_interp":   "insufficient data",
            })
            continue

        # Reshape to long format for pingouin
        long = pivot.reset_index().melt(
            id_vars="dilemma_type",
            var_name="prompt_type",
            value_name="stage",
        ).dropna(subset=["stage"])

        # pingouin.intraclass_corr expects: targets (subjects), raters, ratings
        try:
            icc_res = pg.intraclass_corr(
                data=long,
                targets="dilemma_type",
                raters="prompt_type",
                ratings="stage",
            ).set_index("Type")

            # ICC(2,1) = two-way random, absolute agreement, single measures
            # Note: pingouin column is 'CI95' (list), not 'CI95%'
            row_icc = icc_res.loc["ICC2"]
            icc_val  = float(row_icc["ICC"])
            ci_col   = "CI95%" if "CI95%" in row_icc.index else "CI95"
            ci_val   = row_icc[ci_col]
            icc_lo   = float(ci_val[0])
            icc_hi   = float(ci_val[1])
            icc_p    = float(row_icc["pval"])

            # If ICC is NaN (zero variance — perfect consistency), set to 1.0
            if np.isnan(icc_val) and long["stage"].std() == 0.0:
                icc_val, icc_lo, icc_hi, icc_p = 1.0, 1.0, 1.0, 0.0
        except Exception as e:
            # Zero-variance case: all stages identical → perfect ICC
            if long["stage"].std() == 0.0:
                icc_val, icc_lo, icc_hi, icc_p = 1.0, 1.0, 1.0, 0.0
            else:
                icc_val = icc_lo = icc_hi = np.nan
                icc_p   = np.nan
                print(f"  [WARN] ICC failed for {key}: {e}")

        # Interpretation (Koo & Mae, 2016)
        if np.isnan(icc_val):
            interp = "insufficient data"
        elif icc_val >= 1.0:
            interp = "perfect"
        elif icc_val < 0.50:
            interp = "poor"
        elif icc_val < 0.75:
            interp = "moderate"
        elif icc_val < 0.90:
            interp = "good"
        else:
            interp = "excellent"

        rows.append({
            "model_key":    key,
            "display_name": grp["display_name"].iloc[0],
            "params_B":     grp["params_B"].iloc[0],
            "log_params":   grp["log_params"].iloc[0],
            "provider":     grp["provider"].iloc[0],
            "icc":          icc_val,
            "icc_lo":       icc_lo,
            "icc_hi":       icc_hi,
            "icc_p":        icc_p,
            "icc_interp":   interp,
        })

    icc_df = (
        pd.DataFrame(rows)
        .sort_values("params_B")
        .reset_index(drop=True)
    )

    # Spearman: params_B vs ICC
    valid = icc_df.dropna(subset=["icc"])
    if len(valid) >= 3:
        rho, p_corr = spearmanr(valid["params_B"], valid["icc"])
        icc_df.attrs["scale_icc_corr"] = {"rho": float(rho), "p": float(p_corr)}
    else:
        icc_df.attrs["scale_icc_corr"] = {"rho": np.nan, "p": np.nan}

    return icc_df


# ── 4. Sample agreement ───────────────────────────────────────────────────

def compute_sample_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, dilemma, prompt_type) cell, we have ≥1 sample rows.
    Compute:
      - n_samples        : number of samples per cell
      - exact_agree      : 1 if all samples have same stage, else 0
      - majority_agree   : 1 if ≥2 samples agree (for n=3)
      - mad              : mean absolute deviation from cell mean stage
      - stage_mode       : most common stage in cell

    Aggregated per model:
      - exact_agree_rate    : fraction of cells with exact agreement
      - majority_agree_rate : fraction of cells with majority agreement
      - mean_mad            : mean MAD across cells

    Returns
    -------
    pd.DataFrame — per-model summary (sorted by params_B)
    Also stores raw cell-level df in attrs["cell_df"]
    """
    cell_rows = []
    for (model_key, dilemma, pt), grp in df.groupby(
        ["model_key", "dilemma_type", "prompt_type"]
    ):
        stages = grp["kohlberg_stage"].astype(float).values
        n = len(stages)
        if n == 0:
            continue

        stage_mode_val = int(pd.Series(stages).mode().iloc[0])
        exact  = int(len(set(stages)) == 1)
        major  = int(np.sum(stages == stage_mode_val) >= 2) if n >= 2 else exact
        mad    = float(np.mean(np.abs(stages - np.mean(stages))))

        cell_rows.append({
            "model_key":    model_key,
            "display_name": grp["display_name"].iloc[0],
            "params_B":     grp["params_B"].iloc[0],
            "provider":     grp["provider"].iloc[0],
            "dilemma_type": dilemma,
            "prompt_type":  pt,
            "n_samples":    n,
            "stage_mode":   stage_mode_val,
            "exact_agree":  exact,
            "majority_agree": major,
            "mad":          mad,
        })

    cell_df = pd.DataFrame(cell_rows)

    # Per-model aggregation
    model_rows = []
    for key, grp in cell_df.groupby("model_key"):
        model_rows.append({
            "model_key":           key,
            "display_name":        grp["display_name"].iloc[0],
            "params_B":            grp["params_B"].iloc[0],
            "log_params":          np.log10(grp["params_B"].iloc[0]),
            "provider":            grp["provider"].iloc[0],
            "n_cells":             len(grp),
            "exact_agree_rate":    float(grp["exact_agree"].mean()),
            "majority_agree_rate": float(grp["majority_agree"].mean()),
            "mean_mad":            float(grp["mad"].mean()),
        })

    agree_df = (
        pd.DataFrame(model_rows)
        .sort_values("params_B")
        .reset_index(drop=True)
    )
    agree_df.attrs["cell_df"] = cell_df
    return agree_df


# ── 5. Combined per-model summary ─────────────────────────────────────────

def compute_model_summary(
    sd_df: pd.DataFrame,
    icc_df: pd.DataFrame,
    agree_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge SD, ICC, and agreement stats into a single per-model summary.
    """
    merge_keys = ["model_key", "display_name", "params_B", "log_params", "provider"]

    summary = sd_df[merge_keys + ["mean_stage", "std_stage", "stage_range"]].copy()
    summary = summary.merge(
        icc_df[merge_keys + ["icc", "icc_lo", "icc_hi", "icc_interp"]],
        on=merge_keys, how="left",
    )
    summary = summary.merge(
        agree_df[merge_keys + ["exact_agree_rate", "majority_agree_rate", "mean_mad"]],
        on=merge_keys, how="left",
    )
    return summary.sort_values("params_B").reset_index(drop=True)
