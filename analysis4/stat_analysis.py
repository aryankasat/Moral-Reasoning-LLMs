"""
stat_analysis.py — Statistical analyses for Analysis 4: Stage Distribution Patterns.

Functions
---------
compute_stage_distributions(df)
    Per-model stage count + proportion table.

compare_to_human_baseline(dist_df)
    Chi-square goodness-of-fit vs adult norms + Pearson residuals +
    Jensen-Shannon divergence.

compute_distribution_stats(dist_df)
    Modal stage, Shannon entropy, skewness, kurtosis, pattern label.

compute_kl_matrix(dist_df)
    Symmetric NxN Jensen-Shannon divergence matrix (models + human baseline).

identify_patterns(stat_df)
    Classify each model into a pattern type.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chi2 as chi2_dist
from scipy.special import kl_div
from config import (
    STAGES, HUMAN_ADULT, HUMAN_DIST, MODEL_META,
)

EPS = 1e-9  # smoothing to avoid log(0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Stage distribution table
# ─────────────────────────────────────────────────────────────────────────────

def compute_stage_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (model, stage):
        model_key, display_name, params_B, provider,
        stage, count, proportion
    """
    records = []
    model_order = (
        df.drop_duplicates("model_key")
        .sort_values("params_B")[["model_key", "display_name", "params_B", "log_params", "provider"]]
    )
    total_per_model = df.groupby("model_key")["kohlberg_stage"].count()

    for _, mrow in model_order.iterrows():
        mk = mrow["model_key"]
        n  = int(total_per_model[mk])
        counts = df[df["model_key"] == mk]["kohlberg_stage"].value_counts()
        for s in STAGES:
            cnt = int(counts.get(s, 0))
            records.append({
                "model_key":    mk,
                "display_name": mrow["display_name"],
                "params_B":     mrow["params_B"],
                "log_params":   mrow["log_params"],
                "provider":     mrow["provider"],
                "stage":        s,
                "count":        cnt,
                "n_total":      n,
                "proportion":   cnt / n,
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chi-square vs human adult + JSD
# ─────────────────────────────────────────────────────────────────────────────

def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base-2) — symmetric, bounded [0,1]."""
    p = p + EPS;  p = p / p.sum()
    q = q + EPS;  q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def compare_to_human_baseline(dist_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each model:
      - Chi-square goodness-of-fit vs HUMAN_ADULT (stages with expected=0
        are merged with adjacent stages to avoid zero-expected-cell violations).
      - Pearson residuals per stage.
      - Jensen-Shannon divergence (primary metric).
      - JSD vs Adolescent and Children baselines.
    """
    human_arr = np.array([HUMAN_ADULT[s] for s in STAGES])
    human_adol = np.array([HUMAN_DIST["Adolescent"][s] for s in STAGES])
    human_chil = np.array([HUMAN_DIST["Children"][s]   for s in STAGES])

    records = []
    resid_records = []

    for mk, grp in dist_df.groupby("model_key", sort=False):
        grp   = grp.sort_values("stage")
        props = grp["proportion"].values
        n     = int(grp["n_total"].iloc[0])
        obs   = (props * n).round().astype(int)

        # --- Chi-square (merge S1+S2 since expected=0 for adults) ----------
        # Merge stages with expected < 5 count into neighbours
        # Adult: stages 1,2 have expected=0 → merge with stage 3
        obs_merged  = np.array([obs[0] + obs[1] + obs[2], obs[3], obs[4], obs[5]])
        exp_props_m = np.array([0.00 + 0.00 + 0.15, 0.40, 0.35, 0.10])
        exp_merged  = exp_props_m * n

        # Guard: avoid near-zero expected (clip at 0.5)
        exp_merged = np.clip(exp_merged, 0.5, None)

        chi2_stat = float(np.sum((obs_merged - exp_merged) ** 2 / exp_merged))
        df_chi    = len(obs_merged) - 1          # = 3
        p_val     = float(1 - chi2_dist.cdf(chi2_stat, df_chi))
        sig       = p_val < 0.05

        # --- Pearson residuals per stage (against full 6-stage adult dist) --
        exp_full = human_arr * n + EPS
        residuals = (obs - exp_full) / np.sqrt(exp_full)

        for s, res in zip(STAGES, residuals):
            resid_records.append({
                "model_key": mk,
                "stage":     s,
                "pearson_residual": float(res),
                "obs_count":        int(obs[STAGES.index(s)]),
                "exp_count":        float(exp_full[STAGES.index(s)]),
            })

        # --- JSD -----------------------------------------------------------
        jsd_adult = _jsd(props, human_arr)
        jsd_adol  = _jsd(props, human_adol)
        jsd_child = _jsd(props, human_chil)

        meta = grp.iloc[0]
        records.append({
            "model_key":    mk,
            "display_name": meta["display_name"],
            "params_B":     meta["params_B"],
            "provider":     meta["provider"],
            "chi2_stat":    round(chi2_stat, 4),
            "chi2_df":      df_chi,
            "chi2_p":       round(p_val, 6),
            "significant":  sig,
            "jsd_adult":    round(jsd_adult, 6),
            "jsd_adolescent": round(jsd_adol, 6),
            "jsd_children": round(jsd_child, 6),
        })

    chi_df   = pd.DataFrame(records).sort_values("params_B")
    resid_df = pd.DataFrame(resid_records)
    return chi_df, resid_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Distribution statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_distribution_stats(dist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Modal stage, Shannon entropy, skewness, kurtosis, pattern label.
    """
    records = []
    for mk, grp in dist_df.groupby("model_key", sort=False):
        grp   = grp.sort_values("stage")
        p     = grp["proportion"].values.astype(float)
        s_arr = grp["stage"].values.astype(float)
        meta  = grp.iloc[0]

        modal_stage = int(grp.loc[grp["proportion"].idxmax(), "stage"])

        # Shannon entropy (bits)
        p_safe  = p + EPS
        p_safe /= p_safe.sum()
        entropy = float(-np.sum(p_safe * np.log2(p_safe)))

        # Mean and variance of the distribution
        mean_s  = float(np.sum(p_safe * s_arr))
        var_s   = float(np.sum(p_safe * (s_arr - mean_s) ** 2))
        std_s   = np.sqrt(var_s) if var_s > 0 else 1e-9

        # Skewness and excess kurtosis
        skewness = float(np.sum(p_safe * ((s_arr - mean_s) / std_s) ** 3))
        kurtosis = float(np.sum(p_safe * ((s_arr - mean_s) / std_s) ** 4)) - 3.0

        # Pattern flags
        ceil_flag  = bool((grp.loc[grp["stage"] >= 4, "proportion"].sum()) > 0.90)
        floor_flag = bool((grp.loc[grp["stage"] <= 2, "proportion"].sum()) > 0.50)

        # Bimodal: two local maxima separated by >= 2 stages
        bimodal = False
        peaks = []
        for i in range(1, len(p) - 1):
            if p[i] > p[i - 1] and p[i] > p[i + 1] and p[i] > 0.05:
                peaks.append(i)
        if len(peaks) >= 2 and (peaks[-1] - peaks[0]) >= 2:
            bimodal = True

        # Human-like: JSD-adult < 0.10 AND 3 <= modal_stage <= 5
        # (computed later when chi_df is merged)
        records.append({
            "model_key":    mk,
            "display_name": meta["display_name"],
            "params_B":     meta["params_B"],
            "log_params":   meta["log_params"],
            "provider":     meta["provider"],
            "modal_stage":  modal_stage,
            "mean_stage":   round(mean_s, 3),
            "entropy_bits": round(entropy, 4),
            "skewness":     round(skewness, 4),
            "kurtosis":     round(kurtosis, 4),
            "ceiling_effect": ceil_flag,
            "floor_effect":   floor_flag,
            "bimodal":        bimodal,
        })

    return pd.DataFrame(records).sort_values("params_B")


def label_patterns(stat_df: pd.DataFrame, chi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge stats + chi_df, then assign a human-readable pattern label.
    """
    merged = stat_df.merge(chi_df[["model_key", "jsd_adult"]], on="model_key", how="left")

    def _label(row):
        if row["floor_effect"]:
            return "floor-biased"
        if row["bimodal"]:
            return "bimodal"
        if row["ceiling_effect"] and row["modal_stage"] >= 5:
            return "hyper-principled"
        if row["ceiling_effect"]:
            return "ceiling-biased"
        if row["jsd_adult"] <= 0.10 and 3 <= row["modal_stage"] <= 5:
            return "human-like"
        return "divergent"

    merged["pattern"] = merged.apply(_label, axis=1)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 4. JSD matrix (models + human baselines)
# ─────────────────────────────────────────────────────────────────────────────

def compute_jsd_matrix(dist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a symmetric NxN JSD matrix where N = n_models + 3 baselines.
    Returns a square DataFrame with display_name as index/columns.
    """
    # Gather each entity's proportion vector
    entities: dict[str, np.ndarray] = {}

    for mk, grp in dist_df.groupby("model_key", sort=False):
        grp_s = grp.sort_values("stage")
        entities[grp_s["display_name"].iloc[0]] = grp_s["proportion"].values.astype(float)

    for group_name, baseline in HUMAN_DIST.items():
        entities[f"Human ({group_name})"] = np.array([baseline[s] for s in STAGES])

    names = list(entities.keys())
    n = len(names)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = _jsd(entities[names[i]], entities[names[j]])
            mat[i, j] = mat[j, i] = v

    return pd.DataFrame(mat, index=names, columns=names)
