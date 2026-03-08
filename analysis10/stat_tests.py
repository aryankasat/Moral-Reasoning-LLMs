"""
stat_tests.py — Statistical tests for Analysis 10: Stage Transition Dynamics.

Tests:
  1. Friedman test — non-parametric repeated measures on entropy across models
  2. Kruskal-Wallis — entropy differences across scale groups
  3. Chi-square on transition matrix — test sequential transitions > expected
  4. Transition window summary statistics
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Any

from config import ACTIVE_STAGES, ALPHA


# ── 1. Friedman test on per-dilemma entropy ──────────────────────────────────

def friedman_entropy_test(obs_df: pd.DataFrame) -> dict[str, Any]:
    """
    Friedman test: Are median stage scores significantly different across models?

    Uses each dilemma as a block (repeated measurement unit).
    Block = dilemma_type, treatment = model (ordered by params_B).
    DV = kohlberg_stage.

    Returns stat, p-value, interpretation.
    """
    if "dilemma_type" not in obs_df.columns:
        return {"error": "dilemma_type column missing — Friedman test skipped"}

    # Pivot: rows = dilemma, cols = model (ordered by params_B)
    pivot = (
        obs_df
        .sort_values("model_order")
        .pivot_table(
            index="dilemma_type",
            columns="model_key",
            values="kohlberg_stage",
            aggfunc="mean",
        )
    )
    pivot.dropna(how="any", inplace=True)

    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return {
            "error": (
                f"Insufficient data for Friedman test: "
                f"{pivot.shape[0]} blocks × {pivot.shape[1]} treatments."
            )
        }

    try:
        stat, p = stats.friedmanchisquare(*[pivot[col].values for col in pivot.columns])
    except Exception as e:
        return {"error": str(e)}

    return {
        "statistic":      float(stat),
        "p_value":        float(p),
        "df":             int(pivot.shape[1] - 1),
        "n_blocks":       int(pivot.shape[0]),
        "n_treatments":   int(pivot.shape[1]),
        "significant":    bool(p < ALPHA),
        "interpretation": (
            "Stage scores differ significantly across model scales (reject H₀)"
            if p < ALPHA else
            "No significant difference in stage scores across model scales"
        ),
    }


# ── 2. Kruskal-Wallis on entropy across scale groups ─────────────────────────

def kruskal_entropy_by_group(model_df: pd.DataFrame, group_col: str = "scale_group") -> dict[str, Any]:
    """
    Kruskal-Wallis test: Does entropy differ across scale groups / training types?
    """
    groups = model_df.groupby(group_col)["entropy"].apply(list)
    if len(groups) < 2:
        return {"error": f"Need ≥2 groups in '{group_col}' for KW test"}

    arrays = [np.array(v) for v in groups.values]
    # Need at least 1 observation per group
    arrays = [a for a in arrays if len(a) > 0]
    if len(arrays) < 2:
        return {"error": "After filtering, fewer than 2 groups remain"}

    try:
        stat, p = stats.kruskal(*arrays)
    except Exception as e:
        return {"error": str(e)}

    group_stats = {}
    for gname, gvals in groups.items():
        arr = np.array(gvals)
        group_stats[gname] = {
            "n":      len(arr),
            "median": float(np.median(arr)),
            "mean":   float(np.mean(arr)),
        }

    return {
        "group_col":      group_col,
        "statistic":      float(stat),
        "p_value":        float(p),
        "df":             len(arrays) - 1,
        "significant":    bool(p < ALPHA),
        "group_stats":    group_stats,
        "interpretation": (
            f"Entropy differs significantly across {group_col} groups"
            if p < ALPHA else
            f"No significant entropy difference across {group_col} groups"
        ),
    }


# ── 3. Chi-square on transition matrix ───────────────────────────────────────

def chisquare_sequential_transitions(T: np.ndarray, stages: list[int]) -> dict[str, Any]:
    """
    Chi-square test: Are sequential transitions (i → i+1) more common
    than expected under a uniform distribution?

    H₀: Transition probabilities are uniformly distributed across target stages.
    H₁: Sequential (adjacent) transitions are over-represented.
    """
    n = len(stages)
    if n < 2:
        return {"error": "Need at least 2 stages for chi-square test"}

    # Observed (aggregate row of T, weighted equally across source stages)
    observed = T.mean(axis=0)   # average target distribution
    expected = np.ones(n) / n   # uniform expectation

    # Scale to counts (use 1000 as pseudo-count total)
    pseudo_n  = 1000
    obs_counts = observed * pseudo_n
    exp_counts = expected * pseudo_n

    # Filter near-zero cells
    mask = obs_counts > 0.5
    if mask.sum() < 2:
        return {"error": "Insufficient non-zero cells for chi-square test"}

    try:
        stat, p = stats.chisquare(obs_counts[mask], f_exp=exp_counts[mask])
    except Exception as e:
        return {"error": str(e)}

    # Sequential proportion
    seq_total = sum(
        T[i, j]
        for i, s in enumerate(stages)
        for j, t in enumerate(stages)
        if t == s + 1
    )
    all_total = T.sum()
    seq_prop = seq_total / all_total if all_total > 0 else 0.0

    return {
        "statistic":             float(stat),
        "p_value":               float(p),
        "df":                    int(mask.sum() - 1),
        "sequential_proportion": float(seq_prop),
        "significant":           bool(p < ALPHA),
        "interpretation": (
            "Transition distribution is non-uniform (sequential transitions may dominate)"
            if p < ALPHA else
            "Transition distribution is consistent with uniformity"
        ),
    }


# ── 4. Regression frequency test ─────────────────────────────────────────────

def regression_frequency_test(windows: list[dict]) -> dict[str, Any]:
    """
    Test whether backwards (regression) transitions are rare using a
    binomial test against a uniform baseline.

    H₀: P(regression) = 1/3  (equal probability of forward, stay, backward)
    H₁: P(regression) < 1/3
    """
    if not windows:
        return {"error": "No transition windows to test"}

    n_total     = len(windows)
    n_regression= sum(1 for w in windows if w["is_regression"])
    n_sequential= sum(1 for w in windows if w["is_sequential"])
    n_other     = n_total - n_regression - n_sequential

    try:
        # One-tailed binomial: regression less common than 1/3?
        result = stats.binomtest(n_regression, n_total, p=1/3, alternative="less")
        p_val  = float(result.pvalue)
    except Exception as e:
        p_val  = float("nan")

    return {
        "n_total_transitions": n_total,
        "n_sequential":        n_sequential,
        "n_regression":        n_regression,
        "n_other":             n_other,
        "prop_sequential":     n_sequential / n_total if n_total > 0 else 0.0,
        "prop_regression":     n_regression / n_total if n_total > 0 else 0.0,
        "binomial_p":          p_val,
        "regressions_rare":    bool(p_val < ALPHA) if not np.isnan(p_val) else False,
        "interpretation": (
            "Regressions are significantly rare (consistent with progressive development)"
            if (not np.isnan(p_val) and p_val < ALPHA) else
            "Cannot confirm regressions are rare"
        ),
    }


# ── 5. Spearman correlation: model order vs mean stage ───────────────────────

def spearman_scale_stage(model_df: pd.DataFrame) -> dict[str, Any]:
    """
    Spearman ρ between model_order (scale proxy) and mean_stage.
    Tests if higher-scale models show higher average moral stage.
    """
    x = model_df["model_order"].values
    y = model_df["mean_stage"].values

    try:
        rho, p = stats.spearmanr(x, y)
    except Exception as e:
        return {"error": str(e)}

    return {
        "spearman_rho": float(rho),
        "p_value":      float(p),
        "significant":  bool(p < ALPHA),
        "interpretation": (
            "Significant positive correlation: larger models show higher moral stages"
            if (rho > 0 and p < ALPHA)
            else (
                "Significant negative correlation: larger models show lower moral stages"
                if (rho < 0 and p < ALPHA)
                else "No significant monotonic trend between scale and moral stage"
            )
        ),
    }


# ── Bundle all tests ──────────────────────────────────────────────────────────

def run_all_tests(
    obs_df:     pd.DataFrame,
    model_df:   pd.DataFrame,
    T:          np.ndarray,
    stages:     list[int],
    windows:    list[dict],
) -> dict[str, Any]:
    """Run all statistical tests and return bundled results dict."""
    return {
        "friedman":         friedman_entropy_test(obs_df),
        "kw_scale_group":   kruskal_entropy_by_group(model_df, "scale_group"),
        "kw_training_type": kruskal_entropy_by_group(model_df, "training_type"),
        "chisq_transitions":chisquare_sequential_transitions(T, stages),
        "regression_freq":  regression_frequency_test(windows),
        "spearman_scale":   spearman_scale_stage(model_df),
    }
