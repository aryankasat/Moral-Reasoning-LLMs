"""
stat_tests.py — Statistical tests for Analysis 11: RLHF Causal Analysis.

Tests:
  1. Per-pair Mann-Whitney U test (base vs. instruct stage scores)
  2. Paired t-test across pairs (mean stage: instruct − base)
  3. Wilcoxon signed-rank test across pairs
  4. Chi-square goodness of fit: post-conventional (S5+S6) proportion
  5. Cross-pair sign test (directional consistency)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Any

from config import ALPHA, PAIR_ORDER, STAGES


# ── 1. Per-pair Mann-Whitney U test ───────────────────────────────────────────

def mann_whitney_per_pair(obs_df: pd.DataFrame) -> dict[str, dict]:
    """
    Mann-Whitney U test for each architecture pair:
      H₀: stage distributions of base and instruct are identical
      H₁: instruct stochastically dominates base (one-tailed)

    Returns dict: pair_id → {statistic, p_value, significant, effect_r, interpretation}
    """
    results = {}
    for pair_id in PAIR_ORDER:
        x_base     = obs_df[(obs_df["pair_id"] == pair_id) & (obs_df["variant"] == "base")]["kohlberg_stage"].values
        x_instruct = obs_df[(obs_df["pair_id"] == pair_id) & (obs_df["variant"] == "instruct")]["kohlberg_stage"].values

        if len(x_base) < 3 or len(x_instruct) < 3:
            results[pair_id] = {"error": "insufficient data (n < 3)"}
            continue

        try:
            stat, p = stats.mannwhitneyu(x_instruct, x_base, alternative="greater")
            # Effect size r = z / sqrt(N)
            n1, n2 = len(x_instruct), len(x_base)
            z = (stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            r = float(z / np.sqrt(n1 + n2))
        except Exception as e:
            results[pair_id] = {"error": str(e)}
            continue

        results[pair_id] = {
            "pair_id":      pair_id,
            "n_base":       int(len(x_base)),
            "n_instruct":   int(len(x_instruct)),
            "statistic":    float(stat),
            "z":            float(z),
            "p_value":      float(p),
            "effect_r":     r,
            "significant":  bool(p < ALPHA),
            "interpretation": (
                "Instruct model shows significantly higher moral stages than base (p < α)"
                if p < ALPHA else
                "No significant stochastic dominance of instruct over base"
            ),
        }

    return results


# ── 2. Paired t-test across pairs ─────────────────────────────────────────────

def paired_ttest_across_pairs(pair_metrics: pd.DataFrame) -> dict[str, Any]:
    """
    Paired t-test on mean_stage across architecture pairs.
      H₀: mean delta (instruct − base) = 0 across pairs
      H₁: mean delta > 0

    Uses pair_metrics DataFrame (one row per pair).
    """
    if len(pair_metrics) < 2:
        return {"error": "Need ≥ 2 pairs for paired t-test"}

    base_means     = pair_metrics["base_mean"].values
    instruct_means = pair_metrics["instruct_mean"].values

    try:
        stat, p_two = stats.ttest_rel(instruct_means, base_means)
        p_one = float(p_two / 2) if stat > 0 else 1.0
    except Exception as e:
        return {"error": str(e)}

    deltas = instruct_means - base_means

    return {
        "statistic":       float(stat),
        "p_value_twotail": float(p_two),
        "p_value_onetail": p_one,
        "mean_delta":      float(deltas.mean()),
        "std_delta":       float(deltas.std(ddof=1)),
        "n_pairs":         len(pair_metrics),
        "significant":     bool(p_one < ALPHA),
        "interpretation": (
            "RLHF consistently elevates mean moral stage across architectures (p < α)"
            if p_one < ALPHA else
            "No significant paired uplift across architecture pairs"
        ),
    }


# ── 3. Wilcoxon signed-rank test ───────────────────────────────────────────────

def wilcoxon_across_pairs(pair_metrics: pd.DataFrame) -> dict[str, Any]:
    """
    Wilcoxon signed-rank test on mean stage deltas across pairs.
    Non-parametric alternative to the paired t-test.
    """
    if len(pair_metrics) < 3:
        return {"error": "Need ≥ 3 pairs for Wilcoxon test; skipped"}

    deltas = (pair_metrics["instruct_mean"] - pair_metrics["base_mean"]).values

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = stats.wilcoxon(deltas, alternative="greater")
    except Exception as e:
        return {"error": str(e)}

    return {
        "statistic":  float(stat),
        "p_value":    float(p),
        "n_pairs":    len(pair_metrics),
        "significant": bool(p < ALPHA),
        "interpretation": (
            "Wilcoxon: RLHF uplift is significantly positive across pairs (p < α)"
            if p < ALPHA else
            "Wilcoxon: RLHF uplift not significant across pairs"
        ),
    }


# ── 4. Chi-square: post-conventional proportion ───────────────────────────────

def chisq_postconventional(obs_df: pd.DataFrame) -> dict[str, dict]:
    """
    For each pair: chi-square test comparing post-conventional (stage ≥ 5)
    count proportions between base and instruct.

    H₀: P(stage≥5 | base) = P(stage≥5 | instruct)
    """
    results = {}
    for pair_id in PAIR_ORDER:
        base_grp     = obs_df[(obs_df["pair_id"] == pair_id) & (obs_df["variant"] == "base")]
        instruct_grp = obs_df[(obs_df["pair_id"] == pair_id) & (obs_df["variant"] == "instruct")]

        if base_grp.empty or instruct_grp.empty:
            results[pair_id] = {"error": "missing data"}
            continue

        def _counts(grp):
            n_pc = int((grp["kohlberg_stage"] >= 5).sum())
            n_conv = len(grp) - n_pc
            return np.array([n_conv, n_pc])

        obs_table = np.array([_counts(base_grp), _counts(instruct_grp)])

        try:
            chi2, p, dof, expected = stats.chi2_contingency(obs_table, correction=True)
        except Exception as e:
            results[pair_id] = {"error": str(e)}
            continue

        base_pc_prop     = float(obs_table[0, 1] / obs_table[0].sum())
        instruct_pc_prop = float(obs_table[1, 1] / obs_table[1].sum())

        results[pair_id] = {
            "chi2":               float(chi2),
            "p_value":            float(p),
            "dof":                int(dof),
            "base_postconv_prop": base_pc_prop,
            "instruct_postconv_prop": instruct_pc_prop,
            "delta_postconv_prop": instruct_pc_prop - base_pc_prop,
            "significant":        bool(p < ALPHA),
            "interpretation": (
                "Instruct model shows significantly higher post-conventional proportion (p < α)"
                if p < ALPHA else
                "No significant difference in post-conventional proportions"
            ),
        }

    return results


# ── 5. Sign test: cross-pair consistency ──────────────────────────────────────

def sign_test_consistency(pair_metrics: pd.DataFrame) -> dict[str, Any]:
    """
    Sign test: Is the direction of RLHF uplift (positive delta) consistent across pairs?
    H₀: P(delta > 0) = 0.5 (no consistent direction)
    H₁: P(delta > 0) > 0.5 (RLHF systematically elevates stage)
    """
    from scipy.stats import binomtest

    deltas     = pair_metrics["delta_mean"].values
    n_pairs    = len(deltas)
    n_positive = int((deltas > 0).sum())

    result = binomtest(n_positive, n_pairs, p=0.5, alternative="greater")

    return {
        "n_pairs":    n_pairs,
        "n_positive": n_positive,
        "p_value":    float(result.pvalue),
        "significant": bool(result.pvalue < ALPHA),
        "interpretation": (
            f"Consistent positive RLHF uplift across all {n_pairs} pairs (sign p={result.pvalue:.4f})"
            if result.pvalue < ALPHA else
            "Uplift direction is not statistically consistent across pairs"
        ),
    }


# ── Bundle all tests ───────────────────────────────────────────────────────────

def run_all_tests(obs_df: pd.DataFrame, pair_metrics: pd.DataFrame) -> dict[str, Any]:
    """Run all statistical tests and return bundled results dict."""
    return {
        "mann_whitney_per_pair":  mann_whitney_per_pair(obs_df),
        "paired_ttest":           paired_ttest_across_pairs(pair_metrics),
        "wilcoxon":               wilcoxon_across_pairs(pair_metrics),
        "chisq_postconv_per_pair":chisq_postconventional(obs_df),
        "sign_test":              sign_test_consistency(pair_metrics),
    }
