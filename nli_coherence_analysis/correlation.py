"""
correlation.py — Correlation analysis between NLI coherence and decoupling scores.

Computes:
  1. Observation-level point-biserial correlation (NLI entailment vs. is_consistent)
  2. Model-level Pearson/Spearman correlation (mean NLI vs. consistency %)
  3. Pair-level comparison (base vs. instruct NLI coherence) — RLHF data
  4. Partial correlation (NLI vs. consistency, controlling for Kohlberg stage)
  5. Bootstrap confidence intervals on all correlations
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Any

from config import ALPHA, N_BOOTSTRAP, RLHF_PAIR_ORDER


# ── 1. Observation-level point-biserial correlation ──────────────────────────

def pointbiserial_nli_consistency(df: pd.DataFrame) -> dict[str, Any]:
    """
    Point-biserial correlation between continuous NLI entailment score
    and binary is_consistent flag.

    Tests whether observations judged consistent by the Kohlberg framework
    also have higher NLI entailment (framework-independent validation).
    """
    valid = df.dropna(subset=["nli_entailment", "is_consistent"]).copy()

    if len(valid) < 5:
        return {"error": "Insufficient data (n < 5)"}

    r, p = stats.pointbiserialr(
        valid["is_consistent"].astype(int).values,
        valid["nli_entailment"].values,
    )

    # Mean NLI by consistency group
    consistent = valid[valid["is_consistent"] == 1]["nli_entailment"]
    inconsistent = valid[valid["is_consistent"] == 0]["nli_entailment"]

    return {
        "r":                    float(r),
        "p_value":              float(p),
        "significant":          bool(p < ALPHA),
        "n":                    len(valid),
        "mean_nli_consistent":  float(consistent.mean()) if len(consistent) > 0 else np.nan,
        "mean_nli_inconsistent":float(inconsistent.mean()) if len(inconsistent) > 0 else np.nan,
        "mean_diff":            float(consistent.mean() - inconsistent.mean())
                                if len(consistent) > 0 and len(inconsistent) > 0 else np.nan,
        "interpretation": (
            f"Significant positive correlation (r={r:.3f}, p={p:.4f}): "
            f"consistent observations have higher NLI entailment"
            if p < ALPHA and r > 0 else
            f"No significant correlation (r={r:.3f}, p={p:.4f})"
        ),
    }


# ── 2. Model-level correlation ───────────────────────────────────────────────

def model_level_correlation(
    model_summary: pd.DataFrame,
) -> dict[str, Any]:
    """
    Pearson and Spearman correlation between per-model mean NLI entailment
    and per-model consistency percentage.

    Parameters
    ----------
    model_summary : pd.DataFrame
        Must contain columns: mean_nli_entailment, consistency_pct
    """
    valid = model_summary.dropna(subset=["mean_nli_entailment", "consistency_pct"])

    if len(valid) < 4:
        return {"error": f"Insufficient models (n={len(valid)}, need ≥ 4)"}

    x = valid["mean_nli_entailment"].values
    y = valid["consistency_pct"].values

    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)

    return {
        "n_models":        len(valid),
        "pearson_r":       float(r_pearson),
        "pearson_p":       float(p_pearson),
        "pearson_sig":     bool(p_pearson < ALPHA),
        "spearman_rho":    float(r_spearman),
        "spearman_p":      float(p_spearman),
        "spearman_sig":    bool(p_spearman < ALPHA),
        "interpretation": (
            f"Strong model-level convergence: Pearson r={r_pearson:.3f} (p={p_pearson:.4f}), "
            f"Spearman ρ={r_spearman:.3f} (p={p_spearman:.4f})"
            if p_pearson < ALPHA or p_spearman < ALPHA else
            f"No significant model-level correlation "
            f"(Pearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f})"
        ),
    }


# ── 3. Pair-level RLHF comparison ────────────────────────────────────────────

def rlhf_pair_comparison(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compare NLI coherence between base and instruct variants.

    For RLHF data: tests whether alignment training increases not just
    stage scores but also genuine reasoning-action coherence.
    """
    if "variant" not in df.columns or "pair_id" not in df.columns:
        return {"error": "Not RLHF data — missing variant/pair_id columns"}

    valid = df.dropna(subset=["nli_entailment"])

    base_scores = valid[valid["variant"] == "base"]["nli_entailment"]
    instruct_scores = valid[valid["variant"] == "instruct"]["nli_entailment"]

    if len(base_scores) < 3 or len(instruct_scores) < 3:
        return {"error": "Insufficient data for base/instruct comparison"}

    # Overall comparison
    u_stat, u_p = stats.mannwhitneyu(
        instruct_scores.values, base_scores.values, alternative="greater"
    )

    # Per-pair deltas
    pair_results = []
    for pair_id in RLHF_PAIR_ORDER:
        pair_data = valid[valid["pair_id"] == pair_id]
        base = pair_data[pair_data["variant"] == "base"]["nli_entailment"]
        inst = pair_data[pair_data["variant"] == "instruct"]["nli_entailment"]

        if len(base) > 0 and len(inst) > 0:
            pair_results.append({
                "pair_id":         pair_id,
                "base_mean_nli":   float(base.mean()),
                "instruct_mean_nli": float(inst.mean()),
                "delta_nli":       float(inst.mean() - base.mean()),
                "n_base":          len(base),
                "n_instruct":      len(inst),
            })

    return {
        "overall_base_mean":      float(base_scores.mean()),
        "overall_instruct_mean":  float(instruct_scores.mean()),
        "overall_delta":          float(instruct_scores.mean() - base_scores.mean()),
        "mann_whitney_U":         float(u_stat),
        "mann_whitney_p":         float(u_p),
        "significant":            bool(u_p < ALPHA),
        "per_pair":               pair_results,
        "interpretation": (
            f"Instruct models show significantly higher NLI coherence "
            f"(Δ={instruct_scores.mean() - base_scores.mean():.3f}, p={u_p:.4f})"
            if u_p < ALPHA else
            f"No significant difference in NLI coherence between variants "
            f"(Δ={instruct_scores.mean() - base_scores.mean():.3f}, p={u_p:.4f})"
        ),
    }


# ── 4. Partial correlation ───────────────────────────────────────────────────

def partial_correlation_controlling_stage(df: pd.DataFrame) -> dict[str, Any]:
    """
    Partial correlation between NLI entailment and is_consistent,
    controlling for Kohlberg stage.

    Tests whether NLI coherence captures information beyond what the
    stage label already provides.
    """
    valid = df.dropna(subset=["nli_entailment", "is_consistent", "kohlberg_stage"]).copy()

    if len(valid) < 10:
        return {"error": "Insufficient data (n < 10)"}

    x = valid["nli_entailment"].values
    y = valid["is_consistent"].astype(float).values
    z = valid["kohlberg_stage"].astype(float).values

    # Residualise x and y on z
    # x_resid = x - predicted(x ~ z)
    # y_resid = y - predicted(y ~ z)
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)

    x_resid = x - (slope_xz * z + intercept_xz)
    y_resid = y - (slope_yz * z + intercept_yz)

    r_partial, p_partial = stats.pearsonr(x_resid, y_resid)

    # Also compute zero-order for comparison
    r_zero, p_zero = stats.pearsonr(x, y)

    return {
        "r_partial":           float(r_partial),
        "p_partial":           float(p_partial),
        "partial_sig":         bool(p_partial < ALPHA),
        "r_zero_order":        float(r_zero),
        "p_zero_order":        float(p_zero),
        "n":                   len(valid),
        "r_change":            float(abs(r_partial) - abs(r_zero)),
        "interpretation": (
            f"Partial r={r_partial:.3f} (p={p_partial:.4f}) after controlling for stage "
            f"(zero-order r={r_zero:.3f}). "
            + (
                "NLI captures coherence information beyond stage labels."
                if p_partial < ALPHA else
                "NLI coherence is largely explained by the stage label."
            )
        ),
    }


# ── 5. Bootstrap CI ──────────────────────────────────────────────────────────

def bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "pearson",
    n_boot: int = N_BOOTSTRAP,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap confidence interval on a correlation coefficient.

    Parameters
    ----------
    x, y : arrays
    method : 'pearson' or 'spearman'
    n_boot : number of bootstrap resamples
    ci : confidence level

    Returns
    -------
    dict with: observed_r, ci_lower, ci_upper, se
    """
    rng = np.random.default_rng(rng_seed)
    n = len(x)

    corr_func = stats.pearsonr if method == "pearson" else stats.spearmanr

    observed_r = float(corr_func(x, y)[0])

    boot_rs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boot_rs[i] = corr_func(x[idx], y[idx])[0]
        except Exception:
            boot_rs[i] = np.nan

    boot_rs = boot_rs[~np.isnan(boot_rs)]
    alpha = (1 - ci) / 2

    return {
        "observed_r":  observed_r,
        "ci_lower":    float(np.percentile(boot_rs, alpha * 100)),
        "ci_upper":    float(np.percentile(boot_rs, (1 - alpha) * 100)),
        "se":          float(boot_rs.std(ddof=1)),
        "n_boot":      len(boot_rs),
    }


# ── Bundle all analyses ──────────────────────────────────────────────────────

def build_model_nli_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate NLI scores to model level for correlation analysis.

    Returns one row per model_key with mean/median/std of NLI scores.
    """
    valid = scored_df.dropna(subset=["nli_entailment"])

    records = []
    for mk, grp in valid.groupby("model_key", sort=False):
        meta = grp.iloc[0]
        ent = grp["nli_entailment"].values

        # Consistency metric
        consist_valid = grp[grp["is_consistent"].notna()]
        n_valid = len(consist_valid)
        n_consistent = int(consist_valid["is_consistent"].sum()) if n_valid > 0 else 0
        consistency_pct = (n_consistent / n_valid * 100) if n_valid > 0 else np.nan

        records.append({
            "model_key":          mk,
            "display_name":       meta.get("display_name", mk),
            "params_B":           meta.get("params_B", np.nan),
            "provider":           meta.get("provider", ""),
            "variant":            meta.get("variant", ""),
            "pair_id":            meta.get("pair_id", ""),
            "data_source":        meta.get("data_source", ""),
            "n_scored":           len(grp),
            "mean_nli_entailment":float(ent.mean()),
            "median_nli_entailment": float(np.median(ent)),
            "std_nli_entailment": float(ent.std(ddof=1)) if len(ent) > 1 else 0.0,
            "min_nli_entailment": float(ent.min()),
            "max_nli_entailment": float(ent.max()),
            "mean_nli_contradiction": float(grp["nli_contradiction"].mean()),
            "mean_nli_neutral":   float(grp["nli_neutral"].mean()),
            "consistency_pct":    consistency_pct,
            "n_consistent":       n_consistent,
            "n_valid_actions":    n_valid,
            "mean_stage":         float(grp["kohlberg_stage"].mean()),
        })

    return pd.DataFrame(records)


def run_all_correlations(
    scored_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    data_source: str,
) -> dict[str, Any]:
    """
    Run all correlation analyses and return bundled results.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Observation-level data with NLI scores and consistency flags.
    model_summary : pd.DataFrame
        Model-level aggregation from build_model_nli_summary().
    data_source : str
        'main' or 'rlhf' — determines which analyses to run.
    """
    results: dict[str, Any] = {}

    # 1. Observation-level point-biserial
    print("  [1/4] Point-biserial correlation (NLI entailment vs. consistency)…")
    results["pointbiserial"] = pointbiserial_nli_consistency(scored_df)

    # 2. Model-level correlation
    print("  [2/4] Model-level correlation (mean NLI vs. consistency %)…")
    results["model_level"] = model_level_correlation(model_summary)

    # 3. RLHF pair comparison (only for RLHF data)
    if data_source == "rlhf":
        print("  [3/4] RLHF pair comparison (base vs. instruct NLI coherence)…")
        results["rlhf_pairs"] = rlhf_pair_comparison(scored_df)
    else:
        print("  [3/4] RLHF pair comparison — skipped (using main data)")
        results["rlhf_pairs"] = {"skipped": True, "reason": "Using main project data"}

    # 4. Partial correlation controlling for stage
    print("  [4/4] Partial correlation (NLI vs. consistency | Kohlberg stage)…")
    results["partial_corr"] = partial_correlation_controlling_stage(scored_df)

    # 5. Bootstrap CIs on key correlations
    valid = scored_df.dropna(subset=["nli_entailment", "is_consistent"])
    if len(valid) >= 10:
        x = valid["nli_entailment"].values
        y = valid["is_consistent"].astype(float).values
        results["bootstrap_pointbiserial"] = bootstrap_correlation_ci(x, y, method="pearson")

    if len(model_summary.dropna(subset=["mean_nli_entailment", "consistency_pct"])) >= 4:
        ms = model_summary.dropna(subset=["mean_nli_entailment", "consistency_pct"])
        results["bootstrap_model_level"] = bootstrap_correlation_ci(
            ms["mean_nli_entailment"].values,
            ms["consistency_pct"].values,
            method="spearman",
        )

    return results
