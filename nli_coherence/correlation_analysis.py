"""
correlation_analysis.py — Correlates NLI coherence scores with McNemar
decoupling scores (p_values) from analysis5.

Statistical Methods
-------------------
1. Spearman rank correlation: non-parametric, robust to non-linear monotonic
   relationships; ideal here since both scores are bounded and may not be
   normally distributed.

2. Pearson correlation: included for completeness, assuming linearity.

3. We also compute:
   - Per-model mean NLI coherence scores
   - Coherence-decoupling difference: |coherence - (1 - p_value)|
   - Rank-based concordance analysis

Design Note on Correlation vs. Difference
------------------------------------------
The user considered using a simple difference between coherence and decoupling
scores. We implement both but recommend Spearman correlation because:
  (a) It captures monotonic relationships regardless of scale;
  (b) Difference conflates magnitude and direction;
  (c) Spearman is standard in the behavioral sciences literature.

We also provide a "normalised gap" metric for interpretability:
    gap_i = NLI_coherence_i − (1 − p_value_i)
where (1 − p_value) approximates decoupling strength. A positive gap means
the model's reasoning-action link is stronger than what the Kohlberg
consistency test suggests.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau


def aggregate_coherence_by_model(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-model summary statistics for NLI entailment scores.

    Returns
    -------
    pd.DataFrame with one row per model_key:
        display_name, params_B, provider,
        mean_entailment, median_entailment, std_entailment,
        min_entailment, max_entailment, n_scored
    """
    valid = scored_df.dropna(subset=["entailment_score"]).copy()

    agg = valid.groupby("model_key").agg(
        display_name  = ("display_name",  "first"),
        params_B      = ("params_B",      "first"),
        provider      = ("provider",      "first"),
        mean_entailment   = ("entailment_score", "mean"),
        median_entailment = ("entailment_score", "median"),
        std_entailment    = ("entailment_score", "std"),
        min_entailment    = ("entailment_score", "min"),
        max_entailment    = ("entailment_score", "max"),
        n_scored          = ("entailment_score", "count"),
    ).reset_index()

    return agg.sort_values("params_B")


def aggregate_coherence_by_model_dilemma(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-model × per-dilemma NLI coherence summaries.
    """
    valid = scored_df.dropna(subset=["entailment_score"]).copy()

    agg = valid.groupby(["model_key", "dilemma_type"]).agg(
        display_name    = ("display_name",     "first"),
        params_B        = ("params_B",         "first"),
        mean_entailment = ("entailment_score", "mean"),
        n_scored        = ("entailment_score", "count"),
    ).reset_index()

    return agg.sort_values(["params_B", "dilemma_type"])


def merge_with_decoupling(
    coherence_df: pd.DataFrame,
    mcnemar_df:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge per-model NLI coherence with McNemar p_values (decoupling scores).

    Adds derived columns:
        decoupling_strength : 1 − p_value  (higher → more decoupled)
        coherence_gap       : mean_entailment − (1 − p_value)
                              Positive → reasoning-action link is stronger
                              than Kohlberg-based consistency suggests
    """
    merged = coherence_df.merge(
        mcnemar_df[["model_key", "p_value"]],
        on="model_key",
        how="inner",
    )

    merged["decoupling_strength"] = 1.0 - merged["p_value"]
    merged["coherence_gap"]       = merged["mean_entailment"] - merged["decoupling_strength"]

    return merged.sort_values("params_B")


def run_correlation_tests(merged_df: pd.DataFrame) -> dict:
    """
    Compute Spearman, Pearson, and Kendall correlations between
    NLI coherence scores and McNemar decoupling p_values.

    Tests two relationships:
    1. mean_entailment vs p_value
       (Are more coherent models also more consistent under Kohlberg?)
    2. mean_entailment vs decoupling_strength  (= 1 − p_value)
       (Does higher NLI coherence correspond to stronger decoupling?)

    Returns
    -------
    dict with sub-dicts for each test.
    """
    x_coherence  = merged_df["mean_entailment"].values
    y_pvalue     = merged_df["p_value"].values
    y_decoupling = merged_df["decoupling_strength"].values

    results = {}

    # ── Coherence vs. p_value ──────────────────────────────────────────────
    sp_r, sp_p = spearmanr(x_coherence, y_pvalue)
    pe_r, pe_p = pearsonr(x_coherence, y_pvalue)
    ke_r, ke_p = kendalltau(x_coherence, y_pvalue)

    results["coherence_vs_pvalue"] = {
        "spearman_r": round(sp_r, 4), "spearman_p": round(sp_p, 4),
        "pearson_r":  round(pe_r, 4), "pearson_p":  round(pe_p, 4),
        "kendall_tau": round(ke_r, 4), "kendall_p": round(ke_p, 4),
        "n": len(merged_df),
        "interpretation": (
            "Positive r → higher NLI coherence correlates with higher p_value "
            "(i.e., less Kohlberg-based decoupling). "
            "Negative r → higher coherence correlates with lower p_value "
            "(more Kohlberg-based decoupling)."
        ),
    }

    # ── Coherence vs. decoupling_strength ──────────────────────────────────
    sp_r2, sp_p2 = spearmanr(x_coherence, y_decoupling)
    pe_r2, pe_p2 = pearsonr(x_coherence, y_decoupling)
    ke_r2, ke_p2 = kendalltau(x_coherence, y_decoupling)

    results["coherence_vs_decoupling"] = {
        "spearman_r": round(sp_r2, 4), "spearman_p": round(sp_p2, 4),
        "pearson_r":  round(pe_r2, 4), "pearson_p":  round(pe_p2, 4),
        "kendall_tau": round(ke_r2, 4), "kendall_p": round(ke_p2, 4),
        "n": len(merged_df),
        "interpretation": (
            "Positive r → higher NLI coherence correlates with stronger "
            "Kohlberg-based decoupling (more stage-action mismatch). "
            "Negative r → higher coherence correlates with weaker decoupling."
        ),
    }

    # ── Coherence gap statistics ───────────────────────────────────────────
    gaps = merged_df["coherence_gap"].values
    results["coherence_gap_stats"] = {
        "mean_gap":   round(float(np.mean(gaps)), 4),
        "median_gap": round(float(np.median(gaps)), 4),
        "std_gap":    round(float(np.std(gaps)), 4),
        "min_gap":    round(float(np.min(gaps)), 4),
        "max_gap":    round(float(np.max(gaps)), 4),
        "interpretation": (
            "coherence_gap = NLI_coherence − decoupling_strength. "
            "Positive gap → model's reasoning-action link is stronger "
            "than what Kohlberg-based consistency suggests; "
            "reasoning is internally coherent even if it violates "
            "stage-based expectations."
        ),
    }

    return results
