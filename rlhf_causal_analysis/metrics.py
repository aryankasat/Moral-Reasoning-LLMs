"""
metrics.py — Core metrics for Analysis 11: RLHF Causal Analysis.

Computes:
  - Stage distribution arrays (base vs. instruct per pair)
  - KL divergence (base → instruct) per pair
  - Mean stage delta (instruct − base) per pair
  - Cohen's d effect size per pair
  - Bootstrap 95% CIs on mean stage difference
  - Cross-pair consistency (sign test on delta direction)
  - Post-conventional proportion (stages 5+6) per variant
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from config import STAGES, N_BOOTSTRAP, PAIR_ORDER


# ── KL divergence ─────────────────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-9) -> float:
    """
    KL(P || Q) = Σ p_i * log(p_i / q_i).

    P = base distribution, Q = instruct distribution.
    Smoothed with epsilon to handle zeros.
    """
    p = np.array(p, dtype=float) + epsilon
    q = np.array(q, dtype=float) + epsilon
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    """JS-inspired symmetric divergence: (KL(P||Q) + KL(Q||P)) / 2."""
    return (kl_divergence(p, q) + kl_divergence(q, p)) / 2.0


# ── Cohen's d ─────────────────────────────────────────────────────────────────

def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Cohen's d = (mean2 − mean1) / pooled_std.
    Positive → x2 has higher mean (instruct > base).
    """
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_var = (((n1 - 1) * np.var(x1, ddof=1)) + ((n2 - 1) * np.var(x2, ddof=1))) / (
        n1 + n2 - 2
    )
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1e-9
    return float((np.mean(x2) - np.mean(x1)) / pooled_std)


# ── Bootstrap CI on mean difference ───────────────────────────────────────────

def bootstrap_mean_diff_ci(
    x_base: np.ndarray,
    x_instruct: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap 95% CI on (mean_instruct − mean_base).

    Returns: {observed_diff, ci_lower, ci_upper, se}
    """
    rng = np.random.default_rng(rng_seed)
    observed_diff = float(np.mean(x_instruct) - np.mean(x_base))

    boot_diffs = np.zeros(n_boot)
    for i in range(n_boot):
        b  = rng.choice(x_base,     size=len(x_base),     replace=True)
        bi = rng.choice(x_instruct, size=len(x_instruct), replace=True)
        boot_diffs[i] = bi.mean() - b.mean()

    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(boot_diffs, alpha * 100))
    ci_upper = float(np.percentile(boot_diffs, (1 - alpha) * 100))
    se       = float(boot_diffs.std(ddof=1))

    return {
        "observed_diff": observed_diff,
        "ci_lower":      ci_lower,
        "ci_upper":      ci_upper,
        "se":            se,
    }


# ── Per-pair metrics ───────────────────────────────────────────────────────────

def compute_pair_metrics(obs_df: pd.DataFrame, dist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all metrics per architecture pair.

    Returns a DataFrame with one row per pair:
      pair_id, architecture, params_B,
      base_mean, instruct_mean, delta_mean,
      base_std, instruct_std,
      kl_base_to_instruct, kl_symmetric,
      cohens_d, boot_diff, boot_ci_lower, boot_ci_upper, boot_se,
      base_postconv_prop, instruct_postconv_prop, delta_postconv_prop,
      stage_1..6 cols for base and instruct
    """
    rows = []
    for pair_id in PAIR_ORDER:
        base_row     = dist_df[(dist_df["pair_id"] == pair_id) & (dist_df["variant"] == "base")]
        instruct_row = dist_df[(dist_df["pair_id"] == pair_id) & (dist_df["variant"] == "instruct")]

        if base_row.empty or instruct_row.empty:
            continue

        base_row     = base_row.iloc[0]
        instruct_row = instruct_row.iloc[0]

        # Distribution vectors (stages 1–6)
        p_base     = np.array([float(base_row.get(f"stage_{s}", 0.0))     for s in STAGES])
        p_instruct = np.array([float(instruct_row.get(f"stage_{s}", 0.0)) for s in STAGES])

        # Raw stage arrays for bootstrap / Cohen's d
        obs_base     = obs_df[(obs_df["pair_id"] == pair_id) & (obs_df["variant"] == "base")]["kohlberg_stage"].values
        obs_instruct = obs_df[(obs_df["pair_id"] == pair_id) & (obs_df["variant"] == "instruct")]["kohlberg_stage"].values

        kl_fwd  = kl_divergence(p_base, p_instruct)    # KL(base → instruct)
        kl_sym   = symmetric_kl(p_base, p_instruct)
        d        = cohens_d(obs_base, obs_instruct)
        boot     = bootstrap_mean_diff_ci(obs_base, obs_instruct)

        # Post-conventional proportion (stages 5 + 6)
        base_pc     = float(p_base[4] + p_base[5])      # idx 4=S5, 5=S6
        instruct_pc = float(p_instruct[4] + p_instruct[5])

        row = {
            "pair_id":              pair_id,
            "architecture":         base_row["architecture"],
            "params_B":             base_row["params_B"],
            # Means
            "base_mean":            float(base_row["mean_stage"]),
            "instruct_mean":        float(instruct_row["mean_stage"]),
            "delta_mean":           float(instruct_row["mean_stage"]) - float(base_row["mean_stage"]),
            # Std
            "base_std":             float(base_row["std_stage"]),
            "instruct_std":         float(instruct_row["std_stage"]),
            # KL divergence
            "kl_base_to_instruct":  kl_fwd,
            "kl_symmetric":         kl_sym,
            # Effect size
            "cohens_d":             d,
            # Bootstrap CI
            "boot_diff":            boot["observed_diff"],
            "boot_ci_lower":        boot["ci_lower"],
            "boot_ci_upper":        boot["ci_upper"],
            "boot_se":              boot["se"],
            # Post-conventional proportions
            "base_postconv_prop":    base_pc,
            "instruct_postconv_prop":instruct_pc,
            "delta_postconv_prop":   instruct_pc - base_pc,
            # n_obs
            "n_base":               int(base_row["n_obs"]),
            "n_instruct":           int(instruct_row["n_obs"]),
        }

        # Stage proportions for both variants (for heatmap / delta bars)
        for s in STAGES:
            row[f"base_stage_{s}"]     = float(p_base[s - 1])
            row[f"instruct_stage_{s}"] = float(p_instruct[s - 1])
            row[f"delta_stage_{s}"]    = float(p_instruct[s - 1] - p_base[s - 1])

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


# ── Cross-pair consistency ─────────────────────────────────────────────────────

def cross_pair_consistency(pair_metrics: pd.DataFrame) -> dict[str, Any]:
    """
    Checks whether the RLHF uplift direction (instruct > base mean stage)
    is consistent across all pairs.

    Uses a sign test: if RLHF always increases mean stage, p ~ (0.5)^n_pairs.
    Also computes mean KL divergence across pairs.
    """
    from scipy.stats import binomtest

    deltas    = pair_metrics["delta_mean"].values
    n_pairs   = len(deltas)
    n_positive = int((deltas > 0).sum())
    n_negative = int((deltas < 0).sum())

    # One-tailed binomial: all uplifts positive?
    result = binomtest(n_positive, n_pairs, p=0.5, alternative="greater")

    return {
        "n_pairs":           n_pairs,
        "n_uplift_positive": n_positive,
        "n_uplift_negative": n_negative,
        "binomial_p":        float(result.pvalue),
        "consistent":        bool(n_positive == n_pairs),
        "mean_delta":        float(np.mean(deltas)),
        "mean_kl_fwd":       float(pair_metrics["kl_base_to_instruct"].mean()),
        "mean_kl_sym":        float(pair_metrics["kl_symmetric"].mean()),
        "mean_cohens_d":      float(pair_metrics["cohens_d"].mean()),
        "interpretation": (
            f"RLHF uplift is consistent across all {n_pairs} pairs "
            f"(δ always positive; binomial p={result.pvalue:.4f})"
            if n_positive == n_pairs
            else f"{n_positive}/{n_pairs} pairs show positive RLHF uplift"
        ),
    }


# ── Summary table ──────────────────────────────────────────────────────────────

def build_summary_table(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Human-readable summary table for the report.
    """
    cols = [
        "architecture",
        "base_mean", "instruct_mean", "delta_mean",
        "kl_base_to_instruct",
        "cohens_d",
        "boot_diff", "boot_ci_lower", "boot_ci_upper",
        "base_postconv_prop", "instruct_postconv_prop", "delta_postconv_prop",
    ]
    tbl = pair_metrics[cols].copy()
    tbl = tbl.rename(columns={
        "architecture":          "Architecture",
        "base_mean":             "Base Mean Stage",
        "instruct_mean":         "Instruct Mean Stage",
        "delta_mean":            "Δ Mean Stage",
        "kl_base_to_instruct":   "KL(base→instruct)",
        "cohens_d":              "Cohen's d",
        "boot_diff":             "Boot Δ",
        "boot_ci_lower":         "95% CI Lower",
        "boot_ci_upper":         "95% CI Upper",
        "base_postconv_prop":    "Base PostConv %",
        "instruct_postconv_prop":"Instruct PostConv %",
        "delta_postconv_prop":   "Δ PostConv %",
    })
    # Format percentages
    for col in ["Base PostConv %", "Instruct PostConv %", "Δ PostConv %"]:
        tbl[col] = (tbl[col] * 100).round(1).astype(str) + "%"

    for col in ["Base Mean Stage", "Instruct Mean Stage", "Δ Mean Stage",
                "KL(base→instruct)", "Cohen's d", "Boot Δ", "95% CI Lower", "95% CI Upper"]:
        tbl[col] = tbl[col].round(3)

    return tbl
