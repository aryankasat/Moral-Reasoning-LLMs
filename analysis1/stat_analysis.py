"""
stat_analysis.py — All statistical computations.

Public API
----------
compute_model_stats(df)         -> pd.DataFrame   per-model summary
add_bootstrap_ci(df, summary)   -> pd.DataFrame   adds ci_lo / ci_hi columns
spearman_with_ci(x, y)          -> dict           ρ, p, CI, R², effect, significant
run_nonparametric_tests(df)     -> dict           kw_stat, kw_p, kw_df, kw_eta2,
                                                  dunn (adj. p-values, Bonferroni),
                                                  dunn_sig (boolean significance matrix)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal
import scikit_posthocs as sp  # pip install scikit-posthocs


# ── Per-model summary ────────────────────────────────────────────────────────

def compute_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-model descriptive statistics.

    Returns a DataFrame sorted by params_B (ascending) with columns:
        model_key, display_name, params_B, log_params, provider,
        n_samples, mean_stage, median_stage, mode_stage, std_stage,
        stage_1_pct … stage_6_pct
    """
    rows: list[dict] = []

    for key, grp in df.groupby("model_key"):
        stages = grp["kohlberg_stage"]
        mode_val = int(stages.mode().iloc[0]) if not stages.empty else np.nan

        row: dict = {
            "model_key":    key,
            "display_name": grp["display_name"].iloc[0],
            "params_B":     grp["params_B"].iloc[0],
            "log_params":   grp["log_params"].iloc[0],
            "provider":     grp["provider"].iloc[0],
            "n_samples":    len(stages),
            "mean_stage":   stages.mean(),
            "median_stage": stages.median(),
            "mode_stage":   mode_val,
            "std_stage":    stages.std(ddof=1),
        }

        vcounts = stages.value_counts(normalize=True) * 100
        for s in range(1, 7):
            row[f"stage_{s}_pct"] = vcounts.get(s, 0.0)

        rows.append(row)

    summary = (
        pd.DataFrame(rows)
        .sort_values("params_B")
        .reset_index(drop=True)
    )
    return summary


# ── Bootstrap confidence intervals ───────────────────────────────────────────

def _bootstrap_ci(
    data: np.ndarray,
    stat_fn=np.mean,
    n_boot: int = 5_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for stat_fn applied to data."""
    if rng is None:
        rng = np.random.default_rng(42)
    boot = [
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def add_bootstrap_ci(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    n_boot: int = 5_000,
) -> pd.DataFrame:
    """Add ci_lo and ci_hi columns to summary (in-place and returns summary)."""
    rng = np.random.default_rng(42)
    ci_lo, ci_hi = [], []
    for key in summary["model_key"]:
        stages = df.loc[df["model_key"] == key, "kohlberg_stage"].values
        lo, hi = _bootstrap_ci(stages, n_boot=n_boot, rng=rng)
        ci_lo.append(lo)
        ci_hi.append(hi)
    summary = summary.copy()
    summary["ci_lo"] = ci_lo
    summary["ci_hi"] = ci_hi
    return summary


# ── Spearman correlation ─────────────────────────────────────────────────────

def spearman_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 5_000,
) -> dict:
    """
    Spearman correlation with bootstrapped 95 % CI and effect-size label.

    Returns dict with keys:
        rho, p, ci_lo, ci_hi, r2, effect
    """
    rho, p = spearmanr(x, y)

    rng = np.random.default_rng(42)
    n = len(x)
    boot_rhos = [
        spearmanr(x[idx := rng.choice(n, n, replace=True)], y[idx])[0]
        for _ in range(n_boot)
    ]
    ci_lo = float(np.percentile(boot_rhos, 2.5))
    ci_hi = float(np.percentile(boot_rhos, 97.5))

    abs_rho = abs(rho)
    if abs_rho >= 0.50:
        effect = "large"
    elif abs_rho >= 0.30:
        effect = "medium"
    elif abs_rho >= 0.10:
        effect = "small"
    else:
        effect = "negligible"

    # rho2 = Spearman rho squared (proportion of variance in ranks explained)
    # Keep key as 'r2' for backward compatibility with downstream consumers
    return dict(rho=float(rho), p=float(p),
                ci_lo=ci_lo, ci_hi=ci_hi,
                r2=float(rho ** 2), effect=effect,
                significant=bool(p < 0.05))


# ── Non-parametric tests ─────────────────────────────────────────────────────

def run_nonparametric_tests(df: pd.DataFrame) -> dict:
    """
    Kruskal-Wallis H test across all models, followed by
    Dunn pairwise post-hoc with Bonferroni correction.

    Returns dict with keys:
        kw_stat   – Kruskal-Wallis H statistic
        kw_p      – p-value for Kruskal-Wallis
        kw_df     – degrees of freedom (number of groups − 1)
        kw_eta2   – eta-squared effect size: (H − k + 1) / (n − k)
        dunn      – pd.DataFrame of Bonferroni-adjusted pairwise p-values
                    (display_name labels on both axes)
        dunn_sig  – boolean pd.DataFrame (True where adj. p < 0.05)
    """
    groups = [g["kohlberg_stage"].values for _, g in df.groupby("model_key")]
    k = len(groups)                       # number of groups
    n = sum(len(g) for g in groups)       # total observations
    kw_stat, kw_p = kruskal(*groups)

    # Degrees of freedom and eta-squared effect size
    kw_df   = k - 1
    kw_eta2 = (kw_stat - k + 1) / (n - k)  # standard formula for KW eta²
    kw_eta2 = float(max(kw_eta2, 0.0))      # clamp to [0, 1] (can be slightly negative for tiny H)

    # Dunn pairwise post-hoc: Bonferroni correction (controls FWER)
    dunn_df: pd.DataFrame = sp.posthoc_dunn(
        df,
        val_col="kohlberg_stage",
        group_col="model_key",
        p_adjust="bonferroni",
    )

    # Remap model_key → display_name for readable output
    key_to_name = (
        df[["model_key", "display_name"]]
        .drop_duplicates()
        .set_index("model_key")["display_name"]
        .to_dict()
    )
    dunn_df  = dunn_df.rename(index=key_to_name, columns=key_to_name)
    dunn_sig = (dunn_df < 0.05).astype(bool)  # significance mask

    return dict(
        kw_stat=float(kw_stat),
        kw_p=float(kw_p),
        kw_df=int(kw_df),
        kw_eta2=kw_eta2,
        dunn=dunn_df,
        dunn_sig=dunn_sig,
    )
