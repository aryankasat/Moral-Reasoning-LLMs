"""
stat_analysis.py — Statistical computations for the alignment training analysis.

Public API
----------
compute_model_stats(df)               -> pd.DataFrame
compute_alignment_group_stats(df)     -> pd.DataFrame
run_family_comparisons(df)            -> pd.DataFrame
wilcoxon_effect_size(a, b)            -> dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata
from config import POST_CONV_THRESHOLD, FAMILY_PAIRS, MODEL_META


# ── helpers ───────────────────────────────────────────────────────────────────

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_sd = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 +
                         (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return float((b.mean() - a.mean()) / pooled_sd) if pooled_sd else np.nan


def _rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """
    Rank-biserial correlation r = 1 - (2U) / (n_a * n_b).
    Positive r → b tends to exceed a.
    """
    stat, _ = mannwhitneyu(b, a, alternative="two-sided")
    na, nb = len(a), len(b)
    return float(1 - (2 * stat) / (na * nb))


def _bootstrap_ci_mean(
    data: np.ndarray,
    n_boot: int = 5_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(42)
    boot = [rng.choice(data, len(data), replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad >= 0.8:
        return "large"
    elif ad >= 0.5:
        return "medium"
    elif ad >= 0.2:
        return "small"
    return "negligible"


# ── Per-model summary ─────────────────────────────────────────────────────────

def compute_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for key, grp in df.groupby("model_key"):
        stages = grp["kohlberg_stage"].values
        lo, hi = _bootstrap_ci_mean(stages, rng=rng)
        meta = MODEL_META[key]
        rows.append({
            "model_key":        key,
            "display_name":     meta[0],
            "params_B":         meta[1],
            "family":           meta[2],
            "alignment_type":   meta[3],
            "n":                len(stages),
            "mean_stage":       stages.mean(),
            "median_stage":     float(np.median(stages)),
            "std_stage":        stages.std(ddof=1),
            "ci_lo":            lo,
            "ci_hi":            hi,
            "pct_post_conv":    float((stages >= POST_CONV_THRESHOLD).mean() * 100),
            **{f"stage_{s}_pct": float((stages == s).mean() * 100)
               for s in range(1, 7)},
        })
    return (pd.DataFrame(rows)
            .sort_values(["alignment_type", "params_B"])
            .reset_index(drop=True))


# ── Alignment group summary ───────────────────────────────────────────────────

def compute_alignment_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statistics by alignment_type."""
    rng = np.random.default_rng(42)
    rows = []
    for atype, grp in df.groupby("alignment_type"):
        stages = grp["kohlberg_stage"].values
        lo, hi = _bootstrap_ci_mean(stages, rng=rng)
        rows.append({
            "alignment_type": atype,
            "n_obs":          len(stages),
            "n_models":       grp["model_key"].nunique(),
            "mean_stage":     stages.mean(),
            "median_stage":   float(np.median(stages)),
            "std_stage":      stages.std(ddof=1),
            "ci_lo":          lo,
            "ci_hi":          hi,
            "pct_post_conv":  float((stages >= POST_CONV_THRESHOLD).mean() * 100),
            **{f"stage_{s}_pct": float((stages == s).mean() * 100)
               for s in range(1, 7)},
        })
    return pd.DataFrame(rows)


# ── Within-family pairwise tests ──────────────────────────────────────────────

def wilcoxon_effect_size(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Two-sided Mann-Whitney U test (Wilcoxon rank-sum for independent samples)
    plus Cohen's d and rank-biserial correlation.
    """
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    d        = _cohens_d(a, b)
    rb       = _rank_biserial(a, b)

    # bootstrapped 95% CI on mean difference (b - a)
    rng   = np.random.default_rng(42)
    diffs = [
        rng.choice(b, len(b), replace=True).mean() -
        rng.choice(a, len(a), replace=True).mean()
        for _ in range(5_000)
    ]
    diff_lo = float(np.percentile(diffs, 2.5))
    diff_hi = float(np.percentile(diffs, 97.5))

    return dict(
        mean_a       = float(a.mean()),
        mean_b       = float(b.mean()),
        delta        = float(b.mean() - a.mean()),
        delta_ci_lo  = diff_lo,
        delta_ci_hi  = diff_hi,
        u_stat       = float(stat),
        p_value      = float(p),
        cohens_d     = d,
        rank_biserial= rb,
        effect_label = _effect_label(d),
        pct_post_a   = float((a >= POST_CONV_THRESHOLD).mean() * 100),
        pct_post_b   = float((b >= POST_CONV_THRESHOLD).mean() * 100),
    )


def run_family_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Wilcoxon rank-sum + effect size for every within-family pair
    defined in FAMILY_PAIRS.
    """
    rows = []
    for stem_a, stem_b, label in FAMILY_PAIRS:
        stages_a = df.loc[df["model_key"] == stem_a, "kohlberg_stage"].values
        stages_b = df.loc[df["model_key"] == stem_b, "kohlberg_stage"].values

        if len(stages_a) == 0 or len(stages_b) == 0:
            print(f"  [SKIP] {label}: one or both models not found in data")
            continue

        res = wilcoxon_effect_size(stages_a, stages_b)
        res["comparison"] = label
        res["model_a"]    = MODEL_META[stem_a][0]
        res["model_b"]    = MODEL_META[stem_b][0]
        res["align_a"]    = MODEL_META[stem_a][3]
        res["align_b"]    = MODEL_META[stem_b][3]
        rows.append(res)

    col_order = [
        "comparison", "model_a", "align_a", "model_b", "align_b",
        "mean_a", "mean_b", "delta", "delta_ci_lo", "delta_ci_hi",
        "u_stat", "p_value", "cohens_d", "rank_biserial",
        "effect_label", "pct_post_a", "pct_post_b",
    ]
    return pd.DataFrame(rows)[col_order].reset_index(drop=True)


# ── Overall IT vs RLHF test ───────────────────────────────────────────────────

def run_overall_alignment_test(df: pd.DataFrame) -> dict:
    """Mann-Whitney U test: IT pooled vs RLHF pooled."""
    from config import IT, RLHF
    a = df.loc[df["alignment_type"] == IT,   "kohlberg_stage"].values
    b = df.loc[df["alignment_type"] == RLHF, "kohlberg_stage"].values
    res = wilcoxon_effect_size(a, b)
    res["group_a"] = IT
    res["group_b"] = RLHF
    return res
