"""
reporting.py — Save CSV artefacts and print a structured console report
for the alignment training analysis (analysis2).

Public API
----------
save_results(model_stats, align_stats, family_results, overall, out_dir)
print_report(align_stats, family_results, overall)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd


def save_results(
    model_stats: pd.DataFrame,
    align_stats: pd.DataFrame,
    family_results: pd.DataFrame,
    overall: dict,
    out_dir: Path,
) -> None:
    model_stats.to_csv(out_dir / "model_stats.csv", index=False, float_format="%.4f")
    print("  Saved: model_stats.csv")

    align_stats.to_csv(out_dir / "alignment_group_stats.csv", index=False, float_format="%.4f")
    print("  Saved: alignment_group_stats.csv")

    family_results.to_csv(out_dir / "family_comparisons.csv", index=False, float_format="%.4f")
    print("  Saved: family_comparisons.csv")

    overall_row = {k: v for k, v in overall.items() if not isinstance(v, pd.DataFrame)}
    pd.DataFrame([overall_row]).to_csv(out_dir / "overall_alignment_test.csv",
                                        index=False, float_format="%.6f")
    print("  Saved: overall_alignment_test.csv")


def print_report(
    align_stats: pd.DataFrame,
    family_results: pd.DataFrame,
    overall: dict,
) -> None:
    hr   = "=" * 72
    thin = "-" * 72

    print(f"\n{hr}")
    print("  ALIGNMENT TRAINING vs. MORAL REASONING  —  RESULTS SUMMARY")
    print(hr)

    # ── Group-level summary ────────────────────────────────────────────────
    print("\n▸ Alignment Group Statistics\n")
    cols = ["alignment_type", "n_models", "n_obs",
            "mean_stage", "median_stage", "std_stage", "pct_post_conv"]
    hdrs = ["Alignment",      "N Models", "N Obs",
            "Mean Stage", "Median", "SD", "% Stage5+"]
    print(align_stats[cols].rename(columns=dict(zip(cols, hdrs))).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    # ── Overall IT vs RLHF test ─────────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ Overall Wilcoxon Rank-Sum Test:  IT  vs  RLHF  (pooled)\n")
    sig = "✓ significant" if overall["p_value"] < 0.05 else "✗ not significant"
    print(f"  Mean stage IT     = {overall['mean_a']:.3f}")
    print(f"  Mean stage RLHF   = {overall['mean_b']:.3f}")
    print(f"  Δ (RLHF − IT)    = {overall['delta']:+.3f}  "
          f"95% CI [{overall['delta_ci_lo']:.3f}, {overall['delta_ci_hi']:.3f}]")
    print(f"  U-statistic       = {overall['u_stat']:.1f}")
    print(f"  p-value           = {overall['p_value']:.6f}  {sig}")
    print(f"  Cohen's d         = {overall['cohens_d']:+.3f}  ({overall['effect_label']} effect)")
    print(f"  Rank-biserial r   = {overall['rank_biserial']:+.3f}")
    print(f"  % Stage5+  IT     = {overall['pct_post_a']:.1f}%")
    print(f"  % Stage5+  RLHF   = {overall['pct_post_b']:.1f}%")

    # ── Within-family comparisons ───────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ Within-Family Pairwise Comparisons\n")
    for _, row in family_results.iterrows():
        sig_str = "✓ p<0.05" if row["p_value"] < 0.05 else "n.s."
        print(f"  {row['comparison']:<35}  "
              f"Δ={row['delta']:+.3f}  "
              f"d={row['cohens_d']:+.3f}  "
              f"({row['effect_label']})  {sig_str}")
        print(f"    {row['model_a']} ({row['align_a'][:2]}): mean={row['mean_a']:.3f}  "
              f"Stage5+={row['pct_post_a']:.0f}%")
        print(f"    {row['model_b']} ({row['align_b'][:2]}): mean={row['mean_b']:.3f}  "
              f"Stage5+={row['pct_post_b']:.0f}%")
        print()

    # ── Interpretation ──────────────────────────────────────────────────────
    print(f"{thin}")
    print("▸ Plain-Language Interpretation\n")
    direction = "higher" if overall["delta"] > 0 else "lower"
    sig_label = ("significantly" if overall["p_value"] < 0.05 else "not significantly")
    print(
        f"  RLHF/RL-aligned models score {direction} on average (Δ = {overall['delta']:+.3f})\n"
        f"  and this difference is {sig_label} different from IT models at α = 0.05.\n"
        f"  Cohen's d = {overall['cohens_d']:+.3f} ({overall['effect_label']} effect).\n"
        f"  Post-conventional reasoning (Stage 5+):\n"
        f"    IT models:   {overall['pct_post_a']:.1f}%\n"
        f"    RLHF models: {overall['pct_post_b']:.1f}%\n"
        "  See family_comparisons.csv for pairwise details."
    )
    print(hr)
