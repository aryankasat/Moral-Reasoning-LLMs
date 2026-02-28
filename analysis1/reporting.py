"""
reporting.py — Save CSV artefacts and print a formatted console report.

Public API
----------
save_results(summary, corr, tests, out_dir)
print_report(summary, corr, tests)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ── CSV export ────────────────────────────────────────────────────────────────

def save_results(
    summary: pd.DataFrame,
    corr: dict,
    tests: dict,
    out_dir: Path,
) -> None:
    """Write three CSV files to out_dir."""
    # 1. Model-level statistics
    summary.to_csv(out_dir / "model_stats.csv", index=False, float_format="%.4f")
    print("  Saved: model_stats.csv")

    # 2. Spearman correlation
    corr_row = {k: v for k, v in corr.items()}
    pd.DataFrame([corr_row]).to_csv(
        out_dir / "spearman_correlation.csv", index=False, float_format="%.6f"
    )
    print("  Saved: spearman_correlation.csv")

    # 3. Dunn post-hoc pairwise p-values
    tests["dunn"].to_csv(out_dir / "dunn_posthoc_pvalues.csv", float_format="%.4f")
    print("  Saved: dunn_posthoc_pvalues.csv")


# ── Console report ─────────────────────────────────────────────────────────────

def print_report(
    summary: pd.DataFrame,
    corr: dict,
    tests: dict,
) -> None:
    hr = "=" * 72
    thin = "-" * 72

    print(f"\n{hr}")
    print("  SCALE vs. MORAL REASONING  —  RESULTS SUMMARY")
    print(hr)

    # ── Per-model descriptive statistics ──────────────────────────────────
    print("\n▸ Per-Model Descriptive Statistics  (ordered by parameter scale)\n")
    cols = ["display_name", "params_B", "n_samples",
            "mean_stage", "median_stage", "mode_stage", "std_stage",
            "ci_lo", "ci_hi"]
    col_names = ["Model", "Params(B)", "N",
                 "Mean", "Median", "Mode", "SD",
                 "CI lo", "CI hi"]
    print(summary[cols].rename(columns=dict(zip(cols, col_names))).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    # ── Stage distribution ────────────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ Stage Distribution (% of responses per model)\n")
    dist_cols = ["display_name"] + [f"stage_{i}_pct" for i in range(1, 7)]
    dist_hdrs = ["Model"] + [f"S{i}%" for i in range(1, 7)]
    print(summary[dist_cols].rename(columns=dict(zip(dist_cols, dist_hdrs))).to_string(
        index=False, float_format=lambda x: f"{x:.1f}"
    ))

    # ── Spearman correlation ──────────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ Spearman Correlation  (log₁₀ params vs. mean Kohlberg stage)\n")
    print(f"  ρ            = {corr['rho']:+.4f}")
    print(f"  95% CI       = [{corr['ci_lo']:.4f},  {corr['ci_hi']:.4f}]")
    print(f"  p-value      = {corr['p']:.6f}"
          + ("  ✓ sig. (α=0.05)" if corr["p"] < 0.05 else "  ✗ not sig."))
    print(f"  rho^2        = {corr['r2']:.4f}  ({corr['r2']*100:.1f}% variance in ranks explained)")
    print(f"  Effect size  = {corr['effect']}")

    # ── Kruskal-Wallis ────────────────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ Kruskal-Wallis Test  (between-model differences)\n")
    print(f"  H-statistic  = {tests['kw_stat']:.4f}")
    print(f"  p-value      = {tests['kw_p']:.2e}"
          + ("  ✓" if tests["kw_p"] < 0.05 else "  ✗"))

    # ── Interpretation ────────────────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ Plain-Language Interpretation\n")
    direction = "positive" if corr["rho"] > 0 else "negative"
    sig_label = ("statistically significant" if corr["p"] < 0.05
                 else "not statistically significant at α = 0.05")
    print(
        f"  There is a {direction} {corr['effect']}-effect correlation (ρ = {corr['rho']:.3f})\n"
        f"  between model parameter scale (log) and mean Kohlberg stage.\n"
        f"  This correlation is {sig_label}.\n"
        f"  Scale (rho^2) explains ~{corr['r2']*100:.1f}% of variance in rank order of mean moral reasoning stage.\n"
        "\n"
        f"  Kruskal-Wallis confirms that between-model differences are\n"
        f"  {'highly' if tests['kw_p'] < 0.001 else ''} significant (p = {tests['kw_p']:.2e}),\n"
        f"  indicating real distributional differences beyond scale alone.\n"
        "  See dunn_posthoc_pvalues.csv for pairwise comparisons."
    )
    print(hr)
