"""
reporting.py — Save CSV artefacts and print a formatted console report.

Public API
----------
save_results(summary, corr, tests, out_dir)
    Writes:
        model_stats.csv             – per-model descriptive statistics
        spearman_correlation.csv    – Spearman ρ, CI, p, effect, significant
        kruskal_wallis.csv          – H, df, p, eta-squared
        dunn_posthoc_pvalues.csv    – Bonferroni-adjusted pairwise p-values
        dunn_posthoc_significant.csv – boolean significance matrix (p_adj < 0.05)
print_report(summary, corr, tests)
    Pretty-prints all sections to stdout.
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
    """Write five CSV files to out_dir."""
    # 1. Model-level statistics
    summary.to_csv(out_dir / "model_stats.csv", index=False, float_format="%.4f")
    print("  Saved: model_stats.csv")

    # 2. Spearman correlation (includes significance flag)
    corr_row = {k: v for k, v in corr.items()}
    pd.DataFrame([corr_row]).to_csv(
        out_dir / "spearman_correlation.csv", index=False, float_format="%.6f"
    )
    print("  Saved: spearman_correlation.csv")

    # 3. Kruskal-Wallis summary (H, df, p, eta-squared)
    kw_row = {
        "kw_stat":  tests["kw_stat"],
        "kw_df":    tests["kw_df"],
        "kw_p":     tests["kw_p"],
        "kw_eta2":  tests["kw_eta2"],
        "significant": tests["kw_p"] < 0.05,
    }
    pd.DataFrame([kw_row]).to_csv(
        out_dir / "kruskal_wallis.csv", index=False, float_format="%.6f"
    )
    print("  Saved: kruskal_wallis.csv")

    # 4. Dunn post-hoc Bonferroni-adjusted pairwise p-values
    tests["dunn"].to_csv(out_dir / "dunn_posthoc_pvalues.csv", float_format="%.4f")
    print("  Saved: dunn_posthoc_pvalues.csv")

    # 5. Dunn significance boolean matrix (True = p_adj < 0.05)
    tests["dunn_sig"].to_csv(out_dir / "dunn_posthoc_significant.csv")
    print("  Saved: dunn_posthoc_significant.csv")


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
    print("▸ Spearman's ρ Correlation  (log₁₀ params vs. mean Kohlberg stage)\n")
    sig_star = "✓ p < 0.05  — SIGNIFICANT" if corr["significant"] else "✗ p ≥ 0.05  — not significant"
    print(f"  ρ            = {corr['rho']:+.4f}")
    print(f"  p-value      = {corr['p']:.6f}  ({sig_star})")
    print(f"  95% Boot CI  = [{corr['ci_lo']:.4f},  {corr['ci_hi']:.4f}]")
    print(f"  ρ²           = {corr['r2']:.4f}  ({corr['r2']*100:.1f}% variance in ranks explained)")
    print(f"  Effect size  = {corr['effect']} (negligible <0.10, small <0.30, medium <0.50, large ≥0.50)")

    # ── Kruskal-Wallis ────────────────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ Kruskal-Wallis Test  (non-parametric ANOVA for between-model differences)\n")
    kw_sig = "✓ SIGNIFICANT" if tests["kw_p"] < 0.05 else "✗ not significant"
    print(f"  H-statistic  = {tests['kw_stat']:.4f}")
    print(f"  df           = {tests['kw_df']}  (k − 1, where k = number of models)")
    print(f"  p-value      = {tests['kw_p']:.2e}  ({kw_sig} at α = 0.05)")
    print(f"  η² (eta²)    = {tests['kw_eta2']:.4f}", end="")
    if   tests["kw_eta2"] >= 0.14:
        print("  — large effect")
    elif tests["kw_eta2"] >= 0.06:
        print("  — medium effect")
    elif tests["kw_eta2"] >= 0.01:
        print("  — small effect")
    else:
        print("  — negligible effect")

    # ── Dunn post-hoc ─────────────────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ Dunn Post-Hoc Pairwise Comparisons  (Bonferroni correction, α = 0.05)\n")
    sig_df = tests["dunn_sig"]
    # Count significant pairs (upper-triangle only, exclude diagonal)
    import numpy as np
    upper = sig_df.values
    n_models = len(sig_df)
    n_pairs  = n_models * (n_models - 1) // 2
    n_sig    = sum(
        upper[i, j]
        for i in range(n_models) for j in range(i + 1, n_models)
    )
    print(f"  Significant pairs (p_adj < 0.05): {n_sig} / {n_pairs}")
    print()
    # Compact asterisk table
    labels = list(sig_df.columns)
    short  = [lbl.split()[0] for lbl in labels]          # first word of display_name
    col_w  = max(len(s) for s in short) + 2
    header = " " * (col_w) + "".join(f"{s:>{col_w}}" for s in short)
    print("  " + header)
    for i, row_lbl in enumerate(labels):
        cells = ""
        for j, _ in enumerate(labels):
            if i == j:
                cells += f"{'—':>{col_w}}"
            elif sig_df.iloc[i, j]:
                cells += f"{'*':>{col_w}}"
            else:
                cells += f"{'ns':>{col_w}}"
        print(f"  {short[i]:<{col_w}}{cells}")
    print()
    print("  Legend:  * = significant after Bonferroni correction;  ns = not significant")
    print("  Full p-value tables saved to dunn_posthoc_pvalues.csv / dunn_posthoc_significant.csv")

    # ── Interpretation ────────────────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ Plain-Language Interpretation\n")
    direction = "positive" if corr["rho"] > 0 else "negative"
    sig_label = ("statistically significant" if corr["significant"]
                 else "not statistically significant at α = 0.05")
    eta_label = (
        "large" if tests["kw_eta2"] >= 0.14
        else "medium" if tests["kw_eta2"] >= 0.06
        else "small" if tests["kw_eta2"] >= 0.01
        else "negligible"
    )
    print(
        f"  There is a {direction} {corr['effect']}-effect correlation (ρ = {corr['rho']:.3f})\n"
        f"  between model parameter scale (log) and mean Kohlberg stage.\n"
        f"  This correlation is {sig_label}.\n"
        f"  Spearman ρ² explains ~{corr['r2']*100:.1f}% of variance in rank order of mean moral reasoning stage.\n"
        "\n"
        f"  Kruskal-Wallis (H = {tests['kw_stat']:.3f}, df = {tests['kw_df']},"
        f" p = {tests['kw_p']:.2e}, η² = {tests['kw_eta2']:.3f})\n"
        f"  indicates {'highly ' if tests['kw_p'] < 0.001 else ''}significant between-model differences\n"
        f"  with a {eta_label} overall effect size.\n"
        f"  {n_sig} of {n_pairs} pairwise model comparisons survive Bonferroni correction.\n"
        "  See CSV outputs for full tables."
    )
    print(hr)
