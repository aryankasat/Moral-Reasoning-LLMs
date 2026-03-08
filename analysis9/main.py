"""
main.py — Orchestration for Analysis 9: Capability Correlation Analysis.

Run from project root:
    python analysis9/main.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUT_DIR
from data_loader import load_raw_data, build_model_summary
from capability_metrics import compute_model_capabilities
from stat_analysis import (
    compute_correlation_matrix,
    threshold_detection,
    multi_capability_regression,
    compute_partial_correlations,
)
from visualizations import (
    plot_correlation_heatmap,
    plot_threshold_detection,
    plot_regression_coefficients,
    plot_partial_correlations,
    plot_capability_scatter_panel,
)
from reporting import generate_report


def main() -> None:
    print("=" * 70)
    print("  Analysis 9: Capability Correlation Analysis")
    print("  Research Q: Which capabilities predict moral reasoning stage?")
    print("=" * 70)

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    print("\n[1/6] Loading evaluation data …")
    raw_df = load_raw_data()

    # ── 2. Compute capability metrics ─────────────────────────────────────────
    print("\n[2/6] Computing capability metrics …")
    cap_df = compute_model_capabilities(raw_df)
    print(f"  Capability metrics computed for {len(cap_df)} models.")
    print(cap_df.set_index("model_key").round(3).to_string())

    # ── 3. Build model summary ────────────────────────────────────────────────
    print("\n[3/6] Building model-level summary …")
    model_df = build_model_summary(raw_df, cap_df)
    print("\n  Model summary table:")
    cols_to_show = ["display_name", "params_B", "mean_stage", "post_conv_pct",
                    "post_conv_capable", "coherence", "response_length",
                    "lexical_diversity", "syntactic_complexity", "semantic_density"]
    show_cols = [c for c in cols_to_show if c in model_df.columns]
    print(model_df[show_cols].round(3).to_string(index=False))

    n_capable = model_df["post_conv_capable"].sum()
    print(f"\n  Post-conv capable (≥20% Stage 5+): {n_capable}/{len(model_df)} models")

    # ── 4. Statistical analyses ───────────────────────────────────────────────
    print("\n[4/6] Running statistical analyses …")

    print("  • Correlation matrix (Pearson + Spearman + Bootstrap CIs + FDR) …")
    corr_results = compute_correlation_matrix(model_df)
    pr = corr_results["pearson_r"]
    pp = corr_results["corrected_p_pearson"]
    if "mean_stage" in pr.columns:
        print("\n  Pearson r with mean_stage (FDR-corrected p):")
        for v in corr_results["variables"]:
            if v == "mean_stage":
                continue
            r_val = float(pr.loc["mean_stage", v])
            p_val = float(pp.loc["mean_stage", v])
            sig   = "*" if p_val < 0.05 else " "
            print(f"    {v:<28s}  r={r_val:+.3f}  p={p_val:.4f} {sig}")

    print("\n  • Threshold detection …")
    threshold_results = threshold_detection(model_df)
    for metric, res in threshold_results.items():
        if isinstance(res, dict) and "logistic_auc" in res:
            auc = res.get("logistic_auc", float("nan"))
            lth = res.get("linear_threshold_20pct", float("nan"))
            auc_s = f"{auc:.3f}" if isinstance(auc, float) and not __import__("math").isnan(auc) else "  — "
            lth_s = f"{lth:.3f}" if isinstance(lth, float) and not __import__("math").isnan(lth) else "  — "
            print(f"    {metric:<28s}  AUC={auc_s}  linear-thresh={lth_s}")

    print("\n  • Multi-capability regression …")
    reg_results = multi_capability_regression(model_df)
    if "error" not in reg_results:
        print(f"    R²={reg_results['r_squared']:.4f}  "
              f"Adj-R²={reg_results['adj_r2']:.4f}  "
              f"F={reg_results['f_stat']:.3f}  p={reg_results['f_p']:.4f}")
        print("    Standardised coefficients:")
        for _, row in reg_results["coef_df"].iterrows():
            sig = "*" if row["p_value"] < 0.05 else " "
            print(f"      {row['predictor']:<28s}  β={row['std_coef']:+.3f}  "
                  f"p={row['p_value']:.4f} {sig}")
    else:
        print(f"    {reg_results['error']}")

    print("\n  • Partial correlations (controlling for log_params) …")
    partial_df = compute_partial_correlations(model_df)
    print(partial_df[["metric", "raw_r", "partial_r", "partial_p_fdr"]].round(3).to_string(index=False))

    # ── 5. Visualizations ─────────────────────────────────────────────────────
    print("\n[5/6] Generating figures …")
    plot_correlation_heatmap(corr_results)
    plot_threshold_detection(model_df, threshold_results)
    plot_regression_coefficients(reg_results)
    plot_partial_correlations(partial_df)
    plot_capability_scatter_panel(model_df, corr_results)

    # ── 6. Report ─────────────────────────────────────────────────────────────
    print("\n[6/6] Writing Markdown report …")
    generate_report(
        model_df          = model_df,
        corr_results      = corr_results,
        threshold_results = threshold_results,
        reg_results       = reg_results,
        partial_df        = partial_df,
    )

    print("\n" + "=" * 70)
    print(f"  ✅  Analysis 9 complete. Outputs → {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.png")) + sorted(OUT_DIR.glob("*.md")):
        print(f"       {f.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
