"""
main.py — Entry point for Analysis 7: Emergence Threshold Detection.

Usage:
    cd analysis7
    python main.py

Output (written to analysis7/results/):
    fig1_emergence_curves.png   — Three-panel emergence curves
    fig2_emergence_vs_params.png — Scatter + segmented regression
    fig3_stage_heatmap.png       — Stage distribution heatmap
    fig4_slope_analysis.png      — Pre/post changepoint slope comparison
    model_summary.csv            — Per-model stats
    emergence_metrics.csv        — All key emergence metrics
    analysis_results.json        — Full results bundle
"""

import sys
import warnings
from pathlib import Path

# Allow imports from this directory when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from data_loader    import load_raw_data, build_model_summary
from stat_analysis  import run_full_analysis
from visualizations import run_all_visualizations
from reporting      import export_model_summary, export_emergence_metrics, print_console_report
from config         import OUT_DIR


def main() -> None:
    print("\n" + "=" * 65)
    print("  ANALYSIS 7: Emergence Threshold Detection")
    print("  Moral Reasoning in LLMs — Scale vs. Stage Study")
    print("=" * 65 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────
    print("Step 1 · Loading evaluation data …")
    raw_df   = load_raw_data()
    model_df = build_model_summary(raw_df)

    print(f"\n  Models (sorted by scale):")
    for _, row in model_df.iterrows():
        print(f"    {row['display_name']:30s}  {row['params_B']:>6.0f}B  "
              f" mean_stage={row['mean_stage']:.2f}  post_conv={row['post_conv_pct']:.0%}")

    # ── 2. Statistical analysis ───────────────────────────────────
    print("\nStep 2 · Running statistical analysis …")
    results = run_full_analysis(model_df)

    # ── 3. Visualizations ─────────────────────────────────────────
    print("\nStep 3 · Generating figures …")
    fig_paths = run_all_visualizations(model_df, results)
    print(f"  {len(fig_paths)} figures saved to: {OUT_DIR}")

    # ── 4. Export reports ─────────────────────────────────────────
    print("\nStep 4 · Exporting reports …")
    export_model_summary(model_df)
    export_emergence_metrics(results, model_df)

    # ── 5. Console summary ────────────────────────────────────────
    print_console_report(model_df, results)

    print("\n✓ Analysis 7 complete. All outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
