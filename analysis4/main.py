"""
main.py — Entry point for Analysis 4: Stage Distribution Patterns.

Usage
-----
    python3 analysis4/main.py   (from project root)

Outputs (analysis4/results/)
-----------------------------
Figures:
    fig1_stacked_bar.png          — stacked bar: all models + human baselines
    fig2_histogram_grid.png       — per-model histogram + human baseline overlay
    fig3_jsd_heatmap.png          — N×N Jensen-Shannon divergence clustermap
    fig4_distribution_stats.png   — entropy / skewness / mean-stage / pattern
    fig5_chi_square.png           — χ² bar + Pearson residual heatmap
    fig6_3d_stage_landscape.png   — 3D bar chart of stage proportions

CSVs:
    stage_distributions.csv
    chi_square_results.csv
    pearson_residuals.csv
    distribution_stats.csv
    jsd_matrix.csv
"""

import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import OUT_DIR
from data_loader import load_all_data
from stat_analysis import (
    compute_stage_distributions,
    compare_to_human_baseline,
    compute_distribution_stats,
    label_patterns,
    compute_jsd_matrix,
)
from visualizations import (
    plot_stacked_bar,
    plot_histogram_grid,
    plot_jsd_heatmap,
    plot_distribution_stats,
    plot_chi_square,
    plot_3d_stage_landscape,
)
from reporting import save_results, print_report


def main() -> None:
    import numpy as np
    np.random.seed(42)

    t0 = time.perf_counter()

    print("─" * 65)
    print("Step 1/5  Loading data …")
    df = load_all_data()

    print("Step 2/5  Running statistical analyses …")
    print("          Stage distribution tables …")
    dist_df = compute_stage_distributions(df)

    print("          Chi-square vs human adult + JSD …")
    chi_df, resid_df = compare_to_human_baseline(dist_df)

    print("          Distribution characteristics (entropy, skewness, kurtosis) …")
    stat_df = compute_distribution_stats(dist_df)
    stat_df = label_patterns(stat_df, chi_df)

    print("          Pairwise JSD matrix …")
    jsd_mat = compute_jsd_matrix(dist_df)

    print("Step 3/5  Generating publication-quality figures …")

    print("  Figure 1: Stacked bar — models + human baselines …")
    plot_stacked_bar(dist_df, OUT_DIR)

    print("  Figure 2: Per-model histogram grid …")
    plot_histogram_grid(dist_df, chi_df, OUT_DIR)

    print("  Figure 3: Jensen-Shannon divergence heatmap …")
    plot_jsd_heatmap(jsd_mat, OUT_DIR)

    print("  Figure 4: Distribution stats multi-panel …")
    plot_distribution_stats(stat_df, OUT_DIR)

    print("  Figure 5: Chi-square bar + Pearson residual heatmap …")
    plot_chi_square(chi_df, resid_df, OUT_DIR)

    print("  Figure 6: 3D stage landscape …")
    plot_3d_stage_landscape(dist_df, OUT_DIR)

    print("Step 4/5  Saving CSV reports …")
    save_results(dist_df, chi_df, resid_df, stat_df, jsd_mat, OUT_DIR)

    print("Step 5/5  Printing console report …")
    print_report(dist_df, chi_df, stat_df, jsd_mat)

    elapsed = time.perf_counter() - t0
    print(f"All outputs saved to:  {OUT_DIR}")
    print(f"Total runtime:         {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
