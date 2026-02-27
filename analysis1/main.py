"""
main.py — Entry point for the Scale vs. Moral Reasoning analysis.

Usage
-----
    cd analysis1/
    python main.py

Output
------
    results/
        fig1_box_stage_by_model.png
        fig2_scatter_scale_vs_stage.png
        fig3_heatmap_stage_distribution.png
        fig4_bar_mean_stage.png
        model_stats.csv
        spearman_correlation.csv
        dunn_posthoc_pvalues.csv
"""

import sys
import time
from pathlib import Path

# Allow `python main.py` from within analysis1/ or from the project root
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import OUT_DIR
from data_loader import load_all_data
from stat_analysis import (
    compute_model_stats,
    add_bootstrap_ci,
    spearman_with_ci,
    run_nonparametric_tests,
)
from visualizations import (
    plot_box_by_model,
    plot_scatter_scale_stage,
    plot_stage_heatmap,
    plot_mean_stage_bar,
)
from reporting import save_results, print_report


def main() -> None:
    import numpy as np
    np.random.seed(42)

    t0 = time.perf_counter()

    # ── 1. Load data ───────────────────────────────────────────────────────
    print("─" * 60)
    print("Step 1/5  Loading data …")
    df = load_all_data()

    # ── 2. Per-model statistics ────────────────────────────────────────────
    print("Step 2/5  Computing per-model statistics …")
    summary = compute_model_stats(df)

    print("          Bootstrapping 95% confidence intervals …")
    summary = add_bootstrap_ci(df, summary, n_boot=5_000)

    # ── 3. Correlation & tests ────────────────────────────────────────────
    print("Step 3/5  Running Spearman correlation …")
    corr = spearman_with_ci(
        summary["log_params"].values,
        summary["mean_stage"].values,
        n_boot=5_000,
    )

    print("          Running Kruskal-Wallis + Dunn post-hoc test …")
    tests = run_nonparametric_tests(df)

    # ── 4. Publication-quality figures ────────────────────────────────────
    print("Step 4/5  Generating publication-quality figures …")
    plot_box_by_model(df, summary, OUT_DIR)
    plot_scatter_scale_stage(summary, corr, OUT_DIR)
    plot_stage_heatmap(summary, OUT_DIR)
    plot_mean_stage_bar(summary, OUT_DIR)

    # ── 5. Save results ───────────────────────────────────────────────────
    print("Step 5/5  Saving CSV reports …")
    save_results(summary, corr, tests, OUT_DIR)

    # ── Console report ────────────────────────────────────────────────────
    print_report(summary, corr, tests)

    elapsed = time.perf_counter() - t0
    print(f"\nAll outputs saved to:  {OUT_DIR}")
    print(f"Total runtime:         {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
