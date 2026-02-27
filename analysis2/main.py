"""
main.py — Entry point for the Alignment Training vs. Moral Reasoning analysis.

Usage
-----
    cd analysis2/
    python main.py

Outputs  (written to analysis2/results/)
-----------------------------------------
    fig1_violin_by_alignment.png
    fig2_family_comparisons.png
    fig3_stacked_stage_dist.png
    fig4_pct_postconv.png
    model_stats.csv
    alignment_group_stats.csv
    family_comparisons.csv
    overall_alignment_test.csv
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
    compute_model_stats,
    compute_alignment_group_stats,
    run_family_comparisons,
    run_overall_alignment_test,
)
from visualizations import (
    plot_violin_by_alignment,
    plot_family_comparisons,
    plot_stacked_stage_dist,
    plot_pct_postconv,
)
from reporting import save_results, print_report


def main() -> None:
    import numpy as np
    np.random.seed(42)

    t0 = time.perf_counter()

    # ── 1. Load ────────────────────────────────────────────────────────────
    print("─" * 60)
    print("Step 1/5  Loading data …")
    df = load_all_data()

    # ── 2. Statistics ──────────────────────────────────────────────────────
    print("Step 2/5  Computing statistics …")
    model_stats   = compute_model_stats(df)
    align_stats   = compute_alignment_group_stats(df)
    family_results = run_family_comparisons(df)
    overall        = run_overall_alignment_test(df)

    # ── 3. Figures ─────────────────────────────────────────────────────────
    print("Step 3/5  Generating publication-quality figures …")
    plot_violin_by_alignment(df, OUT_DIR)
    plot_family_comparisons(df, family_results, OUT_DIR)
    plot_stacked_stage_dist(align_stats, OUT_DIR)
    plot_pct_postconv(model_stats, OUT_DIR)

    # ── 4. Save CSVs ───────────────────────────────────────────────────────
    print("Step 4/5  Saving CSV reports …")
    save_results(model_stats, align_stats, family_results, overall, OUT_DIR)

    # ── 5. Console report ──────────────────────────────────────────────────
    print("Step 5/5  Printing results …")
    print_report(align_stats, family_results, overall)

    elapsed = time.perf_counter() - t0
    print(f"\nAll outputs saved to:  {OUT_DIR}")
    print(f"Total runtime:         {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
