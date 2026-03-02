"""
main.py — Entry point for Analysis 5: Action-Reasoning Consistency.

Usage
-----
    python3 analysis5/main.py

Outputs (analysis5/results/)
-----------------------------
Figures:
    fig1_action_by_dilemma.png
    fig2_stage_action_heatmap.png
    fig3_consistency_score_bar.png
    fig4_action_by_stage_model.png
    fig5_inconsistency_network.png
    fig6_3d_stage_action_landscape.png

CSVs:
    consistency_scores_by_model.csv
    stage_action_crosstab.csv
    qualitative_inconsistencies.csv
"""

import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import OUT_DIR
from data_loader import load_and_parse_data
from stat_analysis import (
    compute_overall_consistency,
    compute_action_distributions,
    compute_stage_action_crosstab,
    run_chi_square
)
from visualizations import (
    plot_action_by_dilemma,
    plot_stage_action_heatmap,
    plot_consistency_bar,
    plot_action_by_stage_model,
    plot_stage_action_sankey,
    plot_3d_stage_action_landscape
)
from reporting import save_results, print_report

def main() -> None:
    import numpy as np
    np.random.seed(42)

    t0 = time.perf_counter()

    print("─" * 65)
    print("Step 1/5  Loading and parsing evaluation actions …")
    df = load_and_parse_data()

    print("Step 2/5  Running consistency analytics …")
    consist_df, valid_df = compute_overall_consistency(df)
    action_dists = compute_action_distributions(valid_df)
    ct = compute_stage_action_crosstab(valid_df)
    chi_results = run_chi_square(valid_df)

    print("Step 3/5  Generating 6 publication-quality figures …")
    plot_action_by_dilemma(action_dists, OUT_DIR)
    plot_stage_action_heatmap(ct, OUT_DIR)
    plot_consistency_bar(consist_df, OUT_DIR)
    plot_action_by_stage_model(valid_df, OUT_DIR)
    plot_stage_action_sankey(ct, OUT_DIR)
    plot_3d_stage_action_landscape(ct, OUT_DIR)

    print("Step 4/5  Saving CSV reports …")
    save_results(consist_df, valid_df, ct, OUT_DIR)

    print("Step 5/5  Printing console report …")
    print_report(consist_df, ct, chi_results, len(valid_df))

    elapsed = time.perf_counter() - t0
    print(f"All outputs saved to:  {OUT_DIR}")
    print(f"Total runtime:         {elapsed:.1f}s\n")

if __name__ == "__main__":
    main()
