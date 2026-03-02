"""
main.py — Entry point for Analysis 3: Moral Reasoning Consistency.

Usage
-----
    cd analysis3/
    python3 main.py   (or from project root: python3 analysis3/main.py)

Output (analysis3/results/)
---------------------------
Figures (2D + 3D, publication-grade):
    fig1_clustermap.png               — hierarchically clustered heatmap
    fig2_radar_grid.png               — per-model radar charts (all prompt types)
    fig3_violin_composite.png         — violin + box + strip (w/ SD panel)
    fig4_3d_grouped_bars.png          — 3D grouped bar chart
    fig5_bubble_scale_icc.png         — bubble chart: scale × ICC × SD
    fig6_3d_surface.png               — 3D surface: stage landscape

CSVs:
    within_model_sd.csv
    icc_per_model.csv
    sample_agreement.csv
    sample_agreement_cells.csv
    prompt_anova_per_model.csv
    prompt_anova_global.csv
    model_consistency_summary.csv
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
    compute_within_model_sd,
    run_prompt_anova,
    compute_icc_per_model,
    compute_sample_agreement,
    compute_model_summary,
)
from visualizations import (
    plot_clustermap,
    plot_radar_grid,
    plot_violin_composite,
    plot_3d_grouped_bars,
    plot_bubble_scale_icc,
    plot_3d_surface,
)
from reporting import save_results, print_report


def main() -> None:
    import numpy as np
    np.random.seed(42)

    t0 = time.perf_counter()

    # ── 1. Load data ───────────────────────────────────────────────────────
    print("─" * 65)
    print("Step 1/5  Loading data …")
    df = load_all_data()

    # ── 2. Statistical analyses ────────────────────────────────────────────
    print("Step 2/5  Running statistical analyses …")
    print("          Within-model SD + human baseline t-test …")
    sd_df = compute_within_model_sd(df)
    print("          Prompt-type Kruskal-Wallis ANOVA …")
    anova_results = run_prompt_anova(df)
    print("          ICC(2,1) per model …")
    icc_df = compute_icc_per_model(df)
    print("          Sample agreement …")
    agree_df = compute_sample_agreement(df)
    print("          Merging summary …")
    summary = compute_model_summary(sd_df, icc_df, agree_df)

    # ── 3. Advanced figures ────────────────────────────────────────────────
    print("Step 3/5  Generating publication-quality figures …")

    print("  Figure 1: Clustered heatmap (dendrogram) …")
    plot_clustermap(df, OUT_DIR)

    print("  Figure 2: Radar charts grid (per-model stage profiles) …")
    plot_radar_grid(df, OUT_DIR)

    print("  Figure 3: Violin + box + strip composite …")
    plot_violin_composite(df, sd_df, OUT_DIR)

    print("  Figure 4: 3D grouped bar chart …")
    plot_3d_grouped_bars(df, OUT_DIR)

    print("  Figure 5: Bubble chart — scale × ICC × SD …")
    plot_bubble_scale_icc(icc_df, sd_df, OUT_DIR)

    print("  Figure 6: 3D surface — stage landscape …")
    plot_3d_surface(df, OUT_DIR)

    # ── 4. Save CSVs ───────────────────────────────────────────────────────
    print("Step 4/5  Saving CSV reports …")
    save_results(sd_df, icc_df, agree_df, anova_results, summary, OUT_DIR)

    # ── 5. Console report ──────────────────────────────────────────────────
    print("Step 5/5  Printing console report …")
    print_report(sd_df, icc_df, agree_df, anova_results, summary)

    elapsed = time.perf_counter() - t0
    print(f"All outputs saved to:  {OUT_DIR}")
    print(f"Total runtime:         {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
