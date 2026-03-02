"""
main.py — Entry point for Analysis 6: Reasoning Pattern Analysis.

Usage
-----
    python3 analysis6/main.py

Outputs (analysis6/results/)
-----------------------------
Figures:
    fig1_stage_word_clouds.png
    fig2_model_distinctive_terms.png
    fig3_moral_vocabulary_richness.png
    fig4_target_keyword_heatmap.png
    fig5_pca_linguistic_style.png

CSVs:
    distinctive_terms_by_model.csv
    target_keyword_usage.csv
    linguistic_pca_coordinates.csv
    qualitative_exemplars.csv
"""

import sys
import time
import pandas as pd
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import OUT_DIR
from data_loader import load_and_prepare_text
from stat_analysis import (
    analyze_stage_vocabulary,
    analyze_model_distinctive_terms,
    evaluate_target_keyword_usage,
    compute_linguistic_pca,
    find_qualitative_exemplars
)
from visualizations import (
    plot_stage_word_clouds,
    plot_model_distinctive_terms,
    plot_vocabulary_richness,
    plot_target_keyword_heatmap,
    plot_linguistic_pca
)
from reporting import save_results, print_report

def main() -> None:
    import numpy as np
    np.random.seed(42)

    t0 = time.perf_counter()

    print("─" * 65)
    print("Step 1/5  Loading text and computing vocabulary richness …")
    df = load_and_prepare_text()
    
    if len(df) == 0:
        print("No valid textual data loaded. Exiting.")
        return

    print("Step 2/5  Running TF-IDF and NLP algorithms …")
    stage_terms = analyze_stage_vocabulary(df)
    model_terms_df = analyze_model_distinctive_terms(df)
    kw_usage_df = evaluate_target_keyword_usage(df)
    
    pca_df = pd.DataFrame() # Fallback
    var_exp = [0, 0]
    if len(df["display_name"].unique()) > 2:
        pca_df, var_exp = compute_linguistic_pca(df)
        
    exemplars_df = find_qualitative_exemplars(df)

    print("Step 3/5  Generating 5 publication-quality figures …")
    plot_stage_word_clouds(stage_terms, OUT_DIR)
    plot_model_distinctive_terms(model_terms_df, OUT_DIR)
    plot_vocabulary_richness(df, OUT_DIR)
    
    if len(kw_usage_df) > 0:
        plot_target_keyword_heatmap(kw_usage_df, OUT_DIR)
        
    if len(pca_df) > 0:
        plot_linguistic_pca(pca_df, var_exp, OUT_DIR)

    print("Step 4/5  Saving CSV reports …")
    save_results(model_terms_df, kw_usage_df, pca_df, exemplars_df, OUT_DIR)

    print("Step 5/5  Printing console report …")
    print_report(model_terms_df, exemplars_df, stage_terms, len(df))

    elapsed = time.perf_counter() - t0
    print(f"All outputs saved to:  {OUT_DIR}")
    print(f"Total runtime:         {elapsed:.1f}s\n")

if __name__ == "__main__":
    main()
