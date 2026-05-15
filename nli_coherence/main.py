"""
main.py — Entry point for NLI-Based Coherence Analysis.

This analysis scores entailment between each LLM's stated moral justification
and its endorsed action using DeBERTa-v3-large (NLI fine-tune), producing a
framework-agnostic coherence measure. It then correlates these scores with
the Kohlberg-based decoupling scores (McNemar p-values) from Analysis 5.

Usage
-----
    export HF_TOKEN=hf_...
    python3 nli_coherence/main.py

Outputs (nli_coherence/results/)
---------------------------------
Figures:
    fig1_coherence_by_model.png        – Mean NLI entailment per model
    fig2_coherence_heatmap.png         – Model × dilemma coherence heatmap
    fig3_coherence_vs_decoupling.png   – Scatter: NLI coherence vs decoupling
    fig4_coherence_gap.png             – Signed gap: NLI vs Kohlberg-based
    fig5_entailment_distribution.png   – Box plots of entailment distributions
    fig6_correlation_summary.png       – Two-panel correlation summary

CSVs / JSON:
    nli_scores_all.csv                 – Per-observation entailment scores
    coherence_by_model.csv             – Per-model coherence summaries
    coherence_by_model_dilemma.csv     – Per-model × dilemma summaries
    coherence_vs_decoupling.csv        – Merged coherence + decoupling data
    correlation_results.json           – Full correlation test results
"""

import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import OUT_DIR
from data_loader import load_evaluation_pairs, load_mcnemar_pvalues
from nli_scorer import score_all_pairs
from correlation_analysis import (
    aggregate_coherence_by_model,
    aggregate_coherence_by_model_dilemma,
    merge_with_decoupling,
    run_correlation_tests,
)
from visualizations import (
    plot_coherence_by_model,
    plot_coherence_heatmap,
    plot_coherence_vs_decoupling,
    plot_coherence_gap,
    plot_entailment_distribution,
    plot_correlation_summary,
)
from reporting import save_results, print_report


def main() -> None:
    t0 = time.perf_counter()

    print("═" * 65)
    print("  NLI-Based Coherence Analysis (DeBERTa-v3-large)")
    print("═" * 65)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("\nStep 1/6  Loading evaluation data …")
    df = load_evaluation_pairs()
    mcnemar_df = load_mcnemar_pvalues()

    # ── Step 2: NLI scoring ───────────────────────────────────────────────
    print("\nStep 2/6  Running NLI entailment scoring …")
    scored_df = score_all_pairs(df)

    # ── Step 3: Aggregate ─────────────────────────────────────────────────
    print("\nStep 3/6  Aggregating coherence scores …")
    coherence_by_model   = aggregate_coherence_by_model(scored_df)
    model_dilemma_df     = aggregate_coherence_by_model_dilemma(scored_df)

    # ── Step 4: Correlate with decoupling ─────────────────────────────────
    print("\nStep 4/6  Correlating with Kohlberg decoupling scores …")
    merged_df    = merge_with_decoupling(coherence_by_model, mcnemar_df)
    corr_results = run_correlation_tests(merged_df)

    # ── Step 5: Visualize ─────────────────────────────────────────────────
    print("\nStep 5/6  Generating 6 publication-quality figures …")
    plot_coherence_by_model(coherence_by_model, OUT_DIR)
    plot_coherence_heatmap(model_dilemma_df, OUT_DIR)
    plot_coherence_vs_decoupling(merged_df, corr_results, OUT_DIR)
    plot_coherence_gap(merged_df, OUT_DIR)
    plot_entailment_distribution(scored_df, OUT_DIR)
    plot_correlation_summary(merged_df, corr_results, OUT_DIR)

    # ── Step 6: Save & report ─────────────────────────────────────────────
    print("\nStep 6/6  Saving results …")
    save_results(scored_df, coherence_by_model, model_dilemma_df,
                 merged_df, corr_results, OUT_DIR)
    print_report(coherence_by_model, merged_df, corr_results, len(scored_df))

    elapsed = time.perf_counter() - t0
    print(f"All outputs saved to:  {OUT_DIR}")
    print(f"Total runtime:         {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
