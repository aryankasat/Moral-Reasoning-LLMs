"""
reporting.py — Save CSV/JSON results and print console summary.
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def save_results(
    scored_df:          pd.DataFrame,
    coherence_by_model: pd.DataFrame,
    model_dilemma_df:   pd.DataFrame,
    merged_df:          pd.DataFrame,
    corr_results:       dict,
    out_dir:            Path,
) -> None:
    """Save all CSV and JSON outputs."""

    # 1. Per-observation NLI scores
    out_cols = [
        "model_key", "display_name", "params_B", "provider",
        "dilemma_type", "entailment_score", "not_entailment_score",
    ]
    scored_df[out_cols].to_csv(out_dir / "nli_scores_all.csv", index=False)

    # 2. Per-model coherence summary
    coherence_by_model.to_csv(out_dir / "coherence_by_model.csv", index=False)

    # 3. Per-model × per-dilemma coherence
    model_dilemma_df.to_csv(out_dir / "coherence_by_model_dilemma.csv", index=False)

    # 4. Merged coherence + decoupling
    merged_df.to_csv(out_dir / "coherence_vs_decoupling.csv", index=False)

    # 5. Correlation results (JSON)
    with open(out_dir / "correlation_results.json", "w") as f:
        json.dump(corr_results, f, indent=2, default=str)

    print(f"  → Saved 4 CSV files and 1 JSON file to {out_dir}/")


def print_report(
    coherence_by_model: pd.DataFrame,
    merged_df:          pd.DataFrame,
    corr_results:       dict,
    n_total:            int,
) -> None:
    """Print a console summary of the NLI coherence analysis."""

    print("\n" + "═" * 70)
    print("  NLI-BASED COHERENCE ANALYSIS — SUMMARY REPORT")
    print("═" * 70)

    print(f"\n  Total (reasoning, action) pairs scored: {n_total:,}")
    print(f"  Models evaluated: {len(coherence_by_model)}")

    print("\n─── Per-Model NLI Coherence Scores ─────────────────────────────────")
    for _, row in coherence_by_model.iterrows():
        print(f"  {row['display_name']:>25s}  "
              f"mean={row['mean_entailment']:.3f}  "
              f"sd={row['std_entailment']:.3f}  "
              f"(n={int(row['n_scored'])})")

    print("\n─── Correlation: NLI Coherence × Kohlberg Decoupling ────────────────")

    for key, label in [
        ("coherence_vs_pvalue",     "Coherence vs. McNemar p-value"),
        ("coherence_vs_decoupling", "Coherence vs. Decoupling Strength"),
    ]:
        cv = corr_results.get(key, {})
        print(f"\n  {label}:")
        print(f"    Spearman ρ = {cv.get('spearman_r', '?'):>8}  (p = {cv.get('spearman_p', '?')})")
        print(f"    Pearson  r = {cv.get('pearson_r', '?'):>8}  (p = {cv.get('pearson_p', '?')})")
        print(f"    Kendall  τ = {cv.get('kendall_tau', '?'):>8}  (p = {cv.get('kendall_p', '?')})")

    gap = corr_results.get("coherence_gap_stats", {})
    print(f"\n─── Coherence Gap (NLI − Decoupling) ───────────────────────────────")
    print(f"  Mean gap:   {gap.get('mean_gap', '?')}")
    print(f"  Median gap: {gap.get('median_gap', '?')}")
    print(f"  Std gap:    {gap.get('std_gap', '?')}")
    print(f"  Range:      [{gap.get('min_gap', '?')}, {gap.get('max_gap', '?')}]")

    print("\n─── Interpretation ─────────────────────────────────────────────────")
    cv_dec = corr_results.get("coherence_vs_decoupling", {})
    sp_r = cv_dec.get("spearman_r", 0)
    sp_p = cv_dec.get("spearman_p", 1)

    if isinstance(sp_r, (int, float)) and isinstance(sp_p, (int, float)):
        if sp_p < 0.05:
            direction = "positive" if sp_r > 0 else "negative"
            print(f"  ✓ Significant {direction} correlation detected (ρ={sp_r}, p={sp_p}).")
            if sp_r > 0:
                print("    → Models with higher NLI coherence tend to show MORE")
                print("      Kohlberg-based decoupling (stage-action mismatch).")
                print("    → Reasoning is internally coherent but doesn't follow")
                print("      predicted stage-action patterns.")
            else:
                print("    → Models with higher NLI coherence tend to show LESS")
                print("      Kohlberg-based decoupling (better stage-action alignment).")
        else:
            print(f"  ○ No significant correlation (ρ={sp_r}, p={sp_p}).")
            print("    → NLI coherence and Kohlberg-based consistency appear")
            print("      to be independent dimensions of moral reasoning quality.")

    print("\n" + "═" * 70 + "\n")
