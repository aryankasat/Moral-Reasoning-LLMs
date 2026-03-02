"""
reporting.py — Exports qualitative results and CSVs for Analysis 6.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

def save_results(
    model_terms_df: pd.DataFrame,
    kw_usage_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    exemplars_df: pd.DataFrame,
    out_dir: Path
) -> None:
    model_terms_df.to_csv(out_dir / "distinctive_terms_by_model.csv", index=False)
    print("  Saved: distinctive_terms_by_model.csv")
    
    if len(kw_usage_df) > 0:
        kw_usage_df.to_csv(out_dir / "target_keyword_usage.csv", index=False)
        print("  Saved: target_keyword_usage.csv")
        
    pca_df.to_csv(out_dir / "linguistic_pca_coordinates.csv", index=False)
    print("  Saved: linguistic_pca_coordinates.csv")
    
    if len(exemplars_df) > 0:
        # Sort nicely for Qualitative Review reading
        clean_ex = exemplars_df.sort_values(["kohlberg_stage", "display_name", "rank"])
        clean_ex.to_csv(out_dir / "qualitative_exemplars.csv", index=False)
        print("  Saved: qualitative_exemplars.csv")

def print_report(
    model_terms_df: pd.DataFrame,
    exemplars_df: pd.DataFrame,
    stage_terms: dict,
    total_valid: int
) -> None:
    sep = "─" * 65

    print(f"\n{sep}")
    print("ANALYSIS 6 — REASONING PATTERN ANALYSIS (NLP)")
    print(sep)
    
    print("\n▸ TOP 3 DISTINCTIVE TERMS BY MODEL (TF-IDF Mined)")
    print("  " + "─" * 50)
    for _, row in model_terms_df.sort_values("display_name").iterrows():
        terms = [str(row[f"term_{i}"]) for i in range(1, 4) if pd.notna(row[f"term_{i}"])]
        print(f"  {row['display_name']:<25} | {', '.join(terms)}")
        
    print("\n▸ STAGE LINGUISTIC MARKERS")
    print("  " + "─" * 50)
    for stage in sorted(stage_terms.keys()):
        s_terms = stage_terms[stage][:5] # Top 5
        if s_terms:
            t_str = ", ".join([f"{t} ({w:.2f})" for t, w in s_terms])
            print(f"  S{stage} | {t_str}")
            
    print(f"\n▸ QUALITATIVE REVIEW EXEMPLARS")
    print("  " + "─" * 50)
    print(f"  Automatically selected {len(exemplars_df)} representative responses based on TF-IDF Centroid distance.")
    print("  Review 'qualitative_exemplars.csv' for the full quotes.")

    print(f"\n{sep}")
    print(f"Analyzed {total_valid} textual responses.")
    print(sep + "\n")
