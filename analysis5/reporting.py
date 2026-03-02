"""
reporting.py — Save CSV outputs and print console report for Analysis 5.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

def save_results(
    consist_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    ct: pd.DataFrame,
    out_dir: Path
) -> None:
    consist_df.to_csv(out_dir / "consistency_scores_by_model.csv", index=False)
    print("  Saved: consistency_scores_by_model.csv")
    
    ct.to_csv(out_dir / "stage_action_crosstab.csv")
    print("  Saved: stage_action_crosstab.csv")
    
    # Save a filtered dataset mapping the specific inconsistencies for qualitative review
    inconsist_df = valid_df[~valid_df["is_consistent"]][
        ["model_key", "dilemma_type", "kohlberg_stage", "action_endorsed", "action_category"]
    ]
    inconsist_df.to_csv(out_dir / "qualitative_inconsistencies.csv", index=False)
    print("  Saved: qualitative_inconsistencies.csv")

def print_report(
    consist_df: pd.DataFrame,
    ct: pd.DataFrame,
    chi_results: dict,
    total_valid: int
) -> None:
    sep = "─" * 65

    print(f"\n{sep}")
    print("ANALYSIS 5 — ACTION-REASONING CONSISTENCY")
    print(sep)
    
    print("\n▸ STAGE × ACTION OVERALL CROSSTAB")
    print("  " + "─" * 40)
    # Print neat table
    print(f"  {'Stage':<8} | {'Rule-Following':>14} | {'Rule-Breaking':>14}")
    print("  " + "-" * 40)
    for s in ct.index:
        col1 = ct.loc[s, 'Rule-Following'] if 'Rule-Following' in ct.columns else 0
        col2 = ct.loc[s, 'Rule-Breaking'] if 'Rule-Breaking' in ct.columns else 0
        print(f"  S{s:<7} | {col1:14d} | {col2:14d}")
        
    print(f"\n▸ CHI-SQUARE TEST OF INDEPENDENCE")
    print("  " + "─" * 40)
    if not np.isnan(chi_results['chi2']):
        print(f"  Chi-Square Stat : {chi_results['chi2']:8.2f}")
        print(f"  Degrees of Free : {chi_results['dof']:8.0f}")
        print(f"  P-value         : {chi_results['p_value']:8.6f}")
        print(f"  Significant?    : {'Yes' if chi_results['significant'] else 'No'} (p < 0.05)")
        if chi_results['significant']:
            print("  Interpretation  : The endorsed action depends strongly on the reasoning stage.")
        else:
            print("  Interpretation  : Actions appear independent of reasoning stage (Inconsistent).")
    else:
        print("  Test invalid due to lack of variance in actions/stages.")

    print(f"\n{sep}")
    print("▸ MODEL CONSISTENCY SCORES (% responses aligned with theory)")
    print(f"  {'Model':<28} {'Valid N':>8}  {'Consist %':>10}")
    print("  " + "─" * 50)
    for _, row in consist_df.sort_values("consistency_pct", ascending=False).iterrows():
        print(f"  {row['display_name']:<28} {row['n_valid_actions']:8d}  {row['consistency_pct']:9.1f}%")

    print(f"\n  Analyzed {total_valid} responses categorized into Rule-Following/Breaking.")
    print(sep + "\n")

import numpy as np # Added for isnan check
