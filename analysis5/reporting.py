"""
reporting.py — Save CSV outputs and print console report for Analysis 5.

Public API
----------
save_results(consist_df, valid_df, ct, mcnemar_results, out_dir)
    Writes CSVs including McNemar output files.
print_report(consist_df, ct, chi_results, mcnemar_results, total_valid)
    Prints all sections including McNemar.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def save_results(
    consist_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    ct: pd.DataFrame,
    mcnemar_results: dict,
    out_dir: Path,
) -> None:
    """Write CSV artefacts to out_dir."""
    consist_df.to_csv(out_dir / "consistency_scores_by_model.csv", index=False)
    print("  Saved: consistency_scores_by_model.csv")

    ct.to_csv(out_dir / "stage_action_crosstab.csv")
    print("  Saved: stage_action_crosstab.csv")

    # Qualitative inconsistencies
    inconsist_df = valid_df[~valid_df["is_consistent"]][
        ["model_key", "dilemma_type", "kohlberg_stage", "action_endorsed", "action_category"]
    ]
    inconsist_df.to_csv(out_dir / "qualitative_inconsistencies.csv", index=False)
    print("  Saved: qualitative_inconsistencies.csv")

    # McNemar global test
    g = mcnemar_results["global_test"]
    global_df = pd.DataFrame([{
        "a": g["a"], "b": g["b"], "c": g["c"], "d": g["d"],
        "n_discordant": g["n_discordant"],
        "statistic": g["statistic"],
        "p_value": g["p_value"],
        "significant": g["significant"],
        "method": g["method"],
    }])
    global_df.to_csv(out_dir / "mcnemar_global.csv", index=False, float_format="%.6f")
    print("  Saved: mcnemar_global.csv")

    # McNemar per-model
    mcnemar_results["per_model"].to_csv(
        out_dir / "mcnemar_per_model.csv", index=False, float_format="%.6f"
    )
    print("  Saved: mcnemar_per_model.csv")

    # McNemar per-dilemma
    mcnemar_results["per_dilemma"].to_csv(
        out_dir / "mcnemar_per_dilemma.csv", index=False, float_format="%.6f"
    )
    print("  Saved: mcnemar_per_dilemma.csv")


def print_report(
    consist_df: pd.DataFrame,
    ct: pd.DataFrame,
    chi_results: dict,
    mcnemar_results: dict,
    total_valid: int,
) -> None:
    sep  = "═" * 72
    thin = "─" * 72

    print(f"\n{sep}")
    print("  ANALYSIS 5 — ACTION-REASONING CONSISTENCY")
    print(sep)

    # ── Stage × Action crosstab ───────────────────────────────────────────
    print("\n▸ STAGE × ACTION OVERALL CROSSTAB\n")
    print(f"  {'Stage':<8} | {'Rule-Following':>14} | {'Rule-Breaking':>14}")
    print("  " + "─" * 42)
    for s in ct.index:
        col1 = ct.loc[s, "Rule-Following"] if "Rule-Following" in ct.columns else 0
        col2 = ct.loc[s, "Rule-Breaking"]  if "Rule-Breaking"  in ct.columns else 0
        print(f"  S{s:<7} | {col1:14d} | {col2:14d}")

    # ── Chi-square ────────────────────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ CHI-SQUARE TEST OF INDEPENDENCE  (Stage × Action)\n")
    if not np.isnan(chi_results["chi2"]):
        sig = "✓ SIGNIFICANT" if chi_results["significant"] else "✗ not significant"
        print(f"  χ²-statistic  = {chi_results['chi2']:.3f}")
        print(f"  df            = {chi_results['dof']:.0f}")
        print(f"  p-value       = {chi_results['p_value']:.6f}  ({sig} at α = 0.05)")
        note = ("The endorsed action depends strongly on the reasoning stage."
                if chi_results["significant"]
                else "Actions appear independent of reasoning stage.")
        print(f"  Interpretation: {note}")
    else:
        print("  Test invalid — insufficient variance in actions/stages.")

    # ── McNemar global ────────────────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ McNEMAR'S TEST  (Paired: Expected vs. Actual Action)  — GLOBAL\n")
    g = mcnemar_results["global_test"]
    print("  2×2 Paired Contingency Table")
    print("  " + "─" * 44)
    print(f"  {'':30s}  {'Actual RF':>10}  {'Actual RB':>10}")
    print(f"  {'Expected Rule-Following (RF)':30s}  {g['a']:>10d}  {g['b']:>10d}")
    print(f"  {'Expected Rule-Breaking  (RB)':30s}  {g['c']:>10d}  {g['d']:>10d}")
    print(f"\n  Discordant pairs (b + c)  = {g['n_discordant']}")
    print(f"  Method                    = {g['method']}")
    print(f"  Statistic                 = {g['statistic']:.4f}")
    g_sig = "✓ SIGNIFICANT" if g["significant"] else "✗ not significant"
    print(f"  p-value                   = {g['p_value']:.6f}  ({g_sig} at α = 0.05)")
    if g["significant"]:
        direction = "more" if g["b"] > g["c"] else "fewer"
        print(f"  Interpretation: LLMs endorse {direction} rule-breaking actions than"
              f" their stage predicts (b={g['b']}, c={g['c']}).")
    else:
        print("  Interpretation: No significant asymmetry between expected and actual"
              " action choices across the dataset.")

    # ── McNemar per-model ─────────────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ McNEMAR'S TEST  (Per-Model, Bonferroni-corrected)\n")
    pm = mcnemar_results["per_model"]
    print(f"  {'Model':<26} {'b':>4} {'c':>4} {'statistic':>10} {'p_raw':>10} {'p_adj':>10} {'Sig?':>6}")
    print("  " + "─" * 66)
    for _, row in pm.iterrows():
        flag = "✓" if row["significant_adj"] else "ns"
        print(f"  {row['display_name']:<26} {int(row['b']):>4d} {int(row['c']):>4d} "
              f"{row['statistic']:>10.4f} {row['p_value']:>10.4f} "
              f"{row['p_adj_bonferroni']:>10.4f} {flag:>6}")
    n_sig_model = pm["significant_adj"].sum()
    print(f"\n  {n_sig_model} / {len(pm)} models significant after Bonferroni correction.")
    print("  (b = expected RF but actual RB;  c = expected RB but actual RF)")

    # ── McNemar per-dilemma ───────────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ McNEMAR'S TEST  (Per-Dilemma, Bonferroni-corrected)\n")
    pd_ = mcnemar_results["per_dilemma"]
    print(f"  {'Dilemma':<28} {'b':>4} {'c':>4} {'statistic':>10} {'p_raw':>10} {'p_adj':>10} {'Sig?':>6}")
    print("  " + "─" * 68)
    for _, row in pd_.iterrows():
        flag = "✓" if row["significant_adj"] else "ns"
        print(f"  {row['dilemma_type']:<28} {int(row['b']):>4d} {int(row['c']):>4d} "
              f"{row['statistic']:>10.4f} {row['p_value']:>10.4f} "
              f"{row['p_adj_bonferroni']:>10.4f} {flag:>6}")
    n_sig_dil = pd_["significant_adj"].sum()
    print(f"\n  {n_sig_dil} / {len(pd_)} dilemmas significant after Bonferroni correction.")

    # ── Model consistency scores ──────────────────────────────────────────
    print(f"\n{thin}")
    print("▸ MODEL CONSISTENCY SCORES  (% responses aligned with stage theory)\n")
    print(f"  {'Model':<28} {'Valid N':>8}  {'Consist %':>10}")
    print("  " + "─" * 52)
    for _, row in consist_df.sort_values("consistency_pct", ascending=False).iterrows():
        print(f"  {row['display_name']:<28} {row['n_valid_actions']:8d}  {row['consistency_pct']:9.1f}%")

    print(f"\n  Analyzed {total_valid} responses categorized into Rule-Following / Rule-Breaking.")
    print(sep + "\n")
