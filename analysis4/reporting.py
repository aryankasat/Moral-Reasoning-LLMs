"""
reporting.py — Save CSV outputs and print console report for Analysis 4.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from config import STAGES, HUMAN_ADULT


def save_results(
    dist_df:  pd.DataFrame,
    chi_df:   pd.DataFrame,
    resid_df: pd.DataFrame,
    stat_df:  pd.DataFrame,
    jsd_mat:  pd.DataFrame,
    out_dir:  Path,
) -> None:
    def _save(df, name):
        path = out_dir / name
        df.to_csv(path, index=isinstance(df.index, pd.Index) and df.index.name is not None
                        or not df.index.equals(pd.RangeIndex(len(df))))
        print(f"  Saved: {name}")

    dist_df.to_csv(out_dir / "stage_distributions.csv", index=False)
    print("  Saved: stage_distributions.csv")

    chi_df.to_csv(out_dir / "chi_square_results.csv", index=False)
    print("  Saved: chi_square_results.csv")

    resid_df.to_csv(out_dir / "pearson_residuals.csv", index=False)
    print("  Saved: pearson_residuals.csv")

    stat_df.to_csv(out_dir / "distribution_stats.csv", index=False)
    print("  Saved: distribution_stats.csv")

    jsd_mat.to_csv(out_dir / "jsd_matrix.csv")
    print("  Saved: jsd_matrix.csv")


def print_report(
    dist_df:  pd.DataFrame,
    chi_df:   pd.DataFrame,
    stat_df:  pd.DataFrame,
    jsd_mat:  pd.DataFrame,
) -> None:
    sep = "─" * 65

    print(f"\n{sep}")
    print("ANALYSIS 4 — STAGE DISTRIBUTION PATTERNS")
    print(sep)

    # Human adult baseline
    print("\n▸ HUMAN ADULT BASELINE (Colby & Kohlberg, 1987)")
    for s in STAGES:
        print(f"    Stage {s}: {HUMAN_ADULT[s]*100:.0f}%")

    # Stage distributions
    print(f"\n{sep}")
    print("▸ MODEL STAGE DISTRIBUTIONS (proportions)")
    header = f"  {'Model':<28}" + "".join(f"  S{s}" for s in STAGES) + "  Modal"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for mk, grp in dist_df.groupby("model_key", sort=False):
        grp  = grp.sort_values("stage")
        name = grp["display_name"].iloc[0]
        props = [f"{grp.loc[grp['stage'] == s, 'proportion'].values[0]*100:5.1f}%"
                 if len(grp.loc[grp['stage'] == s]) > 0 else "   0.0%"
                 for s in STAGES]
        modal = stat_df.loc[stat_df["model_key"] == mk, "modal_stage"].values
        modal_str = f"S{modal[0]}" if len(modal) > 0 else "?"
        print(f"  {name:<28}" + "  " + "  ".join(props) + f"   {modal_str}")

    # Chi-square results
    print(f"\n{sep}")
    print("▸ CHI-SQUARE GOODNESS-OF-FIT  (vs. Human Adult Norm; df = 3)")
    print(f"  {'Model':<28} {'χ²':>8} {'p':>10} {'JSD':>8}  Sig")
    print("  " + "─" * 60)
    for _, row in chi_df.sort_values("params_B").iterrows():
        sig = "***" if row["chi2_p"] < 0.001 else ("**" if row["chi2_p"] < 0.01 else
              ("*" if row["chi2_p"] < 0.05 else " n.s."))
        print(f"  {row['display_name']:<28} {row['chi2_stat']:8.2f} {row['chi2_p']:10.5f} "
              f"{row['jsd_adult']:8.4f}  {sig}")

    # Distribution stats
    print(f"\n{sep}")
    print("▸ DISTRIBUTION CHARACTERISTICS")
    print(f"  {'Model':<28} {'Entropy':>8} {'Skew':>8} {'Kurt':>8}  Pattern")
    print("  " + "─" * 68)
    for _, row in stat_df.sort_values("params_B").iterrows():
        print(f"  {row['display_name']:<28} {row['entropy_bits']:8.3f} "
              f"{row['skewness']:8.3f} {row['kurtosis']:8.3f}  {row['pattern']}")

    # JSD vs human
    print(f"\n{sep}")
    print("▸ JENSEN-SHANNON DIVERGENCE (model vs. Human Adult)")
    print(f"  {'Model':<28} {'JSD':>8}  Interpretation")
    print("  " + "─" * 50)
    for _, row in chi_df.sort_values("jsd_adult").iterrows():
        jsd = row["jsd_adult"]
        interp = "very similar" if jsd < 0.05 else \
                 "similar"      if jsd < 0.10 else \
                 "moderate"     if jsd < 0.20 else \
                 "divergent"    if jsd < 0.35 else "highly divergent"
        print(f"  {row['display_name']:<28} {jsd:8.4f}  {interp}")

    print(sep + "\n")
