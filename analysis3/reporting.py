"""
reporting.py — Save CSVs and print console report for Analysis 3.

Public API
----------
save_results(sd_df, icc_df, agree_df, anova_results, summary, out_dir)
print_report(sd_df, icc_df, agree_df, anova_results, summary)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def save_results(
    sd_df: pd.DataFrame,
    icc_df: pd.DataFrame,
    agree_df: pd.DataFrame,
    anova_results: dict,
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Save all analysis outputs as CSV files."""

    # 1. Within-model SD
    sd_df.to_csv(out_dir / "within_model_sd.csv", index=False)
    print(f"  Saved: within_model_sd.csv")

    # 2. ICC per model
    icc_df.to_csv(out_dir / "icc_per_model.csv", index=False)
    print(f"  Saved: icc_per_model.csv")

    # 3. Sample agreement
    agree_df.to_csv(out_dir / "sample_agreement.csv", index=False)
    print(f"  Saved: sample_agreement.csv")

    # 4. Cell-level agreement (if available)
    cell_df = agree_df.attrs.get("cell_df", None)
    if cell_df is not None:
        cell_df.to_csv(out_dir / "sample_agreement_cells.csv", index=False)
        print(f"  Saved: sample_agreement_cells.csv")

    # 5. ANOVA per model
    anova_results["per_model"].to_csv(out_dir / "prompt_anova_per_model.csv", index=False)
    print(f"  Saved: prompt_anova_per_model.csv")

    # 6. Global ANOVA
    global_kw = anova_results["global_kw"]
    pd.DataFrame([global_kw]).to_csv(out_dir / "prompt_anova_global.csv", index=False)
    print(f"  Saved: prompt_anova_global.csv")

    # 7. Combined summary
    summary.to_csv(out_dir / "model_consistency_summary.csv", index=False)
    print(f"  Saved: model_consistency_summary.csv")


def print_report(
    sd_df: pd.DataFrame,
    icc_df: pd.DataFrame,
    agree_df: pd.DataFrame,
    anova_results: dict,
    summary: pd.DataFrame,
) -> None:
    """Print a concise console report of all analysis results."""
    SEP = "─" * 65

    print(f"\n{SEP}")
    print("ANALYSIS 3 — MORAL REASONING CONSISTENCY")
    print(SEP)

    # ── Within-model SD ──────────────────────────────────────────────────
    print("\n▸ WITHIN-MODEL STAGE STANDARD DEVIATION")
    print(f"  Human adult baseline SD (Colby & Kohlberg, 1987): 0.67")
    ttest = sd_df.attrs.get("ttest_vs_human", {})
    if ttest:
        print(f"  One-sample t-test vs. baseline:")
        print(f"    Mean model SD = {ttest['mean_model_sd']:.3f}")
        print(f"    t({ttest['df']}) = {ttest['t']:.3f},  p = {ttest['p']:.4f}")
        print(f"    Cohen's d     = {ttest['cohen_d']:.3f}")

    print(f"\n  {'Model':<28} {'Params (B)':>10} {'SD':>7} {'Range':>6}")
    print(f"  {'─'*28} {'─'*10} {'─'*7} {'─'*6}")
    for _, r in sd_df.iterrows():
        print(f"  {r['display_name']:<28} {r['params_B']:>10.0f} {r['std_stage']:>7.3f} {r['stage_range']:>6}")

    # ── ICC ─────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("▸ INTRACLASS CORRELATION COEFFICIENT (ICC(2,1))")
    corr = icc_df.attrs.get("scale_icc_corr", {})
    if corr:
        sig = "n.s." if corr["p"] >= 0.05 else f"p = {corr['p']:.3f}"
        print(f"  Spearman ρ (params_B vs ICC) = {corr['rho']:.3f}  [{sig}]")

    print(f"\n  {'Model':<28} {'ICC':>7} {'95% CI':>18} {'Interp'}")
    print(f"  {'─'*28} {'─'*7} {'─'*18} {'─'*12}")
    for _, r in icc_df.iterrows():
        if np.isnan(r["icc"]):
            print(f"  {r['display_name']:<28} {'n/a':>7}")
        else:
            ci = f"[{r['icc_lo']:.2f}, {r['icc_hi']:.2f}]"
            print(f"  {r['display_name']:<28} {r['icc']:>7.3f} {ci:>18}  {r['icc_interp']}")

    # ── Prompt ANOVA ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("▸ PROMPT-TYPE EFFECTS (Kruskal-Wallis; H₀: stage dist. same across ZS/COT/RP)")
    gkw = anova_results["global_kw"]
    sig_g = "***" if gkw["p"] < 0.001 else ("**" if gkw["p"] < 0.01 else ("*" if gkw["p"] < 0.05 else "n.s."))
    print(f"  Global: H({gkw['k']-1}) = {gkw['H']:.3f},  p = {gkw['p']:.4f} {sig_g},  η² = {gkw['eta_sq']:.3f}")

    print(f"\n  {'Model':<28} {'H':>7} {'p':>8} {'η²':>7} {'Sig'}")
    print(f"  {'─'*28} {'─'*7} {'─'*8} {'─'*7} {'─'*4}")
    for _, r in anova_results["per_model"].iterrows():
        if np.isnan(r["H_stat"]):
            continue
        sig = "* " if r["significant"] else "  "
        print(f"  {r['display_name']:<28} {r['H_stat']:>7.3f} {r['p_value']:>8.4f} {r['eta_sq']:>7.3f} {sig}")

    # ── Sample agreement ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("▸ SAMPLE AGREEMENT (within each model × dilemma × prompt_type cell)")
    print(f"\n  {'Model':<28} {'Exact':>7} {'Majority':>9} {'MeanMAD':>9}")
    print(f"  {'─'*28} {'─'*7} {'─'*9} {'─'*9}")
    for _, r in agree_df.iterrows():
        print(f"  {r['display_name']:<28} {r['exact_agree_rate']:>7.1%} "
              f"{r['majority_agree_rate']:>9.1%} {r['mean_mad']:>9.3f}")

    print(f"\n{SEP}\n")
