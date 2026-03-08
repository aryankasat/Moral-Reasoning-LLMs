"""
reporting.py — Markdown report generator for Analysis 9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from config import OUT_DIR, ALPHA, POST_CONV_THRESH


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def generate_report(
    model_df: pd.DataFrame,
    corr_results: dict,
    threshold_results: dict,
    reg_results: dict,
    partial_df: pd.DataFrame,
) -> None:
    lines = []
    a = lines.append

    a("# Analysis 9: Capability Correlation Analysis")
    a("")
    a("**Research Question:** Is there a capability threshold for post-conventional")
    a("reasoning (Stage 5+), and which capabilities predict moral reasoning stage?")
    a("")
    a(f"**Dataset:** {corr_results['n_models']} models | "
      f"{model_df['n_obs'].sum():,} total observations")
    a(f"**Post-conventional threshold:** ≥{int(POST_CONV_THRESH*100)}% of responses at Stage 5+")
    a(f"**Significance level:** α = {ALPHA} (FDR-corrected)")
    a("")
    a("---")
    a("")

    # ── Section 1: Model Summary ──────────────────────────────────────────────
    a("## 1. Model-Level Capability Summary")
    a("")

    cap_cols = ["display_name", "params_B", "mean_stage", "post_conv_pct",
                "coherence", "response_length", "lexical_diversity",
                "syntactic_complexity", "semantic_density"]
    available = [c for c in cap_cols if c in model_df.columns]
    a(model_df[available].round(3).to_markdown(index=False))
    a("")
    a("---")
    a("")

    # ── Section 2: Correlation Matrix ─────────────────────────────────────────
    a("## 2. Pearson Correlation Matrix (FDR-corrected p-values)")
    a("")
    r_mat = corr_results["pearson_r"]
    p_mat = corr_results["corrected_p_pearson"]
    vars_ = corr_results["variables"]
    n_v   = len(vars_)

    # Build annotated correlation table
    header = "| Variable | " + " | ".join(vars_) + " |"
    sep    = "|---|" + "---|" * n_v
    a(header); a(sep)
    for v1 in vars_:
        row_cells = []
        for v2 in vars_:
            r_val = float(r_mat.loc[v1, v2])
            p_val = float(p_mat.loc[v1, v2])
            s     = _stars(p_val) if v1 != v2 else "—"
            row_cells.append(f"{r_val:.2f}{s}")
        a(f"| {v1} | " + " | ".join(row_cells) + " |")
    a("")
    a("*Stars: \\* p<0.05 &nbsp;&nbsp; \\*\\* p<0.01 &nbsp;&nbsp; \\*\\*\\* p<0.001 (FDR-corrected)*")
    a("")
    a("### Key correlations with Mean Stage")
    a("")
    a("| Metric | Pearson r | Spearman ρ | FDR-p (Pearson) | Significant? |")
    a("|---|---|---|---|---|")
    sp_r = corr_results["spearman_r"]
    sp_p = corr_results["corrected_p_spearman"]
    for v in vars_:
        if v in ("mean_stage", "post_conv_pct"):
            continue
        r_val = float(r_mat.loc["mean_stage", v]) if "mean_stage" in r_mat.index else np.nan
        p_val = float(p_mat.loc["mean_stage", v]) if "mean_stage" in p_mat.index else np.nan
        sr_val = float(sp_r.loc["mean_stage", v]) if "mean_stage" in sp_r.index else np.nan
        sig = "✓" if p_val < ALPHA else "✗"
        a(f"| {v} | {r_val:.3f} | {sr_val:.3f} | {p_val:.4f} | {sig} |")
    a("")
    a("---")
    a("")

    # ── Section 3: Threshold Detection ───────────────────────────────────────
    a("## 3. Threshold Detection (Post-Conventional Reasoning)")
    a("")
    a("Logistic regression: P(≥20% Stage 5+) ~ Capability Metric")
    a("Linear threshold: Metric value where predicted Post-Conv% = 20%")
    a("")
    a("| Metric | Logistic AUC | Linear Threshold (Stage 5) | Sigmoid Inflection | Linear R² |")
    a("|---|---|---|---|---|")
    for metric, res in threshold_results.items():
        if isinstance(res, dict) and "error" not in res:
            auc   = res.get("logistic_auc", np.nan)
            lth   = res.get("linear_threshold_stage5", np.nan)
            sif   = res.get("sigmoid_inflection", np.nan)
            r2    = res.get("linear_r2", np.nan)
            fmt   = lambda v: f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else "—"
            a(f"| {metric} | {fmt(auc)} | {fmt(lth)} | {fmt(sif)} | {fmt(r2)} |")
    a("")
    a("---")
    a("")

    # ── Section 4: Multi-Capability Regression ────────────────────────────────
    a("## 4. Multi-Capability Regression (Mean Stage ~ All Predictors)")
    a("")
    if "error" in reg_results:
        a(f"> Error: {reg_results['error']}")
    else:
        a(f"**R² = {reg_results['r_squared']:.4f}**  |  "
          f"Adj-R² = {reg_results['adj_r2']:.4f}  |  "
          f"F = {reg_results['f_stat']:.3f}  |  p = {reg_results['f_p']:.4f}  |  "
          f"n = {reg_results['n']}")
        a("")
        a("### Standardised Regression Coefficients")
        a("")
        coef_df = reg_results["coef_df"]
        a("| Predictor | β (std) | 95% CI | t | p | Significant? |")
        a("|---|---|---|---|---|---|")
        for _, row in coef_df.iterrows():
            sig = "✓" if row["p_value"] < ALPHA else "✗"
            ci  = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
            a(f"| {row['predictor']} | {row['std_coef']:.3f} | {ci} | "
              f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {sig} |")
    a("")
    a("---")
    a("")

    # ── Section 5: Partial Correlations ───────────────────────────────────────
    a("## 5. Partial Correlations (Controlling for Model Scale)")
    a("")
    a("Partial r = correlation with Mean Stage after removing variance explained by log₁₀(Parameters)")
    a("")
    a("| Metric | Raw r | Raw p (FDR) | Partial r | Partial p (FDR) | Scale Effect |")
    a("|---|---|---|---|---|---|")
    for _, row in partial_df.iterrows():
        change = abs(float(row["raw_r"])) - abs(float(row["partial_r"]))
        direction = "↓ reduced" if change > 0.05 else ("↑ grows" if change < -0.05 else "≈ stable")
        rr, rp = float(row["raw_r"]), float(row["raw_p_fdr"])
        pr, pp = float(row["partial_r"]), float(row["partial_p_fdr"])
        fmt = lambda v: f"{v:.3f}" if not np.isnan(v) else "—"
        a(f"| {row['metric']} | {fmt(rr)} | {fmt(rp)} | {fmt(pr)} | {fmt(pp)} | {direction} |")
    a("")
    a("---")
    a("")

    # ── Section 6: Interpretation ─────────────────────────────────────────────
    a("## 6. Interpretation")
    a("")
    a("### Strongest Predictors of Moral Reasoning Stage")

    if "error" not in reg_results:
        coef_df = reg_results["coef_df"]
        top3 = coef_df.head(3)["predictor"].tolist()
        a(f"Based on standardised regression coefficients, the top predictors are: "
          f"**{', '.join(top3)}**.")
    a("")
    a("### Post-Conventional Threshold")
    best_auc_metric = max(
        ((m, r.get("logistic_auc", 0)) for m, r in threshold_results.items()
         if isinstance(r, dict) and not np.isnan(r.get("logistic_auc", np.nan))),
        key=lambda x: x[1] if not np.isnan(x[1]) else 0,
        default=("N/A", np.nan),
    )
    a(f"The metric with highest AUC for predicting post-conventional capability "
      f"is **{best_auc_metric[0]}** (AUC = {best_auc_metric[1]:.3f}).")
    a("")
    a("### Scale vs. Capability")
    a("Partial correlations reveal whether capability effects persist after controlling "
      "for model size (log parameters). Metrics showing stable partial correlations "
      "indicate task-specific capability beyond mere scale.")
    a("")
    a("---")
    a("")
    a("*Generated by analysis9/reporting.py*")

    # ── Write file ────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "analysis9_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved: {out_path.name}")
