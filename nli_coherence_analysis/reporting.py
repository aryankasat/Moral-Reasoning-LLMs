"""
reporting.py — Markdown report generator for NLI Coherence Analysis.

Produces a comprehensive report documenting:
  - NLI scoring methodology and model details
  - Summary statistics (per model, per dilemma)
  - Correlation results with existing decoupling metrics
  - Key findings and interpretation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from config import OUT_DIR, NLI_MODEL_ID


def generate_report(
    scored_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    correlation_results: dict[str, Any],
    data_source: str,
) -> Path:
    """
    Generate a Markdown report summarising the NLI coherence analysis.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Observation-level data with NLI scores.
    model_summary : pd.DataFrame
        Model-level aggregation.
    correlation_results : dict
        Output from correlation.run_all_correlations().
    data_source : str
        'main' or 'rlhf'.
    """
    out_path = OUT_DIR / "report.md"
    lines: list[str] = []

    def _w(line: str = "") -> None:
        lines.append(line)

    # ── Header ────────────────────────────────────────────────────────────
    _w("# NLI-Based Coherence Measure — Analysis Report")
    _w()
    _w("> **Framework-independent validation of reasoning–action consistency**")
    _w("> using Natural Language Inference (DeBERTa-v3-large).")
    _w()
    _w("---")
    _w()

    # ── Methodology ───────────────────────────────────────────────────────
    _w("## Methodology")
    _w()
    _w(f"**NLI Model**: `{NLI_MODEL_ID}`")
    _w(f"**Data Source**: {'Main project evaluation data (13 models)' if data_source == 'main' else 'RLHF causal analysis (base vs. instruct pairs)'}")
    _w(f"**N observations scored**: {len(scored_df):,}")
    _w()
    _w("**Pipeline**:")
    _w("1. Extract moral justification from each LLM response (premise)")
    _w("2. Convert `action_endorsed` into a natural-language hypothesis")
    _w("3. Score P(entailment | justification → action) using DeBERTa NLI")
    _w("4. Correlate NLI coherence with existing Kohlberg-based consistency")
    _w()
    _w("---")
    _w()

    # ── NLI Score Summary ─────────────────────────────────────────────────
    _w("## NLI Score Summary")
    _w()

    valid = scored_df.dropna(subset=["nli_entailment"])
    if len(valid) > 0:
        ent = valid["nli_entailment"]
        _w("### Overall Distribution")
        _w()
        _w(f"| Statistic | Value |")
        _w(f"|---|---|")
        _w(f"| Mean | {ent.mean():.4f} |")
        _w(f"| Median | {ent.median():.4f} |")
        _w(f"| Std Dev | {ent.std():.4f} |")
        _w(f"| Min | {ent.min():.4f} |")
        _w(f"| Max | {ent.max():.4f} |")
        _w(f"| N scored | {len(valid):,} |")
        _w()

    # ── Model-Level Table ─────────────────────────────────────────────────
    _w("### Per-Model NLI Scores")
    _w()

    if len(model_summary) > 0:
        cols = ["display_name", "params_B", "mean_nli_entailment",
                "std_nli_entailment", "consistency_pct", "n_scored"]
        available_cols = [c for c in cols if c in model_summary.columns]
        tbl = model_summary[available_cols].copy()

        # Rename for readability
        renames = {
            "display_name": "Model",
            "params_B": "Params (B)",
            "mean_nli_entailment": "Mean NLI Ent.",
            "std_nli_entailment": "Std NLI",
            "consistency_pct": "Consistency %",
            "n_scored": "N",
        }
        tbl = tbl.rename(columns={k: v for k, v in renames.items() if k in tbl.columns})

        # Round
        for col in ["Mean NLI Ent.", "Std NLI"]:
            if col in tbl.columns:
                tbl[col] = tbl[col].round(4)

        _w(tbl.to_markdown(index=False))
        _w()

    # ── Per-Dilemma Summary ───────────────────────────────────────────────
    _w("### Per-Dilemma NLI Scores")
    _w()

    if "dilemma_type" in valid.columns:
        dilemma_stats = valid.groupby("dilemma_type")["nli_entailment"].agg(
            ["mean", "median", "std", "count"]
        ).round(4).reset_index()
        dilemma_stats.columns = ["Dilemma", "Mean", "Median", "Std", "N"]
        _w(dilemma_stats.to_markdown(index=False))
        _w()

    _w("---")
    _w()

    # ── Correlation Results ───────────────────────────────────────────────
    _w("## Correlation Results")
    _w()

    # 1. Point-biserial
    pb = correlation_results.get("pointbiserial", {})
    _w("### 1. Observation-Level: NLI Entailment vs. Consistency (Point-Biserial)")
    _w()
    if "error" not in pb:
        _w(f"| Metric | Value |")
        _w(f"|---|---|")
        _w(f"| r | {pb.get('r', 'N/A'):.4f} |" if isinstance(pb.get('r'), float) else f"| r | {pb.get('r', 'N/A')} |")
        _w(f"| p-value | {pb.get('p_value', 'N/A'):.4f} |" if isinstance(pb.get('p_value'), float) else f"| p-value | {pb.get('p_value', 'N/A')} |")
        _w(f"| Significant (α=0.05) | {'✅ Yes' if pb.get('significant') else '❌ No'} |")
        _w(f"| N | {pb.get('n', 'N/A')} |")
        _w(f"| Mean NLI (consistent) | {pb.get('mean_nli_consistent', 'N/A'):.4f} |" if isinstance(pb.get('mean_nli_consistent'), float) else "")
        _w(f"| Mean NLI (inconsistent) | {pb.get('mean_nli_inconsistent', 'N/A'):.4f} |" if isinstance(pb.get('mean_nli_inconsistent'), float) else "")
        _w()
        _w(f"**Interpretation**: {pb.get('interpretation', '')}")
    else:
        _w(f"⚠️ {pb['error']}")
    _w()

    # 2. Model-level
    ml = correlation_results.get("model_level", {})
    _w("### 2. Model-Level: Mean NLI vs. Consistency %")
    _w()
    if "error" not in ml:
        _w(f"| Metric | Value |")
        _w(f"|---|---|")
        _w(f"| N models | {ml.get('n_models', 'N/A')} |")
        _w(f"| Pearson r | {ml.get('pearson_r', 'N/A'):.4f} |" if isinstance(ml.get('pearson_r'), float) else f"| Pearson r | {ml.get('pearson_r', 'N/A')} |")
        _w(f"| Pearson p | {ml.get('pearson_p', 'N/A'):.4f} |" if isinstance(ml.get('pearson_p'), float) else f"| Pearson p | {ml.get('pearson_p', 'N/A')} |")
        _w(f"| Spearman ρ | {ml.get('spearman_rho', 'N/A'):.4f} |" if isinstance(ml.get('spearman_rho'), float) else f"| Spearman ρ | {ml.get('spearman_rho', 'N/A')} |")
        _w(f"| Spearman p | {ml.get('spearman_p', 'N/A'):.4f} |" if isinstance(ml.get('spearman_p'), float) else f"| Spearman p | {ml.get('spearman_p', 'N/A')} |")
        _w()
        _w(f"**Interpretation**: {ml.get('interpretation', '')}")
    else:
        _w(f"⚠️ {ml['error']}")
    _w()

    # 3. RLHF pairs
    rlhf = correlation_results.get("rlhf_pairs", {})
    _w("### 3. RLHF Pair Comparison: Base vs. Instruct NLI Coherence")
    _w()
    if rlhf.get("skipped"):
        _w(f"⚠️ Skipped — {rlhf.get('reason', 'not RLHF data')}")
    elif "error" not in rlhf:
        _w(f"| Metric | Value |")
        _w(f"|---|---|")
        _w(f"| Base mean NLI | {rlhf.get('overall_base_mean', 'N/A'):.4f} |" if isinstance(rlhf.get('overall_base_mean'), float) else "")
        _w(f"| Instruct mean NLI | {rlhf.get('overall_instruct_mean', 'N/A'):.4f} |" if isinstance(rlhf.get('overall_instruct_mean'), float) else "")
        _w(f"| Δ NLI | {rlhf.get('overall_delta', 'N/A'):+.4f} |" if isinstance(rlhf.get('overall_delta'), float) else "")
        _w(f"| Mann-Whitney U | {rlhf.get('mann_whitney_U', 'N/A')} |")
        _w(f"| p-value | {rlhf.get('mann_whitney_p', 'N/A'):.4f} |" if isinstance(rlhf.get('mann_whitney_p'), float) else "")
        _w(f"| Significant | {'✅ Yes' if rlhf.get('significant') else '❌ No'} |")
        _w()

        # Per-pair breakdown
        per_pair = rlhf.get("per_pair", [])
        if per_pair:
            _w("**Per-Pair Breakdown:**")
            _w()
            _w("| Pair | Base Mean NLI | Instruct Mean NLI | Δ |")
            _w("|---|---|---|---|")
            for pp in per_pair:
                _w(f"| {pp['pair_id']} | {pp['base_mean_nli']:.4f} | {pp['instruct_mean_nli']:.4f} | {pp['delta_nli']:+.4f} |")
            _w()

        _w(f"**Interpretation**: {rlhf.get('interpretation', '')}")
    else:
        _w(f"⚠️ {rlhf.get('error', 'Unknown error')}")
    _w()

    # 4. Partial correlation
    pc = correlation_results.get("partial_corr", {})
    _w("### 4. Partial Correlation: NLI vs. Consistency | Stage")
    _w()
    if "error" not in pc:
        _w(f"| Metric | Value |")
        _w(f"|---|---|")
        _w(f"| Zero-order r | {pc.get('r_zero_order', 'N/A'):.4f} |" if isinstance(pc.get('r_zero_order'), float) else "")
        _w(f"| Partial r (controlling stage) | {pc.get('r_partial', 'N/A'):.4f} |" if isinstance(pc.get('r_partial'), float) else "")
        _w(f"| Partial p | {pc.get('p_partial', 'N/A'):.4f} |" if isinstance(pc.get('p_partial'), float) else "")
        _w(f"| Significant | {'✅ Yes' if pc.get('partial_sig') else '❌ No'} |")
        _w(f"| N | {pc.get('n', 'N/A')} |")
        _w()
        _w(f"**Interpretation**: {pc.get('interpretation', '')}")
    else:
        _w(f"⚠️ {pc['error']}")
    _w()

    # Bootstrap CIs
    _w("### 5. Bootstrap Confidence Intervals")
    _w()

    boot_pb = correlation_results.get("bootstrap_pointbiserial", {})
    if boot_pb:
        _w(f"**Point-biserial bootstrap (5000 iterations):**")
        _w(f"  r = {boot_pb.get('observed_r', 0):.4f}  "
           f"95% CI [{boot_pb.get('ci_lower', 0):.4f}, {boot_pb.get('ci_upper', 0):.4f}]  "
           f"SE = {boot_pb.get('se', 0):.4f}")
        _w()

    boot_ml = correlation_results.get("bootstrap_model_level", {})
    if boot_ml:
        _w(f"**Model-level Spearman bootstrap (5000 iterations):**")
        _w(f"  ρ = {boot_ml.get('observed_r', 0):.4f}  "
           f"95% CI [{boot_ml.get('ci_lower', 0):.4f}, {boot_ml.get('ci_upper', 0):.4f}]  "
           f"SE = {boot_ml.get('se', 0):.4f}")
        _w()

    _w("---")
    _w()

    # ── Key Findings ──────────────────────────────────────────────────────
    _w("## Key Findings")
    _w()

    findings = []

    # Point-biserial finding
    if "error" not in pb:
        if pb.get("significant"):
            findings.append(
                f"✅ **NLI validates Kohlberg consistency**: Observations classified as "
                f"consistent by the Kohlberg framework have significantly higher NLI "
                f"entailment scores (r={pb.get('r', 0):.3f}, p={pb.get('p_value', 1):.4f})."
            )
        else:
            findings.append(
                f"⚠️ **NLI does not significantly correlate with Kohlberg consistency** "
                f"at the observation level (r={pb.get('r', 0):.3f}, p={pb.get('p_value', 1):.4f}). "
                f"This may indicate that NLI coherence captures different aspects of "
                f"reasoning quality than stage-based consistency."
            )

    # Partial correlation finding
    if "error" not in pc:
        if pc.get("partial_sig"):
            findings.append(
                f"✅ **NLI captures information beyond stage labels**: Partial r="
                f"{pc.get('r_partial', 0):.3f} after controlling for Kohlberg stage "
                f"(p={pc.get('p_partial', 1):.4f})."
            )

    # RLHF finding
    if "error" not in rlhf and not rlhf.get("skipped"):
        if rlhf.get("significant"):
            findings.append(
                f"✅ **RLHF increases genuine coherence**: Instruct models show significantly "
                f"higher NLI entailment than base models (Δ={rlhf.get('overall_delta', 0):+.3f}, "
                f"p={rlhf.get('mann_whitney_p', 1):.4f})."
            )
        else:
            findings.append(
                f"⚠️ **RLHF's coherence effect is not significant**: Base and instruct models "
                f"show similar NLI coherence (Δ={rlhf.get('overall_delta', 0):+.3f}, "
                f"p={rlhf.get('mann_whitney_p', 1):.4f}), suggesting RLHF may primarily affect "
                f"vocabulary rather than reasoning-action alignment."
            )

    if not findings:
        findings.append("Analysis completed. See detailed tables above for results.")

    for f in findings:
        _w(f)
        _w()

    _w("---")
    _w()
    _w(f"*Report generated automatically by `nli_coherence_analysis/reporting.py`*")

    # ── Write file ────────────────────────────────────────────────────────
    report_text = "\n".join(lines)
    out_path.write_text(report_text, encoding="utf-8")
    print(f"  ✅ Report saved → {out_path.name}")

    return out_path
