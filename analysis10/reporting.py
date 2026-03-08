"""
reporting.py — Generates the markdown report for Analysis 10.
"""

from __future__ import annotations

import pandas as pd
from typing import Any

from config import OUT_DIR, STAGES, ACTIVE_STAGES


def generate_report(
    model_df: pd.DataFrame,
    metrics_summary: dict[str, Any],
    stats_results: dict[str, Any],
    windows: list[dict],
    qualitative_samples: dict[int, list[dict]] = None
) -> None:
    """Write the Analysis 10 Markdown report."""
    report_path = OUT_DIR / "analysis10_stage_transitions_report.md"

    # Determine pattern based on problem statement criteria
    # Pattern A (Sequential Kohlberg-like): Gradual, consolidated, sequential
    # Pattern B (Abrupt jumps): Low entropy always, skipped stages
    # Pattern C (Non-sequential): High entropy, regressions, non-sequential
    
    mean_h = metrics_summary["mean_entropy"]
    cons_idx = metrics_summary["consolidation_index"]
    chisq = stats_results["chisq_transitions"]
    regressions = stats_results["regression_freq"]
    
    seq_prop = chisq.get("sequential_proportion", 0)
    has_regressions = regressions.get("n_regression", 0) > 0
    
    if cons_idx > 0.7 and seq_prop >= 0.5 and not has_regressions:
        pattern = "Pattern A (Sequential Kohlberg-like)"
        pattern_desc = "Clear sequential progression with stages of consolidation and gradual transitions."
    elif mean_h < 0.5 and seq_prop < 0.5:
        pattern = "Pattern B (Abrupt jumps)"
        pattern_desc = "Sudden mode shifts without buildup, staying highly consolidated (skipping stages)."
    elif mean_h > 1.5 or has_regressions:
         pattern = "Pattern C (Non-sequential / Unstable)"
         pattern_desc = "Stages entered out of order, high sustained entropy, and/or notable regressions."
    else:
        pattern = "Mixed / Hybrid Pattern"
        pattern_desc = "Displays mixed characteristics (e.g., partial consolidation, some sequential steps but also regressions)."


    # Format windows table
    win_lines = []
    if not windows:
        win_lines.append("No modal transitions detected.")
    else:
        win_lines.append("| Transition | From Model | To Model | Start | End | Window Size |")
        win_lines.append("|---|---|---|---|---|---|")
        for w in windows:
            t_str = f"S{w['from_stage']} → S{w['to_stage']}"
            if w['is_regression']: t_str += " ⚠️ (Regress)"
            win_lines.append(
                f"| {t_str} | {w['at_model']} | {w['to_model']} | "
                f"idx {w['win_start_idx']} | idx {w['win_end_idx']} | {w['window_size']} steps |"
            )

    # Format statistical tests
    fri = stats_results["friedman"]
    kw_scale = stats_results["kw_scale_group"]
    kw_train = stats_results["kw_training_type"]
    spearman = stats_results["spearman_scale"]

    md_content = f"""# Analysis 10: Stage Transition Dynamics

## 1. Overview and Hypothesis Evaluation
**Research Question:** How do models transition between stages as scale increases—gradually or suddenly? Do they consolidate at stages before progressing?

**Identified Pattern:** **{pattern}**
_{pattern_desc}_

*Note: Since evaluation data contains static checkpoints of distinct modern LLMs rather than continuous training trajectories, "scale progression" (ordered by parameter count) serves as our proxy for developmental progression.*

## 2. Statistical Findings

### Stage Progression vs Scale
- **Spearman Rank Correlation (Scale vs Mean Stage):**
  - $\rho$ = {spearman.get('spearman_rho', float('nan')):+.3f} (p = {spearman.get('p_value', float('nan')):.4f})
  - **Conclusion:** {spearman.get('interpretation', 'N/A')}

### Variance Across Models
- **Friedman Test (Repeated Measures by Dilemma):**
  - $\chi^2$({fri.get('df', 'N/A')}) = {fri.get('statistic', float('nan')):.2f}, p = {fri.get('p_value', float('nan')):.4f}
  - **Conclusion:** {fri.get('interpretation', 'N/A')}

### Consolidation Differences by Group
- **Kruskal-Wallis (Scale Group):** p = {kw_scale.get('p_value', float('nan')):.4f} — {kw_scale.get('interpretation', 'N/A')}
- **Kruskal-Wallis (Training Type):** p = {kw_train.get('p_value', float('nan')):.4f} — {kw_train.get('interpretation', 'N/A')}

### Transition Characteristics
- **Sequence Analysis:**
  - Sequential transitions (i → i+1): {seq_prop*100:.1f}% of transition mass
  - Chi-square test for uniform transitions: p = {chisq.get('p_value', float('nan')):.4f} ({chisq.get('interpretation', 'N/A')})
- **Regressions:**
  - Observed regressions: {regressions.get('n_regression', 0)} / {regressions.get('n_total_transitions', 0)}
  - Binomial test (H₀: p=1/3): p = {regressions.get('binomial_p', float('nan')):.4f} ({regressions.get('interpretation', 'N/A')})

## 3. Transition Windows

{chr(10).join(win_lines)}

## 4. Key Metrics Summary
- **Mean Entropy:** {metrics_summary['mean_entropy']:.3f} ± {metrics_summary['std_entropy']:.3f} bits (Max theoretical: {metrics_summary['max_entropy']: .3f})
- **Consolidation Index:** {metrics_summary['consolidation_index']*100:.1f}% of models are highly consolidated (Entropy < 1.0)
- **Consistency Score (Agreement across dilemmas):** {metrics_summary['mean_consistency']:.3f} ± {metrics_summary['std_consistency']:.3f} (1.0 = perfect agreement)
- **Mean Gini:** {metrics_summary['mean_gini']:.3f} ± {metrics_summary['std_gini']:.3f}

## 5. Visualizations Index
- **Figure A:** `figA_transition_timing_heatmap.png` — Visualizes exact proportions of each stage across model scale.
- **Figure B:** `figB_entropy_trajectory.png` — Tracks Shannon entropy and Gini coefficients to highlight consolidation periods.
- **Figure C:** `figC_stage_alluvial.png` — Stacked proportions showing stage flow through the scale progression.
- **Figure D:** `figD_stage_residence_times.png` — Indicates which stages act as stable plateaus.
- **Figure E:** `figE_transition_matrix.png` — Outer-product proxy for aggregate transition probability between consecutive scales.
"""

    if qualitative_samples and any(qualitative_samples.values()):
        md_content += "\n## 6. Qualitative Analysis (Transition Windows)\n"
        for idx, samps in qualitative_samples.items():
            if samps:
                w = windows[idx]
                md_content += f"\n### Transition {idx+1}: Stage {w['from_stage']} → Stage {w['to_stage']}\n"
                md_content += f"*Window bounds: {w['at_model']} to {w['to_model']}*\n\n"
                for s in samps:
                    md_content += f"**Model:** {s['model']} | **Dilemma:** {s['dilemma']} | **Assigned Stage:** {s['stage']}\n"
                    # Clean the sample text up nicely for markdown blockquote
                    text_clean = str(s['text']).replace('\n', '\n> ')
                    md_content += f"> {text_clean}\n\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Report cleanly exported to {report_path.name}")

    # Export statistical tests to CSV
    csv_rows = []
    
    # helper to safely extract common keys
    def extract_stats(test_name: str, test_dict: dict) -> dict:
        return {
            "Test": test_name,
            "Statistic": test_dict.get("statistic", float("nan")),
            "p_value": test_dict.get("p_value", test_dict.get("binomial_p", float("nan"))),
            "Significant": test_dict.get("significant", test_dict.get("regressions_rare", False)),
            "Interpretation": test_dict.get("interpretation", "N/A")
        }

    csv_rows.append(extract_stats("Spearman Rank Correlation (Scale vs Mean Stage)", spearman))
    csv_rows.append(extract_stats("Friedman Test (Repeated Measures by Dilemma)", fri))
    csv_rows.append(extract_stats("Kruskal-Wallis (Scale Group)", kw_scale))
    csv_rows.append(extract_stats("Kruskal-Wallis (Training Type)", kw_train))
    csv_rows.append(extract_stats("Chi-square (Transition Sequence)", chisq))
    csv_rows.append(extract_stats("Binomial Test (Regression Frequency)", regressions))

    stats_df = pd.DataFrame(csv_rows)
    csv_path = OUT_DIR / "analysis10_statistical_tests.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"Statistical tests cleanly exported to {csv_path.name}")
