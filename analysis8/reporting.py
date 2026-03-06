"""
reporting.py — Generates the Markdown report for Analysis 8: Scale vs. Training Decomposition.
"""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
from config import OUT_DIR, SCALE_ORDER, TRAINING_ORDER


def generate_report(
    assumptions:  dict,
    anova_table:  pd.DataFrame,
    effect_df:    pd.DataFrame,
    partition:    dict[str, float],
    posthoc:      dict[str, pd.DataFrame],
    hypothesis:   str,
    model_df:     pd.DataFrame,
    r_squared:    float,
    residual_se:  float,
) -> str:
    lines: list[str] = []
    A = lines.append

    A("# Analysis 8: Scale vs. Training Decomposition")
    A(f"*Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    A("")
    A("## Research Question")
    A("Does scale affect moral reasoning stage independent of training, "
       "or is training the primary driver?")
    A("")
    A("**Factorial design:**")
    A("")
    A("| Factor | Levels |")
    A("|--------|--------|")
    A(f"| **Scale** | {', '.join(SCALE_ORDER)} |")
    A(f"| **Training Type** | {', '.join(TRAINING_ORDER)} |")
    A("")

    # ── Model Summary ──────────────────────────────────────────────────────────
    A("## Model Summary")
    A("")
    A("| Model | Scale | Training Type | Params (B) | n | Mean Stage |")
    A("|-------|-------|---------------|-----------|---|-----------|")
    for _, row in model_df.iterrows():
        A(f"| {row['display_name']} | {row['scale_group']} | {row['training_type']} "
           f"| {row['params_B']} | {row['n_obs']} | {row['mean_stage']:.3f} |")
    A("")

    # ── Assumption Checks ──────────────────────────────────────────────────────
    A("## Assumption Checks")
    A("")
    A("### Normality (Shapiro-Wilk per cell)")
    A("")
    A("| Cell | W | p | Status |")
    A("|------|---|---|--------|")
    for cell, W, p, status in assumptions.get("shapiro_results", []):
        W_s = f"{W:.4f}" if W is not None else "—"
        p_s = f"{p:.4f}" if p is not None else "—"
        A(f"| {cell} | {W_s} | {p_s} | {status} |")
    A("")
    A("> **Note:** Ordinal stage data violates strict normality by construction. "
       "The F-test is robust (n ≥ 15/cell). Non-parametric tests are reported as primary evidence.")
    A("")

    A("### Homoscedasticity (Levene's Test)")
    lev_p = assumptions.get("levene_p")
    A(f"- **F** = {assumptions.get('levene_stat')}, **p** = {lev_p}")
    A(f"- **Result:** {'✅ Homoscedasticity assumed' if (lev_p and lev_p > 0.05) else '⚠️ Mild heteroscedasticity — consult Welch ANOVA'}")
    A("")

    A("### Non-parametric Tests")
    kw_s  = assumptions.get("kruskal_scale", {})
    kw_t  = assumptions.get("kruskal_training", {})
    wl_s  = assumptions.get("welch_scale", {})
    wl_t  = assumptions.get("welch_training", {})
    A("")
    A("| Test | Factor | Statistic | p-value | Sig. |")
    A("|------|--------|-----------|---------|------|")
    A(f"| Kruskal-Wallis | Scale | H = {kw_s.get('H', '—')} | {kw_s.get('p', '—')} | "
       f"{'✅' if kw_s.get('p', 1) < 0.05 else ''} |")
    A(f"| Kruskal-Wallis | Training | H = {kw_t.get('H', '—')} | {kw_t.get('p', '—')} | "
       f"{'✅' if kw_t.get('p', 1) < 0.05 else ''} |")
    A(f"| Welch ANOVA | Scale | F({wl_s.get('df1','?')},{wl_s.get('df2','?')}) = {wl_s.get('F','—')} | "
       f"{wl_s.get('p', '—')} | {'✅' if wl_s.get('p', 1) < 0.05 else ''} |")
    A(f"| Welch ANOVA | Training | F({wl_t.get('df1','?')},{wl_t.get('df2','?')}) = {wl_t.get('F','—')} | "
       f"{wl_t.get('p', '—')} | {'✅' if wl_t.get('p', 1) < 0.05 else ''} |")
    A("")

    # ── ANOVA Table ───────────────────────────────────────────────────────────
    A("## Two-Way ANOVA Results (Sequential / Type-I SS)")
    A("")
    A("> **Design note:** The factorial design is incomplete (5 of 9 cells populated). "
       "Sequential SS is used to avoid collinearity. The interaction df is taken from "
       "the actual model rank difference (additive vs. full) to avoid over-stating df_int.")
    A("")
    A(f"**Formula:** `kohlberg_stage ~ Scale + Training_Type + Scale:Training_Type`  ")
    A(f"**R² = {r_squared:.4f}** | **Residual SE = {residual_se:.4f}**")
    A("")
    A("### ANOVA Table")
    A("")
    A("| Effect | df | SS | MS | F | p | η²_p | ω² | Magnitude |")
    A("|--------|----|----|----|----|---|------|-----|-----------|")
    for _, row in effect_df.iterrows():
        ms  = row["SS"] / row["df"] if row["df"] > 0 else float("nan")
        sig = "✅" if row["significant"] else ""
        p_s = f"{row['p_value']:.4f}" if not np.isnan(float(row["p_value"])) else "—"
        f_s = f"{row['F']:.4f}"       if not np.isnan(float(row["F"]))       else "—"
        e2  = f"{row['partial_eta2']:.4f}" if not np.isnan(float(row['partial_eta2'])) else "—"
        w2  = f"{row['omega2']:.4f}"       if not np.isnan(float(row['omega2']))       else "—"
        A(f"| {row['effect']} {sig} | {row['df']} | {row['SS']:.4f} | {ms:.4f} "
           f"| {f_s} | {p_s} | {e2} | {w2} | {row['magnitude']} |")
    if "Residual" in anova_table.index:
        ss_r = anova_table.loc["Residual", "sum_sq"]
        df_r = int(anova_table.loc["Residual", "df"])
        A(f"| Residual | {df_r} | {ss_r:.4f} | {ss_r/df_r:.4f} | — | — | — | — | |")
    A("")
    A("> **Effect size benchmarks (Cohen):** Small: η²_p = 0.01–0.06 | Medium: 0.06–0.14 | Large: >0.14")
    A("")

    # ── Variance Partitioning ─────────────────────────────────────────────────
    A("### Variance Partitioning")
    A("")
    A("| Source | % Total Variance (η²) |")
    A("|--------|----------------------|")
    for source, pct in partition.items():
        A(f"| {source} | {pct:.1f}% |")
    A("")

    # ── Post-hoc ──────────────────────────────────────────────────────────────
    A("## Post-hoc Comparisons")
    A("")
    for title, ph_df in posthoc.items():
        label = (title
                 .replace("Scale_within_", "Scale pairwise — Training: ")
                 .replace("Training_within_", "Training pairwise — Scale: ")
                 .replace("MannWhitney_Scale_Overall", "Mann-Whitney U (Scale, Bonferroni-corrected)")
                 .replace("_", " "))
        A(f"### {label}")
        A("")
        A(ph_df.to_markdown(index=False))
        A("")

    # ── Hypothesis ────────────────────────────────────────────────────────────
    A("## Hypothesis Classification")
    A("")
    A(f"**Supported:** {hypothesis}")
    A("")
    h1 = any(r["effect"] == "Training_Type" and r["significant"] for _, r in effect_df.iterrows())
    h2 = any(r["effect"] == "Scale" and r["significant"] for _, r in effect_df.iterrows())
    hi = any("×" in r["effect"] and r["significant"] for _, r in effect_df.iterrows())
    A("| Hypothesis | Condition | Supported? |")
    A("|-----------|-----------|-----------|")
    A(f"| H1 — Training dominates | Large training F; scale NS | {'✅' if h1 and not h2 else '❌'} |")
    A(f"| H2 — Scale dominates | Large scale F; training NS | {'✅' if h2 and not h1 else '❌'} |")
    A(f"| H3 — Both additive | Both significant; no interaction | {'✅' if h1 and h2 and not hi else '❌'} |")
    A(f"| H4 — Synergistic | Main effects + interaction | {'✅' if hi else '❌'} |")
    A("")

    # ── Implications ──────────────────────────────────────────────────────────
    A("## Practical Implications")
    A("")
    if "H1" in hypothesis:
        A("Training procedure is the primary lever for moral reasoning. "
           "Alignment should focus on RLHF and reasoning-tuning independent of scale.")
    elif "H2" in hypothesis:
        A("Parameter count is the primary driver. Moral reasoning quality appears "
           "capacity-limited; larger models reason at higher stages regardless of training procedure.")
    elif "H3" in hypothesis:
        A("Both scale and training procedure independently contribute. "
           "Optimal systems require both sufficient parameters and appropriate training.")
    elif "H4" in hypothesis:
        A("Scale and training interact synergistically. Reasoning-tuning is most effective "
           "in larger models, suggesting a readiness threshold.")
    A("")
    A("---")
    A("*Analysis 8 — Moral-Reasoning-LLMs*")

    text = "\n".join(lines)
    path = OUT_DIR / "report.md"
    path.write_text(text, encoding="utf-8")
    print(f"  [SAVED] {path.name}")
    return str(path)
