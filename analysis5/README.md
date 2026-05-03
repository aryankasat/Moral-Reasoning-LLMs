# Analysis 5 — Action-Reasoning Consistency (Do Models Practice What They Preach?)

> **Research Question**: When a model reasons at a high moral stage, does it actually choose a correspondingly principled action?

---

## Motivation

A model might use Stage 5 vocabulary ("social contract," "rights") while choosing a Stage 3 action (conforming to authority). This **moral decoupling** — sophisticated reasoning paired with unsophisticated behaviour — would indicate surface-level moral language without genuine principled commitment.

---

## Expected Action Mapping

| Stage | Expected Action | Rationale |
|---|---|---|
| 1–4 (Pre/Conventional) | Rule-Following | Fear, exchange, conformity, law & order |
| 5–6 (Post-Conventional) | Rule-Breaking | Social contract / universal principles override rules when life > property |

---

## Statistical Methods

1. **Consistency Score** — % of observations where expected action matches actual action per model
2. **Stage-Action Cross-Tabulation** — contingency table of Kohlberg stage × action category
3. **Chi-Square Test of Independence** — stage vs. action association
4. **McNemar's Tests** — paired consistency between expected and actual actions:
   - Global (pooled across all models/dilemmas)
   - Per-model with Bonferroni correction
   - Per-dilemma with Bonferroni correction
   - Uses exact binomial when discordant pairs < 25, chi² with continuity otherwise

---

## Outputs

| File | Description |
|---|---|
| `fig1_action_by_dilemma.png` | Rule-Following vs. Rule-Breaking per dilemma |
| `fig2_stage_action_heatmap.png` | Cross-tabulation heatmap (stage × action) |
| `fig3_consistency_score_bar.png` | Per-model consistency percentage |
| `fig4_action_by_stage_model.png` | Action distributions by stage and model |
| `fig5_inconsistency_network.png` | Sankey diagram of stage → action flows |
| `fig6_3d_stage_action_landscape.png` | 3D landscape of stage-action relationships |
| `consistency_scores_by_model.csv` | Per-model consistency metrics |
| `stage_action_crosstab.csv` | Full contingency table |
| `mcnemar_global.csv` | Global McNemar result |
| `mcnemar_per_model.csv` | Per-model McNemar with Bonferroni correction |
| `mcnemar_per_dilemma.csv` | Per-dilemma McNemar with Bonferroni correction |

---

## Key Findings

- Strong statistical alignment overall — high-stage models predominantly choose principled actions
- Some models show **moral decoupling**: high-stage vocabulary with low-stage action choices
- Consistency varies by dilemma — some scenarios elicit more decoupling than others

## Usage

```bash
python analysis5/main.py
```

Requires: `statsmodels` (for McNemar's test)
