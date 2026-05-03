# Analysis 10 — Stage Transition Dynamics (Growth Trajectories)

> **Research Question**: As model scale increases, do moral reasoning stages progress gradually, jump abruptly, or follow non-sequential patterns?

---

## Motivation

Kohlberg's theory predicts that human moral development follows a **sequential, invariant progression** — individuals move from Stage 1 → 2 → 3 → 4 → 5 → 6 without skipping stages or regressing. Do LLMs follow the same pattern across scales, or do they exhibit non-human developmental trajectories?

---

## Metrics

### Distribution Metrics
- **Shannon entropy** — stage distribution spread (higher = more evenly distributed across stages)
- **Gini coefficient** — concentration measure (higher = more concentrated on fewer stages)
- **Consolidation index** — proportion of responses at the modal stage

### Transition Analysis
- **Aggregate transition matrix** — probability of transitioning from stage i to stage j across model scale
- **Residence times** — how many scale steps a model "stays" at each dominant stage
- **Transition windows** — detected intervals where the dominant stage shifts

### Developmental Pattern Classification
- **Pattern A (Sequential)** — stages progress monotonically with scale
- **Pattern B (Step-wise)** — stable plateaus with discrete jumps
- **Pattern C (Non-sequential/Unstable)** — regressions, oscillations, non-monotonic progression

---

## Statistical Tests

1. **Friedman Test** — non-parametric repeated measures on stage scores across models (blocks = dilemmas, treatments = models ordered by scale)
2. **Kruskal-Wallis** — entropy differences across scale groups and training types
3. **Chi-Square on Transition Matrix** — are sequential (i → i+1) transitions more common than expected under uniform distribution?
4. **Binomial Test on Regression Frequency** — are backward transitions significantly rare (H₀: P(regression) = 1/3)?
5. **Spearman Rank Correlation** — model_order (scale proxy) vs. mean_stage

---

## Outputs

| File | Description |
|---|---|
| `fig1_stage_trajectories.png` | Stage trajectory curves across model scale |
| `fig2_transition_matrix.png` | Aggregate transition probability heatmap |
| `fig3_entropy_gini.png` | Entropy and Gini coefficient across scale |
| `fig4_residence_times.png` | Per-stage residence time analysis |
| `fig5_developmental_pattern.png` | Pattern classification visualisation |
| `report.md` | Full statistical report with pattern classification |

---

## Key Findings

- Model progression follows **Pattern C (Non-sequential / Unstable)**
- Transitions between stages are non-sequential — models skip stages and occasionally regress
- High sustained entropy across dilemmas indicates models lack the stable developmental trajectories seen in humans
- This fundamentally distinguishes LLM "moral development" from human Kohlbergian progression

## Usage

```bash
python analysis10/main.py
```
