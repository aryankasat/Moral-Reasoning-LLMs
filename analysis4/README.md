# Analysis 4 — AI vs. Human Stage Distribution Patterns

> **Research Question**: Do AI moral stage distributions resemble human developmental norms?

---

## Motivation

Even if a model scores "Stage 5" on average, the *shape* of its distribution matters. Humans show a bell-shaped curve peaking around Stage 4. If LLMs cluster exclusively at Stage 5–6, they may be producing formulaic post-conventional language rather than genuine moral reasoning diversity.

---

## Human Baselines

| Baseline | S1 | S2 | S3 | S4 | S5 | S6 | Source |
|---|---|---|---|---|---|---|---|
| **Adult** | 0% | 0% | 15% | 40% | 35% | 10% | Colby & Kohlberg, 1987 |
| **Adolescent** | 5% | 10% | 30% | 35% | 15% | 5% | Developmental norms |
| **Children** | 15% | 25% | 30% | 20% | 8% | 2% | Developmental norms |

---

## Statistical Methods

1. **Chi-Square Goodness-of-Fit** — each model vs. adult norms (S1–S3 merged to avoid zero-expected cells)
2. **Pearson Residuals** — per-stage over/under-representation
3. **Jensen-Shannon Divergence (JSD)** — symmetric [0,1] distance vs. adult, adolescent, children baselines
4. **Distribution Characteristics** — modal stage, Shannon entropy, skewness, kurtosis
5. **Pattern Classification** — hyper-principled / ceiling-biased / floor-biased / bimodal / human-like / divergent
6. **Pairwise JSD Matrix** — N×N clustered heatmap (models + 3 baselines)

---

## Outputs

| File | Description |
|---|---|
| `fig1_stacked_bar.png` | Stacked bar: all models + human baselines |
| `fig2_histogram_grid.png` | Per-model histogram with human overlay |
| `fig3_jsd_heatmap.png` | JSD clustermap |
| `fig4_distribution_stats.png` | Multi-panel: entropy / skewness / mean / pattern |
| `fig5_chi_square.png` | Chi-square bar + Pearson residual heatmap |
| `fig6_3d_stage_landscape.png` | 3D bar chart of stage proportions |
| `stage_distributions.csv` | Per-model stage counts/proportions |
| `chi_square_results.csv` | Chi-square + JSD per model |
| `distribution_stats.csv` | Entropy, skewness, kurtosis, pattern labels |
| `jsd_matrix.csv` | Full pairwise JSD matrix |

---

## Key Findings

- Most models show a **ceiling effect** at Stage 5–6
- Very few produce **human-like** distributions (JSD < 0.10)
- RLHF-tuned frontier models converge closest to human adult patterns, but with a post-conventional skew

## Usage

```bash
python analysis4/main.py
```
