# Analysis 3 — Within-Model Moral Reasoning Consistency

> **Research Question**: Does the same model give different moral reasoning for different dilemmas? Are models consistent across dilemmas and prompt types?

---

## Motivation

Human moral reasoning is **context-dependent** — a person might reason at Stage 5 about medical ethics but Stage 3 about interpersonal loyalty. If LLMs genuinely "reason morally," they should show similar contextual variation. Conversely, hyper-consistency would suggest **formulaic pattern reproduction** rather than genuine contextual moral reasoning.

This analysis measures within-model consistency using multiple complementary metrics: Intraclass Correlation Coefficient (ICC), within-model standard deviation, prompt-type sensitivity, and sample-level agreement rates.

---

## Human Baseline

The benchmark for comparison is the adult moral reasoning variability documented by **Colby & Kohlberg (1987)**:
- **Human adult stage SD ≈ 0.67** across different moral dilemmas
- Adults typically show 1–2 stage variation depending on the moral domain

---

## Statistical Methods

### 1. Within-Model Standard Deviation (`compute_within_model_sd`)
- Stage SD across all observations per model
- One-sample t-test comparing model SDs to the human baseline (SD = 0.67)
- Cohen's d for the effect size of the difference

### 2. Prompt-Type Sensitivity (`run_prompt_anova`)
- Per-model **Kruskal-Wallis** test: Stage ~ Prompt Type (Zero-Shot / CoT / Roleplay)
- **η²** effect size = (H − k + 1) / (n − k)
- Global pooled Kruskal-Wallis across all models
- Tests whether prompting style significantly shifts moral reasoning stage

### 3. Intraclass Correlation Coefficient (`compute_icc_per_model`)
- **ICC(2,1)**: two-way random effects, absolute agreement, single measures
- Measurement design: subjects = dilemmas (6), raters = prompt types (3)
- Interpretation thresholds (Koo & Mae, 2016): poor (<0.50), moderate (0.50–0.74), good (0.75–0.89), excellent (≥0.90), perfect (1.0)
- Spearman correlation between params_B and ICC value
- Computed via `pingouin.intraclass_corr`

### 4. Sample Agreement (`compute_sample_agreement`)
- For each (model, dilemma, prompt_type) cell:
  - **Exact agreement**: all samples have identical stage
  - **Majority agreement**: ≥2 of 3 samples agree
  - **Mean Absolute Deviation (MAD)** from cell mean
- Per-model aggregated agreement rates

---

## Code Architecture

```
analysis3/
├── main.py              ← Entry point (5-step pipeline)
├── config.py            ← MODEL_META, colour palette, rcParams
├── data_loader.py       ← Loads evaluation + response data with prompt_type
├── stat_analysis.py     ← SD, ICC, Kruskal-Wallis, agreement metrics
├── visualizations.py    ← 6 figures (2D + 3D, publication-grade)
├── reporting.py         ← CSV export + formatted console report
└── results/             ← Generated outputs
```

---

## Outputs

### Figures

| File | Description |
|---|---|
| `fig1_clustermap.png` | Hierarchically clustered heatmap of stage profiles |
| `fig2_radar_grid.png` | Per-model radar charts showing stage distribution across dilemmas/prompts |
| `fig3_violin_composite.png` | Violin + box + strip with SD panel |
| `fig4_3d_grouped_bars.png` | 3D grouped bar chart of stages |
| `fig5_bubble_scale_icc.png` | Bubble chart: scale × ICC × SD |
| `fig6_3d_surface.png` | 3D surface of the stage landscape |

### CSV Reports

| File | Contents |
|---|---|
| `within_model_sd.csv` | Per-model SD, range, t-test vs. human baseline |
| `icc_per_model.csv` | ICC(2,1) values with CIs and interpretation |
| `sample_agreement.csv` | Per-model exact/majority agreement rates |
| `sample_agreement_cells.csv` | Cell-level (model × dilemma × prompt) agreement |
| `prompt_anova_per_model.csv` | Kruskal-Wallis results per model |
| `prompt_anova_global.csv` | Global prompt-type test |
| `model_consistency_summary.csv` | Merged summary of all consistency metrics |

---

## Key Findings

- Models are **hyper-consistent** — ICC > 0.90 for most models (classified as "excellent" or "perfect")
- Model stage SDs are **significantly lower than the human baseline** of 0.67
- Prompt type (zero-shot vs. CoT vs. roleplay) has **negligible effect** on moral stage
- This consistency is almost **robotic** — suggesting pattern reproduction rather than genuine contextual moral deliberation

---

## Usage

```bash
python analysis3/main.py
# Outputs → analysis3/results/
```

Requires: `pingouin` (for ICC computation), `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`

---

## Caveats

1. **ICC interpretation assumes reliable raters**: Prompt types are used as "raters," but they may not provide truly independent measurements.
2. **Small cell sizes**: With 1 observation per (model × dilemma × prompt_type) cell, ICC estimates have wide confidence intervals.
3. **Perfect consistency can be pathological**: ICC = 1.0 (zero variance) means the model always assigns the same stage — which could indicate rigid template matching rather than reasoning.
