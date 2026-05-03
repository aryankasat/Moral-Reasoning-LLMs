# Analysis 1 — Scale vs. Moral Reasoning Stage

> **Research Question**: Does model size (parameter count) predict a higher Kohlberg moral development stage?

---

## Motivation

If moral reasoning in LLMs is an emergent capability, we should see a positive relationship between model scale and moral sophistication. This analysis tests that hypothesis using Spearman rank correlation between log-transformed parameter count and mean Kohlberg stage, supplemented by non-parametric group-level comparisons.

---

## Theoretical Framework

Kohlberg's Stages of Moral Development (1969) provide a 6-level ordinal scale:

| Level | Stage | Core Logic |
|---|---|---|
| Pre-conventional | 1 – Obedience | Avoid punishment |
| Pre-conventional | 2 – Self-Interest | Instrumental exchange |
| Conventional | 3 – Conformity | Social approval |
| Conventional | 4 – Law & Order | Rules and duty |
| Post-conventional | 5 – Social Contract | Rights and greatest good |
| Post-conventional | 6 – Universal Ethics | Conscience-driven principles |

The **post-conventional threshold** (Stage 5+) is the primary benchmarking target — only ~10–15% of human adults consistently reason at this level.

---

## Data

- **Source**: `evaluation_data/*_evaluation.xlsx` (Kohlberg stage scores) joined with `data/*.xlsx` (response metadata)
- **Models**: 14 models spanning 7B–671B parameters across 6 providers (Anthropic, OpenAI, Meta, Mistral AI, Alibaba, DeepSeek)
- **Observations**: ~18 per model (6 dilemmas × 3 prompt types)

---

## Statistical Methods

### 1. Descriptive Statistics (`compute_model_stats`)
- Per-model: mean, median, mode, SD of Kohlberg stage
- Stage distribution percentages (S1–S6) for each model
- 95% bootstrap confidence intervals on mean stage (5,000 iterations, seed=42)

### 2. Spearman Rank Correlation (`spearman_with_ci`)
- **Variables**: log₁₀(params_B) vs. mean Kohlberg stage
- **Why Spearman?** — Ordinal DV (stage), non-linear scale–stage relationship, robust to outliers
- **Bootstrap CI**: 5,000 resampled ρ values → 95% percentile interval
- **ρ² (rho-squared)**: proportion of rank variance explained
- **Effect size**: negligible (<0.10), small (0.10–0.29), medium (0.30–0.49), large (≥0.50)

### 3. Kruskal-Wallis H Test (`run_nonparametric_tests`)
- Non-parametric omnibus test for between-model stage differences
- **η² effect size**: (H − k + 1) / (n − k)
- **Dunn post-hoc**: pairwise comparisons with Bonferroni correction (FWER control)

---

## Code Architecture

```
analysis1/
├── main.py              ← Entry point: orchestrates the full pipeline
├── config.py            ← MODEL_META registry, paths, Okabe-Ito palette, rcParams
├── data_loader.py       ← Loads and merges evaluation + response data
├── stat_analysis.py     ← Spearman correlation, Kruskal-Wallis, Dunn, bootstrap CIs
├── visualizations.py    ← 4 publication-quality figures (300 DPI, serif fonts)
├── reporting.py         ← CSV export + formatted console report
├── scale_vs_moral_reasoning.py  ← Legacy monolithic script (superseded by modular pipeline)
└── results/             ← Generated outputs
```

---

## Outputs

### Figures

| File | Description |
|---|---|
| `fig1_box_stage_by_model.png` | Horizontal box plots with jittered strip, provider-coloured medians; models ordered by parameter count |
| `fig2_scatter_scale_vs_stage.png` | Scatter of log₁₀(params) vs. mean stage with 95% CI error bars, OLS trend line, auto-repelled labels |
| `fig3_heatmap_stage_distribution.png` | Clustered heatmap of stage % per model (YlOrRd palette, zero cells white) |
| `fig4_bar_mean_stage.png` | Horizontal bar chart of mean stage ± 95% bootstrap CI, coloured by provider |

### CSV Reports

| File | Contents |
|---|---|
| `model_stats.csv` | Per-model descriptive statistics (mean, median, mode, SD, CIs, stage distribution) |
| `spearman_correlation.csv` | ρ, p-value, 95% CI, ρ², effect size label, significance flag |
| `kruskal_wallis.csv` | H statistic, df, p-value, η² |
| `dunn_posthoc_pvalues.csv` | Bonferroni-adjusted pairwise p-values (display name labels) |
| `dunn_posthoc_significant.csv` | Boolean significance matrix (p_adj < 0.05) |

---

## Key Findings

- **Moderate positive correlation** between log-parameter count and mean Kohlberg stage
- Bigger models generally score higher, but with **diminishing returns** past ~70B parameters
- Kruskal-Wallis confirms **highly significant between-model differences**
- The relationship is monotonic but not linear — suggesting a **ceiling effect** at Stages 5–6 for frontier models

---

## Usage

```bash
# From project root
python analysis1/main.py

# All outputs appear in analysis1/results/
```

Requires: `numpy`, `pandas`, `scipy`, `scikit-posthocs`, `matplotlib`, `seaborn`, `adjustText`, `openpyxl`

---

## Design Decisions & Caveats

1. **Log-transformation**: Parameter counts span 7B–671B (nearly 2 orders of magnitude). Log₁₀ linearises the relationship and prevents large models from dominating the correlation.
2. **Spearman over Pearson**: The Kohlberg stage is ordinal (not interval). Spearman tests monotonic association without assuming linearity.
3. **Bootstrap over asymptotic CIs**: With only 14 models, large-sample approximations are unreliable. Bootstrap CIs provide more honest coverage.
4. **Bonferroni correction**: Conservative FWER control for Dunn post-hoc — appropriate given the exploratory nature of 91 pairwise comparisons.
5. **Closed-source parameter estimates**: Parameter counts for Claude and GPT-4o are estimates. Update `config.py` if better numbers become available.
