# Analysis 9 — Capability Correlation Analysis (What Sub-Capabilities Drive Moral Reasoning?)

> **Research Question**: Beyond raw parameter count, which specific linguistic/structural capabilities — coherence, lexical diversity, semantic density, syntactic complexity — predict higher moral reasoning stages?

---

## Motivation

"Scale" is a proxy for many things: more parameters could enable richer vocabulary, longer coherent chains of reasoning, or more abstract conceptual representations. This analysis unpacks the "scale" black box by extracting NLP-based capability metrics from model responses and testing which ones independently predict moral reasoning sophistication.

---

## Capability Metrics

All metrics are derived from response text (no external API calls):

| Metric | Definition | Hypothesis |
|---|---|---|
| `response_length` | Whitespace-split token count | Longer responses → more nuanced reasoning |
| `lexical_diversity` | Type-token ratio (unique/total tokens) | Higher TTR → richer moral vocabulary |
| `syntactic_complexity` | avg_sentence_length × lexical_diversity | Complex syntax → more sophisticated arguments |
| `semantic_density` | Proportion of tokens matching academic/abstract word list (100 curated terms) | Abstract vocabulary → principled reasoning |
| `coherence` | Analysis time in seconds (proxy for deliberation depth) | More deliberation → better reasoning |

---

## Statistical Methods

### 1. Correlation Matrix (`compute_correlation_matrix`)
- Pearson and Spearman correlations among all capability metrics, mean_stage, post_conv_pct, and log_params
- **Bootstrap 95% CIs** on each correlation (5,000 iterations)
- **FDR correction** (Benjamini-Hochberg) for multiple comparisons

### 2. Threshold Detection (`threshold_detection`)
- **Logistic regression**: predicts above-median moral stage from each metric; reports AUC
- **Linear regression**: mean_stage ~ metric; identifies value where predicted stage = 5.0
- **Sigmoid fit**: inflection point as capability threshold

### 3. Multi-Capability Regression (`multi_capability_regression`)
- OLS: mean_stage ~ log_params + all capability metrics
- **Standardised coefficients** (z-scored predictors) for direct comparison
- R², adjusted R², F-test, AIC/BIC
- 95% CIs and significance on each predictor

### 4. Partial Correlations (`compute_partial_correlations`)
- Pearson partial correlation of each metric with mean_stage, **controlling for log_params**
- Tests whether a capability predicts moral reasoning *above and beyond* what scale explains
- FDR correction on partial p-values

---

## Outputs

| File | Description |
|---|---|
| `fig1_correlation_heatmap.png` | Annotated Pearson correlation matrix |
| `fig2_threshold_detection.png` | Logistic/linear/sigmoid threshold curves |
| `fig3_regression_coefficients.png` | Standardised β coefficients with CIs |
| `fig4_partial_correlations.png` | Raw vs. scale-controlled correlations |
| `fig5_capability_scatter_panel.png` | Individual scatter plots per metric |
| `report.md` | Full statistical report |

---

## Key Findings

- Semantic density and syntactic complexity act as **step-function thresholds** for post-conventional reasoning
- After controlling for scale (partial correlations), some metrics retain significant predictive power — suggesting they capture genuine capability dimensions beyond raw parameter count
- Multi-capability regression explains more variance than log_params alone

## Usage

```bash
python analysis9/main.py
```

Requires: `scikit-learn`, `statsmodels`
