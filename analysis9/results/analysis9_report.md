# Analysis 9: Capability Correlation Analysis

**Research Question:** Is there a capability threshold for post-conventional
reasoning (Stage 5+), and which capabilities predict moral reasoning stage?

**Dataset:** 13 models | 234 total observations
**Post-conventional threshold:** ≥20% of responses at Stage 5+
**Significance level:** α = 0.05 (FDR-corrected)

---

## 1. Model-Level Capability Summary

| display_name       |   params_B |   mean_stage |   post_conv_pct |   coherence |   response_length |   lexical_diversity |   syntactic_complexity |   semantic_density |
|:-------------------|-----------:|-------------:|----------------:|------------:|------------------:|--------------------:|-----------------------:|-------------------:|
| Ministral 8B       |          8 |        5.167 |         100     |       4.578 |          1149.17  |               0.375 |                  4.23  |              0.053 |
| Claude 3.5 Haiku   |         20 |        5.5   |         100     |       4.491 |           367.444 |               0.536 |                 15.734 |              0.165 |
| Qwen3-30B Coder    |         30 |        5.333 |         100     |       6.107 |           713.611 |               0.4   |                 10.233 |              0.061 |
| Qwen3-32B          |         32 |        5     |         100     |       4.908 |          1174.5   |               0.331 |                  4.36  |              0.054 |
| Llama 3.3 70B      |         70 |        5.222 |         100     |       5.492 |           576.556 |               0.387 |                  6.718 |              0.054 |
| Qwen3-80B          |         80 |        5.667 |         100     |       4.984 |          1503.78  |               0.351 |                  4.27  |              0.037 |
| Llama 4 Scout 109B |        109 |        5.222 |          88.889 |       3.894 |           546.5   |               0.39  |                  6.569 |              0.048 |
| GPT-OSS 120B       |        120 |        5.278 |         100     |       4.406 |          1115.11  |               0.412 |                  5.934 |              0.043 |
| Claude Sonnet 4.5  |        175 |        5.556 |         100     |       7.533 |          1165.5   |               0.434 |                 10.543 |              0.046 |
| GPT-4o             |        200 |        5.222 |          88.889 |       3.727 |           350.222 |               0.497 |                  7.385 |              0.059 |
| Qwen3-235B (Think) |        235 |        6     |         100     |       5.112 |          1732.17  |               0.412 |                  5.75  |              0.052 |
| DeepSeek-R1 671B   |        671 |        5.333 |         100     |       3.517 |           445.389 |               0.616 |                  5.905 |              0.055 |
| DeepSeek-V3.1 671B |        671 |        5.5   |         100     |       4.338 |           586.222 |               0.475 |                  6.618 |              0.041 |

---

## 2. Pearson Correlation Matrix (FDR-corrected p-values)

| Variable | coherence | response_length | sentence_count | avg_sentence_length | lexical_diversity | syntactic_complexity | semantic_density | mean_stage | post_conv_pct | log_params |
|---|---|---|---|---|---|---|---|---|---|---|
| coherence | 1.00— | 0.40ns | 0.09ns | 0.56ns | -0.39ns | 0.29ns | -0.11ns | 0.26ns | 0.43ns | -0.19ns |
| response_length | 0.40ns | 1.00— | 0.93*** | -0.27ns | -0.61ns | -0.45ns | -0.41ns | 0.48ns | 0.42ns | -0.12ns |
| sentence_count | 0.09ns | 0.93*** | 1.00— | -0.56ns | -0.53ns | -0.65ns | -0.44ns | 0.36ns | 0.38ns | -0.11ns |
| avg_sentence_length | 0.56ns | -0.27ns | -0.56ns | 1.00— | 0.09ns | 0.92*** | 0.62ns | 0.13ns | 0.06ns | -0.32ns |
| lexical_diversity | -0.39ns | -0.61ns | -0.53ns | 0.09ns | 1.00— | 0.43ns | 0.42ns | 0.13ns | -0.06ns | 0.49ns |
| syntactic_complexity | 0.29ns | -0.45ns | -0.65ns | 0.92*** | 0.43ns | 1.00— | 0.81* | 0.17ns | 0.04ns | -0.17ns |
| semantic_density | -0.11ns | -0.41ns | -0.44ns | 0.62ns | 0.42ns | 0.81* | 1.00— | 0.05ns | 0.08ns | -0.40ns |
| mean_stage | 0.26ns | 0.48ns | 0.36ns | 0.13ns | 0.13ns | 0.17ns | 0.05ns | 1.00— | 0.28ns | 0.36ns |
| post_conv_pct | 0.43ns | 0.42ns | 0.38ns | 0.06ns | -0.06ns | 0.04ns | 0.08ns | 0.28ns | 1.00— | -0.15ns |
| log_params | -0.19ns | -0.12ns | -0.11ns | -0.32ns | 0.49ns | -0.17ns | -0.40ns | 0.36ns | -0.15ns | 1.00— |

*Stars: \* p<0.05 &nbsp;&nbsp; \*\* p<0.01 &nbsp;&nbsp; \*\*\* p<0.001 (FDR-corrected)*

### Key correlations with Mean Stage

| Metric | Pearson r | Spearman ρ | FDR-p (Pearson) | Significant? |
|---|---|---|---|---|
| coherence | 0.263 | 0.291 | 0.5985 | ✗ |
| response_length | 0.483 | 0.299 | 0.3881 | ✗ |
| sentence_count | 0.363 | 0.191 | 0.4324 | ✗ |
| avg_sentence_length | 0.129 | 0.136 | 0.8624 | ✗ |
| lexical_diversity | 0.125 | 0.391 | 0.8624 | ✗ |
| syntactic_complexity | 0.170 | 0.197 | 0.8135 | ✗ |
| semantic_density | 0.049 | -0.324 | 0.8944 | ✗ |
| log_params | 0.357 | 0.405 | 0.4324 | ✗ |

---

## 3. Threshold Detection (Post-Conventional Reasoning)

Logistic regression: P(≥20% Stage 5+) ~ Capability Metric
Linear threshold: Metric value where predicted Post-Conv% = 20%

| Metric | Logistic AUC | Linear Threshold (Stage 5) | Sigmoid Inflection | Linear R² |
|---|---|---|---|---|
| coherence | 0.675 | -1.226 | -1.108 | 0.069 |
| response_length | 0.700 | -514.252 | 134719.580 | 0.233 |
| sentence_count | 0.625 | -96.966 | 13110.947 | 0.132 |
| avg_sentence_length | 0.575 | -52.487 | 1710.097 | 0.017 |
| lexical_diversity | 0.650 | -0.511 | -20.205 | 0.016 |
| syntactic_complexity | 0.600 | -20.924 | 191.005 | 0.029 |
| semantic_density | 0.275 | -0.933 | 25.027 | 0.002 |
| log_params | 0.638 | -0.396 | -3.150 | 0.128 |

---

## 4. Multi-Capability Regression (Mean Stage ~ All Predictors)

**R² = 0.8405**  |  Adj-R² = 0.5215  |  F = 2.635  |  p = 0.1826  |  n = 13

### Standardised Regression Coefficients

| Predictor | β (std) | 95% CI | t | p | Significant? |
|---|---|---|---|---|---|
| sentence_count | 0.684 | [-0.767, 2.134] | 1.309 | 0.2607 | ✗ |
| avg_sentence_length | 0.451 | [-1.160, 2.061] | 0.777 | 0.4806 | ✗ |
| response_length | -0.358 | [-1.681, 0.964] | -0.752 | 0.4939 | ✗ |
| log_params | 0.257 | [-0.116, 0.631] | 1.916 | 0.1278 | ✗ |
| syntactic_complexity | -0.058 | [-1.847, 1.730] | -0.091 | 0.9322 | ✗ |
| coherence | -0.034 | [-0.373, 0.305] | -0.278 | 0.7948 | ✗ |
| semantic_density | 0.032 | [-0.605, 0.668] | 0.138 | 0.8967 | ✗ |
| lexical_diversity | 0.007 | [-0.486, 0.499] | 0.037 | 0.9725 | ✗ |

---

## 5. Partial Correlations (Controlling for Model Scale)

Partial r = correlation with Mean Stage after removing variance explained by log₁₀(Parameters)

| Metric | Raw r | Raw p (FDR) | Partial r | Partial p (FDR) | Scale Effect |
|---|---|---|---|---|---|
| coherence | 0.263 | 0.797 | 0.363 | 0.520 | ↑ grows |
| response_length | 0.483 | 0.664 | 0.568 | 0.300 | ↑ grows |
| sentence_count | 0.363 | 0.779 | 0.434 | 0.486 | ↑ grows |
| avg_sentence_length | 0.129 | 0.797 | 0.275 | 0.538 | ↑ grows |
| lexical_diversity | 0.125 | 0.797 | -0.062 | 0.840 | ↓ reduced |
| syntactic_complexity | 0.170 | 0.797 | 0.253 | 0.538 | ↑ grows |
| semantic_density | 0.049 | 0.875 | 0.224 | 0.538 | ↑ grows |

---

## 6. Interpretation

### Strongest Predictors of Moral Reasoning Stage
Based on standardised regression coefficients, the top predictors are: **sentence_count, avg_sentence_length, response_length**.

### Post-Conventional Threshold
The metric with highest AUC for predicting post-conventional capability is **response_length** (AUC = 0.700).

### Scale vs. Capability
Partial correlations reveal whether capability effects persist after controlling for model size (log parameters). Metrics showing stable partial correlations indicate task-specific capability beyond mere scale.

---

*Generated by analysis9/reporting.py*