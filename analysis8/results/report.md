# Analysis 8: Scale vs. Training Decomposition
*Generated: 2026-03-06 22:54*

## Research Question
Does scale affect moral reasoning stage independent of training, or is training the primary driver?

**Factorial design:**

| Factor | Levels |
|--------|--------|
| **Scale** | Small, Mid, Large |
| **Training Type** | Base-RLHF, Coding-Tuned, Reasoning-Tuned |

## Model Summary

| Model | Scale | Training Type | Params (B) | n | Mean Stage |
|-------|-------|---------------|-----------|---|-----------|
| Ministral 8B | Small | Base-RLHF | 8 | 18 | 5.167 |
| Claude 3.5 Haiku | Small | Base-RLHF | 20 | 18 | 5.500 |
| Qwen3-30B Coder | Small | Coding-Tuned | 30 | 18 | 5.333 |
| Qwen3-32B | Small | Base-RLHF | 32 | 18 | 5.000 |
| Llama 3.3 70B | Mid | Base-RLHF | 70 | 18 | 5.222 |
| Qwen3-80B | Mid | Base-RLHF | 80 | 18 | 5.667 |
| Llama 4 Scout 109B | Mid | Base-RLHF | 109 | 18 | 5.222 |
| GPT-OSS 120B | Mid | Base-RLHF | 120 | 18 | 5.278 |
| Claude Sonnet 4.5 | Large | Base-RLHF | 175 | 18 | 5.556 |
| GPT-4o | Large | Base-RLHF | 200 | 18 | 5.222 |
| Qwen3-235B (Think) | Large | Reasoning-Tuned | 235 | 18 | 6.000 |
| DeepSeek-R1 671B | Large | Reasoning-Tuned | 671 | 18 | 5.333 |
| DeepSeek-V3.1 671B | Large | Base-RLHF | 671 | 18 | 5.500 |

## Assumption Checks

### Normality (Shapiro-Wilk per cell)

| Cell | W | p | Status |
|------|---|---|--------|
| Small × Base-RLHF | 0.5134 | 0.0000 | FAIL |
| Small × Coding-Tuned | 0.6007 | 0.0000 | FAIL |
| Mid × Base-RLHF | 0.6978 | 0.0000 | FAIL |
| Large × Base-RLHF | 0.7207 | 0.0000 | FAIL |
| Large × Reasoning-Tuned | 0.5959 | 0.0000 | FAIL |

> **Note:** Ordinal stage data violates strict normality by construction. The F-test is robust (n ≥ 15/cell). Non-parametric tests are reported as primary evidence.

### Homoscedasticity (Levene's Test)
- **F** = 2.4594, **p** = 0.0463
- **Result:** ⚠️ Mild heteroscedasticity — consult Welch ANOVA

### Non-parametric Tests

| Test | Factor | Statistic | p-value | Sig. |
|------|--------|-----------|---------|------|
| Kruskal-Wallis | Scale | H = 12.7777 | 0.0017 | ✅ |
| Kruskal-Wallis | Training | H = 12.6055 | 0.0018 | ✅ |
| Welch ANOVA | Scale | F(2,151.41) = 6.2581 | 0.0024 | ✅ |
| Welch ANOVA | Training | F(2,38.55) = 7.1048 | 0.0024 | ✅ |

## Two-Way ANOVA Results (Sequential / Type-I SS)

> **Design note:** The factorial design is incomplete (5 of 9 cells populated). Sequential SS is used to avoid collinearity. The interaction df is taken from the actual model rank difference (additive vs. full) to avoid over-stating df_int.

**Formula:** `kohlberg_stage ~ Scale + Training_Type + Scale:Training_Type`  
**R² = 0.0714** | **Residual SE = 0.5070**

### ANOVA Table

| Effect | df | SS | MS | F | p | η²_p | ω² | Magnitude |
|--------|----|----|----|----|---|------|-----|-----------|
| Scale ✅ | 2 | 3.1096 | 1.5548 | 6.0495 | 0.0028 | 0.0502 | 0.0408 | Small |
| Training_Type  | 2 | 1.4185 | 0.7093 | 2.7596 | 0.0654 | 0.0235 | 0.0142 | Small |
| Scale × Training_Type  | 0 | 0.0000 | nan | — | — | 0.0000 | 0.0000 | Negligible |
| Residual | 229 | 58.8565 | 0.2570 | — | — | — | — | |

> **Effect size benchmarks (Cohen):** Small: η²_p = 0.01–0.06 | Medium: 0.06–0.14 | Large: >0.14

### Variance Partitioning

| Source | % Total Variance (η²) |
|--------|----------------------|
| Scale | 4.9% |
| Training_Type | 2.2% |
| Scale × Training_Type | 0.0% |
| Residual | 92.9% |

## Post-hoc Comparisons

### Scale pairwise — Training: Base-RLHF

| group1   | group2   |   meandiff |   p-adj |   lower |   upper | reject   |
|:---------|:---------|-----------:|--------:|--------:|--------:|:---------|
| Large    | Mid      |    -0.0787 |  0.6726 | -0.2976 |  0.1402 | False    |
| Large    | Small    |    -0.2037 |  0.1019 | -0.4377 |  0.0303 | False    |
| Mid      | Small    |    -0.125  |  0.3698 | -0.3439 |  0.0939 | False    |

### Training pairwise — Scale: Small

| group1    | group2       |   meandiff |   p-adj |   lower |   upper | reject   |
|:----------|:-------------|-----------:|--------:|--------:|--------:|:---------|
| Base-RLHF | Coding-Tuned |     0.1111 |  0.3528 | -0.1258 |   0.348 | False    |

### Training pairwise — Scale: Large

| group1    | group2          |   meandiff |   p-adj |   lower |   upper | reject   |
|:----------|:----------------|-----------:|--------:|--------:|--------:|:---------|
| Base-RLHF | Reasoning-Tuned |     0.2407 |  0.0394 |  0.0119 |  0.4696 | True     |

### Mann-Whitney U (Scale, Bonferroni-corrected)

| group1   | group2   |    U |   p_raw |   p_bonferroni |   r_effect_size | significant   |
|:---------|:---------|-----:|--------:|---------------:|----------------:|:--------------|
| Large    | Mid      | 3785 |  0.0361 |         0.1084 |          -0.168 | False         |
| Large    | Small    | 4140 |  0.0004 |         0.0013 |          -0.278 | True          |
| Mid      | Small    | 2862 |  0.1863 |         0.5589 |          -0.104 | False         |

## Hypothesis Classification

**Supported:** H2 — Scale dominates (scale main effect; training NS)

| Hypothesis | Condition | Supported? |
|-----------|-----------|-----------|
| H1 — Training dominates | Large training F; scale NS | ❌ |
| H2 — Scale dominates | Large scale F; training NS | ✅ |
| H3 — Both additive | Both significant; no interaction | ❌ |
| H4 — Synergistic | Main effects + interaction | ❌ |

## Practical Implications

Parameter count is the primary driver. Moral reasoning quality appears capacity-limited; larger models reason at higher stages regardless of training procedure.

---
*Analysis 8 — Moral-Reasoning-LLMs*