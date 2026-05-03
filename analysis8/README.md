# Analysis 8 — Scale vs. Training Decomposition (Two-Way ANOVA)

> **Research Question**: Does model scale or training procedure drive moral reasoning stage — and do they interact?

---

## Motivation

Analyses 1 and 2 showed that both scale and alignment type correlate with moral reasoning. But which is the **primary driver**? This analysis formally decomposes the variance using a two-way factorial ANOVA (Scale Group × Training Type), supplemented by non-parametric alternatives for robustness.

---

## Factorial Design

| Factor | Levels |
|---|---|
| **Scale Group** | Small (8–32B), Mid (70–120B), Large (175–671B) |
| **Training Type** | Base-RLHF, Coding-Tuned, Reasoning-Tuned |

**Note**: This is an **incomplete factorial design** — only 5 of 9 cells are populated (not all scale × training combinations exist among the 13 models). The statistical pipeline accounts for this via rank-based df calculation.

---

## Statistical Methods

### 1. Assumption Checks (`check_assumptions`)
- **Shapiro-Wilk** per cell (normality)
- **Levene's test** (homoscedasticity)
- **Kruskal-Wallis** for Scale and Training (non-parametric alternative)
- **Welch ANOVA** (variance-robust parametric) for both factors

### 2. Two-Way ANOVA — Sequential (Type-I) SS (`run_two_way_anova`)
- Nested OLS model comparison:
  - SS_Scale = SSR(null) − SSR(Scale)
  - SS_Training = SSR(Scale) − SSR(Scale + Training)
  - SS_Interaction = SSR(additive) − SSR(full)
- Interaction df derived from actual model rank differences (not nominal k₁×k₂)
- Handles aliased columns from the incomplete design

### 3. Effect Sizes (`compute_effect_sizes`)
- **Partial η²** = SS_effect / (SS_effect + SS_residual)
- **ω²** = (SS_effect − df × MS_resid) / (SS_total + MS_resid) — preferred for reporting
- Magnitude: Negligible (<0.01), Small (0.01–0.06), Medium (0.06–0.14), Large (≥0.14)

### 4. Variance Partitioning (`variance_partition`)
- % of total SS attributed to each source (Scale, Training, Interaction, Residual)

### 5. Post-Hoc Comparisons (`run_posthoc`)
- **Tukey HSD**: Scale pairwise within each Training Type, and vice versa
- **Mann-Whitney U** with Bonferroni correction: non-parametric pairwise (overall)
- **Cohen's d** for scale group pairs

### 6. Hypothesis Classification (`classify_hypothesis`)
- **H1** — Training dominates
- **H2** — Scale dominates
- **H3** — Both matter (additive)
- **H4** — Synergistic interaction

---

## Outputs

| File | Description |
|---|---|
| `summary_panel.png` | Journal-ready 4-panel summary figure |
| `interaction_plot.png` | Scale × Training interaction plot |
| `bar_with_jitter.png` | Bar chart with individual data points |
| `box_violin.png` | Box + violin by factor levels |
| `posthoc_matrix.png` | Post-hoc significance matrix |
| `variance_bars.png` | Variance partitioning bar chart |
| `report.md` | Full Markdown statistical report |

---

## Key Results

| Test | Effect | Statistic | p-value |
|---|---|---|---|
| Sequential ANOVA | Scale | F(2,229)=6.05 | **0.003** ✅ |
| Sequential ANOVA | Training | F(2,229)=2.76 | 0.065 |
| Kruskal-Wallis | Scale | H=12.78 | **0.002** ✅ |
| Welch ANOVA | Scale | F=6.26 | **0.002** ✅ |
| Mann-Whitney (Bonf.) | Large vs. Small | — | **0.001** ✅ |

**Supported hypothesis: H2 — Scale Dominates.** Training shows conditional effects only within the Large group (Reasoning-Tuned > Base-RLHF, Tukey p=.039).

Effect sizes: Scale partial η²=0.050, ω²=0.041, Cohen's d=0.55 (Large vs. Small).

## Usage

```bash
python analysis8/main.py
```

Requires: `statsmodels`, `scipy`
