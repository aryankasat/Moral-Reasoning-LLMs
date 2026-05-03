# Analysis 7 — Emergence Threshold Detection

> **Research Question**: Is there a critical model scale at which post-conventional moral reasoning "emerges," and does the transition happen gradually or abruptly?

---

## Motivation

Some capabilities in LLMs appear suddenly at specific scales ("emergence"). This analysis tests whether moral reasoning follows a similar pattern — is there a parameter-count threshold below which models cannot reach post-conventional (Stage 5+) reasoning, with a sharp transition once that threshold is crossed?

---

## Statistical Methods

### 1. Changepoint Detection (`detect_changepoints`)
- **PELT algorithm** (Pruned Exact Linear Time) via `ruptures` library
- RBF kernel model with BIC-like automatic penalty: pen = log(n) × σ²
- Fallback: sliding-window between-segment variance maximisation if `ruptures` unavailable
- **Bootstrap CI** on changepoint location (5,000 iterations)

### 2. Segmented Regression (`segmented_regression`)
- Two-segment linear model fit at the primary changepoint
- **F-test**: segmented model vs. simple linear model
  - H₀: single linear trend is sufficient
  - H₁: two segments significantly reduce residual SS
- Pre/post changepoint slopes + intercepts reported
- R² for both linear and segmented models

### 3. Emergence Threshold (`find_emergence_threshold`)
- Identifies the smallest model (by params_B) where ≥ threshold % of responses are post-conventional
- Binary step-function detection on cumulative capability

### 4. Scenario Classification (`classify_emergence_scenario`)
- **Scenario A — Gradual**: no clear changepoints, smooth monotonic increase
- **Scenario B — Sharp (Phase Transition)**: single changepoint with large slope change (|Δslope| > 0.3)
- **Scenario C — Multi-Stage**: multiple changepoints, stepwise emergence

### 5. Cross-Scale Correlation
- Spearman ρ between log₁₀(params) and mean_stage
- Confirms monotonic trend direction

---

## Outputs

| File | Description |
|---|---|
| `fig1_emergence_curves.png` | Three-panel emergence curves |
| `fig2_emergence_vs_params.png` | Scatter + segmented regression at changepoint |
| `fig3_stage_heatmap.png` | Stage distribution heatmap |
| `fig4_slope_analysis.png` | Pre/post changepoint slope comparison |
| `model_summary.csv` | Per-model stats |
| `emergence_metrics.csv` | All key emergence metrics |
| `analysis_results.json` | Full results bundle |

---

## Key Findings

- Post-conventional reasoning shows **gradual emergence** rather than a sharp phase transition
- No single dramatic changepoint — improvement is continuous but with diminishing returns
- The smallest model achieving substantial post-conventional reasoning provides a practical emergence threshold estimate

## Usage

```bash
python analysis7/main.py
```

Requires: `ruptures` (optional, for PELT changepoint detection)
