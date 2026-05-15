# NLI-Based Coherence Analysis

## Overview

This module implements a **framework-agnostic coherence measure** that uses Natural Language Inference (NLI) to score the logical entailment between each LLM's stated moral justification and its endorsed action. Unlike Analysis 5 (which evaluates action-reasoning consistency *through the lens of Kohlberg's moral development stages*), this analysis asks a fundamentally different question:

> **Does the model's own reasoning logically entail its chosen action, regardless of any moral development framework?**

The coherence scores are then correlated with the Kohlberg-based **decoupling scores** (McNemar p-values from Analysis 5) to examine whether internal reasoning coherence and stage-action alignment are related or independent dimensions of moral reasoning quality.

---

## Motivation

Analysis 5 measures consistency by checking whether a model's assigned Kohlberg stage predicts its action choice (e.g., Stage 5–6 → Rule-Breaking, Stage 1–4 → Rule-Following). A model can score poorly on this measure even if its reasoning is internally coherent — for example, a model at Stage 6 that endorses rule-following with a logically airtight justification.

This NLI-based approach **decouples the coherence assessment from the Kohlberg framework entirely**, using a pre-trained NLI model (DeBERTa-v3-large) to evaluate whether the stated justification logically supports the declared action.

---

## Method

### Contrastive NLI Scoring

For each `(model, dilemma, trial)` observation, we construct a combined text and use contrastive zero-shot classification:

| Component | Value |
|-----------|-------|
| **Input Text** | `"Reasoning: <kohlberg_reasoning>  Action: <action_endorsed>"` |
| **Label A** | `"the reasoning logically supports the action"` |
| **Label B** | `"the reasoning contradicts the action"` |
| **Template** | `"In this text, {}."` |

The NLI model outputs `P(supports)` ∈ [0, 1], which is the **coherence score**:
- **1.0** → The reasoning perfectly supports the action
- **0.0** → The reasoning contradicts the action

> **Why contrastive labels?** Initial experiments with direct entailment/not-entailment scoring showed poor discrimination (53% vs 47%). The contrastive approach gives dramatically better separation: 73% for coherent pairs vs 0.8% for contradictory pairs.

### Model

| Attribute | Value |
|-----------|-------|
| Model | `MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33` |
| Architecture | DeBERTa-v3-large (304M params) |
| Training | Fine-tuned on 33 NLI/classification datasets (387 classes) |
| API | HuggingFace Inference API (zero-shot classification) |

### Correlation with Decoupling Scores

The McNemar p-value from Analysis 5 serves as the **decoupling score** — lower p-values indicate stronger evidence that a model's actions diverge from its assigned Kohlberg stage predictions.

We compute:

1. **Spearman rank correlation** (ρ) — Primary metric; non-parametric, robust to non-linear monotonic relationships
2. **Pearson correlation** (r) — For completeness, assuming linearity
3. **Kendall's τ** — Robust rank-based alternative
4. **Coherence gap** = `NLI_coherence − (1 − p_value)` — Interpretable signed difference:
   - **Positive gap** → Reasoning is more internally coherent than Kohlberg consistency suggests
   - **Negative gap** → Reasoning is less coherent than Kohlberg consistency suggests

> **Why Correlation over Difference?**
> The user considered using a simple difference between scores. We implement both, but **Spearman correlation is the primary metric** because:
> (a) It captures monotonic relationships regardless of scale;
> (b) Simple difference conflates magnitude and direction;
> (c) Spearman is standard in behavioral sciences literature.
> The coherence gap is provided as a complementary interpretability tool.

---

## File Structure

```
nli_coherence/
├── main.py                  # Entry point — orchestrates the full pipeline
├── config.py                # Paths, model registry, API settings, plot styles
├── data_loader.py           # Loads evaluation data and McNemar p-values
├── nli_scorer.py            # NLI entailment scoring via HF Inference API
├── correlation_analysis.py  # Aggregation + Spearman/Pearson/Kendall tests
├── visualizations.py        # 6 publication-quality matplotlib figures
├── reporting.py             # CSV/JSON output + console summary
├── README.md                # This file
└── results/                 # Generated outputs (after running)
    ├── nli_scores_all.csv
    ├── coherence_by_model.csv
    ├── coherence_by_model_dilemma.csv
    ├── coherence_vs_decoupling.csv
    ├── correlation_results.json
    ├── fig1_coherence_by_model.png
    ├── fig2_coherence_heatmap.png
    ├── fig3_coherence_vs_decoupling.png
    ├── fig4_coherence_gap.png
    ├── fig5_entailment_distribution.png
    └── fig6_correlation_summary.png
```

---

## Usage

### Prerequisites

```bash
# From the repository root
pip install huggingface_hub pandas numpy scipy matplotlib openpyxl
```

### Running

```bash
# Set your HuggingFace API token
export HF_TOKEN=hf_...

# Run the analysis
python3 nli_coherence/main.py
```

### Expected Runtime

The analysis makes one API call per `(model, dilemma, trial)` observation. With ~14 models × 6 dilemmas × 3 trials = ~252 API calls, the pipeline takes approximately **5–10 minutes** depending on API response times and rate limits.

---

## Outputs

### CSV Files

| File | Description |
|------|-------------|
| `nli_scores_all.csv` | Per-observation entailment scores for every `(model, dilemma, trial)` |
| `coherence_by_model.csv` | Per-model aggregated coherence statistics (mean, median, std, min, max) |
| `coherence_by_model_dilemma.csv` | Per-model × per-dilemma mean coherence |
| `coherence_vs_decoupling.csv` | Merged view: NLI coherence + McNemar p-values + coherence gap |

### JSON

| File | Description |
|------|-------------|
| `correlation_results.json` | Full Spearman, Pearson, Kendall correlation test results with interpretations |

### Figures

| Figure | Description |
|--------|-------------|
| `fig1_coherence_by_model.png` | Bar chart: mean NLI entailment score per model (with error bars) |
| `fig2_coherence_heatmap.png` | Heatmap: model × dilemma coherence matrix |
| `fig3_coherence_vs_decoupling.png` | Scatter plot: NLI coherence vs. decoupling strength with Spearman ρ |
| `fig4_coherence_gap.png` | Signed bar chart: coherence gap (positive = more coherent than Kohlberg suggests) |
| `fig5_entailment_distribution.png` | Box plots: per-observation entailment score distributions by model |
| `fig6_correlation_summary.png` | Two-panel correlation summary (vs. p-value and vs. decoupling strength) |

---

## Interpretation Guide

### What the Scores Mean

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Entailment Score | [0, 1] | How strongly the reasoning logically implies the action |
| Decoupling Strength | [0, 1] | = 1 − p_value; how much the action diverges from Kohlberg predictions |
| Coherence Gap | [−1, 1] | Positive = reasoning is more coherent than Kohlberg would predict |

### Observed Results

The analysis of 234 (reasoning, action) pairs across 13 models found:

| Metric | Value | p-value |
|--------|-------|---------|
| **Spearman ρ** | −0.1132 | 0.7128 |
| **Pearson r** | −0.1718 | 0.5747 |
| **Kendall τ** | −0.1307 | 0.5729 |

**Key finding: No significant correlation** (ρ ≈ 0, p > 0.05). NLI coherence and Kohlberg-based consistency are **independent dimensions** of moral reasoning quality.

- All models achieve high NLI coherence (mean = 0.908), indicating that LLMs consistently produce reasoning that logically supports their chosen actions.
- The coherence gap is predominantly positive (mean = +0.15), meaning models are more internally coherent than Kohlberg-based consistency suggests.
- **Qwen3-32B** (0.974) and **Qwen3-80B** (0.966) show the highest coherence; **Llama 4 Scout** (0.795) and **DeepSeek-R1** (0.822) show the lowest.

### What This Means

1. **No correlation** → The two measures capture orthogonal aspects of moral reasoning quality
2. **High NLI coherence across all models** → LLMs are generally good at constructing reasoning that logically supports their chosen actions
3. **Positive coherence gap** → Even when models violate Kohlberg stage predictions, their reasoning-action link remains internally sound

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `huggingface_hub` | HuggingFace Inference API client |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scipy` | Spearman, Pearson, Kendall correlation tests |
| `matplotlib` | Publication-quality figures |
| `openpyxl` | Reading `.xlsx` evaluation files |

---

## Relationship to Other Analyses

| Analysis | Focus | Framework |
|----------|-------|-----------|
| Analysis 5 | Action-reasoning consistency via McNemar's test | Kohlberg-dependent |
| **NLI Coherence** | Justification-action logical entailment | **Framework-agnostic** |

This analysis provides a complementary lens: while Analysis 5 asks *"Does the model act as its Kohlberg stage predicts?"*, NLI Coherence asks *"Does the model's own reasoning support its action?"*. The correlation between these two measures reveals whether stage-based consistency and logical coherence are aligned or orthogonal dimensions.
