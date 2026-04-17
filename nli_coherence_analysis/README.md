# NLI-Based Coherence Measure — DeBERTa-v3-large

> **Framework-independent validation of reasoning–action consistency** using Natural Language Inference (NLI).

---

## Overview

This module provides a **Kohlberg-framework-independent** measure of whether LLM moral reasoning is genuinely coherent — i.e., whether the stated justification logically supports the endorsed action. It uses `cross-encoder/nli-deberta-v3-large` (fine-tuned on MNLI + SNLI) to score:

$$P(\text{entailment} \mid \text{justification} \to \text{action})$$

This NLI coherence score is then **correlated with the existing Kohlberg-based decoupling metric** from Analysis 5, providing convergent validation (or identifying where the two diverge).

### Why This Matters

Analysis 5 found that some models exhibit **moral decoupling** — using Stage 5–6 vocabulary while choosing low-stage actions. However, that consistency metric depends entirely on the Kohlberg framework (mapping stages to expected actions). This NLI measure asks a more fundamental question: *"Does the model's reasoning actually support its conclusion?"* — without assuming any moral framework.

---

## Background

| Metric | Framework | Measures | Source |
|---|---|---|---|
| Kohlberg Consistency % | Stage-dependent | Whether stage-expected action matches actual | Analysis 5 |
| **NLI Entailment Score** | **Framework-free** | **Whether justification logically entails action** | **This module** |

If both metrics agree, we have strong evidence that reasoning–action coherence is real. If they diverge, NLI may reveal coherence (or incoherence) that the stage-based metric misses — e.g., a model that gives thoughtful reasoning for Rule-Breaking at Stage 4 would be "inconsistent" by Kohlberg but "coherent" by NLI.

---

## NLI Model

| Property | Value |
|---|---|
| Model | `cross-encoder/nli-deberta-v3-large` |
| Base | Microsoft DeBERTa-v3-large (~435M params) |
| Training | Fine-tuned on MNLI + SNLI |
| Output | P(contradiction), P(entailment), P(neutral) |
| Max length | 512 tokens |

### Why DeBERTa-v3-large?

1. **State-of-the-art NLI performance** — highest accuracy on MNLI dev-matched among publicly available models of comparable size
2. **Cross-encoder architecture** — processes premise and hypothesis jointly (more accurate than bi-encoder approaches)
3. **Disentangled attention** — DeBERTa's enhanced attention mechanism handles long reasoning texts well
4. **Runs on consumer hardware** — ~2GB VRAM, or CPU-only (~5-10 sec/sample)

---

## Pipeline

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Load Evaluation  │────▶│ NLI Scoring       │────▶│ Correlation      │
│ Data (RLHF or   │     │ (DeBERTa-v3)      │     │ Analysis         │
│ Main Project)    │     │                    │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
                              ┌────────────────────────────┤
                              ▼                            ▼
                     ┌──────────────────┐     ┌──────────────────┐
                     │ Visualizations   │     │ Report           │
                     │ (5 figures)      │     │ (Markdown)       │
                     └──────────────────┘     └──────────────────┘
```

### Step 1: Load Data
- **Primary**: `rlhf_causal_analysis/evaluation/` (base vs. instruct pairs)
- **Fallback**: `evaluation_data/` (13 models from the main project)

### Step 2: NLI Scoring
For each observation:
1. **Extract justification** — moral reasoning sentences from the full LLM response
2. **Build hypothesis** — convert `action_endorsed` into a natural-language sentence using dilemma-specific templates
3. **Score** — `P(entailment | justification → hypothesis)` via DeBERTa

### Step 3: Correlation Analysis
1. **Point-biserial** — NLI entailment vs. binary `is_consistent` (observation-level)
2. **Model-level Pearson/Spearman** — mean NLI vs. consistency % per model
3. **RLHF pair comparison** — base vs. instruct NLI coherence
4. **Partial correlation** — NLI vs. consistency, controlling for Kohlberg stage
5. **Bootstrap 95% CI** on all key correlations

---

## Folder Structure

```
nli_coherence_analysis/
├── README.md                   ← This file
├── config.py                   ← Paths, DeBERTa config, hypothesis templates
├── nli_scorer.py               ← NLI scoring engine + justification extraction
├── data_loader.py              ← Load evaluation data + compute decoupling flags
├── correlation.py              ← Correlation analysis (NLI ↔ decoupling)
├── visualizations.py           ← 5 publication-quality figures
├── reporting.py                ← Markdown report generator
├── main.py                     ← Pipeline orchestrator
├── scores/                     ← [Generated] NLI score output (xlsx)
└── results/                    ← [Generated] Figures (PNG) + report.md
```

---

## Usage

All commands are run from the **project root** directory.

### Full Pipeline

```bash
# Using RLHF data (if collected) — falls back to main data
python nli_coherence_analysis/main.py

# Force use of main project data (13 models)
python nli_coherence_analysis/main.py --use-main-data
```

### Individual Stages

```bash
# NLI scoring only
python nli_coherence_analysis/main.py --score-only
python nli_coherence_analysis/main.py --score-only --use-main-data

# Analysis only (requires pre-computed scores)
python nli_coherence_analysis/main.py --analyze
python nli_coherence_analysis/main.py --analyze --use-main-data
```

---

## Outputs

After a full run, `nli_coherence_analysis/results/` contains:

| File | Description |
|---|---|
| `fig1_nli_score_distribution.png` | Violin + histogram of NLI scores, split by consistency status |
| `fig2_model_scatter.png` | Mean NLI entailment vs. consistency % with regression line |
| `fig3_dilemma_heatmap.png` | Mean NLI entailment across models × dilemmas |
| `fig4_base_vs_instruct_nli.png` | Paired box plots for base vs. instruct (RLHF only) |
| `fig5_correlation_matrix.png` | Annotated correlation heatmap of all measures |
| `report.md` | Full statistical report with tables and interpretation |

Scores are saved to `nli_coherence_analysis/scores/nli_scores_{source}.xlsx`.

---

## Dependencies

All core dependencies are already in the project `requirements.txt`:

```
torch
transformers
scipy
numpy
pandas
matplotlib
openpyxl
tqdm
```

The DeBERTa model will be downloaded automatically from HuggingFace Hub on first run (~1.5 GB download).

---

## Relationship to Other Analyses

| Analysis | Description | Relation |
|---|---|---|
| Analysis 5 | Reasoning-Action Alignment (Kohlberg-based) | Provides the decoupling metric we correlate against |
| RLHF Causal Analysis | Base vs. instruct model comparison | Provides the paired data for RLHF coherence analysis |
| **This module** | **NLI-based coherence (framework-free)** | **Validates and extends decoupling findings** |

---

## Technical Notes

- The `cross-encoder/nli-deberta-v3-large` model outputs logits in order `[contradiction, entailment, neutral]`. We apply softmax to get probabilities.
- Justification extraction uses heuristic moral-keyword filtering to focus on reasoning sentences (not dilemma repetition or preamble).
- Action hypotheses are constructed from dilemma-specific templates to ensure semantic clarity for the NLI model.
- Partial correlation residualises both NLI and consistency on Kohlberg stage to test whether NLI captures information beyond what the stage label provides.
- All figures use the Okabe-Ito colour-blind-safe palette at 300 DPI.
