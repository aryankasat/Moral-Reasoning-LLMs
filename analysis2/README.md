# Analysis 2 — Alignment Training vs. Moral Reasoning

> **Research Question**: Does RLHF / RL-aligned training produce higher moral reasoning stages than instruction-tuning alone, and is this effect consistent across model families?

---

## Motivation

Analysis 1 showed that larger models score higher on Kohlberg's scale. But is this due to **scale** or **training procedure**? Some models undergo basic supervised fine-tuning on instructions (Instruction-Tuned, IT), while others receive Reinforcement Learning from Human Feedback or Constitutional AI training (RLHF / RL-Aligned). This analysis decomposes the alignment contribution by comparing IT vs. RLHF models **within the same model families**, controlling for architectural differences.

---

## Alignment Taxonomy

| Type | Description | Models |
|---|---|---|
| **Instruction-Tuned (IT)** | Supervised fine-tuning on instruction datasets; no RL signal | Mistral Tiny 7B, Ministral 8B, Llama 3.3 70B, Llama 4 Scout, Qwen3-30B Coder, Qwen3-32B, Qwen3-80B, DeepSeek-V3.1, GPT-OSS 120B |
| **RLHF / RL-Aligned** | RLHF, Constitutional AI, or RL-based reasoning training | Qwen3-235B (Think), DeepSeek-R1, GPT-4o, Claude 3.5 Haiku, Claude Sonnet 4.5 |

---

## Experimental Design

### Within-Family Comparison Pairs

Each pair controls for architecture/provider while varying alignment procedure:

| Pair | Model A (less-aligned) | Model B (more-aligned) | What varies |
|---|---|---|---|
| DeepSeek IT → RL | DeepSeek-V3.1 (IT) | DeepSeek-R1 (RL) | Training method |
| Mistral 7B → 8B | Mistral Tiny 7B (IT) | Ministral 8B (IT) | Generation step |
| Llama Gen3 → Gen4 | Llama 3.3 70B (IT) | Llama 4 Scout (IT) | Generation step |
| OpenAI IT → RLHF | GPT-OSS 120B (IT) | GPT-4o (RLHF) | Training method |
| Claude Haiku → Sonnet | Claude 3.5 Haiku (RLHF) | Claude Sonnet 4.5 (RLHF) | Capability tier |
| Qwen3 IT → Think-RL | Qwen3-32B (IT) | Qwen3-235B Think (RL) | Scale + training |

---

## Statistical Methods

### 1. Descriptive Statistics
- Per-model: mean stage, median, SD, 95% bootstrap CI, % post-conventional (Stage ≥ 5)
- Per-alignment-group: aggregated statistics across IT vs. RLHF pools

### 2. Within-Family Pairwise Tests (`wilcoxon_effect_size`)
- **Mann-Whitney U** (Wilcoxon rank-sum): non-parametric two-sided test for stage distribution differences
- **Cohen's d**: pooled-SD standardised effect size (negligible < 0.2, small < 0.5, medium < 0.8, large ≥ 0.8)
- **Rank-biserial correlation**: r = 1 − 2U/(n₁n₂), measures probability of dominance
- **Bootstrap 95% CI** on mean stage difference (Δ = RLHF − IT), 5,000 iterations

### 3. Overall IT vs. RLHF Test (`run_overall_alignment_test`)
- Pooled Mann-Whitney U across all IT observations vs. all RLHF observations
- Tests whether alignment type has a global effect independent of family

---

## Code Architecture

```
analysis2/
├── main.py              ← Entry point
├── config.py            ← MODEL_META with alignment type, FAMILY_PAIRS, colour palette
├── data_loader.py       ← Loads evaluation data + merges alignment metadata
├── stat_analysis.py     ← Mann-Whitney U, Cohen's d, rank-biserial, bootstrap CIs
├── visualizations.py    ← 4 publication-quality figures
├── reporting.py         ← CSV export + console summary
└── results/             ← Generated outputs
```

---

## Outputs

### Figures

| File | Description |
|---|---|
| `fig1_violin_by_alignment.png` | Violin plots comparing IT vs. RLHF stage distributions |
| `fig2_family_comparisons.png` | Within-family paired comparisons with effect sizes |
| `fig3_stacked_stage_dist.png` | Stacked bar chart of stage proportions by alignment group |
| `fig4_pct_postconv.png` | Post-conventional (Stage 5+) percentage per model |

### CSV Reports

| File | Contents |
|---|---|
| `model_stats.csv` | Per-model descriptive statistics with alignment type |
| `alignment_group_stats.csv` | Aggregated IT vs. RLHF group statistics |
| `family_comparisons.csv` | Pairwise test results: U, p, Cohen's d, rank-biserial, ΔCI |
| `overall_alignment_test.csv` | Pooled IT vs. RLHF Mann-Whitney result |

---

## Key Findings

- Prompting style has **negligible effect** on fundamental moral stage
- RLHF-aligned models show **conditionally higher** moral reasoning — but the effect is **entangled with scale** (the largest RLHF models are also the largest overall)
- Within-family comparisons reveal that alignment training makes a measurable difference, particularly the DeepSeek IT→RL and OpenAI IT→RLHF pairs

---

## Usage

```bash
python analysis2/main.py
# Outputs → analysis2/results/
```

---

## Caveats

1. **Confound warning**: Many RLHF models are also larger than their IT counterparts. Analysis 8 formally decomposes this confound via two-way ANOVA.
2. **Alignment taxonomy is simplified**: Real alignment pipelines involve multiple stages (SFT → RLHF → DPO). The IT/RLHF binary is a first approximation.
3. **Within-family pairs are not perfectly matched**: Some pairs differ in generation (Llama 3.3 vs. 4) or scale (Qwen3-32B vs. 235B), not just training.
