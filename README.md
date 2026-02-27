# Moral-Reasoning-LLMs

Research repository for evaluating and comparing large language models' (LLMs) moral reasoning using Kohlberg's stages of moral development. The project covers end-to-end pipelines: prompt generation → LLM inference → Kohlberg-stage evaluation → statistical analysis.

---

## Overview

- **Research question:** Do larger LLMs exhibit higher stages of moral reasoning?
- **Evaluation framework:** Kohlberg's six stages of moral development, applied to structured moral dilemmas.
- **Models covered:** Claude 3.5 Haiku, Claude Sonnet 4.5, GPT-4o, GPT-OSS 120B, Llama 3.3 70B, Llama 4 Scout, Mistral Tiny, Ministral 8B, Qwen3 (32B / 80B / 235B), DeepSeek-R1, DeepSeek-V3.1.

---

## Repository Structure

```
Moral-Reasoning-LLMs/
│
├── LLM calls/                   # Model-specific inference wrappers
│   ├── groq_llm.py
│   ├── mistral_llm.py
│   ├── pythia_llm.py
│   └── puter_llm.py
│
├── data/                        # Raw LLM responses (one .xlsx per model)
│   └── <model_name>.xlsx        # Columns: model_name, dilemma_type, prompt_type,
│                                #   timestamp, response, response_length,
│                                #   inference_time, api_source, temperature
│
├── evaluation_data/             # Kohlberg stage labels (one .xlsx per model)
│   ├── <model_name>_evaluation.xlsx   # Columns: analysis_timestamp, dilemma_type,
│   │                                  #   response, kohlberg_stage, kohlberg_confidence,
│   │                                  #   kohlberg_reasoning, secondary_stage, …
│   ├── puter_evaluation_llm.py  # Auto-evaluation script
│   └── update_excel.py          # Helper to patch evaluation files
│
├── analysis1/                   # Scale vs. moral reasoning analysis (modular)
│   ├── config.py                # Paths, MODEL_META registry, colour palette, rcParams
│   ├── data_loader.py           # Loads & joins data + evaluation xlsx files
│   ├── stat_analysis.py         # Per-model stats, bootstrap CI, Spearman ρ, KW+Dunn
│   ├── visualizations.py        # 4 publication-quality figures (300 dpi)
│   ├── reporting.py             # CSV export + structured console report
│   ├── main.py                  # Entry point — run this
│   └── results/                 # Auto-generated outputs (figures + CSVs)
│       ├── fig1_box_stage_by_model.png
│       ├── fig2_scatter_scale_vs_stage.png
│       ├── fig3_heatmap_stage_distribution.png
│       ├── fig4_bar_mean_stage.png
│       ├── model_stats.csv
│       ├── spearman_correlation.csv
│       └── dunn_posthoc_pvalues.csv
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Set up environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install scikit-posthocs   # required for Dunn post-hoc test
```

### 2. Generate LLM responses

```bash
python "LLM calls/puter_llm.py"
# Outputs saved to data/<model_name>.xlsx
```

### 3. Run Kohlberg-stage evaluation

```bash
python evaluation_data/puter_evaluation_llm.py
# Outputs saved to evaluation_data/<model_name>_evaluation.xlsx
```

### 4. Run the scale vs. moral reasoning analysis

```bash
cd analysis1/
python main.py
# Outputs written to analysis1/results/
```

---

## Analysis: Scale vs. Moral Reasoning

The `analysis1/` module answers:

> **Do larger models show higher Kohlberg moral reasoning stages?**

### Methodology

| Step | Detail |
|---|---|
| **Stage labelling** | Each response is classified into Kohlberg stages 1–6 |
| **Per-model stats** | Mean, median, mode, SD, stage distribution % |
| **Bootstrap CIs** | 95% CIs on mean stage (5,000 resamples) |
| **Correlation** | Spearman ρ between log₁₀(params) and mean stage |
| **Hypothesis tests** | Kruskal-Wallis H + Dunn pairwise (Bonferroni correction) |

### Key Findings (234 observations, 13 models)

| Metric | Value |
|---|---|
| Spearman ρ | **+0.405** (medium effect) |
| 95% CI | [−0.17, 0.80] |
| p-value | 0.170 (not sig. at α = 0.05 with N = 13 models) |
| R² | 16.4% variance explained |
| Kruskal-Wallis H | 54.86, **p < 0.001** |

All models concentrated at Stage 5–6; **Qwen3-235B (Think)** is the only model achieving 100% Stage 6.

### Output Figures

| Figure | Description |
|---|---|
| `fig1_box_stage_by_model.png` | Stage distribution box plots per model |
| `fig2_scatter_scale_vs_stage.png` | log-scale vs. mean stage scatter + OLS trend |
| `fig3_heatmap_stage_distribution.png` | Stage % heat-map across all models |
| `fig4_bar_mean_stage.png` | Mean stage bar chart with 95% bootstrap CIs |

---

## Environment Variables

Add API keys to your `.env` file before running model wrappers:

```
GROQ_API_KEY=...
MISTRAL_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

---

## Notes

- `puter_session/` and `puter_user_data/` are local browser-session cache folders — excluded from commits via `.gitignore`.
- Parameter counts in `analysis1/config.py → MODEL_META` are estimates for closed models (Claude, GPT-4o). Update them if you have more precise figures; the Spearman correlation is sensitive to these values.
- All analysis outputs in `analysis1/results/` are reproducible by running `main.py` from scratch.