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
├── analysis1/                   # Analysis 1: Scale vs. Moral Reasoning Stage
│
├── analysis2/                   # Analysis 2: Prompt Engineering Impact
│
├── analysis3/                   # Analysis 3: Consistency & Stability
│
├── analysis4/                   # Analysis 4: Stage Distribution Patterns
│
├── analysis5/                   # Analysis 5: Action-Reasoning Consistency
│
├── analysis6/                   # Analysis 6: Reasoning Pattern Analysis
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

### 4. Run the Statistical Analyses

Each analysis module operates independently. Change into the desired directory and run its `main.py` entry point.

```bash
cd analysis1/
python main.py
# Outputs written to analysis1/results/
```

---

## Analysis Modules

The project is divided into six main analysis modules, each answering a specific research question about how LLMs reason about morality:

### Analysis 1 — Scale vs. Moral Reasoning

**Research Question:** Do larger models show higher Kohlberg moral reasoning stages?

- **Methods:** Spearman correlation ($\rho$) between parameter count and mean stage, bootstrap confidence intervals, Kruskal-Wallis tests.
- **Key finding:** Moderate positive correlation, but diminishing returns for the largest models.

### Analysis 2 — Prompt Engineering Impact

**Research Question:** Does Chain-of-Thought (CoT) or Roleplay prompting improve moral reasoning compared to Zero-Shot out-of-the-box performance?

- **Methods:** Repeated-measures ANOVA (Friedman test), Wilcoxon signed-rank post-hoc, magnitude of change analysis.
- **Key finding:** Prompt engineering has negligible effect on the fundamental moral reasoning stage of state-of-the-art models.

### Analysis 3 — Consistency & Stability

**Research Question:** Do models show stable moral reasoning across divergent dilemmas and prompt contexts?

- **Methods:** Intraclass Correlation Coefficient (ICC), within-model standard deviation vs. human baseline variance.
- **Key finding:** Models exhibit hyper-consistent (ICC > 0.90) reasoning profiles, lacking the context-dependent variance seen in human populations.

### Analysis 4 — Stage Distribution Patterns

**Research Question:** Do models mirror human stage distributions or exhibit synthetic patterns (e.g., ceiling effects)?

- **Methods:** Chi-square goodness-of-fit vs. human adult norms, Jensen-Shannon Divergence (JSD), entropy and kurtosis.
- **Key finding:** Most models diverge significantly from human adults, showing either ceiling-biased (all Stage 5/6) or human-like patterns depending on RLHF methodology.

### Analysis 5 — Action-Reasoning Consistency

**Research Question:** Do models' moral reasoning stages (Kohlberg) align coherently with the ethical actions they endorse?

- **Methods:** Rule-following vs Principled rule-breaking action extraction, Stage × Action cross-tabulations, Chi-Square Independence.
- **Key finding:** Strong statistical dependency between reasoning stage and endorsed action, though some models display "moral decoupling", adopting post-conventional actions with conventional reasoning.

### Analysis 6 — Reasoning Pattern Analysis

**Research Question:** What qualitative reasoning patterns characterize each model's moral reasoning?

- **Methods:** TF-IDF keyword extraction, Dimensionality reduction (PCA), Vocabulary richness mapping, Qualitative Centroid exemplars.
- **Key finding:** Model families share distinct linguistic "voices" and aligned models tend to manifest significantly richer moral vocabulary independent of parameter scale.

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
