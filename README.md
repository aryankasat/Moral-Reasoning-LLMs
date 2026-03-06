# 🧠 Moral-Reasoning-LLMs

> **Can AI think morally — and does bigger mean wiser?**

This research project investigates whether large language models (LLMs) — the AI systems behind tools like ChatGPT, Claude, and Llama — can reason about ethics in meaningful ways. We use **Kohlberg's Stages of Moral Development**, a well-established framework from developmental psychology, to score how sophisticated an AI's moral reasoning is across 6 moral dilemmas and 3 prompting styles.

**Core finding:** Scale is the primary driver of moral reasoning stage. Larger models consistently score higher — but the relationship is nuanced, reasoning is task-rigid, and training procedure shows conditional effects only at the largest scales.

---

## 📌 What Is This Project?

We asked 13 different LLMs to respond to classic moral dilemmas — stories with no easy right or wrong answer:

| Dilemma | Core Tension |
|---|---|
| **Trolley Problem** | Sacrifice 1 to save 5? |
| **Heinz's Dilemma** | Steal medicine to save a dying spouse? |
| **The Lifeboat** | Who survives when the boat is too full? |
| **Judith Jarvis Thomson's Violinist** | Bodily autonomy vs. life? |
| **The Whistleblower** | Loyalty vs. truth? |
| **The Bystander** | Personal risk vs. duty to help? |

Each model answered under **three prompting styles**:
- **Zero-Shot** — Just answer.
- **Chain-of-Thought** — Think step by step before answering.
- **Roleplay** — Answer as a moral philosopher.

An AI evaluator then scored every response on Kohlberg's 1–6 scale, where higher = more sophisticated ethical reasoning (Stage 6 = universal moral principles).

---

## 🤖 Models Tested

| Model | Provider | Params (B) | Scale Group | Training Type |
|---|---|---|---|---|
| Ministral 8B | Mistral AI | 8 | Small | Base-RLHF |
| Claude 3.5 Haiku | Anthropic | 20 | Small | Base-RLHF |
| Qwen3-30B Coder | Alibaba | 30 | Small | Coding-Tuned |
| Qwen3-32B | Alibaba | 32 | Small | Base-RLHF |
| Llama 3.3 70B | Meta | 70 | Mid | Base-RLHF |
| Qwen3-80B | Alibaba | 80 | Mid | Base-RLHF |
| Llama 4 Scout 109B | Meta | 109 | Mid | Base-RLHF |
| GPT-OSS 120B | OpenAI | 120 | Mid | Base-RLHF |
| Claude Sonnet 4.5 | Anthropic | 175 | Large | Base-RLHF |
| GPT-4o | OpenAI | 200 | Large | Base-RLHF |
| Qwen3-235B (Think) | Alibaba | 235 | Large | Reasoning-Tuned |
| DeepSeek-R1 671B | DeepSeek | 671 | Large | Reasoning-Tuned |
| DeepSeek-V3.1 671B | DeepSeek | 671 | Large | Base-RLHF |

---

## 🗂️ Repository Structure

```
Moral-Reasoning-LLMs/
│
├── LLM calls/              ← Scripts that send dilemmas to each AI model
│   ├── prompt_hub.py       ← All 6 moral dilemmas + 3 prompt styles defined here
│   ├── groq_llm.py         ← Calls models via Groq API (Llama, Mistral, etc.)
│   ├── mistral_llm.py      ← Calls Mistral models
│   ├── puter_llm.py        ← Calls models via Puter (browser-based)
│   └── pythia_llm.py       ← Calls open-source Pythia models
│
├── data/                   ← Raw AI responses (one file per model)
│   └── <model_name>.xlsx   ← dilemma type, prompt style, response, timing…
│
├── evaluation_data/        ← AI-scored Kohlberg ratings of each response
│   ├── <model>_evaluation.xlsx  ← kohlberg_stage, confidence, reasoning…
│   ├── puter_evaluation_llm.py  ← Runs the Kohlberg scoring
│   └── update_excel.py          ← Helper to patch evaluation files
│
├── analysis1/              ← Scale vs. Moral Stage (Spearman correlation)
├── analysis2/              ← Prompt Style Effects (Friedman + Wilcoxon)
├── analysis3/              ← Within-Model Consistency (ICC)
├── analysis4/              ← AI vs. Human Stage Distributions
├── analysis5/              ← Reasoning-Action Alignment
├── analysis6/              ← Linguistic Patterns (TF-IDF + PCA)
├── analysis7/              ← [Extended analysis — response quality metrics]
├── analysis8/              ← Scale vs. Training Decomposition (Two-way ANOVA)
│
├── requirements.txt        ← Python dependencies
└── README.md               ← You are here
```

Each `analysis*/` folder follows a consistent structure:

| File | Purpose |
|---|---|
| `main.py` | Run this to execute the full analysis |
| `config.py` | Settings: model list, parameter counts, group definitions |
| `data_loader.py` | Loads and cleans data from `evaluation_data/` |
| `stat_analysis.py` | Runs all statistical tests |
| `visualizations.py` | Generates charts (300 DPI, publication-quality) |
| `reporting.py` | Writes the Markdown results report |
| `results/` | Output folder: figures (`.png`) + report (`.md`) |

---

## 🔬 The Eight Research Questions

### 📊 Analysis 1 — Do Bigger Models Reason More Morally?
*Does model size (parameter count) predict a higher Kohlberg stage?*

**Method:** Spearman rank correlation between log-parameter count and mean moral stage.  
**Finding:** Moderate positive correlation — bigger models generally score higher, but with diminishing returns past ~70B.

📂 `analysis1/results/`

---

### 💬 Analysis 2 — Does Prompting Style Matter?
*Does asking the AI to "think step by step" or "roleplay as a philosopher" change its moral stage?*

**Method:** Repeated-measures Friedman test + Wilcoxon post-hoc across 3 prompt types.  
**Finding:** Prompting has **negligible effect** on the fundamental moral stage of frontier models. Reasoning is baked in, not prompted out.

📂 `analysis2/results/`

---

### 🔁 Analysis 3 — Are Models Consistent Across Dilemmas?
*Does the same model give different moral reasoning for different dilemmas?*

**Method:** Intraclass Correlation Coefficient (ICC) per model across 6 dilemmas.  
**Finding:** Models are **hyper-consistent** (ICC > 0.90) — almost robotically so. Human moral reasoning varies by context; AI moral reasoning largely doesn't.

📂 `analysis3/results/`

---

### 📉 Analysis 4 — Do AI Models Reason Like Humans?
*Do AI stage distributions resemble how human adults are distributed?*

**Method:** Chi-squared goodness-of-fit against human developmental norms; Jensen-Shannon divergence.  
**Finding:** Most models cluster at Stage 5/6 (ceiling effect) or show non-human distributions. A few RLHF-tuned frontier models converge on human-like patterns.

📂 `analysis4/results/`

---

### ⚖️ Analysis 5 — Do Models Practice What They Preach?
*When a model reasons at Stage 5, does it actually choose a principled action?*

**Method:** Action-reasoning cross-tabulation; Cramér's V for association strength.  
**Finding:** Strong statistical alignment overall — but some models show **moral decoupling**: high-stage vocabulary with low-stage action choices.

📂 `analysis5/results/`

---

### 🔤 Analysis 6 — What Do the Words Reveal?
*Are there patterns in the language different models use when reasoning morally?*

**Method:** TF-IDF keyword extraction per model; PCA dimensionality reduction; stage-wise word clouds.  
**Finding:** Model families share distinct "linguistic voices." Aligned/RLHF models demonstrate richer moral vocabulary regardless of size.

📂 `analysis6/results/`

---

### 🔍 Analysis 7 — Response Quality Metrics
*How do models differ in response depth, confidence, and evaluator agreement?*

**Method:** Response length analysis, evaluator confidence distribution, secondary stage agreement metrics.  
**Finding:** Reasoning-tuned models produce longer, more nuanced responses; evaluator confidence is highest for Stage 5/6 responses.

📂 `analysis7/results/`

---

### ⚗️ Analysis 8 — Scale vs. Training: What Drives Moral Reasoning?
*Does scale affect moral reasoning independent of training type, or is training the primary driver?*

**Method:** Two-way factorial ANOVA (Scale × Training Type) on 234 observations across 13 models. Supplemented by Kruskal-Wallis, Welch ANOVA, Mann-Whitney U (non-parametric), and Cohen's d for convergent validation.

**Factorial design:**

| Factor | Levels |
|---|---|
| **Scale Group** | Small (8–32B), Mid (70–120B), Large (175–671B) |
| **Training Type** | Base-RLHF, Coding-Tuned, Reasoning-Tuned |

**Key results:**

| Test | Effect | Statistic | p-value |
|---|---|---|---|
| Sequential ANOVA | Scale | F(2,229)=6.05 | **0.003** ✅ |
| Sequential ANOVA | Training Type | F(2,229)=2.76 | 0.065 |
| Kruskal-Wallis (non-parametric) | Scale | H=12.78 | **0.002** ✅ |
| Welch ANOVA (variance-robust) | Scale | F=6.26 | **0.002** ✅ |
| Mann-Whitney U (Bonferroni) | Large vs. Small | — | **0.001** ✅ |

**Supported hypothesis: H2 — Scale Dominates.**  
Moral reasoning capacity is primarily parameter-count limited. Training procedure shows conditional effects only among large-scale models (Reasoning-Tuned > Base-RLHF within the Large group, Tukey p=.039), but does not have a significant independent main effect once scale is partialled out.

**Effect sizes:** Scale partial η²=0.050, ω²=0.041, Cohen's d=0.55 (Large vs. Small).

📂 `analysis8/results/` — includes `summary_panel.png` (journal-ready 4-panel figure), `interaction_plot.png`, `variance_bars.png`, `box_violin.png`, `posthoc_matrix.png`, `bar_with_jitter.png`

---

## 🚀 How to Run

> Requires Python 3.9+.

### Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install statsmodels scikit-posthocs   # needed for analysis8 and analysis2
```

### Add API keys (optional — only to collect new data)

```env
# .env
GROQ_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

> ⚠️ `data/` and `evaluation_data/` already contain pre-collected responses and Kohlberg scores. API keys are only needed if you want to add new models.

### Run any analysis

Each analysis is fully independent:

```bash
# Run Analysis 8 (Scale vs. Training Decomposition)
source venv/bin/activate
python3 analysis8/main.py
# → Outputs appear in analysis8/results/

# Run any other analysis the same way
python3 analysis1/main.py   # Scale correlation
python3 analysis2/main.py   # Prompt style effects
# ... etc.
```

### (Optional) Collect new LLM responses

```bash
python "LLM calls/puter_llm.py"
# → Saves to data/<model_name>.xlsx

python evaluation_data/puter_evaluation_llm.py
# → Saves to evaluation_data/<model_name>_evaluation.xlsx
```

---

## 📁 Data Schema

### `data/` — Raw model responses

| Column | Description |
|---|---|
| `model_name` | Which AI model responded |
| `dilemma_type` | Which moral dilemma |
| `prompt_type` | Zero-Shot / Chain-of-Thought / Roleplay |
| `response` | Full model-generated text |
| `response_length` | Word count |
| `inference_time` | Response latency |
| `api_source` | API used |
| `temperature` | Randomness setting |

### `evaluation_data/` — Kohlberg stage scores

| Column | Description |
|---|---|
| `kohlberg_stage` | Stage 1–6 assigned by AI evaluator |
| `kohlberg_confidence` | Evaluator confidence score |
| `kohlberg_reasoning` | Explanation for assigned stage |
| `secondary_stage` | Second-best stage if borderline |

---

## 🧩 Kohlberg's Stages of Moral Development

| Stage | Level | Core Idea |
|---|---|---|
| 1 | Pre-conventional | Avoid punishment |
| 2 | Pre-conventional | Do what benefits you |
| 3 | Conventional | Be a good person / fit in |
| 4 | Conventional | Follow rules and laws |
| 5 | Post-conventional | Respect social contracts and rights |
| 6 | Post-conventional | Follow universal ethical principles |

Most human adults reason at Stages 3–5. The **post-conventional threshold** (Stage 5+) is the key benchmarking target in this project.

---

## ⚙️ Technical Notes

- All `results/` outputs are fully reproducible by running `main.py` — no manual steps.
- Parameter counts for closed-source models (Claude, GPT-4o) are estimates. Update `config.py` in the relevant analysis folder if better numbers are available.
- Analysis 8 uses **sequential (Type-I) SS** for the two-way ANOVA to handle the incomplete factorial design (5 of 9 cells populated). The interaction df is derived from actual model rank differences rather than the nominal k₁×k₂ formula — see `analysis8/stat_analysis.py` for full documentation.
- `puter_session/` and `puter_user_data/` are local browser-cache folders excluded via `.gitignore`.

---

## 📄 License

[MIT License](LICENSE) — free to use, cite, or build upon.
