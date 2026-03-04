# 🧠 Moral-Reasoning-LLMs

> **Can AI think morally — and does bigger mean wiser?**

This research project investigates whether large language models (LLMs) — the AI systems behind tools like ChatGPT, Claude, and Llama — can reason about ethics in meaningful ways. We use a well-known framework from psychology, **Kohlberg's Stages of Moral Development**, to score how sophisticated an AI's moral reasoning is.

**Short answer from our findings:** Larger models tend to reason at higher moral stages, but the relationship is nuanced — and AI reasoning is far more rigid than human reasoning.

---

## 📌 What Is This Project?

We asked 14 different AI models to respond to classic moral dilemmas — stories with no easy right or wrong answer, like:

- *The Trolley Problem* — Pull a lever to save 5 people but kill 1?
- *Heinz's Dilemma* — Should a husband steal medicine to save his dying wife?
- *The Lifeboat* — Who should be sacrificed when a boat is too full?

Each model was given these dilemmas under **three different question styles** (prompt types):
- **Zero-Shot** — Just answer the question.
- **Chain-of-Thought** — Think step by step before answering.
- **Roleplay** — Answer as a moral philosopher.

An AI evaluator then scored each response on Kohlberg's 1–6 scale, where higher scores indicate more sophisticated ethical reasoning (e.g., Stage 6 = reasoning from universal moral principles).

---

## 🤖 Models Tested

| Model Family | Models |
|---|---|
| **Anthropic** | Claude 3.5 Haiku, Claude Sonnet 4.5 |
| **OpenAI** | GPT-4o, GPT-OSS 120B |
| **Meta** | Llama 3.3 70B, Llama 4 Scout |
| **Mistral AI** | Mistral Tiny, Ministral 8B |
| **Alibaba** | Qwen3 32B, Qwen3 80B, Qwen3 235B |
| **DeepSeek** | DeepSeek-R1, DeepSeek-V3.1 |
| **Qwen** | Qwen3 30B Coder Instruct |

---

## 🗂️ How the Repo Is Organized

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
│   └── <model_name>.xlsx   ← Columns: dilemma type, prompt style, response, timing…
│
├── evaluation_data/        ← AI-scored Kohlberg ratings of each response
│   ├── <model>_evaluation.xlsx  ← Columns: Kohlberg stage (1–6), confidence, reasoning…
│   ├── puter_evaluation_llm.py  ← Script that runs the Kohlberg scoring
│   └── update_excel.py          ← Helper to fix/patch evaluation files
│
├── analysis1/              ← Does model size predict moral reasoning? (Spearman correlation)
├── analysis2/              ← Does prompt style/engineering change moral reasoning?
├── analysis3/              ← How consistent/stable is each model across dilemmas?
├── analysis4/              ← Do AI stage distributions look like humans or not?
├── analysis5/              ← Do AI actions match their stated moral reasoning?
├── analysis6/              ← What language patterns reveal about moral thinking?
│
├── requirements.txt        ← Python packages needed to run the project
└── README.md               ← You are here
```

Each `analysis*/` folder contains the same consistent structure:

| File | Purpose |
|---|---|
| `main.py` | Run this to execute the analysis |
| `config.py` | Settings: model list, parameter counts, etc. |
| `data_loader.py` | Loads and cleans data from `evaluation_data/` |
| `stat_analysis.py` | Runs the statistical tests |
| `visualizations.py` | Generates charts and graphs |
| `reporting.py` | Formats the final summary |
| `results/` | Output folder: charts (`.png`) and tables (`.csv`) |

---

## 🔬 The Six Research Questions — Plain English

### 📊 Analysis 1 — Do Bigger AI Models Reason More Morally?
*Does the number of parameters (model "size") predict a higher Kohlberg stage?*

We ran a statistical correlation between model size and average moral reasoning score. We found a moderate positive link — bigger models generally score higher — but with diminishing returns at the top end.

📂 `analysis1/` → `results/fig2_scatter_scale_vs_stage.png`

---

### 💬 Analysis 2 — Does Prompting Style Matter?
*Does asking the AI to "think step by step" or "roleplay as a philosopher" change how it reasons morally?*

Using repeated-measures statistical tests (Friedman + Wilcoxon), we found that prompt engineering has **negligible effect** on the fundamental moral stage of modern frontier models. Their moral reasoning is baked in, not prompted out.

📂 `analysis2/` → `results/`

---

### 🔁 Analysis 3 — Are AI Models Consistent or All Over the Place?
*Does the same model give different moral reasoning for different dilemmas?*

Using Intraclass Correlation Coefficient (ICC), we found models are **hyper-consistent** (ICC > 0.90) — almost robotically so. Human moral reasoning varies by context; AI moral reasoning largely doesn't.

📂 `analysis3/` → `results/`

---

### 📉 Analysis 4 — Do AI Models Reason Like Humans?
*Do the distribution of moral stages across responses resemble how human adults are distributed?*

We compared AI stage distributions to human developmental norms. Most models either cluster at Stage 5/6 (ceiling effect) or show patterns very different from human populations. A few RLHF-tuned models converge on human-like patterns.

📂 `analysis4/` → `results/`

---

### ⚖️ Analysis 5 — Do AI Models Practice What They Preach?
*When a model reasons at Stage 5 (social contract thinking), does it actually choose a principled action?*

We extracted the action recommended by each model and cross-tabulated it against the reasoning stage. We found strong statistical alignment — but some models show "moral decoupling": they use high-stage words while making low-stage choices.

📂 `analysis5/` → `results/`

---

### 🔤 Analysis 6 — What Do the Words Reveal?
*Are there patterns in the language and vocabulary that different models use when reasoning morally?*

Using TF-IDF keyword extraction and PCA dimensionality reduction, we found that model families share distinct "linguistic voices." Aligned/RLHF models demonstrate richer moral vocabulary regardless of model size.

📂 `analysis6/` → `results/`

---

## 🚀 How to Run This Project

> You'll need Python 3.9+. A basic familiarity with the terminal is helpful.

### Step 1 — Set up the environment

```bash
python3 -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install scikit-posthocs        # Needed for Dunn post-hoc tests
```

### Step 2 — Add your API keys

Create a `.env` file in the root folder and paste in your keys:

```
GROQ_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

> ⚠️ The `data/` and `evaluation_data/` folders already contain pre-collected responses and Kohlberg scores. You only need API keys if you want to collect **new** model responses.

### Step 3 — (Optional) Collect new LLM responses

```bash
python "LLM calls/puter_llm.py"
# ↳ Saves responses to data/<model_name>.xlsx
```

### Step 4 — (Optional) Run Kohlberg-stage evaluation

```bash
python evaluation_data/puter_evaluation_llm.py
# ↳ Saves scored data to evaluation_data/<model_name>_evaluation.xlsx
```

### Step 5 — Run any analysis module

All six analyses are **independent** — you can run any one without running the others.

```bash
cd analysis1/
python main.py
# ↳ Charts and tables appear in analysis1/results/
```

Replace `analysis1` with `analysis2` through `analysis6` for the other modules.

---

## 📁 Data Files at a Glance

### `data/` — Raw model responses

Each `.xlsx` file stores everything captured during an LLM call:

| Column | Description |
|---|---|
| `model_name` | Which AI model responded |
| `dilemma_type` | Which moral dilemma (e.g., Heinz, Trolley, Lifeboat) |
| `prompt_type` | Zero-Shot / CoT / Roleplay |
| `response` | The full text the model generated |
| `response_length` | Word count |
| `inference_time` | How long the model took to respond |
| `api_source` | Which API was used |
| `temperature` | Randomness setting (usually 0 for determinism) |

### `evaluation_data/` — Kohlberg stage scores

Each `*_evaluation.xlsx` stores the automated scoring of every response:

| Column | Description |
|---|---|
| `kohlberg_stage` | Stage 1–6 assigned by the evaluator |
| `kohlberg_confidence` | How confident the evaluator was |
| `kohlberg_reasoning` | Explanation for the assigned stage |
| `secondary_stage` | Second-best stage if borderline |

---

## 🧩 What Is Kohlberg's Framework?

Lawrence Kohlberg proposed that humans develop moral reasoning in stages. Here's a simplified breakdown:

| Stage | Level | Core Idea |
|---|---|---|
| 1 | Pre-conventional | Avoid punishment |
| 2 | Pre-conventional | Do what benefits you |
| 3 | Conventional | Be a good person / fit in |
| 4 | Conventional | Follow rules and laws |
| 5 | Post-conventional | Respect social contracts and rights |
| 6 | Post-conventional | Follow universal ethical principles |

Most adults reason at Stages 3–5. This project measures which stages AI models tend to occupy.

---

## ⚙️ Technical Notes

- `puter_session/` and `puter_user_data/` are local browser-cache folders used by the Puter-based model runner — they are already excluded from Git via `.gitignore`.
- Parameter counts in `analysis1/config.py` (`MODEL_META`) are **estimates** for closed-source models (Claude, GPT-4o). The Spearman correlation is sensitive to these, so update them if you have better numbers.
- All analysis outputs in `results/` are fully reproducible by running `main.py` from scratch — no manual steps needed.

---

## 📄 License

[MIT License](LICENSE) — feel free to use, cite, or build on this research.
