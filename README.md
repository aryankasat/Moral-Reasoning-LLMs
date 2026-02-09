# Moral-Reasoning-LLMs

Research repository for evaluating and comparing large language models' moral reasoning using curated prompts, model wrappers, and evaluation utilities.

## Overview

- **Purpose:** Provide prompts, data, and tooling to evaluate how different LLMs respond to moral dilemmas and reasoning tasks.
- **Contents:** prompt collections, LLM wrappers for running model calls, evaluation scripts, and datasets for human/automated evaluation.

## Key Features

- Centralized prompt hub and examples (`prompt_hub.py`).
- Wrappers for different model providers under `LLM calls/` (e.g., `groq_llm.py`, `mistral_llm.py`, `pythia_llm.py`, `puter_llm.py`).
- Evaluation scripts and helpers in `evaluation_data/`.
- Data storage and a place to collect model outputs in `data/`.

## Repository Structure

- `prompt_hub.py` — canonical prompt collection and utilities.
- `LLM calls/` — model-specific wrappers and helpers to call various LLMs.
- `evaluation_data/` — scripts for running evaluations and updating results (e.g., `puter_evaluation_llm.py`, `update_excel.py`).
- `data/` — data used by experiments and evaluations.
- `puter_session/`, `puter_user_data/` — local session/cache folders (browser/profile data, can be ignored or removed from commits).
- `README.md` — this file.

## Quick Start

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (add a `requirements.txt` if you keep track of dependencies):

```bash
pip install -r requirements.txt 
```

3. Run the LLM calling for data generation (example):

```bash
python LLM calls/groq.py
```

4. Run an evaluation script (example):

```bash
python evaluation_data/puter_evaluation_llm.py
```


## Usage Notes

- Add provider-specific credentials as environment variables before running model wrappers.
- Model wrappers are in `LLM calls/` — adapt them to your API keys and desired parameters.
- Save outputs and results under `data/` and use the scripts in `evaluation_data/` to aggregate results.

## Contributing

- Create an issue or PR for new prompts, models, or evaluation metrics.
- Keep large binary/session files out of git; add them to `.gitignore` if needed.