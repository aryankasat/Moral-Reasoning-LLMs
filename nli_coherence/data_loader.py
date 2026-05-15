"""
data_loader.py — Loads evaluation data and extracts (reasoning, action) pairs
for NLI coherence scoring.

This module is framework-agnostic: it treats kohlberg_reasoning as the
model's justification text and action_endorsed as the declared action,
without imposing any stage-based expectations.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from config import DATA_DIR, EVAL_DIR, MODEL_META, MCNEMAR_CSV


def load_evaluation_pairs() -> pd.DataFrame:
    """
    Loads evaluation data and extracts (reasoning, action_endorsed) pairs
    for every model × dilemma × trial.

    Returns
    -------
    pd.DataFrame with columns:
        model_key, display_name, params_B, provider,
        dilemma_type, kohlberg_reasoning, action_endorsed
    """
    frames = []
    eval_files = sorted(EVAL_DIR.glob("*_evaluation.xlsx"))

    for eval_path in eval_files:
        stem = eval_path.stem.replace("_evaluation", "")
        if stem not in MODEL_META:
            continue

        display_name, params_B, provider = MODEL_META[stem]
        edf = pd.read_excel(eval_path)

        required = {"dilemma_type", "kohlberg_reasoning", "action_endorsed"}
        if not required.issubset(edf.columns):
            print(f"  [WARN] {eval_path.name} missing required columns, skipping.")
            continue

        edf["model_key"]     = stem
        edf["display_name"]  = display_name
        edf["params_B"]      = params_B
        edf["log_params"]    = np.log10(params_B)
        edf["provider"]      = provider

        keep = [
            "model_key", "display_name", "params_B", "log_params", "provider",
            "dilemma_type", "kohlberg_reasoning", "action_endorsed",
        ]
        frames.append(edf[keep])

    df = pd.concat(frames, ignore_index=True)

    # Drop rows where either reasoning or action is missing
    df.dropna(subset=["kohlberg_reasoning", "action_endorsed"], inplace=True)

    # Clean up text: strip whitespace, collapse newlines
    for col in ("kohlberg_reasoning", "action_endorsed"):
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    # Filter out empty strings
    df = df[
        (df["kohlberg_reasoning"].str.len() > 10) &
        (df["action_endorsed"].str.len() > 5)
    ].copy()

    print(f"Loaded {len(df):,} (reasoning, action) pairs across "
          f"{df['model_key'].nunique()} models.")
    return df


def load_mcnemar_pvalues() -> pd.DataFrame:
    """
    Loads the McNemar per-model p_values from analysis5 as the
    existing decoupling scores.

    Returns
    -------
    pd.DataFrame with columns: model_key, display_name, params_B, p_value
    """
    mcnemar = pd.read_csv(MCNEMAR_CSV)
    keep = ["model_key", "display_name", "params_B", "p_value"]
    mcnemar = mcnemar[keep].copy()
    print(f"Loaded McNemar p_values for {len(mcnemar)} models.")
    return mcnemar
