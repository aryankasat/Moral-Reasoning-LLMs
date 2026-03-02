"""
data_loader.py — Load and merge evaluation + response data for Analysis 4.
Identical row-index join logic as analysis3 to correctly recover prompt_type.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from config import DATA_DIR, EVAL_DIR, MODEL_META, PROMPT_ORDER


def load_all_data() -> pd.DataFrame:
    """
    Join every evaluation_data/*_evaluation.xlsx with data/*.xlsx by row index.

    Returns
    -------
    pd.DataFrame with columns:
        model_key, display_name, params_B, log_params, provider,
        kohlberg_stage, kohlberg_confidence, dilemma_type, prompt_type,
        sample_id
    """
    frames = []
    eval_files = sorted(EVAL_DIR.glob("*_evaluation.xlsx"))

    for eval_path in eval_files:
        stem = eval_path.stem.replace("_evaluation", "")
        if stem not in MODEL_META:
            print(f"  [SKIP] {eval_path.name} — no entry in MODEL_META")
            continue

        display_name, params_B, provider = MODEL_META[stem]

        edf = pd.read_excel(eval_path).reset_index(drop=True)
        required = {"kohlberg_stage", "kohlberg_confidence", "dilemma_type"}
        missing = required - set(edf.columns)
        if missing:
            print(f"  [WARN] {eval_path.name} missing columns: {missing}")
            continue

        # Row-index join to recover prompt_type
        data_path = DATA_DIR / f"{stem}.xlsx"
        if data_path.exists():
            ddf = pd.read_excel(data_path)[["dilemma_type", "prompt_type"]].reset_index(drop=True)
            if len(ddf) == len(edf):
                edf["prompt_type"] = ddf["prompt_type"].values
            else:
                n_dilemmas = 6
                if len(edf) == n_dilemmas * len(PROMPT_ORDER):
                    pt_labels = [pt for pt in PROMPT_ORDER for _ in range(n_dilemmas)]
                    edf["prompt_type"] = pt_labels
                else:
                    print(f"  [WARN] Shape mismatch for {stem}, prompt_type NaN")
                    edf["prompt_type"] = np.nan
        else:
            print(f"  [WARN] No data file for {stem}, prompt_type NaN")
            edf["prompt_type"] = np.nan

        edf["sample_id"] = edf.groupby(["dilemma_type", "prompt_type"]).cumcount()

        edf = edf.assign(
            model_key    = stem,
            display_name = display_name,
            params_B     = params_B,
            log_params   = np.log10(params_B),
            provider     = provider,
        )

        keep_cols = [
            "model_key", "display_name", "params_B", "log_params", "provider",
            "kohlberg_stage", "kohlberg_confidence",
            "dilemma_type", "prompt_type", "sample_id",
        ]
        frames.append(edf[keep_cols])

    df = pd.concat(frames, ignore_index=True)
    df["kohlberg_stage"] = pd.to_numeric(df["kohlberg_stage"], errors="coerce")
    df.dropna(subset=["kohlberg_stage"], inplace=True)
    df["kohlberg_stage"] = df["kohlberg_stage"].astype(int)

    n_models = df["model_key"].nunique()
    print(f"Loaded {len(df):,} observations from {n_models} models.")
    print(f"Prompt types found: {sorted(df['prompt_type'].dropna().unique())}")
    return df
