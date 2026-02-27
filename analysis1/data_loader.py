"""
data_loader.py — Load and merge evaluation + response data.

Public API
----------
load_all_data() -> pd.DataFrame
    Long-format DataFrame with one row per LLM response, enriched with
    model metadata (display name, parameter count, provider).
"""

import numpy as np
import pandas as pd

from config import DATA_DIR, EVAL_DIR, MODEL_META


def load_all_data() -> pd.DataFrame:
    """
    Join every evaluation_data/*_evaluation.xlsx file with its matching
    data/*.xlsx file (for prompt_type metadata).

    Returns
    -------
    pd.DataFrame with columns:
        model_key, display_name, params_B, log_params, provider,
        kohlberg_stage, kohlberg_confidence, dilemma_type, prompt_type
    """
    frames = []
    eval_files = sorted(EVAL_DIR.glob("*_evaluation.xlsx"))

    for eval_path in eval_files:
        stem = eval_path.stem.replace("_evaluation", "")
        if stem not in MODEL_META:
            print(f"  [SKIP] {eval_path.name} — no entry in MODEL_META")
            continue

        display_name, params_B, provider = MODEL_META[stem]

        # ── Evaluation data ────────────────────────────────────────────────
        edf = pd.read_excel(eval_path)
        required = {"kohlberg_stage", "kohlberg_confidence", "dilemma_type"}
        missing = required - set(edf.columns)
        if missing:
            print(f"  [WARN] {eval_path.name} missing: {missing}")
            continue

        # ── Response metadata (prompt_type) ────────────────────────────────
        data_path = DATA_DIR / f"{stem}.xlsx"
        if data_path.exists():
            ddf = pd.read_excel(data_path)[["dilemma_type", "prompt_type"]]
            edf = edf.merge(
                ddf.drop_duplicates("dilemma_type"),
                on="dilemma_type", how="left",
            )
        else:
            edf["prompt_type"] = np.nan

        # ── Tag with model metadata ────────────────────────────────────────
        edf = edf.assign(
            model_key    = stem,
            display_name = display_name,
            params_B     = params_B,
            log_params   = np.log10(params_B),
            provider     = provider,
        )

        keep_cols = [
            "model_key", "display_name", "params_B", "log_params", "provider",
            "kohlberg_stage", "kohlberg_confidence", "dilemma_type", "prompt_type",
        ]
        frames.append(edf[keep_cols])

    df = pd.concat(frames, ignore_index=True)

    # Coerce and clean stage
    df["kohlberg_stage"] = pd.to_numeric(df["kohlberg_stage"], errors="coerce")
    df.dropna(subset=["kohlberg_stage"], inplace=True)
    df["kohlberg_stage"] = df["kohlberg_stage"].astype(int)

    n_models = df["model_key"].nunique()
    print(f"Loaded {len(df):,} observations from {n_models} models.")
    return df
