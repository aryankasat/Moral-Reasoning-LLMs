"""
data_loader.py — Load and merge evaluation + response data, tagging each row
with alignment_type and family from MODEL_META.

Public API
----------
load_all_data() -> pd.DataFrame
"""

import numpy as np
import pandas as pd

from config import DATA_DIR, EVAL_DIR, MODEL_META


def load_all_data() -> pd.DataFrame:
    """
    Join every evaluation_data/*_evaluation.xlsx with its matching data/*.xlsx.

    Returns
    -------
    pd.DataFrame with columns:
        model_key, display_name, params_B, family, alignment_type,
        kohlberg_stage, kohlberg_confidence, dilemma_type, prompt_type
    """
    frames = []
    for eval_path in sorted(EVAL_DIR.glob("*_evaluation.xlsx")):
        stem = eval_path.stem.replace("_evaluation", "")
        if stem not in MODEL_META:
            continue

        display_name, params_B, family, alignment_type = MODEL_META[stem]

        edf = pd.read_excel(eval_path)
        required = {"kohlberg_stage", "kohlberg_confidence", "dilemma_type"}
        if required - set(edf.columns):
            print(f"  [WARN] {eval_path.name} missing required columns — skipped")
            continue

        data_path = DATA_DIR / f"{stem}.xlsx"
        if data_path.exists():
            ddf = pd.read_excel(data_path)[["dilemma_type", "prompt_type"]]
            edf = edf.merge(ddf.drop_duplicates("dilemma_type"),
                            on="dilemma_type", how="left")
        else:
            edf["prompt_type"] = np.nan

        edf = edf.assign(
            model_key      = stem,
            display_name   = display_name,
            params_B       = params_B,
            family         = family,
            alignment_type = alignment_type,
        )

        keep = ["model_key", "display_name", "params_B", "family",
                "alignment_type", "kohlberg_stage", "kohlberg_confidence",
                "dilemma_type", "prompt_type"]
        frames.append(edf[keep])

    df = pd.concat(frames, ignore_index=True)
    df["kohlberg_stage"] = pd.to_numeric(df["kohlberg_stage"], errors="coerce")
    df.dropna(subset=["kohlberg_stage"], inplace=True)
    df["kohlberg_stage"] = df["kohlberg_stage"].astype(int)

    print(f"Loaded {len(df):,} observations from {df['model_key'].nunique()} models.")
    return df
