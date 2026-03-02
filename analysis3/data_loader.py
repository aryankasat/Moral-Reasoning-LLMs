"""
data_loader.py — Load and merge evaluation + response data for Analysis 3.

Key difference from analysis1: we join eval and data files by *row index*
(not by dilemma_type dedup), so all three prompt types (ZERO_SHOT, COT,
ROLEPLAY) are correctly recovered for all 18 rows per model.

Data structure per model:
  Rows  0– 5 : ZERO_SHOT, dilemmas 0–5, sample 0
  Rows  6–11 : COT,       dilemmas 0–5, sample 0  (same dilemma order)
  Rows 12–17 : ROLEPLAY,  dilemmas 0–5, sample 0

Wait — actually each eval Excel has 18 rows corresponding to 18 LLM responses,
i.e. 3 prompt_types × 6 dilemmas × 1 response each.
Within the same prompt_type block, each dilemma appears once.
Across models, there is exactly 1 sample per (model, prompt_type, dilemma) cell.

Public API
----------
load_all_data() -> pd.DataFrame
    Long-format DataFrame with one row per LLM response, enriched with
    model metadata and a sample_id within each (model, prompt_type, dilemma).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_DIR, EVAL_DIR, MODEL_META, PROMPT_ORDER


def load_all_data() -> pd.DataFrame:
    """
    Join every evaluation_data/*_evaluation.xlsx with data/*.xlsx by row index
    to correctly recover prompt_type for all 18 rows per model.

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

        # ── Evaluation data ────────────────────────────────────────────────
        edf = pd.read_excel(eval_path).reset_index(drop=True)
        required = {"kohlberg_stage", "kohlberg_confidence", "dilemma_type"}
        missing = required - set(edf.columns)
        if missing:
            print(f"  [WARN] {eval_path.name} missing columns: {missing}")
            continue

        # ── Response metadata (prompt_type) — row-index join ───────────────
        data_path = DATA_DIR / f"{stem}.xlsx"
        if data_path.exists():
            ddf = pd.read_excel(data_path)[["dilemma_type", "prompt_type"]].reset_index(drop=True)
            # Validate shapes match
            if len(ddf) == len(edf):
                edf["prompt_type"] = ddf["prompt_type"].values
            else:
                # Fallback: try to infer from row position
                # Assume 3 prompt-type blocks of 6 dilemmas each
                n_dilemmas = 6
                n_prompts   = len(PROMPT_ORDER)
                if len(edf) == n_dilemmas * n_prompts:
                    pt_labels = [pt for pt in PROMPT_ORDER for _ in range(n_dilemmas)]
                    edf["prompt_type"] = pt_labels
                else:
                    print(f"  [WARN] Shape mismatch for {stem}, prompt_type set to NaN")
                    edf["prompt_type"] = np.nan
            
        else:
            print(f"  [WARN] No data file for {stem}, prompt_type set to NaN")
            edf["prompt_type"] = np.nan
            

        # ── Assign sample_id within each (dilemma_type, prompt_type) group ─
        # Since the data has 3 repetitions of the same dilemma×prompt_type,
        # sample_id = 0, 1, 2 in order of appearance.
        edf["sample_id"] = edf.groupby(
            ["dilemma_type", "prompt_type"]
        ).cumcount()

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
            "kohlberg_stage", "kohlberg_confidence",
            "dilemma_type", "prompt_type", "sample_id",
        ]
        frames.append(edf[keep_cols])

    df = pd.concat(frames, ignore_index=True)

    # ── Coerce and clean stage ─────────────────────────────────────────────
    df["kohlberg_stage"] = pd.to_numeric(df["kohlberg_stage"], errors="coerce")
    df.dropna(subset=["kohlberg_stage"], inplace=True)
    df["kohlberg_stage"] = df["kohlberg_stage"].astype(int)

    n_models = df["model_key"].nunique()
    print(f"Loaded {len(df):,} observations from {n_models} models.")
    print(f"Prompt types found: {sorted(df['prompt_type'].dropna().unique())}")
    return df
