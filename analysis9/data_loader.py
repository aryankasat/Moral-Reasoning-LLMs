"""
data_loader.py — Loads evaluation data for Analysis 9: Capability Correlation.

Returns:
  raw_df   : observation-level DataFrame (one row per dilemma × model)
  model_df : model-level aggregates keyed by model_key
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import EVAL_DIR, MODEL_META, STAGES, POST_CONV_STAGE


def load_raw_data() -> pd.DataFrame:
    """Load all evaluation xlsx files; attach model metadata columns."""
    frames: list[pd.DataFrame] = []
    eval_files = sorted(EVAL_DIR.glob("*_evaluation.xlsx"))

    for path in eval_files:
        stem = path.stem.replace("_evaluation", "")
        if stem not in MODEL_META:
            print(f"  [SKIP] {path.name} — not in MODEL_META")
            continue

        display_name, params_B, provider, scale_group, training_type = MODEL_META[stem]
        df = pd.read_excel(path)

        required = {"kohlberg_stage", "response"}
        if not required.issubset(df.columns):
            print(f"  [WARN] {path.name} missing required columns — skipping")
            continue

        df["model_key"]     = stem
        df["display_name"]  = display_name
        df["params_B"]      = params_B
        df["log_params"]    = np.log10(params_B)
        df["provider"]      = provider
        df["scale_group"]   = scale_group
        df["training_type"] = training_type
        frames.append(df)

    raw_df = pd.concat(frames, ignore_index=True)

    # Coerce numeric columns
    raw_df["kohlberg_stage"] = pd.to_numeric(raw_df["kohlberg_stage"], errors="coerce")
    raw_df.dropna(subset=["kohlberg_stage"], inplace=True)
    raw_df["kohlberg_stage"] = raw_df["kohlberg_stage"].astype(int)

    # Confidence (1–5) as coherence proxy
    if "kohlberg_confidence" in raw_df.columns:
        raw_df["kohlberg_confidence"] = pd.to_numeric(
            raw_df["kohlberg_confidence"], errors="coerce"
        )
    else:
        raw_df["kohlberg_confidence"] = np.nan

    # Flag post-conventional responses
    raw_df["is_post_conv"] = (raw_df["kohlberg_stage"] >= POST_CONV_STAGE).astype(int)

    # Per-stage indicator columns
    for s in STAGES:
        raw_df[f"is_stage_{s}"] = (raw_df["kohlberg_stage"] == s).astype(int)

    print(
        f"Loaded {len(raw_df):,} observations across "
        f"{raw_df['model_key'].nunique()} models."
    )
    return raw_df


def build_model_summary(raw_df: pd.DataFrame, cap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge model-level aggregates with pre-computed capability metrics.

    cap_df: output of capability_metrics.compute_model_capabilities(raw_df)
    Returns one row per model with all metrics.
    """
    rows = []
    for model_key, grp in raw_df.groupby("model_key"):
        stages_arr = grp["kohlberg_stage"].values
        n = len(stages_arr)
        meta = MODEL_META[model_key]
        rows.append({
            "model_key":     model_key,
            "display_name":  meta[0],
            "params_B":      meta[1],
            "log_params":    np.log10(meta[1]),
            "provider":      meta[2],
            "scale_group":   meta[3],
            "training_type": meta[4],
            "n_obs":         n,
            "mean_stage":    stages_arr.mean(),
            "std_stage":     stages_arr.std(ddof=1) if n > 1 else 0.0,
            "post_conv_pct": grp["is_post_conv"].mean() * 100,   # percentage
            "post_conv_capable": int(grp["is_post_conv"].mean() >= 0.20),
        })

    model_df = pd.DataFrame(rows)
    model_df = model_df.merge(cap_df, on="model_key", how="left")
    model_df.sort_values("params_B", ignore_index=True, inplace=True)
    print(f"Model summary built: {len(model_df)} models.")
    return model_df
