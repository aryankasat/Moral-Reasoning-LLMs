"""
data_loader.py — Loads evaluation data for Analysis 10: Stage Transition Dynamics.

Returns:
  obs_df   : observation-level DataFrame (one row per dilemma × model)
  model_df : model-level stage-distribution DataFrame, ordered by params_B
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import EVAL_DIR, MODEL_META, MODEL_ORDER, STAGES


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

        required = {"kohlberg_stage"}
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

    obs_df = pd.concat(frames, ignore_index=True)

    # Coerce stage to int
    obs_df["kohlberg_stage"] = pd.to_numeric(obs_df["kohlberg_stage"], errors="coerce")
    obs_df.dropna(subset=["kohlberg_stage"], inplace=True)
    obs_df["kohlberg_stage"] = obs_df["kohlberg_stage"].astype(int)

    # Confidence proxy
    if "kohlberg_confidence" in obs_df.columns:
        obs_df["kohlberg_confidence"] = pd.to_numeric(
            obs_df["kohlberg_confidence"], errors="coerce"
        )
    else:
        obs_df["kohlberg_confidence"] = np.nan

    # Per-stage indicator columns
    for s in STAGES:
        obs_df[f"is_stage_{s}"] = (obs_df["kohlberg_stage"] == s).astype(int)

    # Stage position in MODEL_ORDER (used for ordering)
    order_map = {k: i for i, k in enumerate(MODEL_ORDER)}
    obs_df["model_order"] = obs_df["model_key"].map(order_map)
    obs_df.sort_values("model_order", inplace=True, ignore_index=True)

    print(
        f"Loaded {len(obs_df):,} observations across "
        f"{obs_df['model_key'].nunique()} models."
    )
    return obs_df


def build_stage_distribution(obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-level stage-distribution DataFrame.

    Returns a DataFrame with one row per model (ordered by params_B), columns:
      model_key, display_name, params_B, log_params, scale_group,
      training_type, n_obs, modal_stage, mean_stage, std_stage,
      stage_1 … stage_6  (proportion of responses at each stage)
    """
    rows = []
    for model_key in MODEL_ORDER:
        grp = obs_df[obs_df["model_key"] == model_key]
        if grp.empty:
            continue

        meta = MODEL_META[model_key]
        display_name, params_B, provider, scale_group, training_type = meta
        stages_arr = grp["kohlberg_stage"].values
        n = len(stages_arr)

        # Stage proportions
        props = {}
        for s in STAGES:
            props[f"stage_{s}"] = (stages_arr == s).sum() / n

        modal_stage = int(pd.Series(stages_arr).mode().iloc[0])
        mean_stage  = float(stages_arr.mean())
        std_stage   = float(stages_arr.std(ddof=1)) if n > 1 else 0.0

        # Stage spread (IQR)
        q75, q25 = np.percentile(stages_arr, [75, 25])
        iqr       = float(q75 - q25)
        stage_range = int(stages_arr.max() - stages_arr.min())

        row = {
            "model_key":     model_key,
            "display_name":  display_name,
            "params_B":      params_B,
            "log_params":    np.log10(params_B),
            "provider":      provider,
            "scale_group":   scale_group,
            "training_type": training_type,
            "n_obs":         n,
            "modal_stage":   modal_stage,
            "mean_stage":    mean_stage,
            "std_stage":     std_stage,
            "iqr":           iqr,
            "stage_range":   stage_range,
        }
        row.update(props)
        rows.append(row)

    model_df = pd.DataFrame(rows).reset_index(drop=True)
    # model_order column (integer index in progression)
    model_df["model_order"] = range(len(model_df))
    print(f"Stage distribution table built: {len(model_df)} models.")
    return model_df
