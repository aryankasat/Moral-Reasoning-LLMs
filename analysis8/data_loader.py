"""
data_loader.py — Loads evaluation data for Analysis 8: Scale vs. Training Decomposition.

Returns:
  raw_df  : observation-level DataFrame (one row per dilemma × model)
  cell_df : cell-level aggregates (mean ± CI) keyed by (scale_group, training_type)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats

from config import EVAL_DIR, MODEL_META, STAGES, CI_LEVEL, SCALE_ORDER, TRAINING_ORDER


def load_raw_data() -> pd.DataFrame:
    """Load all evaluation xlsx files; attach scale_group and training_type columns."""
    frames: list[pd.DataFrame] = []
    eval_files = sorted(EVAL_DIR.glob("*_evaluation.xlsx"))

    for path in eval_files:
        stem = path.stem.replace("_evaluation", "")
        if stem not in MODEL_META:
            print(f"  [SKIP] {path.name} — not in MODEL_META")
            continue

        display_name, params_B, provider, scale_group, training_type = MODEL_META[stem]
        df = pd.read_excel(path)

        required = {"kohlberg_stage", "dilemma_type"}
        if not required.issubset(df.columns):
            print(f"  [WARN] {path.name} missing required columns — skipping")
            continue

        df["model_key"]     = stem
        df["display_name"]  = display_name
        df["params_B"]      = params_B
        df["log_params"]    = np.log10(params_B)
        df["provider"]      = provider
        df["scale_group"]   = scale_group    # plain str; cast after concat
        df["training_type"] = training_type  # plain str; cast after concat
        frames.append(df)

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df["kohlberg_stage"] = pd.to_numeric(raw_df["kohlberg_stage"], errors="coerce")
    raw_df.dropna(subset=["kohlberg_stage"], inplace=True)
    raw_df["kohlberg_stage"] = raw_df["kohlberg_stage"].astype(int)

    # Cast factor columns to ordered Categoricals after concat
    raw_df["scale_group"]   = pd.Categorical(raw_df["scale_group"],   categories=SCALE_ORDER,    ordered=True)
    raw_df["training_type"] = pd.Categorical(raw_df["training_type"], categories=TRAINING_ORDER, ordered=True)

    # Stage distribution per-stage percentages at observation level
    for s in STAGES:
        raw_df[f"is_stage_{s}"] = (raw_df["kohlberg_stage"] == s).astype(int)

    print(
        f"Loaded {len(raw_df):,} observations across "
        f"{raw_df['model_key'].nunique()} models, "
        f"{raw_df['scale_group'].nunique()} scale groups, "
        f"{raw_df['training_type'].nunique()} training types."
    )
    return raw_df


def build_cell_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to (scale_group × training_type) cell-level stats.

    Returned columns:
      scale_group, training_type
      n_obs, n_models
      mean_stage, median_stage, std_stage, se_stage, ci_lower, ci_upper
    """
    alpha = 1 - CI_LEVEL
    rows = []

    for (sg, tt), grp in raw_df.groupby(["scale_group", "training_type"], observed=False):
        if len(grp) == 0:
            continue
        stages_arr = grp["kohlberg_stage"].values
        n = len(stages_arr)
        mean_s  = stages_arr.mean()
        std_s   = stages_arr.std(ddof=1) if n > 1 else 0.0
        se_s    = std_s / np.sqrt(n)     if n > 0 else 0.0

        if n > 1:
            t_crit   = stats.t.ppf(1 - alpha / 2, df=n - 1)
            ci_lower = mean_s - t_crit * se_s
            ci_upper = mean_s + t_crit * se_s
        else:
            ci_lower = ci_upper = mean_s

        rows.append({
            "scale_group":   sg,
            "training_type": tt,
            "n_obs":         n,
            "n_models":      grp["model_key"].nunique(),
            "mean_stage":    mean_s,
            "median_stage":  float(np.median(stages_arr)),
            "std_stage":     std_s,
            "se_stage":      se_s,
            "ci_lower":      ci_lower,
            "ci_upper":      ci_upper,
        })

    cell_df = pd.DataFrame(rows)
    cell_df["scale_group"]   = pd.Categorical(cell_df["scale_group"],   categories=SCALE_ORDER,    ordered=True)
    cell_df["training_type"] = pd.Categorical(cell_df["training_type"], categories=TRAINING_ORDER, ordered=True)
    cell_df.sort_values(["scale_group", "training_type"], inplace=True, ignore_index=True)

    print(f"Cell summary built: {len(cell_df)} cells.")
    return cell_df


def build_model_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Model-level summary for supplementary tables."""
    rows = []
    for model_key, grp in raw_df.groupby("model_key"):
        stages_arr = grp["kohlberg_stage"].values
        n  = len(stages_arr)
        meta = MODEL_META[model_key]
        rows.append({
            "model_key":     model_key,
            "display_name":  meta[0],
            "params_B":      meta[1],
            "provider":      meta[2],
            "scale_group":   meta[3],
            "training_type": meta[4],
            "n_obs":         n,
            "mean_stage":    stages_arr.mean(),
            "std_stage":     stages_arr.std(ddof=1) if n > 1 else 0.0,
        })

    model_df = pd.DataFrame(rows).sort_values("params_B", ignore_index=True)
    return model_df
