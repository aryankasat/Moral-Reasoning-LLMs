"""
data_loader.py — Loads evaluation data for Analysis 7: Emergence Threshold Detection.

Returns two DataFrames:
  - raw_df   : full observation-level data (one row per dilemma × model)
  - model_df : model-level summary stats sorted by parameter count
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.stats as stats

from config import EVAL_DIR, MODEL_META, STAGES, POST_CONV_THRESHOLD, CI_LEVEL


def load_raw_data() -> pd.DataFrame:
    """Load all evaluation xlsx files and return a unified observation-level DataFrame."""
    frames: list[pd.DataFrame] = []
    eval_files = sorted(EVAL_DIR.glob("*_evaluation.xlsx"))

    for path in eval_files:
        stem = path.stem.replace("_evaluation", "")
        if stem not in MODEL_META:
            print(f"  [SKIP] {path.name} — not in MODEL_META")
            continue

        display_name, params_B, provider = MODEL_META[stem]
        df = pd.read_excel(path)

        required = {"kohlberg_stage", "dilemma_type"}
        if not required.issubset(df.columns):
            print(f"  [WARN] {path.name} missing required columns — skipping")
            continue

        df["model_key"]    = stem
        df["display_name"] = display_name
        df["params_B"]     = params_B
        df["log_params"]   = np.log10(params_B)
        df["provider"]     = provider
        frames.append(df)

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df["kohlberg_stage"] = pd.to_numeric(raw_df["kohlberg_stage"], errors="coerce")
    raw_df.dropna(subset=["kohlberg_stage"], inplace=True)
    raw_df["kohlberg_stage"] = raw_df["kohlberg_stage"].astype(int)

    print(f"Loaded {len(raw_df):,} observations across {raw_df['model_key'].nunique()} models.")
    return raw_df


def build_model_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to model-level summary statistics.

    Columns in returned DataFrame:
      model_key, display_name, params_B, log_params, provider
      mean_stage, median_stage, std_stage, se_stage, ci_lower, ci_upper
      stage_1_pct … stage_6_pct
      post_conv_pct   (fraction Stage 5+)
      n_obs
    """
    alpha = 1 - CI_LEVEL
    rows = []

    for model_key, grp in raw_df.groupby("model_key"):
        stages_arr = grp["kohlberg_stage"].values
        n = len(stages_arr)

        mean_s  = stages_arr.mean()
        std_s   = stages_arr.std(ddof=1) if n > 1 else 0.0
        se_s    = std_s / np.sqrt(n) if n > 0 else 0.0

        # t-based CI
        if n > 1:
            t_crit   = stats.t.ppf(1 - alpha / 2, df=n - 1)
            ci_lower = mean_s - t_crit * se_s
            ci_upper = mean_s + t_crit * se_s
        else:
            ci_lower = ci_upper = mean_s

        # Per-stage percentages
        stage_pcts = {f"stage_{s}_pct": (stages_arr == s).mean() for s in STAGES}

        # Post-conventional (Stage 5+)
        post_conv_pct = ((stages_arr >= 5).mean())

        meta = MODEL_META[model_key]
        rows.append({
            "model_key":    model_key,
            "display_name": meta[0],
            "params_B":     meta[1],
            "log_params":   np.log10(meta[1]),
            "provider":     meta[2],
            "mean_stage":   mean_s,
            "median_stage": np.median(stages_arr),
            "std_stage":    std_s,
            "se_stage":     se_s,
            "ci_lower":     ci_lower,
            "ci_upper":     ci_upper,
            "post_conv_pct": post_conv_pct,
            "n_obs":        n,
            **stage_pcts,
        })

    model_df = pd.DataFrame(rows)
    model_df.sort_values("params_B", inplace=True, ignore_index=True)

    # For models that share the same params_B (e.g. two 671B models), add a tiny
    # jitter so points are distinguishable on a log-scale plot.
    # ALL models get a params_B_plot value (non-duplicates get exactly params_B).
    params_B_plot = model_df["params_B"].copy().astype(float)
    seen: dict[float, int] = {}
    for idx in model_df.index:
        p = float(model_df.at[idx, "params_B"])
        count = seen.get(p, 0)
        if count > 0:
            params_B_plot.at[idx] = p + count * 2  # slight offset in B
        seen[p] = count + 1
    model_df["params_B_plot"]   = params_B_plot
    model_df["log_params_plot"] = np.log10(model_df["params_B_plot"])

    # Emergence flag
    model_df["emerged"] = model_df["post_conv_pct"] >= POST_CONV_THRESHOLD

    print(f"Model summary built: {len(model_df)} models, "
          f"{model_df['emerged'].sum()} meeting post-conv threshold (≥{POST_CONV_THRESHOLD:.0%} Stage 5+).")
    return model_df
