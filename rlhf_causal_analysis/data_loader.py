"""
data_loader.py — Loads and merges scored evaluation data for Analysis 11.

Returns structured DataFrames for downstream metrics and visualizations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from config import EVAL_DIR, MODEL_PAIRS, PAIR_ORDER, STAGES


def load_scored_data() -> pd.DataFrame:
    """
    Load all scored evaluation xlsx files from EVAL_DIR.

    Returns obs_df: one row per (pair_id, variant, dilemma, prompt_type) observation.
    Skips missing files with a warning.
    """
    frames: list[pd.DataFrame] = []

    for pair_id in PAIR_ORDER:
        for variant in ("base", "instruct"):
            path = EVAL_DIR / f"{pair_id}_{variant}_evaluation.xlsx"
            if not path.exists():
                print(f"  [SKIP] {path.name} — not found")
                continue

            df = pd.read_excel(path)

            # Validate required columns
            if "kohlberg_stage" not in df.columns:
                print(f"  [WARN] {path.name} missing 'kohlberg_stage' — skipping")
                continue

            # Ensure metadata columns (may be absent if loaded from older runs)
            pair_cfg = MODEL_PAIRS[pair_id]
            df["pair_id"]      = df.get("pair_id",      pair_id)
            df["architecture"] = df.get("architecture", pair_cfg["architecture"])
            df["params_B"]     = df.get("params_B",     pair_cfg["params_B"])
            df["variant"]      = df.get("variant",      variant)
            df["model_label"]  = df.get(
                "model_label",
                pair_cfg.get(f"{variant}_label", f"{pair_id}_{variant}"),
            )

            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No scored evaluation files found in rlhf_causal_analysis/evaluation/. "
            "Run evaluator.py first."
        )

    obs_df = pd.concat(frames, ignore_index=True)

    # Coerce stage to int
    obs_df["kohlberg_stage"] = pd.to_numeric(obs_df["kohlberg_stage"], errors="coerce")
    obs_df.dropna(subset=["kohlberg_stage"], inplace=True)
    obs_df["kohlberg_stage"] = obs_df["kohlberg_stage"].astype(int)
    obs_df = obs_df[obs_df["kohlberg_stage"].between(1, 6)]

    # Confidence
    if "kohlberg_confidence" in obs_df.columns:
        obs_df["kohlberg_confidence"] = pd.to_numeric(
            obs_df["kohlberg_confidence"], errors="coerce"
        )
    else:
        obs_df["kohlberg_confidence"] = np.nan

    # Per-stage indicator columns
    for s in STAGES:
        obs_df[f"is_stage_{s}"] = (obs_df["kohlberg_stage"] == s).astype(int)

    # Pair × variant ordering index
    order_map = {
        (pair_id, variant): i * 2 + j
        for i, pair_id   in enumerate(PAIR_ORDER)
        for j, variant   in enumerate(("base", "instruct"))
    }
    obs_df["row_order"] = obs_df.apply(
        lambda r: order_map.get((r["pair_id"], r["variant"]), 999), axis=1
    )
    obs_df.sort_values("row_order", inplace=True, ignore_index=True)

    print(
        f"Loaded {len(obs_df):,} observations "
        f"({obs_df['pair_id'].nunique()} pairs × 2 variants)."
    )
    return obs_df


def build_pair_distributions(obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate obs_df to one row per (pair_id, variant).

    Columns: pair_id, architecture, params_B, variant, model_label,
             n_obs, modal_stage, mean_stage, std_stage,
             stage_1 … stage_6  (proportions)
    """
    rows = []
    for pair_id in PAIR_ORDER:
        pair_cfg = MODEL_PAIRS[pair_id]
        for variant in ("base", "instruct"):
            grp = obs_df[(obs_df["pair_id"] == pair_id) & (obs_df["variant"] == variant)]
            if grp.empty:
                continue

            stages_arr = grp["kohlberg_stage"].values
            n = len(stages_arr)

            props = {f"stage_{s}": (stages_arr == s).sum() / n for s in STAGES}
            modal_stage = int(pd.Series(stages_arr).mode().iloc[0])
            mean_stage  = float(stages_arr.mean())
            std_stage   = float(stages_arr.std(ddof=1)) if n > 1 else 0.0

            row = {
                "pair_id":      pair_id,
                "architecture": pair_cfg["architecture"],
                "params_B":     pair_cfg["params_B"],
                "variant":      variant,
                "model_label":  pair_cfg.get(f"{variant}_label", f"{pair_id}_{variant}"),
                "n_obs":        n,
                "modal_stage":  modal_stage,
                "mean_stage":   mean_stage,
                "std_stage":    std_stage,
            }
            row.update(props)
            rows.append(row)

    dist_df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"Distribution table built: {len(dist_df)} rows ({len(dist_df)//2} pairs × 2 variants).")
    return dist_df
