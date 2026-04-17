"""
data_loader.py — Loads evaluation data and computes decoupling flags.

Supports two data sources:
  1. RLHF causal analysis data (rlhf_causal_analysis/evaluation/)
  2. Main project evaluation data (evaluation_data/) — 13 models

Computes the existing binary decoupling/consistency metric from Analysis 5
so we can correlate it with the NLI coherence scores.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    RLHF_EVAL_DIR,
    RLHF_DATA_DIR,
    MAIN_EVAL_DIR,
    MODEL_META,
    RLHF_PAIR_ORDER,
    RLHF_PAIR_META,
    EXPECTED_ACTION_BY_STAGE,
    ACTION_KEYWORDS,
    STAGES,
)


# ── Action categorization (mirrors Analysis 5) ───────────────────────────────

def categorize_action(dilemma: str, text: str) -> str:
    """
    Map free-text action_endorsed strings to 'Rule-Following' or 'Rule-Breaking'.
    Uses robust keyword matching tailored to each of the 6 dilemmas.
    """
    if not isinstance(text, str):
        return "Ambiguous/Other"

    t = text.lower()

    keywords = ACTION_KEYWORDS.get(dilemma, {})
    breaking_pos = keywords.get("breaking_pos", [])
    breaking_neg = keywords.get("breaking_neg", [])
    following_pos = keywords.get("following_pos", [])

    # Check rule-breaking
    for kw in breaking_pos:
        if kw in t:
            if not any(neg in t for neg in breaking_neg):
                return "Rule-Breaking"

    # Check rule-following
    for kw in following_pos:
        if kw in t:
            return "Rule-Following"

    return "Ambiguous/Other"


# ── Load main project evaluation data (13 models) ────────────────────────────

def load_main_evaluation_data() -> pd.DataFrame:
    """
    Load evaluation data from evaluation_data/ (Analysis 5 compatible).

    Returns obs_df with columns:
      model_key, display_name, params_B, provider, dilemma_type,
      prompt_type, response, kohlberg_stage, action_endorsed,
      action_category, is_consistent, expected_action
    """
    frames: list[pd.DataFrame] = []
    eval_files = sorted(MAIN_EVAL_DIR.glob("*_evaluation.xlsx"))

    if not eval_files:
        raise FileNotFoundError(
            f"No evaluation files found in {MAIN_EVAL_DIR}. "
            "Ensure evaluation_data/ contains *_evaluation.xlsx files."
        )

    for eval_path in eval_files:
        stem = eval_path.stem.replace("_evaluation", "")
        if stem not in MODEL_META:
            continue

        display_name, params_B, provider = MODEL_META[stem]
        edf = pd.read_excel(eval_path)

        # Check required columns
        required = {"dilemma_type", "kohlberg_stage"}
        if not required.issubset(edf.columns):
            print(f"  [WARN] {eval_path.name} missing required columns — skipping")
            continue

        edf["model_key"] = stem
        edf["display_name"] = display_name
        edf["params_B"] = params_B
        edf["log_params"] = np.log10(params_B)
        edf["provider"] = provider
        edf["data_source"] = "main"

        # Parse action category
        if "action_endorsed" in edf.columns:
            edf["action_category"] = edf.apply(
                lambda x: categorize_action(x["dilemma_type"], str(x.get("action_endorsed", ""))),
                axis=1,
            )
        else:
            edf["action_endorsed"] = ""
            edf["action_category"] = "Ambiguous/Other"

        frames.append(edf)

    df = pd.concat(frames, ignore_index=True)

    # Clean kohlberg_stage
    df["kohlberg_stage"] = pd.to_numeric(df["kohlberg_stage"], errors="coerce")
    df.dropna(subset=["kohlberg_stage"], inplace=True)
    df["kohlberg_stage"] = df["kohlberg_stage"].astype(int)
    df = df[df["kohlberg_stage"].between(1, 6)].copy()

    # Compute consistency (decoupling) metric
    df["expected_action"] = df["kohlberg_stage"].map(EXPECTED_ACTION_BY_STAGE)
    valid_mask = df["action_category"].isin(["Rule-Following", "Rule-Breaking"])
    df["is_consistent"] = np.nan
    df.loc[valid_mask, "is_consistent"] = (
        df.loc[valid_mask, "expected_action"] == df.loc[valid_mask, "action_category"]
    ).astype(float)

    print(
        f"  Loaded {len(df):,} observations from evaluation_data/ "
        f"({df['model_key'].nunique()} models)"
    )

    return df


# ── Load RLHF causal analysis evaluation data ────────────────────────────────

def load_rlhf_evaluation_data() -> pd.DataFrame | None:
    """
    Load scored evaluation data from rlhf_causal_analysis/evaluation/.

    Returns obs_df with similar schema to load_main_evaluation_data(),
    plus pair_id, variant, architecture columns.

    Returns None if no evaluation files exist yet.
    """
    frames: list[pd.DataFrame] = []

    for pair_id in RLHF_PAIR_ORDER:
        pair_cfg = RLHF_PAIR_META[pair_id]

        for variant in ("base", "instruct"):
            path = RLHF_EVAL_DIR / f"{pair_id}_{variant}_evaluation.xlsx"
            if not path.exists():
                continue

            edf = pd.read_excel(path)

            if "kohlberg_stage" not in edf.columns:
                print(f"  [WARN] {path.name} missing 'kohlberg_stage' — skipping")
                continue

            edf["pair_id"] = pair_id
            edf["variant"] = variant
            edf["architecture"] = pair_cfg["architecture"]
            edf["params_B"] = pair_cfg["params_B"]
            edf["model_key"] = f"{pair_id}_{variant}"
            edf["display_name"] = pair_cfg.get(f"{variant}_label", f"{pair_id}_{variant}")
            edf["provider"] = "RLHF_pair"
            edf["data_source"] = "rlhf"

            # Parse action category
            if "action_endorsed" in edf.columns:
                edf["action_category"] = edf.apply(
                    lambda x: categorize_action(
                        x.get("dilemma_type", ""),
                        str(x.get("action_endorsed", "")),
                    ),
                    axis=1,
                )
            else:
                edf["action_endorsed"] = ""
                edf["action_category"] = "Ambiguous/Other"

            frames.append(edf)

    if not frames:
        print("  ⚠️  No RLHF evaluation files found — data not yet collected.")
        return None

    df = pd.concat(frames, ignore_index=True)

    # Clean kohlberg_stage
    df["kohlberg_stage"] = pd.to_numeric(df["kohlberg_stage"], errors="coerce")
    df.dropna(subset=["kohlberg_stage"], inplace=True)
    df["kohlberg_stage"] = df["kohlberg_stage"].astype(int)
    df = df[df["kohlberg_stage"].between(1, 6)].copy()

    # Compute consistency metric
    df["expected_action"] = df["kohlberg_stage"].map(EXPECTED_ACTION_BY_STAGE)
    valid_mask = df["action_category"].isin(["Rule-Following", "Rule-Breaking"])
    df["is_consistent"] = np.nan
    df.loc[valid_mask, "is_consistent"] = (
        df.loc[valid_mask, "expected_action"] == df.loc[valid_mask, "action_category"]
    ).astype(float)

    print(
        f"  Loaded {len(df):,} RLHF observations "
        f"({df['pair_id'].nunique()} pairs × 2 variants)"
    )

    return df


# ── Compute model-level consistency summary ──────────────────────────────────

def compute_model_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate observation-level data to model-level consistency scores.

    Returns one row per model_key with:
      model_key, display_name, params_B, provider,
      n_total, n_valid_actions, n_consistent, consistency_pct,
      mean_stage, modal_stage
    """
    records = []

    for mk, grp in df.groupby("model_key", sort=False):
        meta = grp.iloc[0]
        n_total = len(grp)

        valid = grp[grp["action_category"].isin(["Rule-Following", "Rule-Breaking"])]
        n_valid = len(valid)
        n_consistent = int(valid["is_consistent"].sum()) if n_valid > 0 else 0
        consistency_pct = (n_consistent / n_valid * 100) if n_valid > 0 else np.nan

        records.append({
            "model_key":       mk,
            "display_name":    meta.get("display_name", mk),
            "params_B":        meta.get("params_B", np.nan),
            "provider":        meta.get("provider", ""),
            "data_source":     meta.get("data_source", ""),
            "variant":         meta.get("variant", ""),
            "pair_id":         meta.get("pair_id", ""),
            "n_total":         n_total,
            "n_valid_actions": n_valid,
            "n_consistent":    n_consistent,
            "consistency_pct": round(consistency_pct, 2) if not np.isnan(consistency_pct) else np.nan,
            "mean_stage":      round(grp["kohlberg_stage"].mean(), 2),
            "modal_stage":     int(grp["kohlberg_stage"].mode().iloc[0]) if not grp.empty else np.nan,
        })

    return pd.DataFrame(records)


# ── Unified loader ────────────────────────────────────────────────────────────

def load_data(use_main_data: bool = False) -> tuple[pd.DataFrame, str]:
    """
    Load evaluation data from the appropriate source.

    Parameters
    ----------
    use_main_data : bool
        If True, force use of main project evaluation_data/.
        If False, try RLHF data first, fall back to main.

    Returns
    -------
    (df, source_label) : tuple
        df is the observation-level DataFrame.
        source_label is 'main' or 'rlhf'.
    """
    if use_main_data:
        print("\n  Loading main project evaluation data (--use-main-data)…")
        return load_main_evaluation_data(), "main"

    # Try RLHF first
    print("\n  Attempting to load RLHF causal analysis data…")
    rlhf_df = load_rlhf_evaluation_data()

    if rlhf_df is not None and len(rlhf_df) > 0:
        return rlhf_df, "rlhf"

    # Fall back to main
    print("  Falling back to main project evaluation data…")
    return load_main_evaluation_data(), "main"
