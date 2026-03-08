"""
metrics.py — Transition dynamics metrics for Analysis 10.

Computes:
  - Shannon entropy & Gini coefficient per model
  - Transition matrices between consecutive models
  - Stage residence times (how many model-steps each stage is modal)
  - Transition window detection
  - Sequential transition proportion
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from config import STAGES, ACTIVE_STAGES, MAX_ENTROPY


# ── Per-model metrics ──────────────────────────────────────────────────────────

def shannon_entropy(props: dict[int, float]) -> float:
    """H = -Σ(p_i × log₂(p_i)), with 0·log₂(0) := 0."""
    h = 0.0
    for p in props.values():
        if p > 0:
            h -= p * np.log2(p)
    return float(h)


def gini_coefficient(props: dict[int, float]) -> float:
    """G = 1 - Σ(p_i²). G=0 → perfect concentration, G=1 → uniform."""
    return float(1.0 - sum(p ** 2 for p in props.values()))


def compute_model_metrics(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add entropy, gini, and consolidation metrics to model_df.

    model_df must have columns: stage_1 … stage_6, modal_stage, model_order.
    Returns model_df with added columns.
    """
    df = model_df.copy()

    entropies, ginis, norm_entropies = [], [], []
    for _, row in df.iterrows():
        props = {s: float(row.get(f"stage_{s}", 0.0)) for s in STAGES}
        h = shannon_entropy(props)
        g = gini_coefficient(props)
        entropies.append(h)
        ginis.append(g)
        norm_entropies.append(h / MAX_ENTROPY if MAX_ENTROPY > 0 else 0.0)

    df["entropy"]      = entropies
    df["gini"]         = ginis
    df["norm_entropy"] = norm_entropies   # entropy / log₂(6)
    
    # Consistency score: 1 - normalized entropy (perfectly consistent within checkpoint = 1.0)
    df["consistency_score"] = [1.0 - ne for ne in norm_entropies]

    # Consolidation index: entropy < 1.0
    df["is_consolidated"] = (df["entropy"] < 1.0).astype(int)

    return df


# ── Transition matrix ──────────────────────────────────────────────────────────

def build_transition_matrix(model_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Build an aggregate transition matrix by comparing consecutive model pairs.

    For each consecutive pair (model_k, model_k+1), we treat each dilemma as
    an "individual" — the dilemma's stage at model_k transitions to its stage
    at model_k+1. Since we only have aggregate distributions (not dilemma-level
    pairings across models), we use the *outer product* of consecutive
    distributions as a proxy for the transition probabilities.

    T[i, j] = Σ_k (p_{k}^{i} × p_{k+1}^{j}) / (N_pairs)

    where p_{k}^{i} = proportion of model k at stage i.

    Returns:
      T      : (6×6) normalized transition matrix (rows sum to ~1 when weighted)
      labels : stage labels for rows/columns
    """
    stages = ACTIVE_STAGES
    n_stages = len(stages)
    T_accum = np.zeros((n_stages, n_stages), dtype=float)
    n_pairs  = 0

    rows_sorted = model_df.sort_values("model_order")
    prop_matrix = rows_sorted[[f"stage_{s}" for s in stages]].values  # (n_models, n_stages)

    for k in range(len(prop_matrix) - 1):
        p_curr = prop_matrix[k]    # distribution at model k
        p_next = prop_matrix[k+1]  # distribution at model k+1
        # Outer product: expected joint distribution under independence
        T_accum += np.outer(p_curr, p_next)
        n_pairs += 1

    if n_pairs > 0:
        T_accum /= n_pairs

    # Normalize rows so they sum to 1 (conditional prob of next stage given current)
    row_sums = T_accum.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    T_norm = T_accum / row_sums

    labels = [f"S{s}" for s in stages]
    return T_norm, labels


def pairwise_transition_matrices(model_df: pd.DataFrame) -> list[dict]:
    """
    Build transition matrix for each consecutive model pair.

    Returns list of dicts with:
      model_from, model_to, params_from, params_to, matrix, labels
    """
    stages   = ACTIVE_STAGES
    n_stages = len(stages)
    rows_sorted = model_df.sort_values("model_order").reset_index(drop=True)
    results = []

    for k in range(len(rows_sorted) - 1):
        row_k   = rows_sorted.iloc[k]
        row_k1  = rows_sorted.iloc[k + 1]
        p_curr  = np.array([float(row_k.get(f"stage_{s}", 0.0))  for s in stages])
        p_next  = np.array([float(row_k1.get(f"stage_{s}", 0.0)) for s in stages])

        # Outer-product proxy
        T_raw  = np.outer(p_curr, p_next)
        row_s  = T_raw.sum(axis=1, keepdims=True)
        row_s[row_s == 0] = 1
        T_norm = T_raw / row_s

        results.append({
            "model_from":   row_k["model_key"],
            "model_to":     row_k1["model_key"],
            "params_from":  row_k["params_B"],
            "params_to":    row_k1["params_B"],
            "matrix":       T_norm,
            "labels":       [f"S{s}" for s in stages],
        })
    return results


# ── Stage residence times ──────────────────────────────────────────────────────

def compute_residence_times(model_df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series indexed by stage (int) with the count of model-steps
    where that stage is the modal (predominant) stage.
    """
    modal_counts = model_df["modal_stage"].value_counts()
    # Fill in missing stages with 0
    residence = pd.Series(0, index=STAGES, dtype=int)
    for s, c in modal_counts.items():
        if s in STAGES:
            residence[s] = c
    return residence


# ── Transition window detection ────────────────────────────────────────────────

def detect_transition_windows(
    model_df: pd.DataFrame,
    appear_thresh: float = 0.10,
    exit_thresh: float   = 0.30,
) -> list[dict]:
    """
    Detect transition windows between consecutive modal stages.

    A transition from stage i → stage j is detected when:
      - Modal stage changes from i to j between consecutive checkpoints
      - Window start: first checkpoint where stage j > appear_thresh
      - Window end:   last checkpoint where stage i > exit_thresh
      - Window width: number of model-steps in that window

    Returns list of dicts with transition details.
    """
    df = model_df.sort_values("model_order").reset_index(drop=True)
    windows = []

    # Find modal stage change points
    for k in range(len(df) - 1):
        s_from = int(df.loc[k, "modal_stage"])
        s_to   = int(df.loc[k + 1, "modal_stage"])
        if s_from == s_to:
            continue  # no transition

        # Find first model where s_to > appear_thresh (look backward from k+1)
        win_start_idx = k + 1
        for j in range(k, -1, -1):
            p_next = float(df.loc[j, f"stage_{s_to}"] if f"stage_{s_to}" in df.columns else 0)
            if p_next > appear_thresh:
                win_start_idx = j
            else:
                break

        # Find last model where s_from > exit_thresh (look forward from k)
        win_end_idx = k
        for j in range(k + 1, len(df)):
            p_from = float(df.loc[j, f"stage_{s_from}"] if f"stage_{s_from}" in df.columns else 0)
            if p_from > exit_thresh:
                win_end_idx = j
            else:
                break

        window_size = win_end_idx - win_start_idx  # model steps
        is_sequential = (s_to == s_from + 1)
        is_regression  = (s_to < s_from)

        windows.append({
            "from_stage":     s_from,
            "to_stage":       s_to,
            "at_model_idx":   k,
            "at_model":       df.loc[k, "model_key"],
            "to_model":       df.loc[k + 1, "model_key"],
            "win_start_idx":  win_start_idx,
            "win_end_idx":    win_end_idx,
            "window_size":    max(0, window_size),
            "is_sequential":  is_sequential,
            "is_regression":  is_regression,
            "params_at":      float(df.loc[k, "params_B"]),
        })

    return windows


# ── Sequential transition proportion ──────────────────────────────────────────

def sequential_transition_proportion(T: np.ndarray, stages: list[int]) -> float:
    """
    Compute proportion of transitions that are sequential (i → i+1).

    T: normalized transition matrix (n_stages × n_stages)
    stages: ordered list of stages (e.g. [4, 5, 6])
    """
    n = len(stages)
    total = 0.0
    sequential = 0.0
    for i, s in enumerate(stages):
        for j, t in enumerate(stages):
            total += T[i, j]
            if t == s + 1:
                sequential += T[i, j]
    return sequential / total if total > 0 else 0.0


# ── Summary statistics ─────────────────────────────────────────────────────────

def summarize_metrics(model_df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics across all models."""
    entropies = model_df["entropy"].values
    ginis     = model_df["gini"].values
    consistencies = model_df["consistency_score"].values

    return {
        "mean_entropy":         float(np.mean(entropies)),
        "std_entropy":          float(np.std(entropies, ddof=1)),
        "max_entropy":          float(np.max(entropies)),
        "min_entropy":          float(np.min(entropies)),
        "consolidation_index":  float((entropies < 1.0).mean()),
        "mean_gini":            float(np.mean(ginis)),
        "std_gini":             float(np.std(ginis, ddof=1)),
        "mean_consistency":     float(np.mean(consistencies)),
        "std_consistency":      float(np.std(consistencies, ddof=1)),
        "n_models":             len(model_df),
    }

# ── Qualitative Analysis ───────────────────────────────────────────────────────

def extract_qualitative_samples(obs_df: pd.DataFrame, windows: list[dict], max_samples_per_window: int = 1) -> dict[int, list[dict]]:
    """
    Extract sample responses from transition windows to highlight mixed reasoning.
    Returns a dict mapping window index to a list of sample dicts.
    """
    samples = {}
    if "response" not in obs_df.columns:
        return samples

    for i, w in enumerate(windows):
        samples[i] = []
        # A transition window spans from win_start_idx to win_end_idx.
        # We want to extract a response from the "to_stage" during this window's models to show emergence.
        models_in_window = list(range(w["win_start_idx"], w["win_end_idx"] + 1))
        
        # Filter observations that fall in these models and have the target stage
        win_obs = obs_df[obs_df["model_order"].isin(models_in_window) & (obs_df["kohlberg_stage"] == w["to_stage"])]
        
        if not win_obs.empty:
            # Take up to max_samples_per_window
            sample_df = win_obs.head(max_samples_per_window)
            for _, row in sample_df.iterrows():
                samples[i].append({
                    "model": row["model_key"],
                    "dilemma": row.get("dilemma_type", "Unknown"),
                    "stage": row["kohlberg_stage"],
                    "text": str(row["response"])[:500] + "..." if pd.notna(row["response"]) else "No text"
                })
    return samples
