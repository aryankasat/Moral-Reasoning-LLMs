"""
capability_metrics.py — Compute NLP-based capability metrics from LLM responses.

All metrics are derived purely from response text or existing evaluation fields
(no external API calls required).

Per-observation metrics (computed on each row's 'response' column):
  response_length      — whitespace-split token count
  sentence_count       — approximate sentence count (split on . ! ?)
  avg_sentence_length  — response_length / sentence_count
  lexical_diversity    — type-token ratio (unique / total tokens)
  syntactic_complexity — avg_sentence_length × lexical_diversity (composite)
  semantic_density     — proportion of tokens matching an academic/abstract word list
  coherence            — kohlberg_confidence (1–5) as evaluator-rated coherence proxy

Model-level aggregates:
  All metrics averaged across a model's responses → single row per model.
"""

from __future__ import annotations

import re
import string
from typing import List

import numpy as np
import pandas as pd

# ── Academic / abstract vocabulary list (100 curated terms) ───────────────────
# Words associated with abstract, principled, sophisticated moral reasoning.
ACADEMIC_WORDS = frozenset([
    "autonomy", "consequentialist", "deontological", "utilitarian", "ethical",
    "justice", "rights", "obligation", "principle", "universal", "rational",
    "framework", "moral", "normative", "prescriptive", "categorical",
    "imperative", "dignity", "intrinsic", "extrinsic", "social", "contract",
    "welfare", "harm", "virtue", "integrity", "legitimate", "authority",
    "perspective", "impartial", "systemic", "humanitarian", "compassion",
    "fairness", "equity", "inequality", "abstract", "pragmatic", "empirical",
    "inference", "philosophical", "consequentialism", "pluralism", "doctrine",
    "axiom", "hypothesis", "criterion", "cognition", "analysis", "synthesis",
    "evaluation", "application", "coherent", "consistent", "axiological",
    "epistemological", "ontological", "teleological", "foundational",
    "construct", "paradigm", "conceptual", "theoretical", "analogical",
    "contextual", "nuanced", "perspective", "implication", "assumption",
    "complexity", "ambiguity", "relativity", "objectivity", "subjectivity",
    "proposition", "argument", "reasoning", "judgment", "deliberation",
    "consideration", "examination", "interpretation", "assessment",
    "determination", "examination", "reflection", "deliberate", "consider",
    "evaluate", "justify", "substantiate", "articulate", "distinguish",
    "differentiate", "reconcile", "weigh", "balance", "acknowledge",
    "recognize", "perceive", "comprehend", "critique", "challenge",
    "question", "interrogate", "problematize", "contextualize",
])


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on '.', '!', '?'."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def compute_response_metrics(response: str, analysis_time_sec: float) -> dict:
    """
    Compute all per-response capability metrics.

    coherence proxy: analysis_time_sec (time the model spent generating the response).
    Longer processing time is associated with richer, more deliberate reasoning.
    kohlberg_confidence is constant (5) across all models, so it provides no variance.
    """
    coherence_val = float(analysis_time_sec) if (
        isinstance(analysis_time_sec, (int, float)) and not np.isnan(float(analysis_time_sec))
    ) else np.nan
    if not isinstance(response, str) or not response.strip():
        return {
            "response_length":      np.nan,
            "sentence_count":       np.nan,
            "avg_sentence_length":  np.nan,
            "lexical_diversity":    np.nan,
            "syntactic_complexity": np.nan,
            "semantic_density":     np.nan,
            "coherence":            coherence_val,
        }

    tokens    = _tokenize(response)
    sentences = _split_sentences(response)
    n_tokens  = len(tokens)
    n_sents   = max(len(sentences), 1)

    # Core metrics
    response_length     = n_tokens
    sentence_count      = n_sents
    avg_sent_len        = n_tokens / n_sents
    unique_tokens       = len(set(tokens))
    ttr                 = unique_tokens / n_tokens if n_tokens > 0 else 0.0
    syntactic_complexity = avg_sent_len * ttr

    # Semantic density: fraction of tokens matching academic vocab
    acad_count = sum(1 for t in tokens if t in ACADEMIC_WORDS)
    semantic_density = acad_count / n_tokens if n_tokens > 0 else 0.0

    # Coherence proxy: analysis_time_sec
    coherence = coherence_val

    return {
        "response_length":      response_length,
        "sentence_count":       sentence_count,
        "avg_sentence_length":  avg_sent_len,
        "lexical_diversity":    ttr,
        "syntactic_complexity": syntactic_complexity,
        "semantic_density":     semantic_density,
        "coherence":            coherence,
    }


def compute_observation_capabilities(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-observation capability metric columns to raw_df.
    Returns a copy of raw_df with new columns.
    """
    raw_df = raw_df.copy()

    metric_rows = []
    for _, row in raw_df.iterrows():
        ats = row.get("analysis_time_sec", np.nan)
        if pd.isna(ats):
            ats = np.nan
        else:
            ats = float(ats)
        metrics = compute_response_metrics(str(row.get("response", "")), ats)
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)
    for col in metrics_df.columns:
        raw_df[col] = metrics_df[col].values

    return raw_df


def compute_model_capabilities(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-observation capability metrics to model level.

    Returns a DataFrame with one row per model_key and columns for
    each capability metric (mean across observations).
    """
    # First compute per-observation metrics
    obs_df = compute_observation_capabilities(raw_df)

    metric_cols = [
        "response_length", "sentence_count", "avg_sentence_length",
        "lexical_diversity", "syntactic_complexity", "semantic_density",
        "coherence",
    ]

    agg = (
        obs_df.groupby("model_key")[metric_cols]
        .mean()
        .reset_index()
    )
    agg.columns = ["model_key"] + metric_cols
    return agg
