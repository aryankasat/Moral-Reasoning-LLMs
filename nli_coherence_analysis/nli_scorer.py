"""
nli_scorer.py — DeBERTa-v3-large NLI scoring engine.

Extracts justification-action pairs from LLM moral dilemma responses and
scores entailment using a cross-encoder NLI model.  This provides a
framework-independent coherence measure: P(entailment | justification → action).

Pipeline:
  1. Extract justification text from LLM response
  2. Build action hypothesis from action_endorsed field
  3. Score (justification, hypothesis) with DeBERTa-v3-large NLI
"""

from __future__ import annotations

import re
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

from config import (
    NLI_MODEL_ID,
    NLI_LABELS,
    NLI_MAX_LENGTH,
    NLI_BATCH_SIZE,
    NLI_DEVICE,
    ACTION_HYPOTHESIS_TEMPLATES,
    ACTION_KEYWORDS,
    SCORES_DIR,
)


# ── Device selection ──────────────────────────────────────────────────────────

def _resolve_device(device: str = "auto") -> str:
    """Resolve 'auto' to the best available device."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Justification extraction ─────────────────────────────────────────────────

def extract_justification(response: str, max_chars: int = 1500) -> str:
    """
    Extract the reasoning/justification portion from an LLM moral dilemma response.

    Strategy:
      1. Remove direct repetition of the dilemma prompt (if present)
      2. Look for moral reasoning content — sentences containing ethical language
      3. Truncate to max_chars to stay within NLI model limits

    Returns a cleaned justification string suitable as an NLI premise.
    """
    if not isinstance(response, str) or not response.strip():
        return ""

    text = response.strip()

    # Remove common prefixes that models add
    for prefix in [
        r"^(?:As a moral philosopher,?\s*)",
        r"^(?:Let me think.*?step by step\.?\s*)",
        r"^(?:SYSTEM:.*?\n\s*)",
    ]:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)

    # Extract the most reasoning-dense portion
    # Split into sentences and filter for ones with moral reasoning content
    moral_keywords = [
        "moral", "ethic", "right", "wrong", "should", "ought", "duty",
        "obligation", "principle", "justice", "fair", "value", "dignity",
        "autonomy", "harm", "consequence", "utilitarian", "deontolog",
        "virtue", "compassion", "responsib", "rights", "law", "rule",
        "society", "social", "contract", "universal", "categorical",
        "greatest good", "well-being", "welfare", "suffering", "life",
    ]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    reasoning_sentences = []

    for sent in sentences:
        sent_lower = sent.lower()
        if any(kw in sent_lower for kw in moral_keywords):
            reasoning_sentences.append(sent)

    # If we found reasoning sentences, use them; otherwise use the full text
    if reasoning_sentences:
        justification = " ".join(reasoning_sentences)
    else:
        justification = text

    # Truncate to max_chars
    if len(justification) > max_chars:
        # Try to truncate at sentence boundary
        truncated = justification[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.5:
            justification = truncated[: last_period + 1]
        else:
            justification = truncated + "…"

    return justification.strip()


# ── Action hypothesis construction ────────────────────────────────────────────

def _categorize_action_text(dilemma: str, action_text: str) -> str:
    """
    Classify action_endorsed text as 'rule_breaking', 'rule_following', or 'unknown'.
    Mirrors Analysis 5's _categorize_action logic.
    """
    if not isinstance(action_text, str):
        return "unknown"

    t = action_text.lower().strip()
    keywords = ACTION_KEYWORDS.get(dilemma, {})

    breaking_pos = keywords.get("breaking_pos", [])
    breaking_neg = keywords.get("breaking_neg", [])
    following_pos = keywords.get("following_pos", [])

    # Check rule-breaking
    for kw in breaking_pos:
        if kw in t:
            # Check negation
            if not any(neg in t for neg in breaking_neg):
                return "rule_breaking"

    # Check rule-following
    for kw in following_pos:
        if kw in t:
            return "rule_following"

    return "unknown"


def build_action_hypothesis(
    dilemma: str,
    action_endorsed: str,
    action_category: str | None = None,
) -> str:
    """
    Convert an action_endorsed value into a natural-language hypothesis
    for NLI scoring.

    Uses dilemma-specific templates when action can be mapped to
    rule-following/breaking; falls back to a generic template with the
    raw action text.

    Parameters
    ----------
    dilemma : str
        Dilemma name (e.g., "HEINZ_DILEMMA").
    action_endorsed : str
        The action_endorsed field from evaluation data.
    action_category : str, optional
        Pre-computed action category ("Rule-Breaking" / "Rule-Following").
        If not provided, will be inferred from action text.

    Returns
    -------
    str
        Natural-language hypothesis suitable for NLI.
    """
    templates = ACTION_HYPOTHESIS_TEMPLATES.get(dilemma, {})

    # Determine action category if not provided
    if action_category is None:
        cat = _categorize_action_text(dilemma, str(action_endorsed))
    else:
        cat = action_category.lower().replace("-", "_").replace(" ", "_")
        if cat not in ("rule_breaking", "rule_following"):
            cat = _categorize_action_text(dilemma, str(action_endorsed))

    # Use template
    if cat in templates:
        return templates[cat]

    # Fallback: generic template with raw action text
    default_template = templates.get(
        "default",
        "Therefore, the morally correct action is to {action}.",
    )
    action_str = str(action_endorsed).strip()
    if not action_str or action_str.lower() in ("nan", "none", ""):
        return "Therefore, the morally correct action follows from the reasoning above."

    return default_template.format(action=action_str)


# ── DeBERTa NLI model loader ─────────────────────────────────────────────────

_nli_model = None
_nli_tokenizer = None


def _load_nli_model() -> tuple:
    """
    Lazy-load the NLI cross-encoder model and tokenizer.
    Caches globally to avoid reloading.
    """
    global _nli_model, _nli_tokenizer

    if _nli_model is not None:
        return _nli_model, _nli_tokenizer

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"  Loading NLI model: {NLI_MODEL_ID}…")
    device = _resolve_device(NLI_DEVICE)
    print(f"  Device: {device}")

    _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID)
    _nli_model.eval()

    if device != "cpu":
        _nli_model = _nli_model.to(device)

    n_params = sum(p.numel() for p in _nli_model.parameters()) / 1e6
    print(f"  Model loaded: {n_params:.0f}M params on {device}")

    return _nli_model, _nli_tokenizer


# ── Single-pair NLI scoring ──────────────────────────────────────────────────

def score_nli_pair(
    premise: str,
    hypothesis: str,
) -> dict[str, float]:
    """
    Score a single (premise, hypothesis) pair using DeBERTa-v3-large NLI.

    Returns dict with keys: entailment, contradiction, neutral (probabilities).
    """
    model, tokenizer = _load_nli_model()
    device = next(model.parameters()).device

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        max_length=NLI_MAX_LENGTH,
        truncation=True,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    return {label: float(probs[i]) for i, label in enumerate(NLI_LABELS)}


# ── Batch NLI scoring ────────────────────────────────────────────────────────

def score_nli_batch(
    premises: list[str],
    hypotheses: list[str],
    batch_size: int = NLI_BATCH_SIZE,
) -> list[dict[str, float]]:
    """
    Score multiple (premise, hypothesis) pairs in batches.

    Returns list of dicts with keys: entailment, contradiction, neutral.
    """
    assert len(premises) == len(hypotheses), "premises and hypotheses must have same length"

    model, tokenizer = _load_nli_model()
    device = next(model.parameters()).device

    all_results = []

    for start in tqdm(range(0, len(premises), batch_size), desc="NLI scoring"):
        end = min(start + batch_size, len(premises))
        batch_premises = premises[start:end]
        batch_hypotheses = hypotheses[start:end]

        inputs = tokenizer(
            batch_premises,
            batch_hypotheses,
            return_tensors="pt",
            max_length=NLI_MAX_LENGTH,
            truncation=True,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for i in range(len(batch_premises)):
            all_results.append(
                {label: float(probs[i][j]) for j, label in enumerate(NLI_LABELS)}
            )

    return all_results


# ── Score an entire DataFrame ─────────────────────────────────────────────────

def score_dataframe(
    df: pd.DataFrame,
    response_col: str = "response",
    action_col: str = "action_endorsed",
    dilemma_col: str = "dilemma_type",
    action_cat_col: str | None = "action_category",
    batch_size: int = NLI_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Score all rows in a DataFrame using NLI entailment.

    For each row:
      - premise = extract_justification(response)
      - hypothesis = build_action_hypothesis(dilemma, action_endorsed)

    Adds columns: nli_entailment, nli_contradiction, nli_neutral,
                  nli_justification, nli_hypothesis

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with response, action_endorsed, dilemma_type columns.
    response_col : str
        Column containing the full model response text.
    action_col : str
        Column containing the action_endorsed text.
    dilemma_col : str
        Column containing the dilemma type identifier.
    action_cat_col : str, optional
        Column containing pre-computed action category. If None, will be inferred.
    batch_size : int
        Batch size for NLI inference.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with NLI score columns appended.
    """
    result_df = df.copy()

    print(f"\n  Extracting justifications from {len(df)} responses…")
    justifications = []
    hypotheses = []

    for _, row in df.iterrows():
        # Extract justification (premise)
        justification = extract_justification(str(row.get(response_col, "")))
        justifications.append(justification)

        # Build action hypothesis
        dilemma = str(row.get(dilemma_col, ""))
        action = str(row.get(action_col, ""))
        action_cat = str(row.get(action_cat_col, "")) if action_cat_col and action_cat_col in df.columns else None

        hypothesis = build_action_hypothesis(dilemma, action, action_cat)
        hypotheses.append(hypothesis)

    # Store extracted text
    result_df["nli_justification"] = justifications
    result_df["nli_hypothesis"] = hypotheses

    # Filter out empty justifications
    valid_mask = [bool(j.strip()) for j in justifications]
    n_valid = sum(valid_mask)
    n_empty = len(valid_mask) - n_valid

    if n_empty > 0:
        print(f"  ⚠️  {n_empty} rows have empty justifications (will score as NaN)")

    # Score valid pairs
    valid_premises = [j for j, v in zip(justifications, valid_mask) if v]
    valid_hypotheses = [h for h, v in zip(hypotheses, valid_mask) if v]

    if valid_premises:
        print(f"  Scoring {n_valid} pairs with {NLI_MODEL_ID}…")
        nli_results = score_nli_batch(valid_premises, valid_hypotheses, batch_size)
    else:
        nli_results = []
        print("  ⚠️  No valid pairs to score")

    # Map results back
    entailment_scores = []
    contradiction_scores = []
    neutral_scores = []
    valid_idx = 0

    for v in valid_mask:
        if v and valid_idx < len(nli_results):
            result = nli_results[valid_idx]
            entailment_scores.append(result["entailment"])
            contradiction_scores.append(result["contradiction"])
            neutral_scores.append(result["neutral"])
            valid_idx += 1
        else:
            entailment_scores.append(np.nan)
            contradiction_scores.append(np.nan)
            neutral_scores.append(np.nan)

    result_df["nli_entailment"] = entailment_scores
    result_df["nli_contradiction"] = contradiction_scores
    result_df["nli_neutral"] = neutral_scores

    # Summary stats
    valid_ent = [s for s in entailment_scores if not np.isnan(s)]
    if valid_ent:
        print(f"\n  NLI Entailment scores: mean={np.mean(valid_ent):.3f}  "
              f"median={np.median(valid_ent):.3f}  "
              f"std={np.std(valid_ent):.3f}  "
              f"range=[{np.min(valid_ent):.3f}, {np.max(valid_ent):.3f}]")

    return result_df


# ── Save / Load scored data ──────────────────────────────────────────────────

def save_scores(df: pd.DataFrame, filename: str = "nli_scores.xlsx") -> Path:
    """Save scored DataFrame to SCORES_DIR."""
    out_path = SCORES_DIR / filename
    df.to_excel(out_path, index=False)
    print(f"  ✅ Saved NLI scores → {out_path.name} ({len(df)} rows)")
    return out_path


def load_scores(filename: str = "nli_scores.xlsx") -> pd.DataFrame | None:
    """Load previously computed NLI scores, if they exist."""
    path = SCORES_DIR / filename
    if path.exists():
        df = pd.read_excel(path)
        print(f"  Loaded existing NLI scores from {path.name} ({len(df)} rows)")
        return df
    return None
