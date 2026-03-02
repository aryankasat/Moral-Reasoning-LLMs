"""
data_loader.py — Loads responses and tokenizes reasoning text for TF-IDF.
"""

from __future__ import annotations
import ast
import re
import numpy as np
import pandas as pd
from config import DATA_DIR, EVAL_DIR, MODEL_META, CUSTOM_STOP_WORDS


def clean_text(text: str) -> str:
    """Basic NLP cleaning: lowercase and strip punctuation"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_prepare_text() -> pd.DataFrame:
    """
    Loads evaluation data and prepares 'reasoning_corpus' field.
    """
    frames = []
    eval_files = sorted(EVAL_DIR.glob("*_evaluation.xlsx"))

    for eval_path in eval_files:
        stem = eval_path.stem.replace("_evaluation", "")
        if stem not in MODEL_META:
            continue

        display_name, params_B, provider = MODEL_META[stem]
        edf = pd.read_excel(eval_path)
        
        required = {"dilemma_type", "kohlberg_stage", "kohlberg_reasoning", "key_indicators"}
        if not required.issubset(edf.columns):
            print(f"  [WARN] {eval_path.name} missing required textual columns.")
            continue

        edf["model_key"] = stem
        edf["display_name"] = display_name
        edf["params_B"] = params_B
        edf["log_params"] = np.log10(params_B)
        edf["provider"] = provider
        
        # Parse 'key_indicators' list from string if it exists
        def _parse_indicators(x):
            try:
                res = ast.literal_eval(str(x))
                if isinstance(res, list): return " ".join([str(i) for i in res])
                return str(x)
            except:
                return str(x)
                
        edf["indicators_text"] = edf["key_indicators"].apply(_parse_indicators)
        
        # Combine reasoning and indicators
        edf["combined_text"] = edf["kohlberg_reasoning"].fillna("") + " " + edf["indicators_text"].fillna("")
        edf["cleaned_text"] = edf["combined_text"].apply(clean_text)
        
        # Vocabulary richness proxy: Count unique words mapping to length > 3 and not stopwords
        def _count_unique_moral_tokens(txt):
            words = set(txt.split())
            vocab = [w for w in words if len(w) > 3 and w not in CUSTOM_STOP_WORDS]
            return len(vocab)
            
        edf["vocab_richness"] = edf["cleaned_text"].apply(_count_unique_moral_tokens)

        keep_cols = [
            "model_key", "display_name", "params_B", "log_params", "provider",
            "dilemma_type", "kohlberg_stage", "combined_text", "cleaned_text", "vocab_richness",
            "response" # Need raw response for qualitative review
        ]
        frames.append(edf[keep_cols])

    df = pd.concat(frames, ignore_index=True)
    df["kohlberg_stage"] = pd.to_numeric(df["kohlberg_stage"], errors="coerce")
    df.dropna(subset=["kohlberg_stage"], inplace=True)
    df["kohlberg_stage"] = df["kohlberg_stage"].astype(int)

    print(f"Loaded {len(df):,} observations for textual analysis.")
    
    return df
