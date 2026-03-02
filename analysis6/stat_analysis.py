"""
stat_analysis.py — TF-IDF analysis, keyword matching, and linguistic style PCA.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from config import CUSTOM_STOP_WORDS, TARGET_KEYWORDS, STAGES

def extract_distinctive_terms(texts: list[str], n_terms: int = 10) -> list[tuple[str, float]]:
    """Runs TF-IDF over a small corpus and extracts top terms by mean TF-IDF weight."""
    if not texts or len(texts) == 0:
        return []
        
    vectorizer = TfidfVectorizer(
        stop_words=CUSTOM_STOP_WORDS,
        max_df=0.90,     # ignore terms in >90% docs
        min_df=2,        # ignore terms in <2 docs
        ngram_range=(1, 2)
    )
    
    try:
        tfidf_mat = vectorizer.fit_transform(texts)
    except ValueError:
        # Happens if vocab is empty after min_df
        return []
        
    feature_names = vectorizer.get_feature_names_out()
    mean_weights = np.asarray(tfidf_mat.mean(axis=0)).flatten()
    
    # Sort indices by descending weight
    top_indices = mean_weights.argsort()[::-1][:n_terms]
    
    return [(feature_names[i], float(mean_weights[i])) for i in top_indices if mean_weights[i] > 0]

def analyze_stage_vocabulary(df: pd.DataFrame) -> dict[int, list[tuple[str, float]]]:
    """Returns top 20 distinctive terms per Kohlberg stage."""
    results = {}
    for stage in STAGES:
        sub = df[df["kohlberg_stage"] == stage]
        if len(sub) > 5:
            results[stage] = extract_distinctive_terms(sub["cleaned_text"].tolist(), n_terms=20)
        else:
            results[stage] = []
    return results
    
def analyze_model_distinctive_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe of the top 5 distinctive terms per model (TF-IDF weighted)."""
    records = []
    for model_name, grp in df.groupby("display_name"):
        terms = extract_distinctive_terms(grp["cleaned_text"].tolist(), n_terms=5)
        
        row = {"display_name": model_name, "provider": grp["provider"].iloc[0]}
        for i in range(5):
            if i < len(terms):
                row[f"term_{i+1}"] = terms[i][0]
                row[f"weight_{i+1}"] = terms[i][1]
            else:
                row[f"term_{i+1}"] = None
                row[f"weight_{i+1}"] = 0.0
                
        records.append(row)
    return pd.DataFrame(records)

def evaluate_target_keyword_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Computes % of responses per model that contain the expected Stage N keywords when assigning Stage N."""
    records = []
    for model_name, grp in df.groupby("display_name"):
        provider = grp["provider"].iloc[0]
        params = grp["params_B"].iloc[0]
        
        for stage in STAGES:
            s_grp = grp[grp["kohlberg_stage"] == stage]
            if len(s_grp) == 0:
                continue
                
            targets = TARGET_KEYWORDS.get(stage, [])
            if not targets:
                continue
                
            # Count how many responses include at least one target keyword
            def _has_target(text):
                return any(t in text for t in targets)
                
            hits = s_grp["cleaned_text"].apply(_has_target).sum()
            pct = (hits / len(s_grp)) * 100
            
            records.append({
                "display_name": model_name,
                "provider": provider,
                "params_B": params,
                "stage": stage,
                "n_responses": len(s_grp),
                "keyword_hit_pct": pct
            })
            
    return pd.DataFrame(records)

def compute_linguistic_pca(df: pd.DataFrame) -> pd.DataFrame:
    """Runs TF-IDF on all models, then PCA to plot linguistic similarity space."""
    # We want one document per model (concatenate all their reasoning)
    models = []
    texts = []
    providers = []
    params = []
    
    for _, grp in df.groupby("display_name"):
        models.append(grp["display_name"].iloc[0])
        providers.append(grp["provider"].iloc[0])
        params.append(grp["params_B"].iloc[0])
        texts.append(" ".join(grp["cleaned_text"].tolist()))
        
    vectorizer = TfidfVectorizer(
        stop_words=CUSTOM_STOP_WORDS,
        max_df=0.95,
        min_df=2,
        max_features=1000,
        ngram_range=(1, 2)
    )
    
    mat = vectorizer.fit_transform(texts).toarray()
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(mat)
    
    # Calculate explained variance
    var_explained = pca.explained_variance_ratio_
    
    res = pd.DataFrame({
        "display_name": models,
        "provider": providers,
        "params_B": params,
        "pca1": coords[:, 0],
        "pca2": coords[:, 1]
    })
    
    return res, var_explained

def find_qualitative_exemplars(df: pd.DataFrame) -> pd.DataFrame:
    """Finds highly representative quotes per model at its modal stage using centroid distance."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    exemplars = []
    
    for mk, grp in df.groupby("model_key"):
        if len(grp) == 0: continue
        
        model_name = grp["display_name"].iloc[0]
        
        # Determine empirical modal stage for this model
        modal_stage = grp["kohlberg_stage"].mode().iloc[0]
        modal_grp = grp[grp["kohlberg_stage"] == modal_stage]
        
        if len(modal_grp) < 3:
            # Fallback if too few at modal stage
            modal_grp = grp
            
        texts = modal_grp["cleaned_text"].tolist()
        if not texts: continue
        
        vec = TfidfVectorizer(stop_words=CUSTOM_STOP_WORDS, max_features=500)
        try:
            mat = vec.fit_transform(texts)
        except ValueError:
            continue
            
        centroid = mat.mean(axis=0)
        sims = cosine_similarity(mat, np.asarray(centroid))
        
        # Get top 3 indices
        top_idx = sims.flatten().argsort()[::-1][:3]
        
        for i, idx in enumerate(top_idx, 1):
            row = modal_grp.iloc[idx]
            exemplars.append({
                "display_name": model_name,
                "kohlberg_stage": row["kohlberg_stage"],
                "rank": i,
                "similarity_score": round(float(sims[idx][0]), 3),
                "dilemma_type": row["dilemma_type"],
                "raw_response": row["response"],
                "extracted_indicators": row["combined_text"],
                "vocab_richness": row["vocab_richness"]
            })
            
    return pd.DataFrame(exemplars)
