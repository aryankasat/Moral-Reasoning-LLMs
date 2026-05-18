import os
import glob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_lexical_coherence(responses):
    """
    Computes lexical coherence as the mean pairwise cosine similarity 
    of TF-IDF vectors of the model's responses across all dilemmas.
    """
    if len(responses) < 2:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(responses)
        sim_matrix = cosine_similarity(tfidf_matrix)
        
        # Extract the upper triangle (excluding the diagonal)
        upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
        mean_sim = np.mean(sim_matrix[upper_tri_indices])
        return float(mean_sim)
    except Exception as e:
        print(f"Error computing lexical coherence: {e}")
        return 0.0

def compute_stage_coherence(stages):
    """
    Computes stage coherence based on the standard deviation of Kohlberg stages.
    Higher standard deviation means lower coherence.
    We convert it to a score between 0 and 1 using: 1 / (1 + std)
    """
    valid_stages = pd.to_numeric(stages, errors='coerce').dropna()
    if len(valid_stages) < 2:
        return 0.0
    stage_std = np.std(valid_stages)
    return float(1.0 / (1.0 + stage_std))

def score_cross_pair_coherence():
    data_dir = "../data"
    eval_dir = "../evaluation_data"
    
    data_files = glob.glob(f"{data_dir}/*.xlsx")
    results = []
    
    for df_path in data_files:
        model_name = os.path.basename(df_path).replace(".xlsx", "")
        eval_path = os.path.join(eval_dir, f"{model_name}_evaluation.xlsx")
        
        if not os.path.exists(eval_path):
            print(f"Warning: Evaluation data missing for {model_name}, skipping stage coherence.")
            continue
            
        # 1. Lexical Coherence
        df_data = pd.read_excel(df_path)
        responses = df_data['response'].dropna().astype(str).tolist()
        lexical_coh = compute_lexical_coherence(responses)
        
        # 2. Stage Coherence
        df_eval = pd.read_excel(eval_path)
        if 'kohlberg_stage' in df_eval.columns:
            stages = df_eval['kohlberg_stage']
            stage_coh = compute_stage_coherence(stages)
        else:
            stage_coh = 0.0
            
        # 3. Overall Coherence (Normalized mapping to 0-10 scale for easier interpretability)
        # Lexical coh is usually low (0.1 to 0.4). Stage coh is usually (0.5 to 1.0).
        # We will scale them slightly to make a neat 0-10 composite score.
        normalized_lexical = min(lexical_coh * 2.5, 1.0) # Scale up lexical overlap
        composite_score = ((normalized_lexical + stage_coh) / 2.0) * 10.0
        
        results.append({
            "model_name": model_name,
            "lexical_coherence": round(lexical_coh, 3),
            "stage_coherence": round(stage_coh, 3),
            "overall_score": round(composite_score, 2)
        })
        print(f"[{model_name}] Lexical: {lexical_coh:.3f} | Stage: {stage_coh:.3f} | Overall: {composite_score:.2f}/10")
        
    res_df = pd.DataFrame(results)
    res_df.to_csv("results/coherence_scores.csv", index=False)
    print(f"\nSaved {len(res_df)} scores to results/coherence_scores.csv")

if __name__ == "__main__":
    score_cross_pair_coherence()
