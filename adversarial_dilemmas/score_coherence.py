import os
import glob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_lexical_coherence(responses):
    """
    Computes lexical coherence as the mean pairwise cosine similarity 
    of TF-IDF vectors of the model's responses.
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
            print(f"Warning: Evaluation data missing for {model_name}, skipping.")
            continue
            
        df_data = pd.read_excel(df_path)
        df_eval = pd.read_excel(eval_path)
        
        # Align evaluation data into the main dataframe
        if 'kohlberg_stage' in df_eval.columns:
            df_data['kohlberg_stage'] = df_eval['kohlberg_stage']
        else:
            df_data['kohlberg_stage'] = np.nan
            
        # 1. Dilemma-wise Coherence
        dilemma_lexical = []
        dilemma_stage = []
        for _, group in df_data.groupby('dilemma_type'):
            lex = compute_lexical_coherence(group['response'].dropna().astype(str).tolist())
            dilemma_lexical.append(lex)
            if not group['kohlberg_stage'].isna().all():
                st = compute_stage_coherence(group['kohlberg_stage'])
                dilemma_stage.append(st)
                
        mean_dilemma_lexical = np.mean(dilemma_lexical) if dilemma_lexical else 0.0
        mean_dilemma_stage = np.mean(dilemma_stage) if dilemma_stage else 0.0

        # 2. Prompt-wise Coherence
        prompt_lexical = []
        prompt_stage = []
        for _, group in df_data.groupby('prompt_type'):
            lex = compute_lexical_coherence(group['response'].dropna().astype(str).tolist())
            prompt_lexical.append(lex)
            if not group['kohlberg_stage'].isna().all():
                st = compute_stage_coherence(group['kohlberg_stage'])
                prompt_stage.append(st)
                
        mean_prompt_lexical = np.mean(prompt_lexical) if prompt_lexical else 0.0
        mean_prompt_stage = np.mean(prompt_stage) if prompt_stage else 0.0
        
        # 3. Overall Coherence (Normalized mapping to 0-10 scale for easier interpretability)
        # Dilemma-wise lexical coherence is highly representative. 
        # We scale it slightly (max ~0.8) and average with stage coherence (max 1.0).
        normalized_lexical = min(mean_dilemma_lexical * 1.5, 1.0)
        composite_score = ((normalized_lexical + mean_dilemma_stage) / 2.0) * 10.0
        
        results.append({
            "model_name": model_name,
            "dilemma_lexical_coherence": round(mean_dilemma_lexical, 3),
            "dilemma_stage_coherence": round(mean_dilemma_stage, 3),
            "prompt_lexical_coherence": round(mean_prompt_lexical, 3),
            "prompt_stage_coherence": round(mean_prompt_stage, 3),
            "overall_score": round(composite_score, 2)
        })
        print(f"[{model_name}] Dilemma Lexical: {mean_dilemma_lexical:.3f} | Dilemma Stage: {mean_dilemma_stage:.3f} | Overall: {composite_score:.2f}/10")
        
    res_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    res_df.to_csv("results/coherence_scores.csv", index=False)
    print(f"\nSaved {len(res_df)} scores to results/coherence_scores.csv")

if __name__ == "__main__":
    score_cross_pair_coherence()
