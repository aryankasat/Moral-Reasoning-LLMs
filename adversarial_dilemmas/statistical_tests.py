import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import kruskal
import json

def calculate_cronbach_alpha(df):
    """
    df should be a subjects x items dataframe.
    """
    item_variances = df.var(axis=0, ddof=1)
    total_variance = df.sum(axis=1).var(ddof=1)
    
    n_items = df.shape[1]
    
    if total_variance == 0:
        return 0.0
        
    alpha = (n_items / (n_items - 1)) * (1 - (item_variances.sum() / total_variance))
    return alpha

def perform_statistical_tests():
    eval_dir = "../evaluation_data"
    eval_files = glob.glob(f"{eval_dir}/*_evaluation.xlsx")
    
    results = []
    
    # For Cronbach's Alpha: average stage per dilemma per model
    # To build a model x dilemma matrix
    alpha_data = []
    
    for eval_path in eval_files:
        model_name = os.path.basename(eval_path).replace("_evaluation.xlsx", "")
        df = pd.read_excel(eval_path)
        
        if 'kohlberg_stage' not in df.columns or 'dilemma_type' not in df.columns:
            continue
            
        df['kohlberg_stage'] = pd.to_numeric(df['kohlberg_stage'], errors='coerce')
        df = df.dropna(subset=['kohlberg_stage'])
        
        # Group by dilemma
        groups = [group['kohlberg_stage'].values for name, group in df.groupby('dilemma_type')]
        
        # We need at least 2 groups and some variance to run Kruskal-Wallis
        if len(groups) > 1 and len(df) > len(groups):
            # Check if all values are identical (Kruskal-Wallis will fail or return nan/zero variance)
            if df['kohlberg_stage'].nunique() == 1:
                # Perfect consistency
                h_stat = 0.0
                p_val = 1.0 # 1.0 means we fail to reject the null hypothesis (which is great, it's coherent)
                is_coherent = True
            else:
                h_stat, p_val = kruskal(*groups)
                is_coherent = bool(p_val >= 0.05)
        else:
            h_stat, p_val = np.nan, np.nan
            is_coherent = False
            
        results.append({
            "model_name": model_name,
            "h_statistic": round(float(h_stat), 4),
            "p_value": float(p_val),
            "is_statistically_coherent": is_coherent
        })
        
        # Aggregate for Cronbach's Alpha
        dilemma_means = df.groupby('dilemma_type')['kohlberg_stage'].mean().to_dict()
        dilemma_means['model_name'] = model_name
        alpha_data.append(dilemma_means)

    # Save Kruskal-Wallis Results
    res_df = pd.DataFrame(results).sort_values(by="p_value", ascending=False)
    os.makedirs("results", exist_ok=True)
    res_df.to_csv("results/statistical_results.csv", index=False)
    print(f"Saved Kruskal-Wallis test results to results/statistical_results.csv")
    
    # Calculate Cronbach's Alpha
    alpha_df = pd.DataFrame(alpha_data)
    alpha_df = alpha_df.set_index('model_name')
    # Drop any dilemmas that don't have responses from all models
    alpha_df = alpha_df.dropna(axis=1) 
    
    alpha_score = calculate_cronbach_alpha(alpha_df)
    alpha_summary = {
        "cronbach_alpha": float(alpha_score),
        "n_models": alpha_df.shape[0],
        "n_dilemmas": alpha_df.shape[1],
        "interpretation": "Excellent" if alpha_score >= 0.9 else 
                          "Good" if alpha_score >= 0.8 else 
                          "Acceptable" if alpha_score >= 0.7 else 
                          "Questionable" if alpha_score >= 0.6 else 
                          "Poor" if alpha_score >= 0.5 else "Unacceptable"
    }
    
    with open("results/cronbach_alpha.json", "w") as f:
        json.dump(alpha_summary, f, indent=4)
        
    print(f"Cronbach's Alpha: {alpha_score:.3f} ({alpha_summary['interpretation']})")

if __name__ == "__main__":
    perform_statistical_tests()
