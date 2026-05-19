import os
import glob
import pandas as pd
import numpy as np
import json
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pingouin as pg

def calculate_normalized_entropy(stages, base=None):
    """ Computes the normalized Shannon Entropy of the stage distribution. """
    value_counts = pd.Series(stages).value_counts(normalize=True, dropna=True)
    if len(value_counts) <= 1:
        return 0.0
    ent = entropy(value_counts, base=base)
    # Normalize by max possible entropy for the number of unique classes observed
    # or max classes (e.g. 6 Kohlberg stages). Max entropy = ln(6).
    max_ent = np.log(6.0) if base is None else np.log(6.0) / np.log(base)
    return float(ent / max_ent)

def calculate_icc_coherence():
    eval_dir = "../evaluation_data"
    data_dir = "../data"
    eval_files = glob.glob(f"{eval_dir}/*_evaluation.xlsx")

    records = []
    for eval_path in eval_files:
        model_name = os.path.basename(eval_path).replace("_evaluation.xlsx", "")
        df_eval = pd.read_excel(eval_path)
        
        if 'kohlberg_stage' not in df_eval.columns:
            continue
            
        try:
            df_data = pd.read_excel(f"{data_dir}/{model_name}.xlsx")
            df_eval['prompt_type'] = df_data['prompt_type']
        except Exception:
            continue
        
        for idx, row in df_eval.iterrows():
            try:
                stage = float(row['kohlberg_stage'])
                if not np.isnan(stage):
                    records.append({
                        'model': model_name,
                        'context': f"{row['dilemma_type']}_{row['prompt_type']}",
                        'stage': stage
                    })
            except Exception:
                pass

    df = pd.DataFrame(records)
    if len(df) == 0:
        return
    
    # Calculate ICC
    icc_res = pg.intraclass_corr(data=df, targets='model', raters='context', ratings='stage')
    
    # We want ICC2: Two-way random effects, absolute agreement
    icc2_row = icc_res[icc_res['Type'] == 'ICC2'].iloc[0]
    icc2_val = icc2_row['ICC']
    
    # We also want variance components. 
    # Pingouin does not output them directly in intraclass_corr, so we run ANOVA to get MS.
    anova = pg.anova(dv='stage', between=['model', 'context'], data=df)
    
    ms_model = anova.loc[anova['Source'] == 'model', 'MS'].values[0]
    ms_context = anova.loc[anova['Source'] == 'context', 'MS'].values[0]
    ms_error = anova.loc[anova['Source'] == 'model * context', 'MS'].values[0]
    
    k = df['context'].nunique()
    n = df['model'].nunique()
    
    var_error = ms_error
    var_model = max(0, (ms_model - ms_error) / k)
    var_context = max(0, (ms_context - ms_error) / n)
    
    total_var = var_model + var_context + var_error
    
    results = {
        "ICC2_Cross_Pair_Coherence": round(icc2_val, 4),
        "Interpretation": "Excellent" if icc2_val >= 0.9 else "Good" if icc2_val >= 0.75 else "Moderate" if icc2_val >= 0.5 else "Poor",
        "Variance_Components": {
            "Model_Identity_Variance": round(var_model, 4),
            "Contextual_Pairing_Variance": round(var_context, 4),
            "Interaction_Error_Variance": round(var_error, 4)
        },
        "Variance_Proportions": {
            "Model_Identity_Pct": round((var_model / total_var * 100) if total_var > 0 else 0, 2),
            "Contextual_Pairing_Pct": round((var_context / total_var * 100) if total_var > 0 else 0, 2),
            "Interaction_Error_Pct": round((var_error / total_var * 100) if total_var > 0 else 0, 2)
        }
    }
    
    with open("results/icc_coherence_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"ICC Cross-Pair Coherence calculated: {icc2_val:.3f} ({results['Interpretation']})")

def perform_complex_statistical_tests():
    eval_dir = "../evaluation_data"
    eval_files = glob.glob(f"{eval_dir}/*_evaluation.xlsx")
    
    entropy_results = []
    matrix_data = []
    
    for eval_path in eval_files:
        model_name = os.path.basename(eval_path).replace("_evaluation.xlsx", "")
        df = pd.read_excel(eval_path)
        
        if 'kohlberg_stage' not in df.columns or 'dilemma_type' not in df.columns:
            continue
            
        df['kohlberg_stage'] = pd.to_numeric(df['kohlberg_stage'], errors='coerce')
        df = df.dropna(subset=['kohlberg_stage'])
        
        if len(df) == 0:
            continue
            
        # 1. Shannon Entropy
        norm_entropy = calculate_normalized_entropy(df['kohlberg_stage'].values)
        
        entropy_results.append({
            "model_name": model_name,
            "normalized_entropy": round(norm_entropy, 4)
        })
        
        # Prepare data for PCA and CITC (Aggregate by dilemma)
        dilemma_means = df.groupby('dilemma_type')['kohlberg_stage'].mean().to_dict()
        dilemma_means['model_name'] = model_name
        matrix_data.append(dilemma_means)

    # Save Entropy Results
    entropy_df = pd.DataFrame(entropy_results)
    
    # 2. Unidimensionality via PCA
    matrix_df = pd.DataFrame(matrix_data).set_index('model_name')
    # Drop columns that have missing data across models
    matrix_df = matrix_df.dropna(axis=1)
    
    # Standardize data for PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(matrix_df)
    
    pca = PCA()
    pca.fit(scaled_data)
    
    pca_results = pd.DataFrame({
        "principal_component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)
    })
    
    pca_results.to_csv("results/pca_variance.csv", index=False)
    
    # 3. Corrected Item-Total Correlation (CITC)
    citc_results = []
    for col in matrix_df.columns:
        # Sum of all other items
        other_items = matrix_df.drop(columns=[col]).sum(axis=1)
        # Correlation between this item and the sum of others
        corr = matrix_df[col].corr(other_items)
        citc_results.append({
            "dilemma_type": col,
            "citc": round(corr, 4)
        })
        
    citc_df = pd.DataFrame(citc_results).sort_values('citc', ascending=True)
    citc_df.to_csv("results/citc_results.csv", index=False)
    
    # Merge Entropy with existing Kruskal-Wallis stats if available
    if os.path.exists("results/statistical_results.csv"):
        kruskal_df = pd.read_csv("results/statistical_results.csv")
        merged_df = pd.merge(kruskal_df, entropy_df, on="model_name", how="inner")
        merged_df.to_csv("results/complex_statistical_results.csv", index=False)
        print("Saved results/complex_statistical_results.csv (Merged with Kruskal-Wallis)")
    else:
        entropy_df.to_csv("results/complex_statistical_results.csv", index=False)
        print("Saved results/complex_statistical_results.csv")

    print("Saved PCA and CITC results.")
    
    calculate_icc_coherence()
    
if __name__ == "__main__":
    perform_complex_statistical_tests()
