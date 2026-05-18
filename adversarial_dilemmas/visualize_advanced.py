import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text

def create_advanced_visualizations():
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    
    eval_dir = "../evaluation_data"
    eval_files = glob.glob(f"{eval_dir}/*_evaluation.xlsx")
    
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
            
        dilemma_means = df.groupby('dilemma_type')['kohlberg_stage'].mean().to_dict()
        dilemma_means['model_name'] = model_name
        matrix_data.append(dilemma_means)
        
    matrix_df = pd.DataFrame(matrix_data).set_index('model_name')
    matrix_df = matrix_df.dropna(axis=1) # Clean matrix
    
    if len(matrix_df) < 3:
        print("Not enough data for advanced plotting.")
        return
        
    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(matrix_df)
    
    # ---------------------------------------------------------
    # 1. Hierarchical Clustering Correlation Heatmap
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 10))
    # Calculate pairwise correlation between models
    model_corr = pd.DataFrame(scaled_data.T, columns=matrix_df.index).corr()
    
    clustermap = sns.clustermap(
        model_corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        linewidths=1.5,
        linecolor='white',
        figsize=(14, 12),
        cbar_kws={'label': 'Pearson Correlation'}
    )
    
    clustermap.ax_heatmap.set_title("Hierarchical Clustering of LLM Moral Frameworks", fontsize=18, weight='bold', pad=30)
    plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0)
    
    clustermap.savefig("results/fig5_model_clustermap.png", dpi=300, bbox_inches='tight')
    print("Generated results/fig5_model_clustermap.png")
    
    # ---------------------------------------------------------
    # 2. PCA Biplot (Score + Loading Plot)
    # ---------------------------------------------------------
    pca = PCA(n_components=2)
    pca_scores = pca.fit_transform(scaled_data)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    plt.figure(figsize=(12, 10))
    
    # Plot models (scores)
    sns.scatterplot(
        x=pca_scores[:, 0], 
        y=pca_scores[:, 1], 
        s=250, 
        color="#3498db", 
        edgecolor="black", 
        alpha=0.85
    )
    
    # Annotate models
    texts = []
    for i, model in enumerate(matrix_df.index):
        texts.append(plt.text(pca_scores[i, 0], pca_scores[i, 1], model, fontsize=10, weight='bold'))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    # Plot variables (loadings) as vectors
    for i, feature in enumerate(matrix_df.columns):
        # Scale vectors slightly for visibility
        scale_factor = max(np.max(np.abs(pca_scores[:, 0])), np.max(np.abs(pca_scores[:, 1]))) / np.max(np.abs(loadings))
        vec_x = loadings[i, 0] * scale_factor * 0.8
        vec_y = loadings[i, 1] * scale_factor * 0.8
        
        plt.arrow(0, 0, vec_x, vec_y, color='r', alpha=0.6, width=0.015, head_width=0.08)
        # Clean up feature name (e.g. HEINZ_DILEMMA -> HEINZ)
        feat_clean = feature.replace('_DILEMMA', '').replace('_DILLEMA', '').replace('_', ' ')
        plt.text(vec_x * 1.15, vec_y * 1.15, feat_clean, color='r', fontsize=12, weight='bold', ha='center', va='center')
        
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title("PCA Biplot of LLM Responses & Dilemma Loadings", fontsize=16, weight='bold', pad=15)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=14, labelpad=10)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=14, labelpad=10)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig("results/fig6_pca_biplot.png", dpi=300, bbox_inches='tight')
    print("Generated results/fig6_pca_biplot.png")

if __name__ == "__main__":
    create_advanced_visualizations()
