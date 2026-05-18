import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

def visualize_complex_statistics():
    # Academic publication theme
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    
    # 1. Entropy vs Kruskal-Wallis p-value Scatter Plot
    if os.path.exists("results/complex_statistical_results.csv"):
        df = pd.read_csv("results/complex_statistical_results.csv")
        
        plt.figure(figsize=(10, 8))
        
        # Color by coherence
        df['status'] = ["Coherent ($p \geq 0.05$)" if p >= 0.05 else "Incoherent ($p < 0.05$)" for p in df['p_value']]
        
        sns.scatterplot(
            data=df,
            x="p_value",
            y="normalized_entropy",
            hue="status",
            palette={"Coherent ($p \geq 0.05$)": "#2ecc71", "Incoherent ($p < 0.05$)": "#e74c3c"},
            s=250,
            alpha=0.85,
            edgecolor="black"
        )
        
        plt.axvline(x=0.05, color='gray', linestyle='--', linewidth=1.5, label='Significance Threshold ($p=0.05$)')
        plt.xscale('log')
        
        # Add labels with adjustText
        texts = []
        for i, row in df.iterrows():
            texts.append(plt.text(row['p_value'], row['normalized_entropy'], row['model_name'], fontsize=10, alpha=0.9))
            
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
            
        plt.title("Moral Predictability: Entropy vs. Statistical Coherence", fontsize=16, weight='bold', pad=15)
        plt.xlabel("Kruskal-Wallis $p$-value (log scale)", fontsize=14, labelpad=10)
        plt.ylabel("Normalized Shannon Entropy ($H$)\n(0 = Perfect Predictability, 1 = Max Uncertainty)", fontsize=14, labelpad=10)
        
        plt.legend(title="Statistical Status", frameon=True, shadow=True, fancybox=True)
        sns.despine()
        plt.tight_layout()
        plt.savefig("results/fig3_entropy_scatter.png", dpi=300, bbox_inches='tight')
        print("Generated results/fig3_entropy_scatter.png")
        
    # 2. PCA Scree Plot
    if os.path.exists("results/pca_variance.csv"):
        pca_df = pd.read_csv("results/pca_variance.csv")
        
        plt.figure(figsize=(10, 6))
        
        # Bar plot for individual variance
        sns.barplot(
            data=pca_df,
            x="principal_component",
            y="explained_variance_ratio",
            color="#3498db",
            alpha=0.85,
            edgecolor="black",
            label="Individual Explained Variance"
        )
        
        # Line plot for cumulative variance
        plt.plot(
            pca_df["principal_component"],
            pca_df["cumulative_variance"],
            color="#e74c3c",
            marker="o",
            markersize=8,
            linewidth=2.5,
            label="Cumulative Explained Variance"
        )
        
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        plt.title("Dimensionality of LLM Moral Frameworks (PCA Scree Plot)", fontsize=16, weight='bold', pad=15)
        plt.xlabel("Principal Components", fontsize=14, labelpad=10)
        plt.ylabel("Proportion of Variance Explained", fontsize=14, labelpad=10)
        
        # Annotate cumulative variance points
        for i, row in pca_df.iterrows():
            plt.text(i, row['cumulative_variance'] + 0.03, f"{row['cumulative_variance']:.2f}", ha='center', fontsize=11, weight='bold')
            
        plt.legend(frameon=True, shadow=True, fancybox=True)
        sns.despine()
        plt.tight_layout()
        plt.savefig("results/fig4_pca_scree.png", dpi=300, bbox_inches='tight')
        print("Generated results/fig4_pca_scree.png")

if __name__ == "__main__":
    visualize_complex_statistics()
