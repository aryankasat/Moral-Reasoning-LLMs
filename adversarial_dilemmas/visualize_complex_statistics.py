import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

def visualize_complex_statistics():
    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    
    # 1. Entropy vs Kruskal-Wallis p-value Jointplot
    if os.path.exists("results/complex_statistical_results.csv"):
        df = pd.read_csv("results/complex_statistical_results.csv")
        
        df['status'] = ["Coherent ($p \geq 0.05$)" if p >= 0.05 else "Incoherent ($p < 0.05$)" for p in df['p_value']]
        df['log_p'] = np.log10(np.maximum(df['p_value'], 1e-10))
        
        # We use jointplot to show marginal distributions
        g = sns.jointplot(
            data=df,
            x="log_p",
            y="normalized_entropy",
            hue="status",
            palette={"Coherent ($p \geq 0.05$)": "#55A868", "Incoherent ($p < 0.05$)": "#C44E52"},
            s=200,
            alpha=0.85,
            edgecolor="white",
            height=9,
            ratio=4,
            marginal_ticks=False
        )
        
        # Adjust axes
        g.ax_joint.axvline(x=np.log10(0.05), color='#333333', linestyle='--', linewidth=2, label='Threshold ($p=0.05$)')
        g.ax_joint.set_xlim(-4, 0.5)
        
        # Add labels with adjustText inside the joint plot
        texts = []
        for i, row in df.iterrows():
            texts.append(g.ax_joint.text(row['log_p'], row['normalized_entropy'], row['model_name'], fontsize=11, weight='bold'))
            
        adjust_text(texts, ax=g.ax_joint, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
            
        g.ax_joint.set_xlabel("Kruskal-Wallis $p$-value (log10)", fontsize=14, weight='bold')
        g.ax_joint.set_ylabel("Normalized Shannon Entropy ($H$)", fontsize=14, weight='bold')
        g.fig.suptitle("Moral Predictability: Entropy vs. Statistical Coherence", fontsize=18, weight='bold', y=1.03)
        
        # Move legend
        g.ax_joint.legend(title="", frameon=False, loc='lower left')
        
        plt.savefig("results/fig3_entropy_scatter.png", dpi=400, bbox_inches='tight')
        plt.close()
        print("Generated results/fig3_entropy_scatter.png")
        
    # 2. PCA Scree Plot
    if os.path.exists("results/pca_variance.csv"):
        pca_df = pd.read_csv("results/pca_variance.csv")
        
        plt.figure(figsize=(10, 6))
        
        # Line plot for cumulative variance
        plt.fill_between(
            pca_df["principal_component"],
            0,
            pca_df["cumulative_variance"],
            color="#2b5c8f",
            alpha=0.2
        )
        
        plt.plot(
            pca_df["principal_component"],
            pca_df["cumulative_variance"],
            color="#2b5c8f",
            marker="o",
            markersize=10,
            linewidth=3,
            label="Cumulative Explained Variance"
        )
        
        # Bar plot for individual variance
        sns.barplot(
            data=pca_df,
            x="principal_component",
            y="explained_variance_ratio",
            color="#d97746",
            alpha=0.9,
            edgecolor="white",
            label="Individual Explained Variance"
        )
        
        plt.axhline(y=1.0, color='#333333', linestyle='--', alpha=0.5, linewidth=2)
        
        plt.title("Dimensionality of LLM Moral Frameworks (PCA Scree Plot)", fontsize=16, weight='bold', pad=20)
        plt.xlabel("Principal Components", fontsize=14, weight='bold', labelpad=10)
        plt.ylabel("Proportion of Variance Explained", fontsize=14, weight='bold', labelpad=10)
        
        # Annotate cumulative variance points
        for i, row in pca_df.iterrows():
            plt.text(i, row['cumulative_variance'] + 0.04, f"{row['cumulative_variance']:.2f}", ha='center', fontsize=12, weight='bold')
            
        plt.legend(frameon=False, loc='center right')
        sns.despine(left=True, bottom=True)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig("results/fig4_pca_scree.png", dpi=400, bbox_inches='tight')
        plt.close()
        print("Generated results/fig4_pca_scree.png")

if __name__ == "__main__":
    visualize_complex_statistics()
