import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def visualize_statistics():
    # Academic publication theme
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    
    # 1. Forest Plot of p-values
    if os.path.exists("results/statistical_results.csv"):
        df_stats = pd.read_csv("results/statistical_results.csv")
        df_stats['log_p'] = -np.log10(np.maximum(df_stats['p_value'], 1e-10))
        
        plt.figure(figsize=(10, 8))
        # Color based on significance
        colors = ['#2ecc71' if p >= 0.05 else '#e74c3c' for p in df_stats['p_value']]
        
        # Sort by p-value
        df_stats = df_stats.sort_values('p_value', ascending=True)
        
        sns.barplot(x='p_value', y='model_name', data=df_stats, palette=colors)
        
        plt.axvline(x=0.05, color='gray', linestyle='--', linewidth=1.5, label='Significance Threshold ($p=0.05$)')
        plt.xscale('log')
        
        # Add labels
        for idx, row in df_stats.reset_index().iterrows():
            plt.text(row['p_value'], idx, f" $p={row['p_value']:.3f}$", va='center', ha='left' if row['p_value'] < 0.05 else 'right', fontsize=11, weight='bold')
            
        plt.title("Kruskal-Wallis H-Test for Cross-Dilemma Coherence", fontsize=16, weight='bold', pad=15)
        plt.xlabel("$p$-value (log scale)\n$< 0.05$ indicates INCOHERENCE (Moral framework shifts by dilemma)", fontsize=14, labelpad=10)
        plt.ylabel("Model", fontsize=14, labelpad=10)
        plt.legend(frameon=True, shadow=True, fancybox=True)
        
        sns.despine()
        plt.tight_layout()
        plt.savefig("results/fig1_kruskal_forest_plot.png", dpi=300, bbox_inches='tight')
        print("Generated results/fig1_kruskal_forest_plot.png")
        
    # 2. Stage Distribution Violin Plots
    eval_dir = "../evaluation_data"
    eval_files = glob.glob(f"{eval_dir}/*_evaluation.xlsx")
    
    all_data = []
    for eval_path in eval_files:
        model_name = os.path.basename(eval_path).replace("_evaluation.xlsx", "")
        # Only plot a subset of models to keep the plot readable. Let's pick 3 coherent and 3 incoherent.
        df_stats = pd.read_csv("results/statistical_results.csv")
        
        df = pd.read_excel(eval_path)
        if 'kohlberg_stage' in df.columns:
            df['kohlberg_stage'] = pd.to_numeric(df['kohlberg_stage'], errors='coerce')
            df['model_name'] = model_name
            
            # Label as coherent or incoherent based on the stats
            stat_row = df_stats[df_stats['model_name'] == model_name]
            if not stat_row.empty:
                is_coh = stat_row.iloc[0]['is_statistically_coherent']
                df['status'] = "Coherent ($p \geq 0.05$)" if is_coh else "Incoherent ($p < 0.05$)"
                
            all_data.append(df[['model_name', 'dilemma_type', 'kohlberg_stage', 'status']])
            
    if all_data:
        full_df = pd.concat(all_data).dropna()
        
        plt.figure(figsize=(14, 10))
        sns.violinplot(
            data=full_df, 
            x="kohlberg_stage", 
            y="model_name", 
            hue="status",
            split=False,
            inner="quart",
            palette={"Coherent ($p \geq 0.05$)": "#2ecc71", "Incoherent ($p < 0.05$)": "#e74c3c"},
            linewidth=1.5
        )
        
        plt.title("Distribution of Kohlberg Stages Across Dilemmas", fontsize=16, weight='bold', pad=15)
        plt.xlabel("Kohlberg Stage (1-6)", fontsize=14, labelpad=10)
        plt.ylabel("Model", fontsize=14, labelpad=10)
        plt.xticks(range(1, 7))
        plt.legend(title="Statistical Status", frameon=True, shadow=True, fancybox=True, loc='lower right')
        
        sns.despine(left=True)
        plt.tight_layout()
        plt.savefig("results/fig2_stage_distributions.png", dpi=300, bbox_inches='tight')
        print("Generated results/fig2_stage_distributions.png")

if __name__ == "__main__":
    visualize_statistics()
