import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_statistics():
    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    
    # 1. Forest Plot of p-values
    if os.path.exists("results/statistical_results.csv"):
        df_stats = pd.read_csv("results/statistical_results.csv")
        
        plt.figure(figsize=(10, 8))
        df_stats = df_stats.sort_values('p_value', ascending=True)
        
        # Color based on significance
        colors = ['#55A868' if p >= 0.05 else '#C44E52' for p in df_stats['p_value']]
        
        # Draw horizontal lines for the forest plot
        for i in range(len(df_stats)):
            plt.hlines(y=i, xmin=1e-5, xmax=df_stats.iloc[i]['p_value'], color='lightgrey', zorder=1)
            
        sns.scatterplot(
            x='p_value', 
            y='model_name', 
            data=df_stats, 
            hue='model_name',
            palette=colors,
            s=150,
            zorder=2,
            legend=False
        )
        
        plt.axvline(x=0.05, color='#333333', linestyle='--', linewidth=2, label='Significance Threshold ($p=0.05$)')
        plt.xscale('log')
        plt.xlim(1e-4, 1.5)
        
        # Add labels
        for idx, row in df_stats.reset_index().iterrows():
            plt.text(row['p_value'] * 1.2, idx, f" $p={row['p_value']:.3f}$", va='center', ha='left', fontsize=11, weight='bold')
            
        plt.title("Kruskal-Wallis H-Test for Cross-Dilemma Coherence\n(Forest Plot)", fontsize=16, weight='bold', pad=20)
        plt.xlabel("$p$-value (log scale)\n$< 0.05$ indicates INCOHERENCE (Moral framework shifts by dilemma)", fontsize=14, labelpad=10)
        plt.ylabel("Model", fontsize=14, labelpad=10)
        plt.legend(frameon=False, loc='lower left')
        
        sns.despine(left=True, bottom=True)
        plt.grid(axis='x', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig("results/fig1_kruskal_forest_plot.png", dpi=400, bbox_inches='tight')
        plt.close()
        print("Generated results/fig1_kruskal_forest_plot.png")
        
    # 2. Stage Distribution KDE Plots (Ridge style)
    eval_dir = "../evaluation_data"
    eval_files = glob.glob(f"{eval_dir}/*_evaluation.xlsx")
    
    all_data = []
    for eval_path in eval_files:
        model_name = os.path.basename(eval_path).replace("_evaluation.xlsx", "")
        df_stats = pd.read_csv("results/statistical_results.csv")
        
        df = pd.read_excel(eval_path)
        if 'kohlberg_stage' in df.columns:
            df['kohlberg_stage'] = pd.to_numeric(df['kohlberg_stage'], errors='coerce')
            df['model_name'] = model_name
            
            stat_row = df_stats[df_stats['model_name'] == model_name]
            if not stat_row.empty:
                is_coh = stat_row.iloc[0]['is_statistically_coherent']
                df['status'] = "Coherent ($p \geq 0.05$)" if is_coh else "Incoherent ($p < 0.05$)"
                
            all_data.append(df[['model_name', 'dilemma_type', 'kohlberg_stage', 'status']])
            
    if all_data:
        full_df = pd.concat(all_data).dropna()
        
        # FacetGrid for KDE Ridge Plot
        g = sns.FacetGrid(full_df, row="model_name", hue="status", aspect=10, height=0.8, 
                          palette={"Coherent ($p \geq 0.05$)": "#55A868", "Incoherent ($p < 0.05$)": "#C44E52"})
        
        # Draw the densities
        g.map(sns.kdeplot, "kohlberg_stage", bw_adjust=.5, clip_on=False, fill=True, alpha=0.7, linewidth=1.5)
        g.map(sns.kdeplot, "kohlberg_stage", clip_on=False, color="w", lw=2, bw_adjust=.5)
        
        # Add labels to the left
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="right", va="center", transform=ax.transAxes, fontsize=12)
                    
        g.map(label, "kohlberg_stage")
        
        # Overlap the plots
        g.figure.subplots_adjust(hspace=-0.25)
        
        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        
        g.figure.suptitle("Probability Density of Kohlberg Stages Across Contexts", fontsize=18, weight='bold', y=1.02)
        g.set_axis_labels("Kohlberg Stage (1-6)", fontsize=14, weight='bold')
        
        plt.xlim(1, 6)
        plt.savefig("results/fig2_stage_distributions.png", dpi=400, bbox_inches='tight')
        plt.close()
        print("Generated results/fig2_stage_distributions.png")

if __name__ == "__main__":
    visualize_statistics()
