import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np

def visualize_icc_results():
    if not os.path.exists('results/icc_coherence_results.json'):
        print("ICC results not found.")
        return

    with open('results/icc_coherence_results.json', 'r') as f:
        icc_data = json.load(f)

    sns.set_theme(style="ticks")

    # 1. Variance Partitioning Stacked Bar Chart
    var_props = icc_data['Variance_Proportions']
    labels = ['Model Identity\n(True Coherence)', 'Contextual Pairing\n(Prompt x Dilemma)', 'Interaction / Error\n(Noise)']
    sizes = [var_props['Model_Identity_Pct'], var_props['Contextual_Pairing_Pct'], var_props['Interaction_Error_Pct']]
    
    colors = ['#2b5c8f', '#d97746', '#878787']  # Muted, academic color palette

    fig, ax = plt.subplots(figsize=(10, 2.5))
    
    ax.barh([0], sizes[0], color=colors[0], edgecolor='white', label=labels[0])
    ax.barh([0], sizes[1], left=sizes[0], color=colors[1], edgecolor='white', label=labels[1])
    ax.barh([0], sizes[2], left=sizes[0]+sizes[1], color=colors[2], edgecolor='white', label=labels[2])

    ax.set_yticks([])
    ax.set_xlabel('Percentage of Total Variance Explained (%)', fontsize=12, weight='bold')
    ax.set_title(f"Cross-Pair Coherence Variance Partitioning\n(Global ICC = {icc_data['ICC2_Cross_Pair_Coherence']:.3f})", fontsize=14, weight='bold', pad=15)
    ax.set_xlim(0, 100)
    
    sns.despine(left=True, top=True, right=True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3, frameon=False, fontsize=11)

    for i, v in enumerate(sizes):
        if v > 5:
            x_pos = sum(sizes[:i]) + v/2
            ax.text(x_pos, 0, f"{v:.1f}%", ha='center', va='center', color='white', weight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig('results/fig7_variance_partitioning.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("Generated results/fig7_variance_partitioning.png")
    
    # Load data for Heatmaps
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
                        'Model': model_name,
                        'Context': f"{row['dilemma_type']} | {row['prompt_type']}",
                        'Stage': stage
                    })
            except Exception:
                pass

    df = pd.DataFrame(records)
    if len(df) > 0:
        heatmap_data = df.pivot(index='Model', columns='Context', values='Stage')
        
        # 2a. Clustermap (Absolute Stages)
        g = sns.clustermap(
            heatmap_data, 
            cmap="viridis", 
            annot=True, 
            fmt=".1f", 
            figsize=(16, 10),
            cbar_kws={'label': 'Absolute Kohlberg Stage'},
            dendrogram_ratio=(0.1, 0.2),
            linewidths=.5
        )
        g.fig.suptitle('Hierarchical Clustermap of Model Reasoning Across Contexts', fontsize=18, weight='bold', y=1.05)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10)
        g.savefig('results/fig8a_model_context_clustermap.png', dpi=400, bbox_inches='tight')
        plt.close()
        print("Generated results/fig8a_model_context_clustermap.png")

        # 2b. Deviation Matrix (Contextual Variance)
        # Calculate deviation from the model's own mean
        deviation_data = heatmap_data.sub(heatmap_data.mean(axis=1), axis=0)
        
        plt.figure(figsize=(16, 8))
        ax = sns.heatmap(
            deviation_data, 
            cmap="vlag", 
            center=0, 
            annot=True, 
            fmt="+.1f", 
            cbar_kws={'label': 'Deviation from Model Mean Stage'},
            linewidths=.5,
            linecolor='lightgrey'
        )
        plt.title('Deviation Matrix: Contextual Variance Penalized by ICC', fontsize=16, weight='bold', pad=20)
        plt.xlabel('Contextual Pairing (Dilemma | Prompt)', fontsize=12, weight='bold')
        plt.ylabel('Model Identity', fontsize=12, weight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('results/fig8b_deviation_matrix.png', dpi=400, bbox_inches='tight')
        plt.close()
        print("Generated results/fig8b_deviation_matrix.png")

if __name__ == "__main__":
    visualize_icc_results()
