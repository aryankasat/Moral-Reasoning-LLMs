import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_results():
    if not os.path.exists("results/coherence_scores.csv"):
        print("No coherence scores found to visualize.")
        return

    df = pd.read_csv("results/coherence_scores.csv")
    df_sorted = df.sort_values(by="overall_score", ascending=False)
    
    # Set the style
    sns.set_theme(style="whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left subplot: Overall composite score
    sns.barplot(
        x="overall_score", 
        y="model_name", 
        data=df_sorted,
        palette="mako",
        hue="model_name",
        legend=False,
        ax=ax1
    )
    ax1.set_title("Overall Cross-Pair Coherence Score", fontsize=16, weight='bold', pad=15)
    ax1.set_xlabel("Composite Score (0-10)", fontsize=12)
    ax1.set_ylabel("Model", fontsize=12)
    ax1.set_xlim(0, 10.5)
    
    for p in ax1.patches:
        width = p.get_width()
        ax1.text(width + 0.1, p.get_y() + p.get_height()/2. + 0.1, '{:1.2f}'.format(width), ha="left")

    # Right subplot: Lexical vs Stage Coherence Scatter
    sns.scatterplot(
        x="dilemma_lexical_coherence", 
        y="dilemma_stage_coherence", 
        hue="model_name",
        s=200,
        data=df,
        palette="tab20",
        ax=ax2
    )
    ax2.set_title("Lexical Overlap vs. Stage Consistency", fontsize=16, weight='bold', pad=15)
    ax2.set_xlabel("Lexical Coherence (Mean TF-IDF Cosine Similarity)", fontsize=12)
    ax2.set_ylabel("Stage Coherence (1 / (1 + StdDev))", fontsize=12)
    
    # Move legend outside
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.savefig("results/coherence_chart.png", dpi=300, bbox_inches='tight')
    print("Successfully generated and saved coherence_chart.png")

if __name__ == "__main__":
    visualize_results()
