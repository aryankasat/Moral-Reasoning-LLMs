"""
visualizations.py — 5 publication-grade figures for Analysis 6 (Language & Patterns).
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from adjustText import adjust_text

from config import (
    STAGES, STAGE_COLORS, PROVIDER_COLORS, 
    TARGET_KEYWORDS, apply_publication_style
)

apply_publication_style()
MM = 1 / 25.4

def _save(fig, path: Path, name: str) -> None:
    fig.savefig(path / name, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Stage Word Clouds
# ═══════════════════════════════════════════════════════════════════════════

def plot_stage_word_clouds(stage_terms: dict[int, list[tuple[str, float]]], out_dir: Path) -> None:
    """Multi-panel grid of Word Clouds derived from TF-IDF weights."""
    fig, axes = plt.subplots(2, 3, figsize=(210 * MM, 140 * MM))
    axes = axes.flatten()
    
    for i, stage in enumerate(STAGES):
        ax = axes[i]
        terms = stage_terms.get(stage, [])
        if not terms:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center', fontsize=10, color='gray')
            ax.axis('off')
            ax.set_title(f"Stage {stage}", fontsize=11, fontweight="bold", color=STAGE_COLORS[stage])
            continue
            
        freq_dict = {word: weight for word, weight in terms}
        
        # Color function keyed to the Stage Color
        base_color = STAGE_COLORS[stage]
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return base_color
            
        wc = WordCloud(
            width=800, height=600,
            background_color='white',
            color_func=color_func,
            prefer_horizontal=0.8,
            max_words=15
        ).generate_from_frequencies(freq_dict)
        
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Stage {stage} Distinctive Terms", fontsize=10, fontweight="bold", color=STAGE_COLORS[stage])
        
    fig.suptitle("Qualitative Linguistic Patterns by Moral Reasoning Stage (TF-IDF weighted)", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, "fig1_stage_word_clouds.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Distinctive Terms by Model
# ═══════════════════════════════════════════════════════════════════════════

def plot_model_distinctive_terms(model_terms_df: pd.DataFrame, out_dir: Path) -> None:
    """Horizontal bar charts faceted by model showing top 5 TF-IDF terms."""
    models = model_terms_df["display_name"].tolist()
    n_models = len(models)
    
    n_cols = 4
    n_rows = int(np.ceil(n_models / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(210 * MM, max(60, n_rows * 40) * MM), sharex=True)
    axes = axes.flatten()
    
    for idx, row in model_terms_df.iterrows():
        ax = axes[idx]
        provider = row.get("provider", "Meta")
        color = PROVIDER_COLORS[provider]
        
        terms = []
        weights = []
        for i in range(1, 6):
            if pd.notna(row[f"term_{i}"]):
                terms.append(row[f"term_{i}"])
                weights.append(row[f"weight_{i}"])
                
        # Reverse to plot highest on top
        y = np.arange(len(terms))
        ax.barh(y, weights[::-1], color=color, alpha=0.8)
        
        ax.set_yticks(y)
        ax.set_yticklabels(terms[::-1], fontsize=8)
        ax.set_title(row["display_name"], fontsize=9, fontweight="bold")
        ax.set_xlim(0, max(weights) * 1.1 if weights else 1.0)
        
        if idx >= n_models - n_cols:
            ax.set_xlabel("Mean TF-IDF Weight", fontsize=8)
            
    for ax in axes[n_models:]:
        ax.set_visible(False)
        
    fig.suptitle("Top 5 Distinctive Lexical Terms used by each Model (Model 'Tells')", fontsize=12, fontweight="bold", y=1.02)
    
    # Provider Legend
    handles = [mpatches.Patch(color=c, label=p) for p, c in PROVIDER_COLORS.items()]
    fig.legend(handles=handles, title="Provider", loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=8)
    
    fig.tight_layout()
    _save(fig, out_dir, "fig2_model_distinctive_terms.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Vocabulary Richness
# ═══════════════════════════════════════════════════════════════════════════

def plot_vocabulary_richness(df: pd.DataFrame, out_dir: Path) -> None:
    """Box plot of unique non-trivial word counts by model, ordered by params."""
    fig, ax = plt.subplots(figsize=(160 * MM, 90 * MM))
    
    # Sort models by size
    order = df.drop_duplicates("model_key").sort_values("params_B")["display_name"].tolist()
    
    palette = {row["display_name"]: PROVIDER_COLORS[row["provider"]] for _, row in df.drop_duplicates("display_name").iterrows()}
    
    sns.boxplot(
        data=df, x="vocab_richness", y="display_name", order=order,
        palette=palette, width=0.6, fliersize=2, linewidth=1.0, ax=ax,
        boxprops=dict(alpha=0.7)
    )
    
    ax.set_xlabel("Vocabulary Richness (Unique Moral/Reasoning Tokens > 3 chars)", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=8)
    
    ax.set_title("Distribution of Moral Vocabulary Richness across LLMs\n(Do larger/aligned models use more expansive language?)", fontsize=11, fontweight="bold", pad=12)
    
    # Legend
    handles = [mpatches.Patch(color=c, label=p) for p, c in PROVIDER_COLORS.items()]
    ax.legend(handles=handles, title="Provider", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    
    _save(fig, out_dir, "fig3_moral_vocabulary_richness.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Target Keyword Usage Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_target_keyword_heatmap(kw_usage_df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap showing what % of a model's S(x) responses actually use S(x) keywords."""
    if len(kw_usage_df) == 0: return
    
    pivot = kw_usage_df.pivot(index="display_name", columns="stage", values="keyword_hit_pct")
    
    # Sort index by mean usage across all available stages
    pivot["mean_usage"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean_usage", ascending=False).drop(columns="mean_usage")
    
    fig, ax = plt.subplots(figsize=(100 * MM, max(100, len(pivot)*6) * MM))
    
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': '% of Responses'}, 
        ax=ax, linewidths=.5, cbar=True, vmin=0, vmax=100
    )
    
    # Add percent signs to annotations for clarity
    for t in ax.texts:
        if t.get_text():
            t.set_text(t.get_text() + "%")
            
    ax.set_xlabel("Kohlberg Stage Evaluated", fontsize=10)
    ax.set_ylabel("")
    ax.set_title("Theoretical Vocabulary Compliance\n(% of responses utilizing canonical stage keywords)", fontsize=10, fontweight="bold", pad=12)
    
    _save(fig, out_dir, "fig4_target_keyword_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Linguistic Style PCA
# ═══════════════════════════════════════════════════════════════════════════

def plot_linguistic_pca(pca_df: pd.DataFrame, var_explained: np.ndarray, out_dir: Path) -> None:
    """2D Scatter of the entire model corpus TF-IDF PCA projection."""
    fig, ax = plt.subplots(figsize=(120 * MM, 100 * MM))
    
    texts = []
    for _, row in pca_df.iterrows():
        color = PROVIDER_COLORS[row["provider"]]
        size = np.log10(row["params_B"]) * 50 # scale dot size
        
        ax.scatter(row["pca1"], row["pca2"], color=color, s=size, alpha=0.8, edgecolors="white", linewidth=0.5)
        texts.append(ax.text(row["pca1"], row["pca2"], row["display_name"], fontsize=7, ha='center', va='center'))
        
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
        force_text=(0.5, 1.0),
        force_points=(1.5, 1.5),
        expand_points=(2, 2),
        expand_text=(2, 2),
        max_iterations=2000
    )
    
    ax.set_xlabel(f"Linguistic PC1 ({var_explained[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"Linguistic PC2 ({var_explained[1]*100:.1f}%)", fontsize=10)
    
    ax.set_title("Linguistic Similarity Space (PCA of TF-IDF vectors)\n(Do model families share a 'voice'?)", fontsize=11, fontweight="bold", pad=12)
    
    # Legend
    handles = [mpatches.Patch(color=c, label=p) for p, c in PROVIDER_COLORS.items()]
    ax.legend(handles=handles, title="Provider Family", loc="upper left", fontsize=8)
    
    # Zero lines
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    
    _save(fig, out_dir, "fig5_pca_linguistic_style.png")

