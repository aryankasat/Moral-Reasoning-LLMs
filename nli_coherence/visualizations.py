"""
visualizations.py — Publication-quality figures for NLI coherence analysis.

Figures
-------
fig1_coherence_by_model.png       – Mean NLI entailment score per model (bar chart)
fig2_coherence_by_dilemma.png     – NLI coherence heatmap: model × dilemma
fig3_coherence_vs_decoupling.png  – Scatter: NLI coherence vs. decoupling strength
fig4_coherence_gap.png            – Signed coherence gap bar chart per model
fig5_entailment_distribution.png  – Violin / box plot of entailment score distributions
fig6_correlation_matrix.png       – Annotated correlation summary panel
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text
from config import apply_publication_style, PROVIDER_COLORS


def _provider_color(provider: str) -> str:
    return PROVIDER_COLORS.get(provider, "#888888")


# ── Fig 1: Mean NLI Coherence by Model ────────────────────────────────────

def plot_coherence_by_model(coherence_df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of mean NLI entailment score per model, sorted by params."""
    apply_publication_style()

    df = coherence_df.sort_values("params_B")
    fig, ax = plt.subplots(figsize=(12, 5.5))

    colors = [_provider_color(p) for p in df["provider"]]
    bars = ax.bar(
        range(len(df)), df["mean_entailment"],
        yerr=df["std_entailment"],
        color=colors, edgecolor="white", linewidth=0.5,
        capsize=3, error_kw={"linewidth": 0.8},
    )

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["display_name"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean NLI Entailment Score")
    ax.set_title("NLI Coherence: Justification → Action Entailment by Model")
    ax.set_ylim(0, 1.05)

    # Add value labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row["mean_entailment"] + row["std_entailment"] + 0.02,
                f"{row['mean_entailment']:.2f}", ha="center", va="bottom", fontsize=7)

    # Legend for providers
    handles = [mpatches.Patch(color=c, label=p) for p, c in PROVIDER_COLORS.items()
               if p in df["provider"].values]
    ax.legend(handles=handles, title="Provider", loc="lower right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_coherence_by_model.png")
    plt.close(fig)
    print("  → fig1_coherence_by_model.png")


# ── Fig 2: Coherence Heatmap (model × dilemma) ────────────────────────────

def plot_coherence_heatmap(
    model_dilemma_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Heatmap of mean NLI coherence scores, models vs dilemmas."""
    apply_publication_style()

    pivot = model_dilemma_df.pivot_table(
        index="display_name", columns="dilemma_type",
        values="mean_entailment", aggfunc="mean",
    )

    # Sort index by params_B
    params_order = (model_dilemma_df[["display_name", "params_B"]]
                    .drop_duplicates()
                    .sort_values("params_B")["display_name"].tolist())
    pivot = pivot.reindex(params_order)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [d.replace("_DILEMMA", "").replace("_DILLEMA", "").replace("_", " ").title()
         for d in pivot.columns],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Mean Entailment Score")
    ax.set_title("NLI Coherence Heatmap: Model × Dilemma")

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_coherence_heatmap.png")
    plt.close(fig)
    print("  → fig2_coherence_heatmap.png")


# ── Fig 3: Coherence vs Decoupling Scatter ─────────────────────────────────

def plot_coherence_vs_decoupling(merged_df: pd.DataFrame, corr_results: dict, out_dir: Path) -> None:
    """Scatter plot: NLI coherence (x) vs decoupling strength (y)."""
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(9, 7))
    df = merged_df.copy()

    colors = [_provider_color(p) for p in df["provider"]]
    ax.scatter(df["mean_entailment"], df["decoupling_strength"],
               c=colors, s=80, edgecolors="white", linewidth=0.5, zorder=5)

    # Label each point — collect texts for adjustText
    texts = []
    for _, row in df.iterrows():
        texts.append(ax.text(
            row["mean_entailment"], row["decoupling_strength"],
            row["display_name"], fontsize=6.5,
        ))
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.5),
        force_text=(1.5, 2.0),
        force_points=(1.5, 2.0),
        expand=(1.4, 1.6),
    )

    # Trend line
    z = np.polyfit(df["mean_entailment"], df["decoupling_strength"], 1)
    x_line = np.linspace(df["mean_entailment"].min() - 0.02,
                         df["mean_entailment"].max() + 0.02, 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="#888888", linewidth=1, alpha=0.6)

    # Annotation box with correlation
    cv = corr_results.get("coherence_vs_decoupling", {})
    sp_r = cv.get("spearman_r", "N/A")
    sp_p = cv.get("spearman_p", "N/A")
    textstr = f"Spearman ρ = {sp_r}\np = {sp_p}"
    props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", bbox=props)

    ax.set_xlabel("Mean NLI Coherence (Entailment Score)")
    ax.set_ylabel("Decoupling Strength (1 − McNemar p-value)")
    ax.set_title("NLI Coherence vs. Kohlberg-Based Decoupling")

    # Provider legend
    handles = [mpatches.Patch(color=c, label=p) for p, c in PROVIDER_COLORS.items()
               if p in df["provider"].values]
    ax.legend(handles=handles, title="Provider", loc="lower right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_coherence_vs_decoupling.png")
    plt.close(fig)
    print("  → fig3_coherence_vs_decoupling.png")


# ── Fig 4: Coherence Gap Bar Chart ─────────────────────────────────────────

def plot_coherence_gap(merged_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Signed bar chart of coherence_gap = NLI_coherence − decoupling_strength.
    Positive → reasoning is more coherent than Kohlberg consistency suggests.
    """
    apply_publication_style()

    df = merged_df.sort_values("params_B")
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ["#2ca02c" if g > 0 else "#d62728" for g in df["coherence_gap"]]
    ax.bar(range(len(df)), df["coherence_gap"],
           color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["display_name"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Coherence Gap\n(NLI Coherence − Decoupling Strength)")
    ax.set_title("Coherence Gap: NLI-Based vs. Kohlberg-Based Assessment")

    # Annotate
    for i, (_, row) in enumerate(df.iterrows()):
        offset = 0.01 if row["coherence_gap"] >= 0 else -0.03
        ax.text(i, row["coherence_gap"] + offset,
                f"{row['coherence_gap']:+.2f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    # Custom legend
    pos_patch = mpatches.Patch(color="#2ca02c", label="Positive: More coherent than Kohlberg suggests")
    neg_patch = mpatches.Patch(color="#d62728", label="Negative: Less coherent than Kohlberg suggests")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_coherence_gap.png")
    plt.close(fig)
    print("  → fig4_coherence_gap.png")


# ── Fig 5: Entailment Score Distributions ──────────────────────────────────

def plot_entailment_distribution(scored_df: pd.DataFrame, out_dir: Path) -> None:
    """Box plots of per-observation entailment scores by model."""
    apply_publication_style()

    valid = scored_df.dropna(subset=["entailment_score"]).copy()

    # Sort by median entailment
    order = (valid.groupby("display_name")["entailment_score"]
             .median().sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(12, 6))

    data_by_model = [
        valid[valid["display_name"] == m]["entailment_score"].values
        for m in order
    ]

    bp = ax.boxplot(
        data_by_model, patch_artist=True, notch=True,
        widths=0.6,
        boxprops=dict(linewidth=0.6),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )

    # Color boxes by provider
    provider_map = valid[["display_name", "provider"]].drop_duplicates().set_index("display_name")
    for i, m in enumerate(order):
        prov = provider_map.loc[m, "provider"]
        bp["boxes"][i].set_facecolor(_provider_color(prov))
        bp["boxes"][i].set_alpha(0.7)

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Entailment Score")
    ax.set_title("Distribution of NLI Entailment Scores by Model")
    ax.set_ylim(-0.05, 1.1)

    handles = [mpatches.Patch(color=c, label=p) for p, c in PROVIDER_COLORS.items()
               if p in valid["provider"].values]
    ax.legend(handles=handles, title="Provider", loc="lower left", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_entailment_distribution.png")
    plt.close(fig)
    print("  → fig5_entailment_distribution.png")


# ── Fig 6: Correlation Summary Panel ───────────────────────────────────────

def plot_correlation_summary(merged_df: pd.DataFrame, corr_results: dict, out_dir: Path) -> None:
    """
    Two-panel figure:
      Left:  coherence vs p_value (with Spearman ρ)
      Right: coherence vs decoupling_strength
    """
    apply_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (y_col, y_label, corr_key, title) in enumerate([
        ("p_value", "McNemar p-value\n(Decoupling Score)",
         "coherence_vs_pvalue", "Coherence vs. p-value"),
        ("decoupling_strength", "Decoupling Strength (1 − p)",
         "coherence_vs_decoupling", "Coherence vs. Decoupling Strength"),
    ]):
        ax = axes[ax_idx]
        colors = [_provider_color(p) for p in merged_df["provider"]]

        ax.scatter(
            merged_df["mean_entailment"], merged_df[y_col],
            c=colors, s=70, edgecolors="white", linewidth=0.5, zorder=5,
        )

        # Collect text objects for adjustText to resolve overlaps
        texts = []
        for _, row in merged_df.iterrows():
            texts.append(ax.text(
                row["mean_entailment"], row[y_col],
                row["display_name"], fontsize=6,
            ))
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", linewidth=0.4),
            force_text=(1.5, 2.0),
            force_points=(1.5, 2.0),
            expand=(1.5, 1.8),
        )

        # Trend line
        z = np.polyfit(merged_df["mean_entailment"], merged_df[y_col], 1)
        x_line = np.linspace(
            merged_df["mean_entailment"].min() - 0.02,
            merged_df["mean_entailment"].max() + 0.02, 100,
        )
        ax.plot(x_line, np.polyval(z, x_line), "--", color="#888", linewidth=1, alpha=0.6)

        # Stats box
        cv = corr_results.get(corr_key, {})
        textstr = (
            f"Spearman ρ = {cv.get('spearman_r', '?')}\n"
            f"p = {cv.get('spearman_p', '?')}\n"
            f"Kendall τ = {cv.get('kendall_tau', '?')}"
        )
        props = dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=8, verticalalignment="top", bbox=props)

        ax.set_xlabel("Mean NLI Coherence")
        ax.set_ylabel(y_label)
        ax.set_title(title)

    handles = [mpatches.Patch(color=c, label=p) for p, c in PROVIDER_COLORS.items()
               if p in merged_df["provider"].values]
    axes[1].legend(handles=handles, title="Provider", loc="lower right", fontsize=6)

    fig.suptitle("Correlation Analysis: NLI Coherence × Kohlberg Decoupling",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_correlation_summary.png")
    plt.close(fig)
    print("  → fig6_correlation_summary.png")
