"""
visualizations.py — Publication-quality figures for NLI Coherence Analysis.

Generates 5 figures:
  1. NLI Score Distribution — violin plot split by consistent vs. inconsistent
  2. Model-Level Scatter — mean NLI entailment vs. consistency % with regression
  3. Per-Dilemma Heatmap — mean NLI entailment across models × dilemmas
  4. Base vs. Instruct NLI — paired comparison for RLHF data
  5. Correlation Matrix — NLI, Kohlberg, consistency, action measures
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Any

from config import (
    apply_publication_style,
    OUT_DIR,
    OI,
    CONSISTENCY_COLORS,
    VARIANT_COLORS,
    PROVIDER_COLORS,
    SINGLE_COL,
    DOUBLE_COL,
    TALL_DOUBLE,
    WIDE_TRIPLE,
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save a figure to the results directory."""
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"    ✅ Saved → {name}")


# ── Figure 1: NLI Score Distribution ─────────────────────────────────────────

def fig1_nli_score_distribution(scored_df: pd.DataFrame) -> None:
    """
    Violin + strip plot of NLI entailment scores, split by consistency status.
    Shows whether framework-independent NLI agrees with Kohlberg-based consistency.
    """
    apply_publication_style()

    valid = scored_df.dropna(subset=["nli_entailment", "is_consistent"]).copy()
    valid["consistency_label"] = valid["is_consistent"].map(
        {1.0: "Consistent", 0.0: "Inconsistent"}
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE_COL, gridspec_kw={"width_ratios": [3, 2]})

    # Left: violin plot
    groups = ["Consistent", "Inconsistent"]
    data = [
        valid[valid["consistency_label"] == g]["nli_entailment"].values
        for g in groups
    ]

    parts = ax1.violinplot(data, positions=[1, 2], showmeans=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(CONSISTENCY_COLORS[groups[i]])
        pc.set_alpha(0.6)
    parts["cmeans"].set_color(OI["black"])
    parts["cmeans"].set_linewidth(1.5)

    # Add individual points (jittered)
    rng = np.random.default_rng(42)
    for i, (g, d) in enumerate(zip(groups, data)):
        jitter = rng.uniform(-0.08, 0.08, size=len(d))
        ax1.scatter(
            np.full(len(d), i + 1) + jitter,
            d,
            c=CONSISTENCY_COLORS[g],
            alpha=0.25,
            s=8,
            edgecolors="none",
        )

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(groups)
    ax1.set_ylabel("NLI Entailment Score")
    ax1.set_title("NLI Coherence by Consistency Status")

    # Add means as text
    for i, d in enumerate(data):
        if len(d) > 0:
            ax1.text(
                i + 1, np.mean(d) + 0.03,
                f"μ={np.mean(d):.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

    # Right: histogram of all NLI scores
    all_ent = valid["nli_entailment"].values
    ax2.hist(all_ent, bins=30, color=OI["sky_blue"], alpha=0.7, edgecolor="white", linewidth=0.5)
    ax2.axvline(np.mean(all_ent), color=OI["vermillion"], linestyle="--", linewidth=1.2,
                label=f"Mean = {np.mean(all_ent):.3f}")
    ax2.axvline(np.median(all_ent), color=OI["green"], linestyle=":", linewidth=1.2,
                label=f"Median = {np.median(all_ent):.3f}")
    ax2.set_xlabel("NLI Entailment Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Score Distribution")
    ax2.legend(fontsize=7, framealpha=0.8)

    fig.suptitle("NLI-Based Coherence Measure", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_nli_score_distribution.png")


# ── Figure 2: Model-Level Scatter ────────────────────────────────────────────

def fig2_model_scatter(model_summary: pd.DataFrame) -> None:
    """
    Scatter plot: mean NLI entailment (x) vs. consistency % (y) per model.
    Includes regression line with CI band.
    """
    apply_publication_style()

    valid = model_summary.dropna(subset=["mean_nli_entailment", "consistency_pct"])

    if len(valid) < 3:
        print("    ⚠️  Too few models for scatter plot — skipping fig2")
        return

    fig, ax = plt.subplots(figsize=SINGLE_COL)

    x = valid["mean_nli_entailment"].values
    y = valid["consistency_pct"].values

    # Color by provider if available
    for _, row in valid.iterrows():
        provider = row.get("provider", "")
        color = PROVIDER_COLORS.get(provider, OI["black"])
        ax.scatter(
            row["mean_nli_entailment"],
            row["consistency_pct"],
            c=color,
            s=40,
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
        )
        # Label each point
        ax.annotate(
            row.get("display_name", row["model_key"])[:12],
            (row["mean_nli_entailment"], row["consistency_pct"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=5.5,
            alpha=0.8,
        )

    # Regression line
    if len(x) >= 3:
        from scipy import stats as sp_stats
        slope, intercept, r, p, se = sp_stats.linregress(x, y)
        x_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 50)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=OI["vermillion"], linewidth=1.2, alpha=0.8)

        # CI band
        n = len(x)
        y_pred = slope * x + intercept
        mse = np.sum((y - y_pred) ** 2) / (n - 2) if n > 2 else 0
        x_mean = x.mean()
        se_line = np.sqrt(mse * (1.0 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)))
        t_val = sp_stats.t.ppf(0.975, n - 2) if n > 2 else 1.96
        ax.fill_between(
            x_line,
            y_line - t_val * se_line,
            y_line + t_val * se_line,
            alpha=0.15,
            color=OI["vermillion"],
        )

        ax.text(
            0.05, 0.95,
            f"r = {r:.3f}\np = {p:.4f}",
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#ccc"),
        )

    ax.set_xlabel("Mean NLI Entailment Score")
    ax.set_ylabel("Kohlberg Consistency (%)")
    ax.set_title("NLI Coherence vs. Decoupling Score")

    fig.tight_layout()
    _save(fig, "fig2_model_scatter.png")


# ── Figure 3: Per-Dilemma Heatmap ────────────────────────────────────────────

def fig3_dilemma_heatmap(scored_df: pd.DataFrame) -> None:
    """
    Heatmap of mean NLI entailment across models × dilemmas.
    Reveals dilemma-specific coherence patterns.
    """
    apply_publication_style()

    valid = scored_df.dropna(subset=["nli_entailment"])

    pivot = valid.pivot_table(
        values="nli_entailment",
        index="display_name",
        columns="dilemma_type",
        aggfunc="mean",
    )

    if pivot.empty:
        print("    ⚠️  Empty pivot table — skipping fig3")
        return

    # Shorten dilemma names for display
    col_renames = {
        "HEINZ_DILEMMA": "Heinz",
        "LIFEBOAT_DILEMMA": "Lifeboat",
        "TROLLEY_DILLEMA": "Trolley",
        "DOCTOR_DILLEMA": "Doctor",
        "STOLEN_FOOD_DILEMMA": "Stolen Food",
        "PROMISE_DILEMMA": "Promise",
    }
    pivot = pivot.rename(columns=col_renames)

    fig, ax = plt.subplots(figsize=TALL_DOUBLE)

    cmap = plt.cm.YlOrRd
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Axis labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Mean NLI Entailment")
    ax.set_title("NLI Coherence by Model × Dilemma", fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig3_dilemma_heatmap.png")


# ── Figure 4: Base vs. Instruct NLI (RLHF) ──────────────────────────────────

def fig4_base_vs_instruct(scored_df: pd.DataFrame) -> None:
    """
    Paired bar/box comparing NLI coherence for base vs. instruct variants.
    Only generated for RLHF data.
    """
    if "variant" not in scored_df.columns or "pair_id" not in scored_df.columns:
        print("    ⚠️  Not RLHF data — skipping fig4")
        return

    apply_publication_style()

    valid = scored_df.dropna(subset=["nli_entailment"])

    pairs = valid["pair_id"].unique()
    if len(pairs) == 0:
        print("    ⚠️  No pair data — skipping fig4")
        return

    fig, axes = plt.subplots(1, len(pairs) + 1, figsize=WIDE_TRIPLE,
                              gridspec_kw={"width_ratios": [2] * len(pairs) + [3]})

    for i, pair_id in enumerate(pairs):
        ax = axes[i]
        pair_data = valid[valid["pair_id"] == pair_id]

        for j, variant in enumerate(["base", "instruct"]):
            variant_data = pair_data[pair_data["variant"] == variant]["nli_entailment"]
            if len(variant_data) > 0:
                bp = ax.boxplot(
                    [variant_data.values],
                    positions=[j],
                    widths=0.5,
                    patch_artist=True,
                    showmeans=True,
                    meanprops=dict(
                        marker="D", markerfacecolor=OI["black"],
                        markeredgecolor=OI["black"], markersize=4,
                    ),
                )
                bp["boxes"][0].set_facecolor(VARIANT_COLORS[variant])
                bp["boxes"][0].set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Base", "Instruct"], fontsize=7)
        ax.set_title(pair_id.replace("_", " ").title(), fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.set_ylabel("NLI Entailment Score")

    # Overall comparison in last panel
    ax_all = axes[-1]
    for j, variant in enumerate(["base", "instruct"]):
        variant_data = valid[valid["variant"] == variant]["nli_entailment"]
        if len(variant_data) > 0:
            bp = ax_all.boxplot(
                [variant_data.values],
                positions=[j],
                widths=0.5,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(
                    marker="D", markerfacecolor=OI["black"],
                    markeredgecolor=OI["black"], markersize=4,
                ),
            )
            bp["boxes"][0].set_facecolor(VARIANT_COLORS[variant])
            bp["boxes"][0].set_alpha(0.7)

    ax_all.set_xticks([0, 1])
    ax_all.set_xticklabels(["Base", "Instruct"], fontsize=7)
    ax_all.set_title("All Pairs Combined", fontsize=8, fontweight="bold")
    ax_all.set_ylim(-0.05, 1.05)

    fig.suptitle("NLI Coherence: Base vs. RLHF-Instruct", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_base_vs_instruct_nli.png")


# ── Figure 5: Correlation Matrix ─────────────────────────────────────────────

def fig5_correlation_matrix(scored_df: pd.DataFrame) -> None:
    """
    Annotated heatmap of correlations between NLI entailment, Kohlberg stage,
    consistency, and action category metrics.
    """
    apply_publication_style()

    valid = scored_df.dropna(subset=["nli_entailment", "kohlberg_stage"]).copy()

    # Build correlation variables
    corr_data = pd.DataFrame({
        "NLI Entailment": valid["nli_entailment"],
        "Kohlberg Stage": valid["kohlberg_stage"].astype(float),
        "NLI Contradiction": valid.get("nli_contradiction", np.nan),
        "NLI Neutral": valid.get("nli_neutral", np.nan),
    })

    if "is_consistent" in valid.columns:
        corr_data["Consistent (binary)"] = valid["is_consistent"].astype(float)

    # Drop columns that are all NaN
    corr_data = corr_data.dropna(axis=1, how="all")

    corr_matrix = corr_data.corr()

    fig, ax = plt.subplots(figsize=(5, 4))

    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    n = len(corr_matrix)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr_matrix.index, fontsize=7)

    # Annotate
    for i in range(n):
        for j in range(n):
            val = corr_matrix.values[i, j]
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Correlation Matrix: NLI & Kohlberg Measures", fontweight="bold", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig5_correlation_matrix.png")


# ── Generate all ──────────────────────────────────────────────────────────────

def generate_all_visualizations(
    scored_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    data_source: str,
) -> None:
    """Generate all publication-quality figures."""
    print("\n  Generating figures…")

    fig1_nli_score_distribution(scored_df)
    fig2_model_scatter(model_summary)
    fig3_dilemma_heatmap(scored_df)

    if data_source == "rlhf":
        fig4_base_vs_instruct(scored_df)

    fig5_correlation_matrix(scored_df)

    print(f"\n  All figures saved to {OUT_DIR}/")
