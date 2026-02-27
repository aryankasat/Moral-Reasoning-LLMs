"""
visualizations.py — Publication-quality figures for the alignment training
analysis (analysis2).

Figures
-------
  fig1_violin_by_alignment.png   — Violin + strip: IT vs RLHF stage distribution
  fig2_family_comparisons.png    — Grouped bar chart: within-family Δ stage
  fig3_stacked_stage_dist.png    — Stacked bar: stage % by alignment type
  fig4_pct_postconv.png          — Dot-plot: % Stage 5+ per model, coloured by alignment
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    ALIGN_COLORS, FAMILY_COLORS, FAMILY_PAIRS, MODEL_META,
    IT, RLHF, POST_CONV_THRESHOLD, ALL_STAGES,
    apply_publication_style,
)

apply_publication_style()

MM = 1 / 25.4


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    fig.savefig(path / name, dpi=300, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}")


def _align_legend_handles() -> list:
    return [
        mpatches.Patch(facecolor=ALIGN_COLORS[t], edgecolor="#444",
                       linewidth=0.6, label=t, alpha=0.88)
        for t in [IT, RLHF]
    ]


# ── Figure 1: Violin plot — IT vs RLHF ───────────────────────────────────────

def plot_violin_by_alignment(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Paired violin plots (IT left, RLHF right) showing full stage distributions.
    Individual model mean stages overlaid as jittered dots.
    """
    fig, ax = plt.subplots(figsize=(120 * MM, 100 * MM))

    # Map alignment to numeric position for seaborn
    order = [IT, RLHF]
    align_order_map = {v: i for i, v in enumerate(order)}

    # Violin — use hue= to avoid seaborn palette deprecation warning
    sns.violinplot(
        data=df, x="alignment_type", y="kohlberg_stage",
        order=order,
        hue="alignment_type", hue_order=order,
        palette=ALIGN_COLORS,
        inner=None, cut=0, linewidth=0.8,
        alpha=0.55, ax=ax, legend=False,
    )

    # Per-model mean dots
    model_stats = (
        df.groupby(["model_key", "alignment_type"])["kohlberg_stage"]
        .mean().reset_index()
    )
    rng = np.random.default_rng(42)
    for _, row in model_stats.iterrows():
        xpos = align_order_map[row["alignment_type"]] + rng.uniform(-0.12, 0.12)
        ax.scatter(xpos, row["kohlberg_stage"],
                   color=ALIGN_COLORS[row["alignment_type"]],
                   s=28, zorder=5, alpha=0.85, edgecolors="#333",
                   linewidths=0.5)

    # Group median lines
    for i, atype in enumerate(order):
        med = df.loc[df["alignment_type"] == atype, "kohlberg_stage"].median()
        ax.hlines(med, i - 0.3, i + 0.3, color="#333", linewidth=1.8,
                  linestyle="-", zorder=6, label="_")
        ax.text(i + 0.32, med, f"  Mdn={med:.1f}",
                va="center", fontsize=8, color="#333")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([IT, RLHF], fontsize=9.5)
    ax.set_yticks(ALL_STAGES)
    ax.set_yticklabels([f"Stage {s}" for s in ALL_STAGES], fontsize=8.5)
    ax.set_ylim(0.3, 6.8)
    ax.set_xlabel("Alignment Training Type", labelpad=6)
    ax.set_ylabel("Kohlberg Moral Reasoning Stage", labelpad=5)
    ax.set_title(
        "Stage Distribution by Alignment Training Type\n"
        "(dots = per-model means; horizontal bar = group median)",
        fontsize=10, pad=10, fontweight="bold",
    )

    # Light horizontal guides
    for s in ALL_STAGES:
        ax.axhline(s, color="#eeeeee", linewidth=0.5, zorder=0)

    ax.legend(handles=_align_legend_handles(),
              title="Alignment Type", fontsize=8.5, loc="lower right",
              framealpha=0.92, edgecolor="#cccccc")

    fig.tight_layout()
    _save(fig, out_dir, "fig1_violin_by_alignment.png")


# ── Figure 2: Within-family comparison bar chart ──────────────────────────────

def plot_family_comparisons(
    df: pd.DataFrame,
    family_results: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Grouped horizontal bar chart: one row per family comparison,
    showing mean stage for model A (left) and model B (right).
    Effect size and significance annotated.
    """
    n = len(family_results)
    fig, ax = plt.subplots(figsize=(160 * MM, max(80, n * 22) * MM))

    bar_h = 0.32
    y_base = np.arange(n) * 1.1   # spacing between pairs

    for i, row in family_results.iterrows():
        ya = y_base[i] - bar_h / 2
        yb = y_base[i] + bar_h / 2

        col_a = ALIGN_COLORS[row["align_a"]]
        col_b = ALIGN_COLORS[row["align_b"]]

        ax.barh(ya, row["mean_a"], height=bar_h, color=col_a, alpha=0.80,
                edgecolor="#555", linewidth=0.6)
        ax.barh(yb, row["mean_b"], height=bar_h, color=col_b, alpha=0.80,
                edgecolor="#555", linewidth=0.6)

        # Δ annotation + significance (plain ASCII — avoids missing-glyph warning)
        sig = "*" if row["p_value"] < 0.05 else "n.s."
        d_str = f"d={row['cohens_d']:+.2f}, {sig}"
        max_val = max(row["mean_a"], row["mean_b"])
        ax.text(max_val + 0.05, y_base[i], d_str,
                va="center", fontsize=7.5, color="#333")

        # Model labels at bar start
        ax.text(-0.08, ya, row["model_a"], va="center", ha="right", fontsize=7.5,
                color=col_a, style="italic")
        ax.text(-0.08, yb, row["model_b"], va="center", ha="right", fontsize=7.5,
                color=col_b, style="italic")

    ax.set_yticks(y_base)
    ax.set_yticklabels(family_results["comparison"], fontsize=9)
    ax.set_xlim(0, 7.0)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"S{s}" for s in range(1, 7)], fontsize=9)
    ax.set_xlabel("Mean Kohlberg Stage", labelpad=5)
    ax.set_title(
        "Within-Family Comparisons: Mean Moral Reasoning Stage\n"
        "(Δ = model B − model A; ✓ = p < 0.05 Wilcoxon rank-sum)",
        fontsize=10, pad=10, fontweight="bold",
    )

    for s in range(1, 7):
        ax.axvline(s, color="#eeeeee", linewidth=0.6, zorder=0)

    ax.legend(handles=_align_legend_handles(),
              title="Alignment Type", fontsize=8.5, loc="lower right",
              framealpha=0.92, edgecolor="#cccccc")

    fig.tight_layout()
    _save(fig, out_dir, "fig2_family_comparisons.png")


# ── Figure 3: Stacked bar — stage distribution by alignment type ──────────────

def plot_stacked_stage_dist(align_stats: pd.DataFrame, out_dir: Path) -> None:
    """
    Horizontal stacked bar chart: stage percentage per alignment category.
    """
    order = [IT, RLHF]
    stage_cols = [f"stage_{s}_pct" for s in ALL_STAGES]

    # Sequential palette for Kohlberg stages (low=warm, high=cool)
    stage_palette = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4", "#1a237e"]

    fig, ax = plt.subplots(figsize=(140 * MM, 70 * MM))

    lefts = np.zeros(len(order))
    for s, col, color in zip(ALL_STAGES, stage_cols, stage_palette):
        vals = [align_stats.loc[align_stats["alignment_type"] == at, col].values[0]
                for at in order]
        bars = ax.barh(order, vals, left=lefts, color=color, alpha=0.90,
                       label=f"Stage {s}", edgecolor="white", linewidth=0.5,
                       height=0.45)
        # Annotate segment if wide enough
        for bar, v, l in zip(bars, vals, lefts):
            if v > 4:
                ax.text(l + v / 2, bar.get_y() + bar.get_height() / 2,
                        f"{v:.0f}%", va="center", ha="center",
                        fontsize=8, color="white", fontweight="bold")
        lefts += np.array(vals)

    ax.set_xlim(0, 105)
    ax.set_xlabel("Percentage of Responses (%)", labelpad=5)
    ax.set_title(
        "Stage Distribution by Alignment Training Type",
        fontsize=10, pad=10, fontweight="bold",
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9.5)
    ax.set_xticks(range(0, 101, 20))

    ax.legend(title="Kohlberg Stage", fontsize=8.5, title_fontsize=9,
              loc="lower right", ncol=3, framealpha=0.92, edgecolor="#cccccc")

    fig.tight_layout()
    _save(fig, out_dir, "fig3_stacked_stage_dist.png")


# ── Figure 4: % Post-conventional (Stage 5+) dot plot ────────────────────────

def plot_pct_postconv(model_stats: pd.DataFrame, out_dir: Path) -> None:
    """
    Dot-plot of % responses at Stage 5+ per model.
    Models sorted by pct_post_conv; coloured by alignment type.
    """
    s = model_stats.sort_values("pct_post_conv")
    n = len(s)

    fig, ax = plt.subplots(figsize=(130 * MM, max(80, n * 13) * MM))

    y_pos = np.arange(n)
    colors = [ALIGN_COLORS[a] for a in s["alignment_type"]]

    ax.scatter(s["pct_post_conv"], y_pos,
               color=colors, s=55, zorder=5, edgecolors="#333", linewidths=0.5)

    # Reference lines
    it_mean   = model_stats.loc[model_stats["alignment_type"] == IT,
                                "pct_post_conv"].mean()
    rlhf_mean = model_stats.loc[model_stats["alignment_type"] == RLHF,
                                "pct_post_conv"].mean()
    ax.axvline(it_mean,   color=ALIGN_COLORS[IT],   linewidth=1.2,
               linestyle="--", alpha=0.7, label=f"IT mean = {it_mean:.1f}%")
    ax.axvline(rlhf_mean, color=ALIGN_COLORS[RLHF], linewidth=1.2,
               linestyle="--", alpha=0.7, label=f"RLHF mean = {rlhf_mean:.1f}%")

    # Annotate values
    for xi, yi, val in zip(s["pct_post_conv"], y_pos, s["pct_post_conv"]):
        ax.text(xi + 1.2, yi, f"{val:.0f}%", va="center", fontsize=7.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(s["display_name"], fontsize=8.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel(f"% of Responses at Stage ≥ {POST_CONV_THRESHOLD} (Post-Conventional)",
                  labelpad=5)
    ax.set_title(
        "Post-Conventional Reasoning Rate per Model\n"
        "(Stage 5 = Social Contract, Stage 6 = Universal Ethics)",
        fontsize=10, pad=10, fontweight="bold",
    )

    ax.legend(handles=_align_legend_handles() + [
        mlines.Line2D([], [], color=ALIGN_COLORS[IT],   linewidth=1.2,
                      linestyle="--", label=f"IT mean = {it_mean:.1f}%"),
        mlines.Line2D([], [], color=ALIGN_COLORS[RLHF], linewidth=1.2,
                      linestyle="--", label=f"RLHF mean = {rlhf_mean:.1f}%"),
    ], fontsize=7.5, title="Legend", loc="lower right",
              framealpha=0.92, edgecolor="#cccccc")

    fig.tight_layout()
    _save(fig, out_dir, "fig4_pct_postconv.png")
