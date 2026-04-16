"""
visualizations.py — Figures for Analysis 11: RLHF Causal Analysis.

Plots:
  Fig 1: Side-by-side stacked bar charts (base vs. instruct) per architecture pair
  Fig 2: Mean stage comparison + bootstrap CI per pair
  Fig 3: Stage-proportion delta heatmap (instruct − base) across pairs
  Fig 4: KL divergence bar chart across pairs
  Fig 5: Post-conventional proportion comparison per pair
  Fig 6: Cohen's d effect size across pairs with interpretation bands
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from config import (
    STAGES, PAIR_ORDER, MODEL_PAIRS,
    STAGE_COLORS, VARIANT_COLORS, PAIR_COLORS,
    SINGLE_COL, DOUBLE_COL, TALL_DOUBLE, WIDE_TRIPLE,
    OUT_DIR, apply_publication_style,
)

apply_publication_style()


# ── Helper ─────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Fig 1: Side-by-side stacked bar charts ────────────────────────────────────

def fig_stacked_stage_distributions(dist_df: pd.DataFrame) -> None:
    """
    For each architecture pair: two stacked bar charts side by side (base|instruct).
    Arranged as 3 pairs × 2 bars in one wide figure.
    """
    n_pairs = len(PAIR_ORDER)
    fig, axes = plt.subplots(1, n_pairs, figsize=(WIDE_TRIPLE[0], WIDE_TRIPLE[1] + 0.5),
                              sharey=True)
    if n_pairs == 1:
        axes = [axes]

    for ax, pair_id in zip(axes, PAIR_ORDER):
        arch = MODEL_PAIRS[pair_id]["architecture"]

        for x_pos, variant in enumerate(("base", "instruct")):
            row = dist_df[(dist_df["pair_id"] == pair_id) & (dist_df["variant"] == variant)]
            if row.empty:
                continue
            row = row.iloc[0]

            bottom = 0.0
            for s in STAGES:
                prop = float(row.get(f"stage_{s}", 0.0))
                if prop > 0:
                    ax.bar(
                        x_pos, prop, bottom=bottom,
                        color=STAGE_COLORS[s], width=0.55,
                        edgecolor="white", linewidth=0.5,
                        label=f"Stage {s}" if (x_pos == 0 and pair_id == PAIR_ORDER[0]) else "",
                    )
                    if prop > 0.07:
                        ax.text(
                            x_pos, bottom + prop / 2,
                            f"S{s}\n{prop*100:.0f}%",
                            ha="center", va="center",
                            fontsize=6.5, color="white", fontweight="bold",
                        )
                    bottom += prop

        ax.set_title(arch, fontsize=9, pad=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Base", "Instruct"], fontsize=8)
        ax.set_ylim(0, 1.05)
        if ax is axes[0]:
            ax.set_ylabel("Stage proportion", fontsize=8)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.tick_params(left=False if ax is not axes[0] else True)

    # Legend
    handles = [mpatches.Patch(color=STAGE_COLORS[s], label=f"Stage {s}") for s in STAGES]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Stage Distributions: Base vs. RLHF-tuned (by Architecture)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, "fig1_stacked_stage_distributions.png")


# ── Fig 2: Mean stage + bootstrap CI ──────────────────────────────────────────

def fig_mean_stage_comparison(pair_metrics: pd.DataFrame) -> None:
    """
    Grouped bar chart: mean stage (base / instruct) per architecture pair.
    Error bars = bootstrap 95% CI on the delta.
    """
    fig, ax = plt.subplots(figsize=DOUBLE_COL)

    x   = np.arange(len(PAIR_ORDER))
    w   = 0.32
    archs = [MODEL_PAIRS[p]["architecture"] for p in PAIR_ORDER]

    for pm_pair_id, row in pair_metrics.set_index("pair_id").iterrows():
        xi = PAIR_ORDER.index(pm_pair_id)
        # Base bar
        ax.bar(xi - w/2, row["base_mean"],     w, color=VARIANT_COLORS["base"],
               label="Base" if xi == 0 else "", edgecolor="white", linewidth=0.5)
        # Instruct bar with CI whisker
        ax.bar(xi + w/2, row["instruct_mean"], w, color=VARIANT_COLORS["instruct"],
               label="Instruct (RLHF)" if xi == 0 else "", edgecolor="white", linewidth=0.5)
        ci_lo = row["instruct_mean"] - row["boot_ci_lower"] + row["boot_diff"]
        ci_hi = row["boot_ci_upper"] - row["boot_diff"] + row["instruct_mean"]
        ax.errorbar(xi + w/2, row["instruct_mean"],
                    yerr=[[row["instruct_mean"] - row["boot_ci_lower"]],
                          [row["boot_ci_upper"] - row["instruct_mean"]]],
                    fmt="none", color="black", capsize=3, linewidth=1)
        # Delta annotation
        delta = row["delta_mean"]
        ax.annotate(
            f"Δ={delta:+.2f}",
            xy=(xi, max(row["base_mean"], row["instruct_mean"]) + 0.08),
            ha="center", fontsize=7.5, color="black",
            arrowprops=None,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=8.5)
    ax.set_ylabel("Mean Kohlberg Stage", fontsize=9)
    ax.set_title("Mean Moral Stage: Base vs. RLHF-tuned", fontsize=10)
    ax.set_ylim(1, 6.5)
    ax.axhline(5, color="#aaaaaa", linewidth=0.6, linestyle="--", label="Post-Conv. threshold")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    _save(fig, "fig2_mean_stage_comparison.png")


# ── Fig 3: Stage-proportion delta heatmap ─────────────────────────────────────

def fig_delta_heatmap(pair_metrics: pd.DataFrame) -> None:
    """
    Heatmap of Δproportion (instruct − base) for each stage × architecture pair.
    Blue = instruct higher, Red = base higher.
    """
    n_pairs  = len(PAIR_ORDER)
    n_stages = len(STAGES)

    delta_matrix = np.zeros((n_stages, n_pairs))
    for j, pair_id in enumerate(PAIR_ORDER):
        row = pair_metrics[pair_metrics["pair_id"] == pair_id]
        if row.empty:
            continue
        row = row.iloc[0]
        for i, s in enumerate(STAGES):
            delta_matrix[i, j] = float(row.get(f"delta_stage_{s}", 0.0))

    archs = [MODEL_PAIRS[p]["architecture"] for p in PAIR_ORDER]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL[0] * 1.1, DOUBLE_COL[1] + 0.5))
    im = ax.imshow(delta_matrix, cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto")

    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels(archs, fontsize=8.5)
    ax.set_yticks(range(n_stages))
    ax.set_yticklabels([f"Stage {s}" for s in STAGES], fontsize=8)
    ax.set_title("Δ Stage Proportion (Instruct − Base)", fontsize=10)

    for i in range(n_stages):
        for j in range(n_pairs):
            v = delta_matrix[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(v) < 0.3 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Δ Proportion (Instruct − Base)", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig3_delta_heatmap.png")


# ── Fig 4: KL divergence bar chart ────────────────────────────────────────────

def fig_kl_divergence(pair_metrics: pd.DataFrame) -> None:
    """
    Bar chart of KL(base → instruct) per architecture pair.
    Annotates with Cohen's d on second axis.
    """
    fig, ax1 = plt.subplots(figsize=DOUBLE_COL)
    ax2 = ax1.twinx()

    archs = [MODEL_PAIRS[p]["architecture"] for p in PAIR_ORDER]
    x     = np.arange(len(PAIR_ORDER))
    w     = 0.38

    kl_vals  = []
    d_vals   = []
    for pair_id in PAIR_ORDER:
        row = pair_metrics[pair_metrics["pair_id"] == pair_id].iloc[0]
        kl_vals.append(float(row["kl_base_to_instruct"]))
        d_vals.append(float(row["cohens_d"]))

    bars = ax1.bar(x - w/2, kl_vals, w, color=PAIR_COLORS.get(PAIR_ORDER[0], "#0072B2"),
                   label="KL(base→instruct)", alpha=0.85)
    for bar, pair_id in zip(bars, PAIR_ORDER):
        bar.set_color(PAIR_COLORS.get(pair_id, "#0072B2"))

    ax2.plot(x + w/2, d_vals, "D--", color=VARIANT_COLORS["instruct"],
             label="Cohen's d", markersize=7, linewidth=1.4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(archs, fontsize=8.5)
    ax1.set_ylabel("KL Divergence (nats)", fontsize=8.5, color=PAIR_COLORS.get(PAIR_ORDER[0], "#0072B2"))
    ax2.set_ylabel("Cohen's d", fontsize=8.5, color=VARIANT_COLORS["instruct"])

    # Cohen's d interpretation bands
    ax2.axhline(0.2, color="#cccccc", linewidth=0.7, linestyle=":")
    ax2.axhline(0.5, color="#aaaaaa", linewidth=0.7, linestyle=":")
    ax2.axhline(0.8, color="#888888", linewidth=0.7, linestyle=":")
    for y, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax2.text(len(PAIR_ORDER) - 0.45, y + 0.02, lbl, fontsize=6.5, color="#888888")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=7.5, frameon=False)
    ax1.set_title("KL Divergence & Effect Size (Base → Instruct)", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig4_kl_divergence_effect_size.png")


# ── Fig 5: Post-conventional proportion ───────────────────────────────────────

def fig_postconventional_proportion(pair_metrics: pd.DataFrame) -> None:
    """
    Grouped bar chart: post-conventional (Stage 5+6) proportion per pair.
    """
    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    x = np.arange(len(PAIR_ORDER))
    w = 0.32
    archs = [MODEL_PAIRS[p]["architecture"] for p in PAIR_ORDER]

    for j, pair_id in enumerate(PAIR_ORDER):
        row = pair_metrics[pair_metrics["pair_id"] == pair_id].iloc[0]
        ax.bar(j - w/2, row["base_postconv_prop"]     * 100, w,
               color=VARIANT_COLORS["base"],
               label="Base" if j == 0 else "")
        ax.bar(j + w/2, row["instruct_postconv_prop"] * 100, w,
               color=VARIANT_COLORS["instruct"],
               label="Instruct (RLHF)" if j == 0 else "")
        delta_pc = (row["delta_postconv_prop"]) * 100
        ax.annotate(
            f"Δ={delta_pc:+.1f}pp",
            xy=(j, max(row["base_postconv_prop"], row["instruct_postconv_prop"]) * 100 + 1.5),
            ha="center", fontsize=7.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=8.5)
    ax.set_ylabel("Post-conventional proportion (%)", fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.set_title("Post-conventional Reasoning (Stage 5–6): Base vs. Instruct", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    _save(fig, "fig5_postconventional_proportion.png")


# ── Fig 6: Cohen's d with effect size bands ────────────────────────────────────

def fig_cohens_d_panel(pair_metrics: pd.DataFrame) -> None:
    """
    Horizontal bar chart of Cohen's d per pair with interpretation bands.
    """
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    archs    = [MODEL_PAIRS[p]["architecture"] for p in PAIR_ORDER]
    d_values = []
    for pair_id in PAIR_ORDER:
        row = pair_metrics[pair_metrics["pair_id"] == pair_id].iloc[0]
        d_values.append(float(row["cohens_d"]))

    colors = [PAIR_COLORS.get(p, "#0072B2") for p in PAIR_ORDER]
    y = np.arange(len(PAIR_ORDER))

    ax.barh(y, d_values, 0.5, color=colors, edgecolor="white", linewidth=0.5)

    # Interpretation bands
    bands = [(0.2, 0.5, "#ffe5b4", "small"), (0.5, 0.8, "#b4d8ff", "medium"), (0.8, 3.0, "#b4ffcc", "large")]
    for lo, hi, color, lbl in bands:
        ax.axvspan(lo, hi, alpha=0.15, color=color, zorder=0)
        ax.text((lo + min(hi, d_values[0] + 1)) / 2, len(PAIR_ORDER) - 0.2,
                lbl, fontsize=7, ha="center", color="#888888")

    ax.axvline(0, color="black", linewidth=0.8)
    for i, (d, pair_id) in enumerate(zip(d_values, PAIR_ORDER)):
        ax.text(d + 0.03, i, f"{d:.2f}", va="center", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(archs, fontsize=8)
    ax.set_xlabel("Cohen's d (instruct − base)", fontsize=8)
    ax.set_title("RLHF Effect Size per Architecture", fontsize=9)
    ax.set_xlim(left=min(0, min(d_values) - 0.1))
    fig.tight_layout()
    _save(fig, "fig6_cohens_d_panel.png")


# ── Full generation ────────────────────────────────────────────────────────────

def generate_all_visualizations(
    obs_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    pair_metrics: pd.DataFrame,
) -> None:
    print("\n[VIZ] Generating figures…")
    fig_stacked_stage_distributions(dist_df)
    fig_mean_stage_comparison(pair_metrics)
    fig_delta_heatmap(pair_metrics)
    fig_kl_divergence(pair_metrics)
    fig_postconventional_proportion(pair_metrics)
    fig_cohens_d_panel(pair_metrics)
    print("[VIZ] All figures saved.")
