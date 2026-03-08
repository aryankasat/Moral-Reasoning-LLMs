"""
visualizations.py — Visualizations for Analysis 10: Stage Transition Dynamics.

Research paper grade figures matching Analysis 8 styling.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

from config import (
    OUT_DIR, STAGES, ACTIVE_STAGES, STAGE_COLORS, SHORT_NAMES,
    MAX_ENTROPY, apply_publication_style, SINGLE_COL, DOUBLE_COL
)


def _setup_figure(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax


def _get_model_labels(model_keys: list[str]) -> list[str]:
    return [SHORT_NAMES.get(k, k).replace('\n', ' ') for k in model_keys]


def plot_transition_timing_heatmap(model_df: pd.DataFrame) -> None:
    fig, ax = _setup_figure(DOUBLE_COL)
    stages   = ACTIVE_STAGES
    prop_col = [f"stage_{s}" for s in stages]
    data     = model_df.sort_values("model_order")[prop_col].values

    # Using an Okabe-Ito inspired Sequential colormap
    from matplotlib.colors import LinearSegmentedColormap
    from config import OI
    cmap = LinearSegmentedColormap.from_list("stage_dens", ["#ffffff", OI["sky_blue"], OI["blue"]])

    sns.heatmap(
        data, ax=ax, cmap=cmap, vmin=0, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Proportion of Responses", "pad": 0.02}
    )

    ax.set_title("A. Moral Stage Distribution by Model Scale", fontsize=10, fontweight="bold", pad=10, loc="left")
    ax.set_xlabel("Kohlberg Stage", fontsize=9, labelpad=5)
    ax.set_ylabel("Model (Ordered by Parameter Scale)", fontsize=9, labelpad=5)

    ax.set_xticks(np.arange(len(stages)) + 0.5)
    ax.set_xticklabels([f"Stage {s}" for s in stages], fontsize=8)

    ylabels = _get_model_labels(model_df.sort_values("model_order")["model_key"].tolist())
    ax.set_yticks(np.arange(len(ylabels)) + 0.5)
    ax.set_yticklabels(ylabels, rotation=0, va="center", fontsize=8)

    ax.tick_params(left=False, bottom=False)
    
    out = OUT_DIR / "figA_transition_timing_heatmap.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_entropy_trajectories(model_df: pd.DataFrame) -> None:
    fig, ax = _setup_figure(DOUBLE_COL)

    df = model_df.sort_values("model_order")
    x  = df["model_order"].values
    labels = [SHORT_NAMES.get(k, k) for k in df["model_key"]]

    from config import OI
    ax.plot(x, df["entropy"], marker="o", color=OI["vermillion"], lw=1.6, markersize=5, label="Shannon Entropy (bits)", zorder=3)
    ax.plot(x, df["gini"], marker="s", color=OI["blue"], lw=1.6, markersize=5, label="Gini Coefficient", linestyle="--", zorder=3)

    ax.axhline(1.0, color="#888888", linestyle=":", lw=1.2, alpha=0.8, label="Consolidation Threshold (H=1.0)", zorder=1)
    max_h_active = np.log2(len(ACTIVE_STAGES))
    ax.axhline(max_h_active, color="#444444", linestyle="-.", lw=1.0, alpha=0.5, label=f"Max H for 3 stages ({max_h_active:.2f})", zorder=1)

    ax.set_title("B. Stage Consolidation Trajectories", fontsize=10, fontweight="bold", pad=10, loc="left")
    ax.set_xlabel("Model (Ordered by Parameter Scale)", fontsize=9, labelpad=5)
    ax.set_ylabel("Metric Value", fontsize=9, labelpad=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(-0.05, max_h_active + 0.2)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="#cccccc", fontsize=8)

    out = OUT_DIR / "figB_entropy_trajectory.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_stage_alluvial(model_df: pd.DataFrame) -> None:
    fig, ax = _setup_figure(DOUBLE_COL)

    df = model_df.sort_values("model_order")
    x  = df["model_order"].values
    labels = [SHORT_NAMES.get(k, k) for k in df["model_key"]]

    y_data = []
    colors = []
    poly_labels = []

    for s in ACTIVE_STAGES:
        y_data.append(df[f"stage_{s}"].values)
        colors.append(STAGE_COLORS[s])
        poly_labels.append(f"Stage {s}")

    ax.stackplot(x, y_data, labels=poly_labels, colors=colors, alpha=0.85, edgecolor="white", lw=0.4)

    ax.set_title("C. Stage Transition Flows Across Scale", fontsize=10, fontweight="bold", pad=10, loc="left")
    ax.set_xlabel("Model (Ordered by Parameter Scale)", fontsize=9, labelpad=5)
    ax.set_ylabel("Cumulative Proportion", fontsize=9, labelpad=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1)

    ax.grid(False)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="#cccccc", fontsize=8, reverse=True)

    out = OUT_DIR / "figC_stage_alluvial.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_stage_residence_times(residence: pd.Series) -> None:
    fig, ax = _setup_figure(SINGLE_COL)

    plot_stages = [s for s in STAGES if residence.get(s, 0) > 0 or s in ACTIVE_STAGES]
    counts = [residence.get(s, 0) for s in plot_stages]
    colors = [STAGE_COLORS[s] for s in plot_stages]

    bars = ax.bar(plot_stages, counts, color=colors, edgecolor="white", linewidth=0.8, alpha=0.9, width=0.6)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{int(h)}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333333")

    ax.set_title("D. Stage Residence Times", fontsize=10, fontweight="bold", pad=10, loc="left")
    ax.set_xlabel("Kohlberg Stage", fontsize=9, labelpad=5)
    ax.set_ylabel("Number of Models (where stage is Modal)", fontsize=9, labelpad=5)

    ax.set_xticks(plot_stages)
    ax.set_xticklabels([f"Stage {s}" for s in plot_stages], fontsize=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(0, max(counts) + 1.2 if counts else 10)

    ax.grid(axis='y', linestyle=":", alpha=0.5)
    ax.grid(axis='x', visible=False)

    out = OUT_DIR / "figD_stage_residence_times.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_transition_matrix(T: np.ndarray, labels: list[str]) -> None:
    fig, ax = _setup_figure(SINGLE_COL)
    
    from config import OI
    cmap = LinearSegmentedColormap.from_list("trans", ["#ffffff", OI["yellow"], OI["vermillion"]])

    sns.heatmap(
        T, ax=ax, cmap=cmap, vmin=0, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Transition Prob. Proxy", "pad": 0.04}
    )

    ax.set_title("E. Aggregate Stage Transition Matrix", fontsize=10, fontweight="bold", pad=10, loc="left")
    ax.set_xlabel("Model $k+1$ Stage (Target)", fontsize=9, labelpad=5)
    ax.set_ylabel("Model $k$ Stage (Source)", fontsize=9, labelpad=5)

    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, rotation=0, fontsize=8)

    ax.tick_params(left=False, bottom=False)

    out = OUT_DIR / "figE_transition_matrix.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def generate_all_visualizations(
    model_df: pd.DataFrame,
    residence: pd.Series,
    T: np.ndarray,
    T_labels: list[str]
) -> None:
    plot_transition_timing_heatmap(model_df)
    plot_entropy_trajectories(model_df)
    plot_stage_alluvial(model_df)
    plot_stage_residence_times(residence)
    plot_transition_matrix(T, T_labels)
