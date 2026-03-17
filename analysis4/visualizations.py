"""
visualizations.py — 6 publication-grade figures for Analysis 4.

Fig 1: Stacked bar — all models + human baseline (sorted by params_B)
Fig 2: Histogram grid — per-model stage proportions + human baseline overlay
Fig 3: JSD heatmap — N×N symmetric matrix with dendrograms (seaborn clustermap)
Fig 4: Distribution stats panel — entropy, skewness, modal stage, pattern
Fig 5: Chi-square bar + Pearson residual heatmap
Fig 6: 3D stage landscape — X=stage, Y=model, Z=proportion
"""

from __future__ import annotations
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    STAGES, STAGE_LABELS, STAGE_LABELS_SHORT, STAGE_COLORS,
    PROVIDER_COLORS, HUMAN_DIST, HUMAN_ADULT,
    apply_publication_style,
)

apply_publication_style()

MM = 1 / 25.4   # mm → inch


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pc(prov: str) -> str:
    return PROVIDER_COLORS.get(prov, "#888888")

def _save(fig, path: Path, name: str) -> None:
    fig.savefig(path / name, dpi=300, bbox_inches="tight",
                pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}")

def _stage_handles() -> list:
    return [
        mpatches.Patch(facecolor=STAGE_COLORS[s], edgecolor="#444",
                       lw=0.5, label=f"S{s} — {STAGE_LABELS[s].replace(chr(10), ' ')}",
                       alpha=0.90)
        for s in STAGES
    ]

def _prov_handles() -> list:
    return [
        mpatches.Patch(facecolor=c, edgecolor="#444", lw=0.5, label=p, alpha=0.90)
        for p, c in PROVIDER_COLORS.items()
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Stacked bar: all models + human adult baseline
# ═══════════════════════════════════════════════════════════════════════════

def plot_stacked_bar(dist_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Horizontal stacked bar chart. One bar per model (sorted params_B, smallest
    at bottom) + a reference bar for Human Adult baseline at the top separated
    by a gap. Bar width proportional to stage proportion.
    """
    # Build proportion matrix: rows = models (sorted), cols = stages
    model_order = (
        dist_df.drop_duplicates("model_key")
        .sort_values("params_B")["display_name"]
        .tolist()
    )
    # Add human baselines as extra rows at the top
    baselines   = list(HUMAN_DIST.keys())   # Adult, Adolescent, Children
    all_rows    = model_order + ["─" * 12] + baselines   # separator then baselines

    def _row_props(name):
        if name in baselines:
            return np.array([HUMAN_DIST[name][s] for s in STAGES])
        if "─" in name:
            return np.zeros(len(STAGES))
        sub = dist_df[dist_df["display_name"] == name]
        return sub.sort_values("stage")["proportion"].values

    fig_h = max(130, len(all_rows) * 11) * MM
    fig, ax = plt.subplots(figsize=(185 * MM, fig_h))

    y_pos = np.arange(len(all_rows))
    bar_h = 0.72

    for row_idx, row_name in enumerate(all_rows):
        if "─" in row_name:
            ax.axhline(row_idx, color="#cccccc", lw=0.8, ls="--", zorder=0)
            continue
        props = _row_props(row_name)
        left  = 0.0
        for s_idx, s in enumerate(STAGES):
            w = props[s_idx]
            if w <= 0:
                continue
            is_baseline = row_name in baselines
            ax.barh(
                row_idx, w, left=left, height=bar_h,
                color=STAGE_COLORS[s], alpha=0.88 if not is_baseline else 0.60,
                edgecolor="white", linewidth=0.4,
                hatch="///" if is_baseline else "",
            )
            if w >= 0.08:
                ax.text(
                    left + w / 2, row_idx,
                    f"{w*100:.0f}%",
                    ha="center", va="center", fontsize=9.0,
                    color="white" if STAGE_COLORS[s] in ("#4575b4", "#1a237e", "#d73027") else "#222222",
                    fontweight="bold",
                )
            left += w

    # Y-axis labels + provider colour dot
    meta = dist_df.drop_duplicates("display_name").set_index("display_name")
    y_labels = []
    y_colors = []
    for row_name in all_rows:
        if "─" in row_name:
            y_labels.append("")
            y_colors.append("#ffffff")
        elif row_name in baselines:
            y_labels.append(f"Human ({row_name})")
            y_colors.append("#555555")
        else:
            prov = meta.loc[row_name, "provider"] if row_name in meta.index else "?"
            y_labels.append(row_name)
            y_colors.append(_pc(prov))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=12.0)
    for ytick, col in zip(ax.get_yticklabels(), y_colors):
        ytick.set_color(col)

    ax.set_xlim(0, 1.01)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels([f"{int(v*100)}%" for v in np.arange(0, 1.1, 0.1)], fontsize=12)
    ax.set_xlabel("Proportion of Responses", labelpad=5, fontsize=14)
    ax.set_title(
        "Kohlberg Stage Distribution — All Models vs. Human Baselines\n"
        "(hatch = human baseline; y-axis colour = provider; "
        "bars sorted by model scale small→large)",
        fontsize=16, pad=8, fontweight="bold",
    )

    leg = ax.legend(
        handles=_stage_handles(),
        title="Kohlberg Stage", title_fontsize=12,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=11.0, framealpha=0.92, edgecolor="#cccccc",
        ncol=1,
    )

    fig.tight_layout()
    _save(fig, out_dir, "fig1_stacked_bar.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Histogram grid: per-model stage proportions
# ═══════════════════════════════════════════════════════════════════════════

def plot_histogram_grid(dist_df: pd.DataFrame, chi_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Grid of bar charts (one per model). Human adult baseline overlaid as a
    step line. Each panel annotated with χ² p-value + JSD.
    """
    model_order = (
        dist_df.drop_duplicates("model_key")
        .sort_values("params_B")
    )
    n_models = len(model_order)
    n_cols   = 4
    n_rows   = int(np.ceil(n_models / n_cols))

    human_props = np.array([HUMAN_ADULT[s] for s in STAGES])
    x = np.arange(len(STAGES))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(205 * MM, n_rows * 42 * MM),
                              sharey=False)
    axes = axes.flatten()

    chi_idx = chi_df.set_index("model_key")

    for ax_idx, (_, mrow) in enumerate(model_order.iterrows()):
        ax  = axes[ax_idx]
        mk  = mrow["model_key"]
        sub = dist_df[dist_df["model_key"] == mk].sort_values("stage")
        props = sub["proportion"].values

        bar_colors = [STAGE_COLORS[s] for s in STAGES]
        ax.bar(x, props, color=bar_colors, alpha=0.85, edgecolor="white",
               linewidth=0.4, zorder=3)

        # Human baseline step line
        ax.step(
            np.append(x - 0.5, x[-1] + 0.5),
            np.append(human_props, human_props[-1]),
            where="post", color="#cc3333", linewidth=1.2,
            linestyle="--", label="Human Adult", zorder=5,
        )

        # Annotate χ² and JSD
        if mk in chi_idx.index:
            p_val = chi_idx.loc[mk, "chi2_p"]
            jsd   = chi_idx.loc[mk, "jsd_adult"]
            sig   = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
            ax.text(0.97, 0.97,
                    f"χ² {sig}\nJSD={jsd:.3f}",
                    transform=ax.transAxes, fontsize=8.0,
                    va="top", ha="right", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#cccccc", lw=0.5, alpha=0.90))

        prov_col = _pc(mrow["provider"])
        ax.set_title(
            mrow["display_name"],
            fontsize=11.0, pad=4, fontweight="bold", color=prov_col,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}" for s in STAGES], fontsize=10.0)
        ax.set_ylim(0, 1.05)
        ax.set_yticklabels([])
        ax.yaxis.set_visible(ax_idx % n_cols == 0)
        if ax_idx % n_cols == 0:
            ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10.0)
        ax.tick_params(axis="x", length=2)

    # Hide unused panels
    for ax in axes[n_models:]:
        ax.set_visible(False)

    # Single legend for the human baseline line
    handles = [
        mlines.Line2D([], [], color="#cc3333", lw=1.2, ls="--", label="Human Adult Baseline"),
    ] + [
        mpatches.Patch(facecolor=STAGE_COLORS[s], edgecolor="#444", lw=0.4,
                       label=f"S{s}", alpha=0.85)
        for s in STAGES
    ]
    fig.legend(
        handles=handles,
        title="Stage / Baseline", title_fontsize=12.0,
        loc="upper center", ncol=7,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=11.0, framealpha=0.92, edgecolor="#cccccc",
    )

    fig.suptitle(
        "Per-Model Stage Proportion  (bars = model; dashed red = Human Adult baseline)\n"
        "* p<0.05  ** p<0.01  *** p<0.001  n.s. = not significant",
        fontsize=16.0, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig2_histogram_grid.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — JSD clustermap (symmetric N×N)
# ═══════════════════════════════════════════════════════════════════════════

def plot_jsd_heatmap(jsd_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Seaborn clustermap of the symmetric JSD matrix.
    Human baseline rows/cols at the bottom with a distinct colour strip.
    """
    n = len(jsd_df)
    is_human = jsd_df.index.str.startswith("Human")
    row_colors = ["#cc4444" if h else "#4477aa" for h in is_human]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.clustermap(
            jsd_df,
            cmap="YlOrRd", vmin=0, vmax=0.55,
            annot=True, fmt=".2f",
            annot_kws={"size": 10.0, "fontweight": "bold"},
            linewidths=0.4, linecolor="#e0e0e0",
            row_colors=row_colors,
            col_colors=row_colors,
            row_cluster=True, col_cluster=True,
            figsize=(195 * MM, 195 * MM),
            dendrogram_ratio=(0.10, 0.10),
            cbar_pos=(1.04, 0.30, 0.025, 0.35),
            cbar_kws={"label": "Jensen-Shannon Divergence"},
            method="average", metric="euclidean",
        )

    g.ax_cbar.tick_params(labelsize=12)
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), fontsize=12.0, rotation=40, ha="right"
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), fontsize=12.0, rotation=0
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    # Legend strips
    handles = [
        mpatches.Patch(facecolor="#4477aa", edgecolor="#444", lw=0.5, label="LLM model"),
        mpatches.Patch(facecolor="#cc4444", edgecolor="#444", lw=0.5, label="Human baseline"),
    ]
    g.figure.legend(
        handles=handles, title="Row / Col type",
        loc="upper left", bbox_to_anchor=(0.01, 0.99),
        fontsize=12, framealpha=0.92, edgecolor="#cccccc",
    )

    g.figure.suptitle(
        "Pairwise Jensen-Shannon Divergence\n"
        "(0 = identical distributions; 1 = maximally different)",
        fontsize=16, fontweight="bold", y=1.03,
    )
    _save(g.figure, out_dir, "fig3_jsd_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Distribution stats multi-panel
# ═══════════════════════════════════════════════════════════════════════════

def plot_distribution_stats(stat_df: pd.DataFrame, out_dir: Path) -> None:
    """
    4-panel figure:
      (a) Shannon entropy (horizontal bars, coloured by provider)
      (b) Skewness (diverging bar: negative = left-skewed = high stages)
      (c) Mean stage vs log(params) scatter, sized by entropy
      (d) Pattern label annotation strip
    """
    from adjustText import adjust_text
    s = stat_df.sort_values("params_B").reset_index(drop=True)
    n = len(s)
    colors = [_pc(p) for p in s["provider"]]

    fig = plt.figure(figsize=(215 * MM, 165 * MM))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
    ax_ent  = fig.add_subplot(gs[0, 0])
    ax_skew = fig.add_subplot(gs[0, 1])
    ax_scat = fig.add_subplot(gs[1, 0])
    ax_pat  = fig.add_subplot(gs[1, 1])

    y = np.arange(n)

    # ── (a) Entropy ────────────────────────────────────────────────────────
    max_ent = np.log2(6)   # uniform distribution over 6 stages ≈ 2.585 bits
    ax_ent.barh(y, s["entropy_bits"], color=colors, alpha=0.85, height=0.65)
    ax_ent.axvline(max_ent, color="#cc3333", lw=1.0, ls="--",
                   label=f"Max entropy (uniform) = {max_ent:.2f} bits")
    human_ent = float(-np.sum([v * np.log2(v + 1e-9) for v in HUMAN_ADULT.values()]))
    ax_ent.axvline(human_ent, color="#006699", lw=1.0, ls=":",
                   label=f"Human adult = {human_ent:.2f} bits")
    ax_ent.set_yticks(y)
    ax_ent.set_yticklabels(s["display_name"], fontsize=7.0)
    ax_ent.set_xlabel("Shannon Entropy (bits)", fontsize=8.5)
    ax_ent.set_title("(a) Distribution Entropy", fontsize=9, fontweight="bold")
    ax_ent.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    ax_ent.set_xlim(0, max_ent + 0.25)

    # ── (b) Skewness ───────────────────────────────────────────────────────
    skew_colors = ["#4575b4" if v < 0 else "#d73027" for v in s["skewness"]]
    ax_skew.barh(y, s["skewness"], color=skew_colors, alpha=0.82, height=0.65)
    ax_skew.axvline(0, color="#333333", lw=0.8, ls="-")
    ax_skew.set_yticks(y)
    ax_skew.set_yticklabels(s["display_name"], fontsize=7.0)
    ax_skew.set_xlabel("Skewness  (−= left tail = high stages)", fontsize=8.5)
    ax_skew.set_title("(b) Skewness of Stage Distribution", fontsize=9, fontweight="bold")
    # Annotation arrows
    ax_skew.text(ax_skew.get_xlim()[0], -0.9, "← Higher stages", fontsize=6.5, color="#4575b4")
    ax_skew.text(0.01, -0.9, "Lower stages →",  fontsize=6.5, color="#d73027")

    # ── (c) Mean stage vs scale scatter ────────────────────────────────────
    sizes = ((s["entropy_bits"] / max_ent) * 200).clip(lower=30)
    ax_scat.scatter(s["log_params"], s["mean_stage"],
                    s=sizes, c=colors, alpha=0.87,
                    edgecolors="#333", linewidths=0.6, zorder=5)
    # Human adult baseline horizontal line
    human_mean = float(np.sum([k * v for k, v in HUMAN_ADULT.items()]))
    ax_scat.axhline(human_mean, color="#cc3333", lw=1.0, ls="--",
                    label=f"Human Adult mean = {human_mean:.2f}")
    texts = [
        ax_scat.text(float(s.loc[i, "log_params"]), float(s.loc[i, "mean_stage"]),
                     s.loc[i, "display_name"], fontsize=6.0, color="#222222")
        for i in range(n)
    ]
    adjust_text(
        texts, ax=ax_scat,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4, shrinkA=3),
        expand=(2.0, 2.5), force_text=(1.2, 1.5), min_arrow_len=4, max_move=5.0,
    )
    nice_P = [8, 10, 30, 100, 300, 700]
    ax_scat.set_xticks([np.log10(v) for v in nice_P if np.log10(v) <= s["log_params"].max() + 0.1])
    ax_scat.set_xticklabels([f"{v}B" for v in nice_P if np.log10(v) <= s["log_params"].max() + 0.1], fontsize=7.5)
    ax_scat.set_yticks(STAGES)
    ax_scat.set_yticklabels([f"S{st}" for st in STAGES], fontsize=7.5)
    ax_scat.set_xlabel("Model Scale (log)", fontsize=8.5)
    ax_scat.set_ylabel("Mean Stage", fontsize=8.5)
    ax_scat.set_title("(c) Mean Stage vs Scale\n(size ∝ entropy)", fontsize=9, fontweight="bold")
    ax_scat.legend(fontsize=6.5, loc="lower right", framealpha=0.9)

    # ── (d) Pattern strip ──────────────────────────────────────────────────
    PATTERN_COLORS = {
        "human-like":       "#009E73",
        "ceiling-biased":   "#56B4E9",
        "hyper-principled": "#0072B2",
        "bimodal":          "#E69F00",
        "floor-biased":     "#D55E00",
        "divergent":        "#CC79A7",
    }
    pattern_col = s["pattern"].map(PATTERN_COLORS).fillna("#bbbbbb")
    ax_pat.barh(y, np.ones(n), color=pattern_col.values, alpha=0.80, height=0.7, edgecolor="white")
    ax_pat.set_yticks(y)
    ax_pat.set_yticklabels(s["display_name"], fontsize=7.0)
    for i, (_, row) in enumerate(s.iterrows()):
        ax_pat.text(0.5, i, row["pattern"],
                    ha="center", va="center", fontsize=6.8,
                    color="white", fontweight="bold")
    ax_pat.set_xlim(0, 1)
    ax_pat.set_xticks([])
    ax_pat.set_title("(d) Distribution Pattern Type", fontsize=9, fontweight="bold")
    pat_handles = [
        mpatches.Patch(facecolor=c, edgecolor="#444", lw=0.5, label=p, alpha=0.85)
        for p, c in PATTERN_COLORS.items()
    ]
    ax_pat.legend(handles=pat_handles, title="Pattern", title_fontsize=7,
                  loc="lower left", bbox_to_anchor=(1.02, 0.0),
                  fontsize=7, framealpha=0.92, edgecolor="#cccccc")

    fig.suptitle(
        "Stage Distribution Characteristics per Model",
        fontsize=11, fontweight="bold", y=1.01,
    )
    _save(fig, out_dir, "fig4_distribution_stats.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Chi-square bar + Pearson residual heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_chi_square(chi_df: pd.DataFrame, resid_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Left: horizontal bar of χ² statistic per model, coloured by significance.
    Right: heatmap of Pearson residuals (model × stage).
    """
    s = chi_df.sort_values("params_B").reset_index(drop=True)
    n = len(s)
    y = np.arange(n)

    def _sig_color(row):
        if row["chi2_p"] < 0.001: return "#d73027"
        if row["chi2_p"] < 0.01:  return "#fc8d59"
        if row["chi2_p"] < 0.05:  return "#fee090"
        return "#91bfdb"

    bar_colors = [_sig_color(r) for _, r in s.iterrows()]

    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(215 * MM, max(110, n * 12) * MM),
        gridspec_kw={"width_ratios": [2.2, 3.0], "wspace": 0.35},
    )

    # ── Left: χ² bar ───────────────────────────────────────────────────────
    ax_bar.barh(y, s["chi2_stat"], color=bar_colors, alpha=0.88, height=0.65, edgecolor="white")
    critical = 7.815   # χ²(df=3, α=0.05)
    ax_bar.axvline(critical, color="#333333", lw=0.9, ls="--",
                   label=f"χ²(3) critical = {critical:.2f}  (α = 0.05)")
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(s["display_name"], fontsize=8.0)
    ax_bar.set_xlabel("χ² Statistic", labelpad=5)
    ax_bar.set_title(
        "Chi-Square Goodness-of-Fit\n(vs. Human Adult Norm; df = 3)",
        fontsize=9, pad=6, fontweight="bold",
    )
    ax_bar.legend(fontsize=7.0, loc="lower right")
    sig_handles = [
        mpatches.Patch(facecolor="#d73027", label="p < 0.001 ***", alpha=0.85),
        mpatches.Patch(facecolor="#fc8d59", label="p < 0.01 **",   alpha=0.85),
        mpatches.Patch(facecolor="#fee090", label="p < 0.05 *",    alpha=0.85),
        mpatches.Patch(facecolor="#91bfdb", label="n.s.",           alpha=0.85),
    ]
    ax_bar.legend(handles=sig_handles, title="Significance",
                  loc="lower right", fontsize=7, title_fontsize=7,
                  framealpha=0.92, edgecolor="#cccccc")

    # ── Right: Pearson residual heatmap ────────────────────────────────────
    # Pivot residuals
    pivot = (
        resid_df
        .merge(chi_df[["model_key", "display_name", "params_B"]], on="model_key")
        .sort_values("params_B")
        .pivot(index="display_name", columns="stage", values="pearson_residual")
        .reindex(s["display_name"])
    )
    pivot.columns = [f"S{c}" for c in pivot.columns]

    sns.heatmap(
        pivot,
        ax=ax_heat,
        cmap="RdBu_r",
        center=0, vmin=-8, vmax=8,
        annot=True, fmt=".1f",
        annot_kws={"size": 7.0},
        linewidths=0.4, linecolor="#e8e8e8",
        cbar_kws={"label": "Pearson Residual\n(+= over-represented vs norm)", "shrink": 0.8},
    )
    ax_heat.set_xlabel("Stage", labelpad=5)
    ax_heat.set_ylabel("")
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), fontsize=7.5, rotation=0)
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), fontsize=8.0)
    ax_heat.set_title(
        "Pearson Residuals by Stage\n(which stages drive misfit?)",
        fontsize=9, pad=6, fontweight="bold",
    )

    fig.suptitle(
        "Statistical Comparison vs. Human Adult Norms\n"
        "Left: χ² per model   |   Right: Stage-level Pearson residuals",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig5_chi_square.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 — 3D stage landscape
# ═══════════════════════════════════════════════════════════════════════════

def plot_3d_stage_landscape(dist_df: pd.DataFrame, out_dir: Path) -> None:
    """
    3D bar chart: X = stage (1–6), Y = model index (small→large), Z = proportion.
    A semi-transparent reference plane for Human Adult proportions overlaid.
    """
    model_order = (
        dist_df.drop_duplicates("model_key")
        .sort_values("params_B")["display_name"]
        .tolist()
    )
    n_models = len(model_order)
    n_stages  = len(STAGES)

    # Proportion matrix: (n_stages × n_models)
    Z = np.zeros((n_stages, n_models))
    for j, model_name in enumerate(model_order):
        sub = dist_df[dist_df["display_name"] == model_name].sort_values("stage")
        Z[:, j] = sub["proportion"].values

    human_z = np.array([HUMAN_ADULT[s] for s in STAGES])

    fig = plt.figure(figsize=(220 * MM, 165 * MM))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3.2, 1.0], figure=fig,
                             left=0.03, right=0.97, wspace=0.06)
    ax     = fig.add_subplot(gs[0], projection="3d")
    ax_key = fig.add_subplot(gs[1])
    ax_key.axis("off")

    bar_w = 0.55
    bar_d = 0.55
    x_base = np.arange(n_stages)
    y_base = np.arange(n_models)

    for i_s, s in enumerate(STAGES):
        col = STAGE_COLORS[s]
        for j_m in range(n_models):
            z_val = float(Z[i_s, j_m])
            if z_val <= 0:
                continue
            ax.bar3d(
                x=i_s - bar_w / 2,
                y=j_m - bar_d / 2,
                z=0,
                dx=bar_w, dy=bar_d, dz=z_val,
                color=col, alpha=0.78, shade=True,
            )

    # Human adult reference plane (transparent surface)
    y_grid = np.linspace(-0.5, n_models - 0.5, 2)
    for i_s, s in enumerate(STAGES):
        h = float(human_z[i_s])
        if h > 0:
            xx = np.array([[i_s - 0.5, i_s + 0.5], [i_s - 0.5, i_s + 0.5]])
            yy = np.array([[y_grid[0],  y_grid[0]],  [y_grid[1],  y_grid[1]]])
            zz = np.full_like(xx, h)
            ax.plot_surface(xx, yy, zz, color="#cc3333", alpha=0.18)

    # Axes
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"S{s}" for s in STAGES], fontsize=7.5)
    ax.set_yticks(np.arange(0, n_models, 2))
    y_sparse = [str(i + 1) if i % 2 == 0 else "" for i in range(n_models)]
    ax.set_yticklabels([y_sparse[i] for i in range(0, n_models, 2)], fontsize=7.0)
    ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_zticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7.0)
    ax.set_zlim(0, 1.05)
    ax.set_xlabel("Kohlberg Stage", labelpad=7, fontsize=8.5)
    ax.set_ylabel("Model Index (small→large)", labelpad=8, fontsize=8.5)
    ax.set_zlabel("Proportion", labelpad=5, fontsize=8.5)
    ax.set_title(
        "3D Stage Landscape\n(red plane = Human Adult norm)",
        fontsize=10, pad=10, fontweight="bold",
    )
    ax.view_init(elev=25, azim=-50)
    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.04)

    # Stage colour legend
    stage_handles = [
        mpatches.Patch(facecolor=STAGE_COLORS[s], edgecolor="#444",
                       lw=0.5, label=f"S{s}", alpha=0.85)
        for s in STAGES
    ]
    human_handle = mpatches.Patch(facecolor="#cc3333", edgecolor="#444",
                                   lw=0.5, label="Human Adult  (ref plane)", alpha=0.35)
    ax.legend(handles=stage_handles + [human_handle],
              title="Stage / Ref", title_fontsize=7.5,
              loc="upper left", bbox_to_anchor=(-0.02, 0.98),
              fontsize=7.5, framealpha=0.88, edgecolor="#cccccc")

    # Model key panel
    meta = dist_df.drop_duplicates("display_name").set_index("display_name")
    ax_key.set_xlim(0, 1); ax_key.set_ylim(0, 1)
    ax_key.text(0.0, 0.99, "Model Key", fontsize=8.5, fontweight="bold",
                va="top", ha="left", transform=ax_key.transAxes)
    ax_key.plot([0, 1], [0.955, 0.955], color="#aaaaaa", lw=0.6,
                transform=ax_key.transAxes, clip_on=False)
    row_h = 0.93 / n_models
    for j, model_name in enumerate(model_order):
        y_frac = 0.93 - j * row_h
        prov   = meta.loc[model_name, "provider"] if model_name in meta.index else "?"
        ax_key.text(0.02, y_frac, str(j + 1), fontsize=7.5, va="center",
                    ha="left", transform=ax_key.transAxes,
                    fontweight="bold", color=_pc(prov))
        ax_key.text(0.20, y_frac, model_name, fontsize=6.5, va="center",
                    ha="left", transform=ax_key.transAxes, color="#222222")
        if j % 2 == 0:
            ax_key.axhspan(y_frac - row_h * 0.5, y_frac + row_h * 0.5,
                           xmin=0, xmax=1, color="#f5f5f5", alpha=0.7,
                           transform=ax_key.transAxes, zorder=0)

    _save(fig, out_dir, "fig6_3d_stage_landscape.png")
