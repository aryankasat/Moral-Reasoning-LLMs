"""
visualizations.py — Research-paper-quality figures for Analysis 8.

Design principles (Nature / APA / PLOS ONE style):
  - Okabe-Ito colour palette (colour-blind safe)
  - Single-column width (3.5 in) or double-column (7.2 in) figures
  - 300 DPI output; vector-compatible line weights
  - Minimal grid: only light horizontal tick lines (no full grid box)
  - Significance brackets drawn inside figure area
  - All fonts 8–11 pt (readable at 100% scale in a two-column journal)
  - No chartjunk: no unnecessary borders, no 3-D, no background fill
  - All axes labelled with units / scale information

Figures:
  F1  interaction_plot.png   — mean ± 95% CI per cell, 3 Scale lines × Training Type
  F2  bar_with_jitter.png    — grouped bar chart with individual model dots overlaid
  F3  box_violin.png         — combined box + violin per Scale group (full distribution)
  F4  posthoc_matrix.png     — Tukey HSD significance matrix (dot plot style)
  F5  variance_bar.png       — horizontal bar chart of variance partition (replaces pie)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy import stats as scipy_stats

from config import (
    OUT_DIR, SCALE_ORDER, TRAINING_ORDER,
    apply_publication_style,
)

# ── Okabe-Ito palette (colour-blind safe) ─────────────────────────────────────
OI = {
    "orange":      "#E69F00",
    "sky_blue":    "#56B4E9",
    "green":       "#009E73",
    "yellow":      "#F0E442",
    "blue":        "#0072B2",
    "vermillion":  "#D55E00",
    "pink":        "#CC79A7",
    "black":       "#000000",
}

SCALE_COLORS  = {"Small": OI["blue"], "Mid": OI["vermillion"], "Large": OI["green"]}
SCALE_MARKERS = {"Small": "o", "Mid": "s", "Large": "^"}
TRAINING_COLORS = {
    "Base-RLHF":       OI["blue"],
    "Coding-Tuned":    OI["orange"],
    "Reasoning-Tuned": OI["vermillion"],
}
TRAINING_HATCHES = {"Base-RLHF": "", "Coding-Tuned": "///", "Reasoning-Tuned": "xxx"}

# Figure geometry constants
SINGLE_COL = (3.5, 2.8)   # inches — single journal column
DOUBLE_COL = (7.2, 3.6)   # inches — double journal column
TALL_SINGLE = (3.5, 4.0)
TALL_DOUBLE = (7.2, 4.8)


def _setup() -> None:
    """Apply publication rcParams."""
    apply_publication_style()
    mpl.rcParams.update({
        "font.family":          "DejaVu Sans",   # widely installed, clean
        "font.size":            9,
        "axes.titlesize":       10,
        "axes.labelsize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.fontsize":      8,
        "legend.title_fontsize":8,
        "axes.linewidth":       0.7,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "grid.linestyle":       ":",
        "grid.linewidth":       0.4,
        "grid.alpha":           0.55,
        "grid.color":           "#bbbbbb",
        "axes.axisbelow":       True,
        "lines.linewidth":      1.6,
        "lines.markersize":     6,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
    })


def _sig_label(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _bracket(ax, x1, x2, y, height, text, color="black", fontsize=8):
    """Draw a significance bracket between x1 and x2 at height y."""
    dy = height * 0.25
    ax.plot([x1, x1, x2, x2],
            [y, y + dy, y + dy, y],
            lw=0.8, color=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + dy * 1.05, text,
            ha="center", va="bottom", fontsize=fontsize,
            color=color, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# F1: Interaction Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_interaction(cell_df: pd.DataFrame,
                     posthoc_results: dict | None = None) -> str:
    """
    Three lines (one per Scale group) with 95% CI error bars.
    X-axis: Training Type.  Y-axis: Mean Kohlberg Stage.
    """
    _setup()
    fig, ax = plt.subplots(figsize=DOUBLE_COL, constrained_layout=True)

    x_pos = {tt: i for i, tt in enumerate(TRAINING_ORDER)}
    x_arr  = np.arange(len(TRAINING_ORDER))

    for sg in SCALE_ORDER:
        sg_data = cell_df[cell_df["scale_group"] == sg].copy()
        sg_data = sg_data.set_index("training_type").reindex(TRAINING_ORDER).reset_index()

        xs   = np.array([x_pos[tt] for tt in sg_data["training_type"]])
        ys   = sg_data["mean_stage"].values.astype(float)
        cil  = (sg_data["mean_stage"] - sg_data["ci_lower"]).values.astype(float)
        ciu  = (sg_data["ci_upper"] - sg_data["mean_stage"]).values.astype(float)

        mask = ~np.isnan(ys)
        col  = SCALE_COLORS[sg]
        mrk  = SCALE_MARKERS[sg]

        ax.errorbar(
            xs[mask], ys[mask],
            yerr=[cil[mask], ciu[mask]],
            color=col, marker=mrk, markersize=7,
            linewidth=1.8, capsize=4, capthick=1.2, elinewidth=1.0,
            label=f"{sg} (n={int(cell_df[cell_df['scale_group']==sg]['n_obs'].sum())})",
            zorder=3,
        )
        # Shade the CI band for existing points
        if mask.sum() >= 2:
            ax.fill_between(
                xs[mask], ys[mask] - cil[mask], ys[mask] + ciu[mask],
                alpha=0.10, color=col, zorder=2,
            )

    # Reference line at Stage 5 (post-conventional threshold)
    ax.axhline(5, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(len(TRAINING_ORDER) - 0.03, 5.03,
            "Post-conventional\nthreshold (Stage 5)",
            ha="right", va="bottom", fontsize=7, color="#777777", style="italic")

    # Significant bracket for Large: Reasoning vs Base
    # (p=0.039 from Tukey HSD — Training within Large)
    tt_x = {tt: i for i, tt in enumerate(TRAINING_ORDER)}
    sig_y = 6.15
    _bracket(ax, tt_x["Base-RLHF"], tt_x["Reasoning-Tuned"],
             sig_y, 0.06, "* (p = .039)", color=SCALE_COLORS["Large"], fontsize=7.5)

    ax.set_xticks(x_arr)
    ax.set_xticklabels(TRAINING_ORDER, fontsize=9)
    ax.set_xlim(-0.55, len(TRAINING_ORDER) - 0.45)
    ax.set_ylim(4.7, 6.45)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.125))
    ax.set_xlabel("Training Procedure", fontsize=9, labelpad=5)
    ax.set_ylabel("Mean Kohlberg Stage (± 95% CI)", fontsize=9, labelpad=5)
    ax.set_title(
        "Scale × Training Type Interaction Plot\n"
        r"(parallel lines $\Rightarrow$ additive effects; non-parallel $\Rightarrow$ interaction)",
        fontsize=10, fontweight="bold", pad=8,
    )

    leg = ax.legend(title="Parameter Scale", loc="lower left",
                    framealpha=0.92, edgecolor="#cccccc",
                    handlelength=2.0, borderpad=0.6)
    leg.get_title().set_fontsize(8)

    # Panel label
    ax.text(-0.08, 1.04, "A", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

    out = OUT_DIR / "interaction_plot.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [SAVED] {out.name}")
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# F2: Grouped Bar Chart with Individual Model Jitter
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar_with_jitter(raw_df: pd.DataFrame, cell_df: pd.DataFrame) -> str:
    """
    Grouped bars for cell means; individual model dots superimposed.
    Groups on x-axis = Training Type; bar colour = Scale group.
    """
    _setup()
    fig, ax = plt.subplots(figsize=DOUBLE_COL, constrained_layout=True)

    n_tt   = len(TRAINING_ORDER)
    n_sg   = len(SCALE_ORDER)
    width  = 0.22
    group_w = n_sg * width + 0.08  # total width of one training-type cluster
    offsets = np.linspace(-(n_sg - 1) / 2 * width, (n_sg - 1) / 2 * width, n_sg)

    # model-level means for jitter
    model_means = (raw_df.groupby(["display_name", "scale_group", "training_type"],
                                   observed=True)["kohlberg_stage"]
                   .mean().reset_index()
                   .rename(columns={"kohlberg_stage": "mean_stage"}))

    for si, sg in enumerate(SCALE_ORDER):
        col = SCALE_COLORS[sg]
        for ti, tt in enumerate(TRAINING_ORDER):
            cell = cell_df[(cell_df["scale_group"] == sg) & (cell_df["training_type"] == tt)]
            if cell.empty:
                continue
            x      = ti + offsets[si]
            mean_v = float(cell["mean_stage"].iloc[0])
            ci_lo  = float(cell["ci_lower"].iloc[0])
            ci_hi  = float(cell["ci_upper"].iloc[0])

            bar = ax.bar(x, mean_v, width=width * 0.92,
                         color=col, alpha=0.72, edgecolor="white",
                         linewidth=0.5, zorder=3,
                         label=sg if ti == 0 else "_nolegend_")
            ax.errorbar(x, mean_v, yerr=[[mean_v - ci_lo], [ci_hi - mean_v]],
                        fmt="none", color="black", capsize=3, capthick=0.9,
                        elinewidth=0.9, zorder=4)

            # Jitter individual model points
            mdots = model_means[(model_means["scale_group"] == sg) &
                                 (model_means["training_type"] == tt)]
            if not mdots.empty:
                jitter = np.random.default_rng(42).uniform(-width * 0.25, width * 0.25, len(mdots))
                ax.scatter(x + jitter, mdots["mean_stage"].values,
                           color="white", edgecolors=col, s=28, linewidths=1.0,
                           zorder=5, alpha=0.95)

    ax.axhline(5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xticks(range(n_tt))
    ax.set_xticklabels(TRAINING_ORDER, fontsize=9)
    ax.set_ylim(4.0, 6.55)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))
    ax.set_xlabel("Training Procedure", fontsize=9, labelpad=5)
    ax.set_ylabel("Mean Kohlberg Stage (± 95% CI)", fontsize=9, labelpad=5)
    ax.set_title("Cell Means by Scale Group & Training Type\n"
                 "(open circles = individual model means)",
                 fontsize=10, fontweight="bold", pad=8)

    patches = [mpatches.Patch(facecolor=SCALE_COLORS[sg], label=sg, alpha=0.8)
               for sg in SCALE_ORDER]
    ax.legend(handles=patches, title="Parameter Scale",
              loc="lower right", framealpha=0.92, edgecolor="#cccccc")

    ax.text(-0.08, 1.04, "B", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

    out = OUT_DIR / "bar_with_jitter.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [SAVED] {out.name}")
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# F3: Box + Violin per Scale Group (full distribution)
# ─────────────────────────────────────────────────────────────────────────────

def plot_box_violin(raw_df: pd.DataFrame) -> str:
    """
    Combined half-violin + box plot per Scale group.
    Shows full distribution of kohlberg_stage scores.
    """
    _setup()
    fig, ax = plt.subplots(figsize=SINGLE_COL, constrained_layout=True)

    for si, sg in enumerate(SCALE_ORDER):
        data = raw_df[raw_df["scale_group"] == sg]["kohlberg_stage"].values
        col  = SCALE_COLORS[sg]
        x    = si

        # Violin (half — right side)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vp = ax.violinplot([data], positions=[x], widths=0.6,
                               showmeans=False, showextrema=False, showmedians=False)
        for body in vp["bodies"]:
            body.set_facecolor(col)
            body.set_alpha(0.30)
            body.set_edgecolor(col)
            body.set_linewidth(0.8)

        # Box plot
        bp = ax.boxplot([data], positions=[x], widths=0.25,
                        patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=1.6),
                        boxprops=dict(facecolor=col, alpha=0.55, linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8, color=col),
                        capprops=dict(linewidth=0.8, color=col),
                        flierprops=dict(marker="o", color=col, markersize=3, alpha=0.6))

        # Mean dot
        ax.scatter([x], [data.mean()], color="black", s=22, zorder=5,
                   marker="D", linewidths=0, label="Mean" if si == 0 else "_")

        # n label
        ax.text(x, data.min() - 0.12, f"n={len(data)}",
                ha="center", va="top", fontsize=7, color="#555555")

    # Kruskal-Wallis annotation
    sg_groups = [raw_df[raw_df["scale_group"] == sg]["kohlberg_stage"].values
                 for sg in SCALE_ORDER]
    H, p_kw = scipy_stats.kruskal(*sg_groups)
    ax.text(0.97, 0.97, f"Kruskal-Wallis H={H:.2f}, p={p_kw:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_xticks(range(len(SCALE_ORDER)))
    ax.set_xticklabels([f"{sg}\nScale" for sg in SCALE_ORDER], fontsize=9)
    ax.set_ylim(3.2, 7.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
    ax.set_ylabel("Kohlberg Stage", fontsize=9, labelpad=5)
    ax.set_title("Distribution of Moral Reasoning Stage\nby Parameter Scale Group",
                 fontsize=10, fontweight="bold", pad=8)
    ax.axhline(5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.text(len(SCALE_ORDER) - 0.03, 5.05, "Stage 5",
            ha="right", va="bottom", fontsize=7, color="#888888", style="italic")

    leg = ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9, edgecolor="#cccccc")
    ax.text(-0.18, 1.04, "C", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

    out = OUT_DIR / "box_violin.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [SAVED] {out.name}")
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# F4: Post-hoc Significance Matrix (dot-plot style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_posthoc_matrix(posthoc_results: dict[str, pd.DataFrame]) -> str:
    """
    For each Tukey HSD result, draw a compact significance matrix:
    square for each pair, colour = adjusted p-value, dot if significant.
    """
    _setup()

    # Filter to Tukey tables only
    tukey_keys = [k for k in posthoc_results if not k.startswith("MannWhitney")]
    n = len(tukey_keys)
    if n == 0:
        return ""

    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.8 * ncols, 3.2 * nrows),
                             constrained_layout=True)
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    cmap = mpl.colormaps["RdYlGn_r"].resampled(256)

    for ax_idx, key in enumerate(tukey_keys):
        ax  = axes[ax_idx // ncols][ax_idx % ncols]
        ph  = posthoc_results[key].copy()
        ph.columns = [c.strip() for c in ph.columns]

        g1_col = next((c for c in ph.columns if "group1" in c.lower()), None)
        g2_col = next((c for c in ph.columns if "group2" in c.lower()), None)
        pv_col = next((c for c in ph.columns if "p-adj" in c.lower()
                        or "padj" in c.lower() or "p_adj" in c.lower()), None)
        md_col = next((c for c in ph.columns if "meandiff" in c.lower()
                        or "mean_diff" in c.lower()), None)
        if not (g1_col and g2_col and pv_col):
            ax.text(0.5, 0.5, "Parse error", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        groups = sorted(set(ph[g1_col].astype(str)) | set(ph[g2_col].astype(str)))
        m      = len(groups)
        M      = np.ones((m, m))  # default p=1
        D      = np.zeros((m, m))

        for _, row in ph.iterrows():
            i = groups.index(str(row[g1_col]))
            j = groups.index(str(row[g2_col]))
            try:
                p = float(row[pv_col])
                d = float(row[md_col]) if md_col else 0.0
            except Exception:
                p, d = 1.0, 0.0
            M[i, j] = M[j, i] = p
            D[i, j] = D[j, i] = d

        im = ax.imshow(M, vmin=0, vmax=0.15, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Tukey adj. p-value")

        ax.set_xticks(range(m)); ax.set_yticks(range(m))
        ax.set_xticklabels(groups, fontsize=8, rotation=30, ha="right")
        ax.set_yticklabels(groups, fontsize=8)

        for i in range(m):
            for j in range(m):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=8.5, color="#555555")
                else:
                    sym  = _sig_label(M[i, j])
                    diff = f"Δ={D[i,j]:+.2f}" if D[i, j] != 0 else ""
                    txt  = f"{sym}\n{diff}" if diff else sym
                    col  = "white" if M[i, j] < 0.05 else "#333333"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=7.5, color=col, fontweight="bold")

        label = (key.replace("Scale_within_", "Scale comparisons — Training: ")
                    .replace("Training_within_", "Training comparisons — Scale: ")
                    .replace("_", " "))
        ax.set_title(label, fontsize=9, fontweight="bold", pad=6)

    # Hide unused axes
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle(
        "Post-hoc Pairwise Comparisons (Tukey HSD)\n"
        "*** p<.001  ** p<.01  * p<.05  ns = not significant",
        fontsize=10, fontweight="bold", y=1.01,
    )

    out = OUT_DIR / "posthoc_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out.name}")
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# F5: Variance Partition Horizontal Bar (replaces pie)
# ─────────────────────────────────────────────────────────────────────────────

def plot_variance_bars(partition: dict[str, float],
                       effect_df: pd.DataFrame) -> str:
    """
    Horizontal stacked bar of % SS (η²-type) per source.
    A companion bar shows partial η² for each effect.
    """
    _setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), constrained_layout=True)

    # ── Left: % total SS stacked bar ──────────────────────────────────────────
    ax = axes[0]
    clean = {k: v for k, v in partition.items() if not np.isnan(v) and v > 0}
    labels = list(clean.keys())
    vals   = list(clean.values())

    colors_map = {
        "Scale":               OI["blue"],
        "Training_Type":       OI["orange"],
        "Scale × Training_Type": OI["vermillion"],
        "Residual":            "#cccccc",
    }
    colors = [colors_map.get(l, "#aaaaaa") for l in labels]

    left = 0
    for lbl, val, col in zip(labels, vals, colors):
        ax.barh(0, val, left=left, color=col, edgecolor="white", linewidth=0.5,
                label=f"{lbl.replace('Training_Type','Training').replace('Scale × Training_Type','Interaction')}"
                      f" ({val:.1f}%)")
        if val > 2.5:
            ax.text(left + val / 2, 0,
                    f"{val:.1f}%", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white" if col != "#cccccc" else "#333333")
        left += val

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Percentage of Total Variance (%)", fontsize=9)
    ax.set_title("Variance Partition (SS decomposition)", fontsize=9,
                 fontweight="bold", pad=6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.65),
              fontsize=7.5, framealpha=0.9, edgecolor="#cccccc",
              ncol=2, columnspacing=0.8, handlelength=1.2)
    ax.text(-0.06, 1.08, "D", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

    # ── Right: partial η² per effect ──────────────────────────────────────────
    ax2 = axes[1]
    eff_colors = {
        "Scale":               OI["blue"],
        "Training_Type":       OI["orange"],
        "Scale × Training_Type": OI["vermillion"],
    }
    eta_effects = effect_df[effect_df["effect"] != "Residual"].copy()
    y_pos = range(len(eta_effects))

    bars = ax2.barh(
        y_pos,
        eta_effects["partial_eta2"].values,
        color=[eff_colors.get(e, "#999999") for e in eta_effects["effect"]],
        edgecolor="white", linewidth=0.5,
        alpha=0.85,
    )

    # significance asterisk
    for i, row in eta_effects.reset_index(drop=True).iterrows():
        sym = _sig_label(float(row["p_value"])) if not np.isnan(float(row["p_value"])) else ""
        if sym and sym != "ns":
            ax2.text(float(row["partial_eta2"]) + 0.003, i, sym,
                     va="center", fontsize=9, color="#222222", fontweight="bold")
        # value label
        ax2.text(max(float(row["partial_eta2"]) / 2, 0.003), i,
                 f"η² = {float(row['partial_eta2']):.3f}",
                 va="center", ha="center", fontsize=7.5,
                 color="white" if float(row["partial_eta2"]) > 0.01 else "#555555",
                 fontweight="bold")

    # Effect size reference lines
    for thresh, lbl in [(0.01, "small"), (0.06, "medium"), (0.14, "large")]:
        ax2.axvline(thresh, color="#888888", linestyle=":", linewidth=0.8)
        ax2.text(thresh + 0.001, len(eta_effects) - 0.15, lbl,
                 fontsize=6.5, color="#777777", rotation=90, va="top")

    ax2.set_yticks(y_pos)
    display_names = (eta_effects["effect"]
                     .str.replace("Training_Type", "Training", regex=False)
                     .str.replace("Scale × Training_Type", "Interaction", regex=False))
    ax2.set_yticklabels(display_names, fontsize=8.5)
    ax2.set_xlabel("Partial η²", fontsize=9)
    ax2.set_title("Effect Sizes (partial η²)", fontsize=9, fontweight="bold", pad=6)
    ax2.set_xlim(0, max(0.12, eta_effects["partial_eta2"].max() * 1.35))

    ax2.text(-0.18, 1.08, "E", transform=ax2.transAxes,
             fontsize=12, fontweight="bold", va="top")

    out = OUT_DIR / "variance_bars.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [SAVED] {out.name}")
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# F6: Comprehensive Figure (4-panel summary — journal-ready)
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_panel(raw_df: pd.DataFrame,
                       cell_df: pd.DataFrame,
                       effect_df: pd.DataFrame,
                       partition: dict,
                       posthoc_results: dict) -> str:
    """
    4-panel journal figure:
      A — Interaction plot (mean ± CI, scale lines)
      B — Box/violin per scale group
      C — Grouped bar with model dots
      D — Effect size + variance bar
    """
    _setup()
    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=True)
    gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38)

    ax_A = fig.add_subplot(gs[0, 0])  # Interaction plot
    ax_B = fig.add_subplot(gs[0, 1])  # Box/violin scale
    ax_C = fig.add_subplot(gs[1, 0])  # Grouped bars
    ax_D = fig.add_subplot(gs[1, 1])  # Effect sizes

    # ── A: Interaction plot ───────────────────────────────────────────────────
    x_pos = {tt: i for i, tt in enumerate(TRAINING_ORDER)}
    for sg in SCALE_ORDER:
        sg_data = (cell_df[cell_df["scale_group"] == sg]
                   .set_index("training_type").reindex(TRAINING_ORDER).reset_index())
        xs  = np.array([x_pos[tt] for tt in sg_data["training_type"]])
        ys  = sg_data["mean_stage"].values.astype(float)
        cil = (sg_data["mean_stage"] - sg_data["ci_lower"]).values.astype(float)
        ciu = (sg_data["ci_upper"] - sg_data["mean_stage"]).values.astype(float)
        mask = ~np.isnan(ys)
        col  = SCALE_COLORS[sg]
        ax_A.errorbar(xs[mask], ys[mask], yerr=[cil[mask], ciu[mask]],
                      color=col, marker=SCALE_MARKERS[sg], markersize=6.5,
                      linewidth=1.8, capsize=3.5, capthick=1.1, elinewidth=0.9,
                      label=f"{sg}", zorder=3)
        if mask.sum() >= 2:
            ax_A.fill_between(xs[mask], ys[mask]-cil[mask], ys[mask]+ciu[mask],
                              alpha=0.08, color=col, zorder=2)

    ax_A.axhline(5, color="#999999", linestyle="--", linewidth=0.8)
    _bracket(ax_A, x_pos["Base-RLHF"], x_pos["Reasoning-Tuned"],
             6.10, 0.05, "* p = .039", color=SCALE_COLORS["Large"], fontsize=7)
    ax_A.set_xticks(range(len(TRAINING_ORDER)))
    ax_A.set_xticklabels(TRAINING_ORDER, fontsize=7.5, rotation=10, ha="right")
    ax_A.set_ylim(4.7, 6.4)
    ax_A.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax_A.set_ylabel("Mean Kohlberg Stage\n(± 95% CI)", fontsize=8)
    ax_A.set_xlabel("Training Procedure", fontsize=8)
    ax_A.legend(title="Scale", fontsize=7, title_fontsize=7,
                loc="lower left", framealpha=0.9, edgecolor="#cccccc")
    ax_A.text(-0.16, 1.06, "(A)", transform=ax_A.transAxes,
              fontsize=10, fontweight="bold", va="top")
    ax_A.set_title("Interaction Plot", fontsize=9, fontweight="bold", pad=4)

    # ── B: Box/Violin ────────────────────────────────────────────────────────
    sg_groups = [raw_df[raw_df["scale_group"] == sg]["kohlberg_stage"].values
                 for sg in SCALE_ORDER]
    H, p_kw = scipy_stats.kruskal(*sg_groups)

    for si, sg in enumerate(SCALE_ORDER):
        data = sg_groups[si]
        col  = SCALE_COLORS[sg]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vp = ax_B.violinplot([data], positions=[si], widths=0.55,
                                 showmeans=False, showextrema=False, showmedians=False)
        for body in vp["bodies"]:
            body.set_facecolor(col); body.set_alpha(0.28)
            body.set_edgecolor(col); body.set_linewidth(0.7)
        bp = ax_B.boxplot([data], positions=[si], widths=0.22, patch_artist=True,
                          notch=False,
                          medianprops=dict(color="black", linewidth=1.5),
                          boxprops=dict(facecolor=col, alpha=0.55, linewidth=0.7),
                          whiskerprops=dict(linewidth=0.7, color=col),
                          capprops=dict(linewidth=0.7, color=col),
                          flierprops=dict(marker="o", color=col, markersize=2.5, alpha=0.5))
        ax_B.scatter([si], [data.mean()], color="black", s=18, zorder=5, marker="D")

    ax_B.axhline(5, color="#999999", linestyle="--", linewidth=0.8)
    ax_B.set_xticks(range(len(SCALE_ORDER)))
    ax_B.set_xticklabels(SCALE_ORDER, fontsize=8)
    ax_B.set_ylim(3.5, 7.0)
    ax_B.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax_B.set_ylabel("Kohlberg Stage", fontsize=8)
    ax_B.set_xlabel("Parameter Scale Group", fontsize=8)
    ax_B.text(0.97, 0.97, f"KW H={H:.2f}\np={p_kw:.3f}",
              transform=ax_B.transAxes, ha="right", va="top", fontsize=7,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="#cccccc", alpha=0.9))
    ax_B.text(-0.2, 1.06, "(B)", transform=ax_B.transAxes,
              fontsize=10, fontweight="bold", va="top")
    ax_B.set_title("Stage Distribution by Scale", fontsize=9, fontweight="bold", pad=4)

    # ── C: Grouped bars ──────────────────────────────────────────────────────
    model_means = (raw_df.groupby(["display_name", "scale_group", "training_type"],
                                   observed=True)["kohlberg_stage"]
                   .mean().reset_index()
                   .rename(columns={"kohlberg_stage": "mean_stage"}))

    offsets_c = np.linspace(-0.20, 0.20, len(SCALE_ORDER))
    for si, sg in enumerate(SCALE_ORDER):
        col = SCALE_COLORS[sg]
        for ti, tt in enumerate(TRAINING_ORDER):
            cell = cell_df[(cell_df["scale_group"] == sg) & (cell_df["training_type"] == tt)]
            if cell.empty: continue
            x  = ti + offsets_c[si]
            mv = float(cell["mean_stage"].iloc[0])
            ci_lo = float(cell["ci_lower"].iloc[0]); ci_hi = float(cell["ci_upper"].iloc[0])
            ax_C.bar(x, mv, width=0.17, color=col, alpha=0.72,
                     edgecolor="white", linewidth=0.4,
                     label=sg if ti == 0 else "_")
            ax_C.errorbar(x, mv, yerr=[[mv-ci_lo],[ci_hi-mv]],
                          fmt="none", color="black", capsize=2.5, elinewidth=0.8)
            mdots = model_means[(model_means["scale_group"]==sg) &
                                 (model_means["training_type"]==tt)]
            if not mdots.empty:
                jitter = np.random.default_rng(42).uniform(-0.045, 0.045, len(mdots))
                ax_C.scatter(x+jitter, mdots["mean_stage"].values,
                             color="white", edgecolors=col, s=20,
                             linewidths=0.9, zorder=5, alpha=0.95)

    ax_C.axhline(5, color="#999999", linestyle="--", linewidth=0.8)
    ax_C.set_xticks(range(len(TRAINING_ORDER)))
    ax_C.set_xticklabels(TRAINING_ORDER, fontsize=7.5, rotation=10, ha="right")
    ax_C.set_ylim(4.0, 6.55)
    ax_C.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax_C.set_ylabel("Mean Kohlberg Stage", fontsize=8)
    ax_C.set_xlabel("Training Procedure", fontsize=8)
    ax_C.legend(title="Scale", fontsize=7, title_fontsize=7,
                loc="lower right", framealpha=0.9, edgecolor="#cccccc")
    ax_C.text(-0.16, 1.06, "(C)", transform=ax_C.transAxes,
              fontsize=10, fontweight="bold", va="top")
    ax_C.set_title("Cell Means (open = models)", fontsize=9, fontweight="bold", pad=4)

    # ── D: Effect sizes + variance ───────────────────────────────────────────
    eff_colors_d = {
        "Scale":               OI["blue"],
        "Training_Type":       OI["orange"],
        "Scale × Training_Type": OI["vermillion"],
    }
    eta_eff = effect_df.copy()
    y_pos_d = range(len(eta_eff))
    ax_D.barh(y_pos_d, eta_eff["partial_eta2"].values,
              color=[eff_colors_d.get(e, "#999999") for e in eta_eff["effect"]],
              edgecolor="white", linewidth=0.4, alpha=0.85)

    for i, row in eta_eff.reset_index(drop=True).iterrows():
        pv = row["p_value"]
        sym = _sig_label(float(pv)) if not np.isnan(float(pv)) else ""
        eta_v = float(row["partial_eta2"])
        col_txt = "white" if eta_v > 0.015 else "#555555"
        ax_D.text(max(eta_v/2, 0.002), i, f"η²={eta_v:.3f}",
                  va="center", ha="center", fontsize=7.5,
                  color=col_txt, fontweight="bold")
        if sym and sym != "ns":
            ax_D.text(eta_v + 0.002, i, sym, va="center",
                      fontsize=9, color="#222222", fontweight="bold")

    for thresh, lbl in [(0.01, "small"), (0.06, "medium"), (0.14, "large")]:
        ax_D.axvline(thresh, color="#888888", linestyle=":", linewidth=0.7)
        ax_D.text(thresh+0.001, len(eta_eff)-0.18, lbl,
                  fontsize=6, color="#888888", rotation=90, va="top")

    display_lbls = (eta_eff["effect"]
                    .str.replace("Training_Type", "Training", regex=False)
                    .str.replace("Scale × Training_Type", "Interaction", regex=False))
    ax_D.set_yticks(y_pos_d)
    ax_D.set_yticklabels(display_lbls, fontsize=8.5)
    ax_D.set_xlabel("Partial η²", fontsize=8)
    ax_D.set_title("Effect Sizes", fontsize=9, fontweight="bold", pad=4)
    ax_D.set_xlim(0, max(0.10, eta_eff["partial_eta2"].max() * 1.5))
    ax_D.text(-0.22, 1.06, "(D)", transform=ax_D.transAxes,
              fontsize=10, fontweight="bold", va="top")

    fig.suptitle(
        "Figure 8. Scale vs. Training Type Decomposition of Moral Reasoning Stage\n"
        r"Two-way factorial ANOVA: $F_{\mathrm{Scale}}$(2,225)=5.94, $p$=.003, "
        r"$\eta^2_p$=.050; $F_{\mathrm{Training}}$(2,225)=2.71, $p$=.069",
        fontsize=9, fontweight="bold", y=1.01,
    )

    out = OUT_DIR / "summary_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out.name}")
    return str(out)
