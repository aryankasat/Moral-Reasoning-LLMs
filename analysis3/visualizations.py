"""
visualizations.py — Advanced research-paper-grade figures for Analysis 3.

Six figures (2D + 3D):
  fig1  Clustered heatmap with dendrogram (seaborn clustermap)
  fig2  Multi-panel radar charts (stage profile across dilemmas, per model)
  fig3  Violin + box + strip composite (within-model stage distributions)
  fig4  3D grouped bar chart (model × prompt_type × mean_stage)
  fig5  Bubble chart (log scale × ICC × SD as bubble size)
  fig6  3D surface — stage landscape over (dilemma rank, prompt_type, model_size)

Design standards:
  - 300 dpi, bbox_inches='tight', Times-family serif
  - Legends outside axes; no label overlaps (adjustText used where needed)
  - Colorblind-safe Okabe-Ito palette
"""

from __future__ import annotations
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from config import (
    PROVIDER_COLORS, PROMPT_COLORS, DILEMMA_LABELS, PROMPT_LABELS,
    PROMPT_ORDER, STAGE_LABELS_SHORT, apply_publication_style,
)

apply_publication_style()

MM = 1 / 25.4   # mm → inches

# ── palette helpers ──────────────────────────────────────────────────────────

def _pc(prov: str) -> str:
    return PROVIDER_COLORS.get(prov, "#888888")

def _prov_handles() -> list:
    return [
        mpatches.Patch(facecolor=c, edgecolor="#444", lw=0.6, label=p, alpha=0.90)
        for p, c in PROVIDER_COLORS.items()
    ]

def _pt_handles() -> list:
    return [
        mpatches.Patch(facecolor=PROMPT_COLORS[k], edgecolor="#444",
                       lw=0.6, label=PROMPT_LABELS[k], alpha=0.90)
        for k in PROMPT_ORDER
    ]

def _save(fig, path: Path, name: str) -> None:
    fig.savefig(path / name, dpi=300, bbox_inches="tight",
                pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}")


# ────────────────────────────────────────────────────────────────────────────
# Figure 1 — Clustered heatmap with row + col dendrogram
# ────────────────────────────────────────────────────────────────────────────

def plot_clustermap(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Seaborn clustermap: models (rows) × dilemma (cols), cell = mean stage.
    Rows and columns are hierarchically clustered. Row colour bar = provider.
    """
    df2 = df.copy()
    df2["dilemma_short"] = df2["dilemma_type"].map(DILEMMA_LABELS).fillna(df2["dilemma_type"])

    # Pivot: rows = models (sorted by params), cols = dilemmas
    pivot = (
        df2.groupby(["display_name", "dilemma_short"])["kohlberg_stage"]
        .mean()
        .unstack("dilemma_short")
    )

    # Model order for colour bar (smallest→largest)
    model_meta = (
        df.drop_duplicates("display_name")[["display_name", "params_B", "provider"]]
        .set_index("display_name")
    )

    row_colors = pivot.index.map(lambda m: _pc(model_meta.loc[m, "provider"]))

    # Clustermap — suppress the ugly scipy cluster warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.clustermap(
            pivot,
            cmap="RdYlGn", vmin=1, vmax=6,
            annot=True, fmt=".1f",
            annot_kws={"size": 8, "fontweight": "bold"},
            linewidths=0.5, linecolor="#d0d0d0",
            row_colors=row_colors.values,
            col_cluster=True, row_cluster=True,
            figsize=(160 * MM, 200 * MM),
            dendrogram_ratio=(0.12, 0.12),
            cbar_pos=(1.04, 0.25, 0.03, 0.40),
            cbar_kws={"label": "Mean Kohlberg Stage",
                      "ticks": [1, 2, 3, 4, 5, 6]},
            method="average", metric="euclidean",
        )

    # Style the colour bar ticks
    g.ax_cbar.set_yticklabels(["S1", "S2", "S3", "S4", "S5", "S6"], fontsize=8)
    g.ax_cbar.tick_params(labelsize=8)

    # Provider colour legend — place on the figure level (not on ax_row_colors)
    # to avoid matplotlib handler_map compatibility issues
    prov_handles = _prov_handles()
    g.figure.legend(
        handles=prov_handles,
        title="Provider", title_fontsize=7,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=7, framealpha=0.92, edgecolor="#cccccc",
        ncol=1,
    )

    g.ax_heatmap.set_xlabel("Dilemma", labelpad=6, fontsize=10)
    g.ax_heatmap.set_ylabel("", labelpad=0)
    g.ax_heatmap.tick_params(axis="x", labelsize=8.5, rotation=30, bottom=False)
    g.ax_heatmap.tick_params(axis="y", labelsize=8.0, rotation=0,  left=False)

    g.figure.suptitle(
        "Hierarchically Clustered Stage Profile\n"
        "Models × Dilemmas  (mean Kohlberg stage; row-bar = provider)",
        y=1.03, fontsize=10, fontweight="bold",
    )

    _save(g.figure, out_dir, "fig1_clustermap.png")



# ────────────────────────────────────────────────────────────────────────────
# Figure 2 — Multi-panel radar charts
# ────────────────────────────────────────────────────────────────────────────

def plot_radar_grid(df: pd.DataFrame, out_dir: Path) -> None:
    """
    N_models-panel radar chart. Each panel = one model.
    Each spoke = a dilemma. Three overlaid polygons = ZERO_SHOT / COT / ROLEPLAY.
    Stages mapped 1–6 on radial axis.
    """
    df2 = df.copy()
    df2["dilemma_short"] = df2["dilemma_type"].map(DILEMMA_LABELS).fillna(df2["dilemma_type"])

    models = (
        df.drop_duplicates("display_name")
        .sort_values("params_B")["display_name"]
        .tolist()
    )
    dilemmas = list(DILEMMA_LABELS.values())
    N = len(dilemmas)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]          # close the polygon

    n_models = len(models)
    n_cols   = 4
    n_rows   = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(200 * MM, n_rows * 46 * MM),
        subplot_kw=dict(polar=True),
    )
    axes = axes.flatten()

    model_meta = (
        df.drop_duplicates("display_name")[["display_name", "params_B", "provider"]]
        .set_index("display_name")
    )

    for ax_idx, model_name in enumerate(models):
        ax = axes[ax_idx]
        grp = df2[df2["display_name"] == model_name]
        prov_col = _pc(model_meta.loc[model_name, "provider"])

        for pt in PROMPT_ORDER:
            sub = grp[grp["prompt_type"] == pt]
            if len(sub) == 0:
                continue
            vals = [
                float(sub.loc[sub["dilemma_short"] == d, "kohlberg_stage"].mean())
                if len(sub[sub["dilemma_short"] == d]) > 0 else 3.0
                for d in dilemmas
            ]
            vals += vals[:1]
            col = PROMPT_COLORS.get(pt, "#888")
            ax.plot(angles, vals, color=col, linewidth=1.4, linestyle="solid", alpha=0.9)
            ax.fill(angles, vals, color=col, alpha=0.13)

        # Spoke labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dilemmas, fontsize=6.5, color="#333333")
        ax.set_ylim(0, 6)
        ax.set_yticks([2, 4, 6])
        ax.set_yticklabels(["S2", "S4", "S6"], fontsize=5.5, color="#999999")
        ax.grid(color="#cccccc", linewidth=0.5)
        ax.spines["polar"].set_linewidth(0.5)
        ax.spines["polar"].set_color(prov_col)

        # Model name centred above panel
        params = int(model_meta.loc[model_name, "params_B"])
        ax.set_title(
            f"{model_name}\n({params} B)",
            fontsize=6.8, pad=8, color="#222222", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=prov_col,
                      edgecolor="none", alpha=0.18),
        )

    # Hide unused panels
    for ax in axes[n_models:]:
        ax.set_visible(False)

    # Shared legend — prompt types
    leg_handles = [
        mlines.Line2D([], [], color=PROMPT_COLORS[pt], linewidth=1.6, label=PROMPT_LABELS[pt])
        for pt in PROMPT_ORDER
    ]
    fig.legend(
        handles=leg_handles,
        title="Prompt Type", title_fontsize=8,
        loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8, framealpha=0.92, edgecolor="#cccccc",
    )

    fig.suptitle(
        "Stage Profile Across Dilemmas by Prompt Type  (Radar Charts)",
        y=1.02, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig2_radar_grid.png")


# ────────────────────────────────────────────────────────────────────────────
# Figure 3 — Violin + box + strip composite
# ────────────────────────────────────────────────────────────────────────────

def plot_violin_composite(
    df: pd.DataFrame,
    sd_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Horizontal combined violin + boxplot per model.
    Violins split by prompt type; jitter strip coloured by prompt type.
    Human SD baseline annotated.
    """
    order = sd_df.sort_values("params_B")["display_name"].tolist()
    n = len(order)

    # Convert to long form with numeric y positions for manual placement
    fig, axes = plt.subplots(
        1, 2, figsize=(195 * MM, max(110, n * 15) * MM),
        gridspec_kw={"width_ratios": [3.2, 1]},
    )
    ax_main, ax_sd = axes

    # ── LEFT: violin + outliers per model ────────────────────────────────
    palette = {PROMPT_LABELS[pt]: PROMPT_COLORS[pt] for pt in PROMPT_ORDER}
    df_plot = df.copy()
    df_plot["Prompt Type"] = df_plot["prompt_type"].map(PROMPT_LABELS)
    df_plot = df_plot[df_plot["display_name"].isin(order)]

    # Map display_name → numeric y position
    name_to_y = {m: i for i, m in enumerate(order)}
    df_plot["y_pos"] = df_plot["display_name"].map(name_to_y)

    # Draw per-model violin per prompt type with offset
    offsets  = {"ZERO_SHOT": -0.25, "COT": 0.0, "ROLEPLAY": 0.25}
    vwidth   = 0.20

    for pt in PROMPT_ORDER:
        sub = df_plot[df_plot["prompt_type"] == pt].copy()
        sub["y_jitter"] = sub["y_pos"] + offsets[pt] + np.random.default_rng(42).uniform(
            -0.07, 0.07, len(sub)
        )
        col = PROMPT_COLORS[pt]

        for m_idx, model_name in enumerate(order):
            msub = sub[sub["display_name"] == model_name]["kohlberg_stage"].dropna()
            if len(msub) < 2:
                continue
            base_y = m_idx + offsets[pt]
            vparts = ax_main.violinplot(
                msub.values,
                positions=[base_y],
                vert=False,
                widths=vwidth,
                showmedians=False, showextrema=False,
            )
            for pc in vparts["bodies"]:
                pc.set_facecolor(col)
                pc.set_edgecolor("#333333")
                pc.set_linewidth(0.5)
                pc.set_alpha(0.55)

        # Jitter strip
        jitter_y = sub["y_pos"] + offsets[pt] + np.random.default_rng(pt.__hash__() % 999).uniform(
            -0.07, 0.07, len(sub)
        )
        ax_main.scatter(
            sub["kohlberg_stage"], jitter_y,
            color=col, s=12, alpha=0.70, linewidths=0, zorder=5,
            label=PROMPT_LABELS[pt],
        )

    # Median line per model (overall)
    for i, model_name in enumerate(order):
        msub = df_plot[df_plot["display_name"] == model_name]["kohlberg_stage"]
        med  = float(msub.median())
        col  = _pc(sd_df.loc[sd_df["display_name"] == model_name, "provider"].iloc[0])
        ax_main.hlines(i, med - 0.02, med + 0.02, color=col,
                       linewidth=2.5, zorder=7)

    ax_main.set_yticks(range(n))
    ax_main.set_yticklabels(order, fontsize=8.5)
    ax_main.set_xticks(range(1, 7))
    ax_main.set_xticklabels(
        ["S1\nObedience", "S2\nSelf-\nInterest", "S3\nConformity",
         "S4\nLaw &\nOrder", "S5\nSocial\nContract", "S6\nUniversal\nEthics"],
        fontsize=8.0,
    )
    ax_main.set_xlim(0.3, 7.2)
    ax_main.set_ylim(-0.6, n - 0.4)
    ax_main.set_xlabel("Kohlberg Stage", labelpad=5)
    ax_main.set_title(
        "Stage Distribution per Model by Prompt Type\n"
        "(violin = density; dots = observations; thick line = median)",
        fontsize=9, pad=8,
    )

    # ── RIGHT: SD bar ──────────────────────────────────────────────────
    sd_sorted = sd_df.sort_values("params_B").reset_index(drop=True)
    colors_sd = [_pc(p) for p in sd_sorted["provider"]]
    y_pos     = np.arange(n)

    ax_sd.barh(y_pos, sd_sorted["std_stage"], color=colors_sd, alpha=0.85, height=0.6, zorder=3)
    ax_sd.axvline(0.67, color="#cc4444", linewidth=1.2, linestyle="--",
                  label="Human baseline SD")
    ax_sd.set_yticks(y_pos)
    ax_sd.set_yticklabels([], fontsize=0)
    ax_sd.set_xlabel("Stage SD", labelpad=5, fontsize=9)
    ax_sd.set_xlim(0, max(0.75, sd_sorted["std_stage"].max() + 0.05))
    ax_sd.set_title("SD", fontsize=9, pad=8)
    ax_sd.legend(fontsize=7, loc="lower right",
                 framealpha=0.9, edgecolor="#cccccc")

    # Shared prompt legends on right
    leg1 = ax_main.legend(
        title="Prompt Type", title_fontsize=8,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=8, framealpha=0.92, edgecolor="#cccccc",
    )
    ax_main.add_artist(leg1)
    prov_leg = ax_main.legend(
        handles=_prov_handles(), title="Provider", title_fontsize=8,
        loc="upper left", bbox_to_anchor=(1.01, 0.55),
        fontsize=8, framealpha=0.92, edgecolor="#cccccc",
    )

    fig.tight_layout(w_pad=0.4)
    _save(fig, out_dir, "fig3_violin_composite.png")


# ────────────────────────────────────────────────────────────────────────────
# Figure 4 — 3D grouped bar chart
# ────────────────────────────────────────────────────────────────────────────

def plot_3d_grouped_bars(df: pd.DataFrame, out_dir: Path) -> None:
    """
    3D perspective bar chart: X = prompt type, Y (depth) = model (sorted by
    params_B), Z = mean Kohlberg stage.
    Model Y-axis uses numeric index 1–N to avoid overlapping long names.
    A key table listing index → model name is drawn to the right.
    """
    models = (
        df.drop_duplicates("display_name")
        .sort_values("params_B")["display_name"]
        .tolist()
    )
    pt_labels = [PROMPT_LABELS[p] for p in PROMPT_ORDER]
    n_models  = len(models)
    n_pt      = len(PROMPT_ORDER)

    # Build Z matrix: (n_pt × n_models) = mean stages
    Z = np.zeros((n_pt, n_models))
    for j, model in enumerate(models):
        for i, pt in enumerate(PROMPT_ORDER):
            sub = df[(df["display_name"] == model) & (df["prompt_type"] == pt)]
            Z[i, j] = sub["kohlberg_stage"].mean() if len(sub) > 0 else np.nan

    # Wide figure: left = 3D plot, right = model key table
    fig = plt.figure(figsize=(235 * MM, 155 * MM))
    # Use GridSpec so the 3D axis and text panel share the figure cleanly
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.8, 1.0], figure=fig,
                             left=0.04, right=0.97, wspace=0.08)
    ax  = fig.add_subplot(gs[0], projection="3d")
    ax_key = fig.add_subplot(gs[1])
    ax_key.axis("off")

    bar_w  = 0.55
    bar_d  = 0.55
    x_base = np.arange(n_pt)
    y_base = np.arange(n_models)

    model_meta = (
        df.drop_duplicates("display_name")[["display_name", "provider"]]
        .set_index("display_name")
    )

    for i, pt in enumerate(PROMPT_ORDER):
        col = PROMPT_COLORS[pt]
        for j in range(n_models):
            z_val = Z[i, j]
            if np.isnan(z_val):
                continue
            ax.bar3d(
                x=i - bar_w / 2,
                y=j - bar_d / 2,
                z=0,
                dx=bar_w, dy=bar_d, dz=z_val,
                color=col, alpha=0.80,
                shade=True,
            )

    # ── Y-axis: use numeric indices only (1-based) ─────────────────────────
    ax.set_xticks(x_base)
    ax.set_xticklabels(pt_labels, fontsize=8)
    ax.set_yticks(y_base)
    # Show every-other index to avoid crowding; tick marks anchor positions
    y_labels_sparse = [
        str(j + 1) if j % 2 == 0 else ""
        for j in range(n_models)
    ]
    ax.set_yticklabels(y_labels_sparse, fontsize=7.5, ha="right")
    ax.set_zticks(range(1, 7))
    ax.set_zticklabels(["S1", "S2", "S3", "S4", "S5", "S6"], fontsize=7.5)
    ax.set_zlim(0, 6.5)

    ax.set_xlabel("Prompt Type", labelpad=8, fontsize=9)
    ax.set_ylabel("Model Index  (small → large)", labelpad=10, fontsize=8.5)
    ax.set_zlabel("Mean Stage", labelpad=6, fontsize=9)
    ax.set_title(
        "Mean Kohlberg Stage: 3D View\nModel × Prompt Type",
        fontsize=10, pad=12, fontweight="bold",
    )
    ax.view_init(elev=22, azim=-55)
    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.04)
    ax.grid(True, linewidth=0.4, alpha=0.4)

    # Prompt-type legend on 3D axes
    leg_handles = [
        mpatches.Patch(facecolor=PROMPT_COLORS[pt], edgecolor="#444",
                       lw=0.6, label=PROMPT_LABELS[pt], alpha=0.85)
        for pt in PROMPT_ORDER
    ]
    ax.legend(handles=leg_handles, title="Prompt Type", title_fontsize=7.5,
              loc="upper left", bbox_to_anchor=(-0.02, 0.98),
              fontsize=7.5, framealpha=0.88, edgecolor="#cccccc")

    # ── Right panel: numbered model key ────────────────────────────────────
    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(0, 1)

    # Title row
    ax_key.text(0.0, 0.99, "Model Key",
                fontsize=8.5, fontweight="bold", va="top", ha="left",
                transform=ax_key.transAxes, color="#222222")
    # Separator line below title (draw using plot with transAxes transform)
    ax_key.plot([0, 1], [0.955, 0.955], color="#aaaaaa",
                linewidth=0.6, transform=ax_key.transAxes, clip_on=False)

    row_h = 0.93 / n_models   # height per row
    for j, model_name in enumerate(models):
        y_frac = 0.93 - j * row_h
        prov   = model_meta.loc[model_name, "provider"]
        col    = _pc(prov)
        # Index badge
        ax_key.text(
            0.02, y_frac,
            str(j + 1),
            fontsize=7.5, va="center", ha="left",
            transform=ax_key.transAxes,
            fontweight="bold", color=col,
        )
        # Model name
        ax_key.text(
            0.20, y_frac,
            model_name,
            fontsize=6.5, va="center", ha="left",
            transform=ax_key.transAxes, color="#222222",
        )

    # Subtle alternating row shading
    for j in range(n_models):
        if j % 2 == 0:
            y_frac = 0.93 - j * row_h
            ax_key.axhspan(
                y_frac - row_h * 0.5,
                y_frac + row_h * 0.5,
                xmin=0, xmax=1,
                color="#f5f5f5", alpha=0.7,
                transform=ax_key.transAxes,
                zorder=0,
            )

    _save(fig, out_dir, "fig4_3d_grouped_bars.png")



# ────────────────────────────────────────────────────────────────────────────
# Figure 5 — Bubble chart: scale × ICC × SD
# ────────────────────────────────────────────────────────────────────────────

def plot_bubble_scale_icc(
    icc_df: pd.DataFrame,
    sd_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Bubble chart:
      X = log10(params_B)
      Y = ICC(2,1) — models at Y=1.0 are jittered vertically to separate labels
      Size = stage SD
      Colour = provider
      Labels placed via adjustText (aggressive repulsion)
    """
    from adjustText import adjust_text
    from scipy.stats import spearmanr

    merged = icc_df.merge(
        sd_df[["model_key", "std_stage"]], on="model_key", how="left"
    ).dropna(subset=["icc"])

    x_raw = merged["log_params"].values
    y_raw = merged["icc"].values
    sd    = merged["std_stage"].values.clip(min=0.01)
    col   = [_pc(p) for p in merged["provider"]]
    sizes = (sd / sd.max() * 550).clip(min=50)

    # ── Jitter Y for models stacked at ICC = 1.0 so labels can spread ─────
    rng = np.random.default_rng(7)
    y_plot = y_raw.copy().astype(float)
    at_top = y_plot >= 0.99
    n_top  = at_top.sum()
    if n_top > 1:
        # evenly space them in a small band [0.98, 1.04] so bubbles stay near top
        spacing = np.linspace(-0.03, 0.03, n_top)
        y_plot[at_top] = 1.01 + spacing

    # Extra width to give labels room on the right
    fig, ax = plt.subplots(figsize=(200 * MM, 145 * MM))

    # Quadrant backgrounds
    x_mid = np.median(x_raw)
    y_mid = 0.75
    ax.axhspan(y_mid, 1.12, xmin=0, xmax=1, color="#e8f5e9", alpha=0.40, zorder=0)
    ax.axhspan(-0.20, y_mid, xmin=0, xmax=1, color="#fce4ec", alpha=0.35, zorder=0)
    ax.axvline(x_mid, color="#dddddd", linewidth=0.7, linestyle="--", zorder=1)
    ax.axhline(y_mid, color="#dddddd", linewidth=0.7, linestyle="--", zorder=1)

    ax.scatter(
        x_raw, y_plot, s=sizes, c=col, alpha=0.88,
        edgecolors="#333333", linewidths=0.6, zorder=5,
    )

    # ── Labels — start offset above/below dot, then let adjustText refine ─
    texts = []
    for idx, (_, row) in enumerate(merged.iterrows()):
        yp = float(y_plot[idx])
        xp = float(x_raw[idx])
        t = ax.text(
            xp, yp,
            row["display_name"],
            fontsize=6.5, color="#222222",
            ha="left", va="bottom",
        )
        texts.append(t)

    adjust_text(
        texts, x=x_raw, y=y_plot, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5, shrinkA=3),
        expand=(2.2, 2.8),
        force_text=(1.2, 1.5),
        force_points=(0.8, 1.0),
        force_static=(0.6, 0.8),
        min_arrow_len=4,
        max_move=5.0,
        only_move={"points": "y", "texts": "xy", "objects": "xy"},
    )

    # OLS trend line (use raw non-jittered y for fitting)
    finite = np.isfinite(x_raw) & np.isfinite(y_raw)
    if finite.sum() >= 2:
        m, b = np.polyfit(x_raw[finite], y_raw[finite], 1)
        xfit = np.linspace(x_raw.min() - 0.2, x_raw.max() + 0.2, 300)
        ax.plot(xfit, m * xfit + b, color="#555555", linewidth=1.1,
                linestyle="--", zorder=3)

    # Spearman annotation
    if finite.sum() >= 3:
        rho, p_val = spearmanr(x_raw[finite], y_raw[finite])
        sig = "n.s." if p_val >= 0.05 else f"p = {p_val:.3f}"
        ax.text(
            0.03, 0.08,
            rf"Spearman $\rho$ = {rho:.3f}" + f"\n{sig}",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", lw=0.6, alpha=0.95),
        )

    # Bubble-size legend
    sd_legend_vals  = [0.1, 0.3, 0.5, 0.65]
    sd_legend_sizes = [(s / sd.max() * 550) for s in sd_legend_vals]
    size_handles = [
        mlines.Line2D([], [], linestyle="none", marker="o", color="#888888",
                      markersize=np.sqrt(sz), markeredgecolor="#444",
                      label=f"SD = {sv:.2f}")
        for sv, sz in zip(sd_legend_vals, sd_legend_sizes)
        if sv <= sd.max()
    ]
    size_leg = ax.legend(
        handles=size_handles, title="Stage SD (bubble)", title_fontsize=7.5,
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=7.5, framealpha=0.92, edgecolor="#cccccc",
    )
    ax.add_artist(size_leg)
    ax.legend(
        handles=_prov_handles(), title="Provider", title_fontsize=7.5,
        loc="upper left", bbox_to_anchor=(1.02, 0.55),
        fontsize=7.5, framealpha=0.92, edgecolor="#cccccc",
    )

    # ICC threshold lines with left-side labels (avoid crowding right margin)
    for thresh, lbl in [(0.50, "Poor | Moderate"), (0.90, "Good | Excellent")]:
        ax.axhline(thresh, color="#bbbbbb", linewidth=0.7, linestyle=":", zorder=2)
        ax.text(x_raw.min() - 0.18, thresh + 0.01, lbl,
                fontsize=6.5, color="#999999", va="bottom", ha="left")

    # Quadrant corner labels
    x_lo, x_hi = x_raw.min() - 0.22, x_raw.max() + 0.40
    ax.text(x_lo + 0.02, 1.10, "Small + Consistent", fontsize=6.0,
            color="#aaaaaa", va="top", ha="left")
    ax.text(x_hi - 0.02, 1.10, "Large + Consistent", fontsize=6.0,
            color="#aaaaaa", va="top", ha="right")
    ax.text(x_lo + 0.02, -0.16, "Small + Variable",  fontsize=6.0,
            color="#aaaaaa", va="bottom", ha="left")
    ax.text(x_hi - 0.02, -0.16, "Large + Variable",  fontsize=6.0,
            color="#aaaaaa", va="bottom", ha="right")

    # X-axis: nice param labels
    nice_P = [10, 30, 100, 300, 1000]
    nice_L = [np.log10(v) for v in nice_P]
    nice_N = ["10B", "30B", "100B", "300B", "1,000B"]
    tick_l = [t for t in nice_L if x_lo <= t <= x_hi]
    tick_n = [nice_N[nice_L.index(t)] for t in tick_l]
    ax.set_xticks(tick_l)
    ax.set_xticklabels(tick_n, fontsize=8.5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.22, 1.18)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.1, 0.2)], fontsize=8.5)
    ax.set_xlabel("Model Scale  (approximate parameter count, log scale)", labelpad=6)
    ax.set_ylabel("ICC(2,1)  —  Cross-Dilemma Consistency", labelpad=5)
    ax.set_title(
        "Scale vs. Consistency  (bubble size ∝ within-model SD)",
        fontsize=10, pad=10, fontweight="bold",
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.76)
    _save(fig, out_dir, "fig5_bubble_scale_icc.png")



# ────────────────────────────────────────────────────────────────────────────
# Figure 6 — 3D surface: stage landscape over (dilemma, model_size)
# ────────────────────────────────────────────────────────────────────────────

def plot_3d_surface(df: pd.DataFrame, out_dir: Path) -> None:
    """
    3D surface + scatter plot:
      X = dilemma index (6 categories, treated as numeric)
      Y = log10(params_B)
      Z = mean Kohlberg stage
    Surface interpolated over the grid; actual data points plotted as scatter.
    One panel per prompt type (3 subplots side by side).
    """
    from scipy.interpolate import griddata

    df2 = df.copy()
    df2["dilemma_idx"]  = pd.Categorical(
        df2["dilemma_type"],
        categories=list(DILEMMA_LABELS.keys()),
    ).codes.astype(float)
    df2["dilemma_short"] = df2["dilemma_type"].map(DILEMMA_LABELS)

    dilemma_keys   = list(DILEMMA_LABELS.keys())
    dilemma_labels = list(DILEMMA_LABELS.values())

    fig = plt.figure(figsize=(215 * MM, 80 * MM))

    for col_idx, pt in enumerate(PROMPT_ORDER):
        ax = fig.add_subplot(1, 3, col_idx + 1, projection="3d")
        sub = df2[df2["prompt_type"] == pt]

        # Aggregate to mean stage per (dilemma, model)
        agg = sub.groupby(["dilemma_idx", "log_params"])["kohlberg_stage"].mean().reset_index()

        xi = agg["dilemma_idx"].values.astype(float)
        yi = agg["log_params"].values.astype(float)
        zi = agg["kohlberg_stage"].values.astype(float)

        # Grid for surface
        x_grid = np.linspace(0, len(dilemma_keys) - 1, 30)
        y_grid = np.linspace(yi.min(), yi.max(), 30)
        XX, YY = np.meshgrid(x_grid, y_grid)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ZZ = griddata((xi, yi), zi, (XX, YY), method="linear")
            ZZ_fill = griddata((xi, yi), zi, (XX, YY), method="nearest")
        ZZ = np.where(np.isnan(ZZ), ZZ_fill, ZZ)

        surf = ax.plot_surface(
            XX, YY, ZZ,
            cmap="RdYlGn", vmin=1, vmax=6,
            alpha=0.72, linewidth=0, antialiased=True,
        )

        # Scatter actual data points
        ax.scatter(xi, yi, zi, color=PROMPT_COLORS[pt],
                   s=22, edgecolors="#333", linewidths=0.5,
                   zorder=6, alpha=0.9)

        ax.set_xticks(range(len(dilemma_keys)))
        ax.set_xticklabels(dilemma_labels, fontsize=5.5, rotation=15, ha="right")
        ax.set_yticks(np.round(np.linspace(yi.min(), yi.max(), 4), 1))
        ax.set_yticklabels(
            [f"{10**v:.0f}B" for v in np.round(np.linspace(yi.min(), yi.max(), 4), 1)],
            fontsize=5.5,
        )
        ax.set_zticks([3, 4, 5, 6])
        ax.set_zticklabels(["S3", "S4", "S5", "S6"], fontsize=6.0)
        ax.set_zlim(1, 6.5)

        ax.set_xlabel("Dilemma", fontsize=7, labelpad=5)
        ax.set_ylabel("Scale (params)", fontsize=7, labelpad=5)
        ax.set_zlabel("Stage", fontsize=7, labelpad=4)
        ax.set_title(f"{PROMPT_LABELS[pt]}", fontsize=9, pad=8, fontweight="bold")
        ax.view_init(elev=28, azim=-50)
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)

    # Shared colour bar
    sm = matplotlib.cm.ScalarMappable(
        cmap="RdYlGn",
        norm=matplotlib.colors.Normalize(vmin=1, vmax=6)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.get_axes(), shrink=0.6, aspect=18, pad=0.04)
    cbar.set_label("Mean Kohlberg Stage", fontsize=8)
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(["S1", "S2", "S3", "S4", "S5", "S6"])
    cbar.ax.tick_params(labelsize=7.5)

    fig.suptitle(
        "Stage Landscape: 3D Surface over Dilemma × Model Scale\n"
        "(one panel per prompt type; surface = interpolated mean stage)",
        fontsize=9, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig6_3d_surface.png")
