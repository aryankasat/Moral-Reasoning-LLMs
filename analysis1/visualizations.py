"""
visualizations.py — Publication-quality figures for the Scale vs. Moral
Reasoning analysis.

All figures follow:
  - Single-column (88 mm) or double-column (180 mm) journal widths
  - 300 dpi PNG output
  - Times-family serif font via rcParams (set by config.apply_publication_style)
  - Okabe-Ito / colorblind-safe palette
  - Minimal chartjunk (top/right spines removed, light grid)

Public API
----------
plot_box_by_model(df, summary, out_dir)
plot_scatter_scale_stage(summary, corr, out_dir)
plot_stage_heatmap(summary, out_dir)
plot_mean_stage_bar(summary, out_dir)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns

from config import PROVIDER_COLORS, STAGE_LABELS, apply_publication_style

# Activate journal style once at import time
apply_publication_style()

# ── helpers ──────────────────────────────────────────────────────────────────

MM = 1 / 25.4          # mm → inches conversion

def _provider_color(provider: str) -> str:
    return PROVIDER_COLORS.get(provider, "#888888")

def _provider_legend_handles() -> list:
    return [
        mpatches.Patch(facecolor=color, edgecolor="#444444",
                       linewidth=0.6, label=prov, alpha=0.90)
        for prov, color in PROVIDER_COLORS.items()
    ]

def _save(fig: plt.Figure, path: Path, name: str) -> None:
    fig.savefig(path / name, dpi=300, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}")


# ── Figure 1: Box plots ──────────────────────────────────────────────────────

def plot_box_by_model(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Horizontal box plots of Kohlberg stage per model.
    Models are ordered bottom→top by parameter count (smallest to largest).
    Figure width: 180 mm (double column), height scales with model count.

    Design choices
    --------------
    - White box with provider-coloured median line for discriminability
    - Individual observations shown as jittered strip (α = 0.35)
    - Median annotated as text for precision
    """
    order = summary.sort_values("params_B")["display_name"].tolist()
    n = len(order)

    fig_h = max(90, n * 14) * MM   # ≥90 mm, 14 mm per model
    fig, ax = plt.subplots(figsize=(180 * MM, fig_h))

    # Stage data in display order
    stage_data = [
        df.loc[df["display_name"] == m, "kohlberg_stage"].values
        for m in order
    ]
    providers = [
        summary.loc[summary["display_name"] == m, "provider"].iloc[0]
        for m in order
    ]

    bp = ax.boxplot(
        stage_data,
        vert=False,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(linewidth=0),   # hide default median; we draw it manually
        whiskerprops=dict(linewidth=0.8, color="#444444"),
        capprops=dict(linewidth=0.8, color="#444444"),
        flierprops=dict(marker="d", markersize=3.5,
                        markerfacecolor="#bbbbbb", markeredgecolor="none",
                        alpha=0.5),
        boxprops=dict(linewidth=0.7, edgecolor="#444444"),
    )

    for i, (patch, prov) in enumerate(zip(bp["boxes"], providers)):
        color = _provider_color(prov)
        patch.set_facecolor("white")

        # Coloured median line
        med_val = np.median(stage_data[i])
        ax.vlines(med_val, i + 0.625, i + 1.375,
                  color=color, linewidth=2.2, zorder=5)

        # Jittered strip
        rng = np.random.default_rng(i)
        jitter = rng.uniform(-0.18, 0.18, len(stage_data[i]))
        ax.scatter(stage_data[i], np.full(len(stage_data[i]), i + 1) + jitter,
                   color=color, s=14, alpha=0.40, zorder=4, linewidths=0)

        # Annotate median
        ax.text(med_val + 0.04, i + 1.32, f"{med_val:.1f}",
                va="bottom", ha="left", fontsize=7.5,
                color=color, fontweight="bold")

    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels(order, fontsize=8.5)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(
        ["S1\nObedience", "S2\nSelf-Interest", "S3\nConformity",
         "S4\nLaw & Order", "S5\nSocial Contract", "S6\nUniversal Ethics"],
        fontsize=8.5,
    )
    ax.set_xlabel("Kohlberg Moral Development Stage", labelpad=6)
    ax.set_title(
        "Distribution of Kohlberg Moral Reasoning Stage by Model\n"
        r"(ordered by parameter scale; $\diamondsuit$ = outlier, "
        "coloured line = median)",
        fontsize=10, pad=10,
    )
    ax.set_xlim(0.5, 6.7)

    # Provider legend — outside the axes on the right so no box is obscured
    leg = ax.legend(
        handles=_provider_legend_handles(),
        title="Provider", title_fontsize=8,
        loc="upper left", bbox_to_anchor=(1.02, 1),
        fontsize=8, framealpha=0.92, edgecolor="#cccccc",
        borderaxespad=0,
    )
    leg.get_frame().set_linewidth(0.6)

    # Light reference lines at each integer stage
    for s in range(1, 7):
        ax.axvline(s, color="#dddddd", linewidth=0.6, zorder=0)

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    _save(fig, out_dir, "fig1_box_stage_by_model.png")


# ── Figure 2: Scatter – scale vs. mean stage ─────────────────────────────────

def plot_scatter_scale_stage(
    summary: pd.DataFrame,
    corr: dict,
    out_dir: Path,
) -> None:
    """
    Scatter plot of log₁₀(params_B) vs. mean Kohlberg stage with
    95 % bootstrap CI error bars, OLS trend line, and annotated r and p.

    Figure width: 120 mm (single-column wide / 1.5-column).
    """
    # Wider figure for breathing room on x-axis + extra height for label spreading
    fig, ax = plt.subplots(figsize=(185 * MM, 125 * MM))

    x = summary["log_params"].values
    y = summary["mean_stage"].values

    texts = []
    for _, row in summary.iterrows():
        col = _provider_color(row["provider"])
        ax.errorbar(
            row["log_params"], row["mean_stage"],
            yerr=[[row["mean_stage"] - row["ci_lo"]],
                  [row["ci_hi"] - row["mean_stage"]]],
            fmt="o", color=col,
            ecolor=col, elinewidth=1.0, capsize=3.0, capthick=0.8,
            markersize=7, zorder=5,
        )
        # Collect text objects — adjustText will reposition them automatically
        t = ax.text(
            row["log_params"], row["mean_stage"],
            row["display_name"],
            fontsize=6.5, alpha=0.88,
        )
        texts.append(t)

    # Auto-repel overlapping labels and draw thin connector arrows back to dots
    from adjustText import adjust_text
    adjust_text(
        texts,
        x=x, y=y,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5, shrinkA=5),
        expand=(1.4, 1.6),
        force_text=(0.6, 0.8),
        force_points=(0.5, 0.6),
        ensure_inside_axes=True,
        min_arrow_len=6,
        max_move=3.0,
    )

    # OLS trend (visual only — Spearman used for statistics)
    m, b = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min() - 0.12, x.max() + 0.12, 300)
    ax.plot(xfit, m * xfit + b, color="#555555", linewidth=1.2,
            linestyle="--", label="Linear trend (OLS)", zorder=3)

    # Annotate correlation stats — upper-left box
    sig_str = "n.s." if corr["p"] >= 0.05 else f"p = {corr['p']:.3f}"
    corr_text = (
        rf"Spearman $\rho$ = {corr['rho']:.3f}"
        "\n"
        rf"95% CI [{corr['ci_lo']:.3f}, {corr['ci_hi']:.3f}]"
        "\n"
        rf"$\rho^2$ = {corr['r2']:.3f},  {sig_str}"
        "\n"
        rf"Effect: {corr['effect']}"
    )
    ax.text(
        0.03, 0.97, corr_text,
        transform=ax.transAxes,
        fontsize=7.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", linewidth=0.6, alpha=0.92),
    )

    # ── X-axis: clean round-number ticks (avoids per-point crowding) ──────
    # Use 5 fixed reference points spanning the data range
    nice_params_B = [10, 30, 100, 300, 1000]
    nice_log      = [np.log10(v) for v in nice_params_B]
    nice_labels   = ["10 B", "30 B", "100 B", "300 B", "1,000 B"]

    # Only show ticks that fall within the plotted x-range (add 10% margin)
    x_lo = x.min() - 0.15
    x_hi = x.max() + 0.15
    tick_log = [t for t in nice_log if x_lo <= t <= x_hi]
    tick_lbl = [nice_labels[nice_log.index(t)] for t in tick_log]

    ax.set_xticks(tick_log)
    ax.set_xticklabels(tick_lbl, fontsize=9)
    ax.set_xlim(x_lo, x_hi)

    # Light vertical grid lines at the nice ticks
    for tl in tick_log:
        ax.axvline(tl, color="#e8e8e8", linewidth=0.6, zorder=0)

    # ── Y-axis ────────────────────────────────────────────────────────────
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([f"Stage {i}" for i in range(1, 7)], fontsize=8.5)
    ax.set_ylim(4.5, 6.7)

    ax.set_xlabel("Model Scale  (approximate parameter count, log scale)", labelpad=6)
    ax.set_ylabel("Mean Kohlberg Stage  +/- 95% Bootstrap CI", labelpad=5)
    ax.set_title(
        "Model Scale vs. Moral Reasoning Stage",
        fontsize=11, pad=10, fontweight="bold",
    )

    # Legend — anchor to the right of the axes; data cluster is at lower-right
    leg_handles = _provider_legend_handles()
    leg_handles.append(
        mlines.Line2D([], [], color="#555555", linewidth=1.2,
                      linestyle="--", label="Linear trend (OLS)")
    )
    ax.legend(handles=leg_handles, title="Provider", fontsize=7.5,
              title_fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1),
              framealpha=0.92, edgecolor="#cccccc", borderaxespad=0)

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    _save(fig, out_dir, "fig2_scatter_scale_vs_stage.png")


# ── Figure 3: Stage distribution heat-map ────────────────────────────────────

def plot_stage_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    """
    Heat-map of stage % distribution per model.

    Rows = models (ordered by params_B), columns = Kohlberg stages 1–6.
    Cells annotated with percentage; only non-zero cells are annotated to
    reduce clutter.
    """
    stage_cols = [f"stage_{i}_pct" for i in range(1, 7)]
    heat = (
        summary.set_index("display_name")[stage_cols]
        .rename(columns={f"stage_{i}_pct": f"Stage {i}" for i in range(1, 7)})
    )

    n_models = len(heat)
    fig_h = max(80, n_models * 13) * MM
    fig, ax = plt.subplots(figsize=(130 * MM, fig_h))

    # Mask zeros so their cells stay white
    mask = heat == 0

    sns.heatmap(
        heat, ax=ax,
        cmap="YlOrRd",
        mask=mask,
        annot=heat.map(lambda v: f"{v:.0f}%" if v > 0 else ""),
        fmt="",
        annot_kws={"size": 8, "fontweight": "bold"},
        linewidths=0.4, linecolor="#e0e0e0",
        cbar_kws={"label": "% of responses", "shrink": 0.75,
                  "aspect": 20, "pad": 0.02},
        vmin=0, vmax=100,
    )

    # Outline non-zero cells more prominently
    for i in range(n_models):
        for j in range(6):
            val = heat.iloc[i, j]
            if val > 0:
                ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1,
                    fill=False, edgecolor="#888888", linewidth=0.7, zorder=5,
                ))

    ax.set_title(
        "Stage Distribution (%) per Model\n"
        "(rows ordered by parameter scale, small → large; zero cells are white)",
        fontsize=10, pad=12,
    )
    ax.set_xlabel("Kohlberg Moral Development Stage", labelpad=5)
    ax.set_ylabel("")
    ax.tick_params(axis="x", bottom=False)
    ax.tick_params(axis="y", left=False)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=8, rotation=0)

    fig.tight_layout()
    _save(fig, out_dir, "fig3_heatmap_stage_distribution.png")


# ── Figure 4: Mean stage bar chart ───────────────────────────────────────────

def plot_mean_stage_bar(summary: pd.DataFrame, out_dir: Path) -> None:
    """
    Horizontal bar chart of mean stage per model.

    - Bars coloured by provider
    - 95 % bootstrap CI shown as error caps
    - Stage labels on x-axis
    - Models ordered bottom→top by params_B
    """
    s = summary.sort_values("params_B")
    n = len(s)

    fig_h = max(90, n * 14) * MM
    fig, ax = plt.subplots(figsize=(130 * MM, fig_h))

    colours = [_provider_color(p) for p in s["provider"]]
    err_lo = (s["mean_stage"] - s["ci_lo"]).values
    err_hi = (s["ci_hi"] - s["mean_stage"]).values

    y_pos = np.arange(n)
    bars = ax.barh(
        y_pos, s["mean_stage"],
        xerr=[err_lo, err_hi],
        color=colours, alpha=0.88,
        height=0.62,
        error_kw=dict(ecolor="#333333", capsize=3.5, elinewidth=0.9,
                      capthick=0.9),
        zorder=3,
    )

    # Annotate bar ends — offset past the CI cap so text doesn't clip
    for bar, val, hi in zip(bars, s["mean_stage"], s["ci_hi"]):
        ax.text(
            hi + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center", fontsize=7.5, color="#333333",
        )

    ax.set_xlim(4.0, 7.0)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(
        ["S1", "S2", "S3", "S4\nLaw &\nOrder",
         "S5\nSocial\nContract", "S6\nUniversal\nEthics"],
        fontsize=8.5,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(s["display_name"], fontsize=8.5)
    ax.set_xlabel("Mean Kohlberg Stage  ± 95% Bootstrap CI", labelpad=5)
    ax.set_title(
        "Mean Moral Reasoning Stage per Model\n"
        "(ordered by scale: small → large)",
        fontsize=10, pad=10, fontweight="bold",
    )

    # Vertical reference lines at integer stages
    for s_val in range(4, 7):
        ax.axvline(s_val, color="#dddddd", linewidth=0.6, zorder=0)

    # Provider legend — outside axes to the right, clear of all bars
    leg = ax.legend(
        handles=_provider_legend_handles(),
        title="Provider", title_fontsize=8,
        loc="upper left", bbox_to_anchor=(1.02, 1),
        fontsize=8, framealpha=0.92, edgecolor="#cccccc",
        borderaxespad=0,
    )
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    _save(fig, out_dir, "fig4_bar_mean_stage.png")
