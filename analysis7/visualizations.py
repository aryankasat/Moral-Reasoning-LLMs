"""
visualizations.py — Publication-quality figures for Analysis 7: Emergence Threshold Detection.

Figures produced:
  fig1_emergence_curves.png   — Three-panel emergence curves with CI and changepoints
  fig2_emergence_vs_params.png — Scatter: mean stage vs log(params) with regression
  fig3_stage_heatmap.png      — Stage distribution heatmap across models
  fig4_slope_analysis.png     — Slope comparison bar chart + R² panel
"""

from __future__ import annotations
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from matplotlib.lines import Line2D

from config import (
    OUT_DIR, STAGE_COLORS, PROVIDER_COLORS, SCALE_GROUPS,
    STAGES, POST_CONV_THRESHOLD, apply_publication_style,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

SCALE_GROUP_COLORS = {
    "Small (8B–32B)":    "#6a994e",
    "Mid   (70B–120B)":  "#e76f51",
    "Large (175B–671B)": "#457b9d",
}


def _format_params(p: float) -> str:
    """Human-readable parameter count label."""
    if p >= 1000:
        return f"{p/1000:.0f}T"
    if p == int(p):
        return f"{int(p)}B"
    return f"{p:.0f}B"


def _log_param_xticks(ax, params_values: np.ndarray, fontsize=8) -> None:
    """Set x-ticks at the exact (plot) parameter counts with human-readable labels."""
    log_vals = np.log10(params_values)
    ax.set_xticks(log_vals)
    ax.set_xticklabels([_format_params(p) for p in params_values],
                       rotation=35, ha="right", fontsize=fontsize)


def _shade_ci(ax, x_vals, ci_lower, ci_upper, color="#457b9d", alpha=0.18) -> None:
    """Shade 95% confidence interval band."""
    ax.fill_between(x_vals, ci_lower, ci_upper, color=color, alpha=alpha, zorder=1)


def _provider_legend_handles():
    return [mpatches.Patch(color=v, label=k) for k, v in PROVIDER_COLORS.items()]


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1: Three-panel Emergence Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_emergence_curves(
    model_df: pd.DataFrame,
    analysis_results: dict,
) -> str:
    """
    Three panels (small / mid / large scale).
    X-axis uses log10(params_B_plot) — always populated, no NaN.
    """
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(22, 9.5), sharey=True)
    fig.suptitle(
        "Figure 1: Emergence Curves — Moral Reasoning Stage vs. Model Scale",
        fontsize=18, fontweight="bold", y=1.02,
    )

    changepoints        = analysis_results.get("changepoints", [])
    all_log_params_plot = model_df["log_params_plot"].values  # no NaN
    emrg                = analysis_results.get("emergence_threshold", {})
    emrg_p              = emrg.get("emergence_params_B", np.nan)

    group_items = list(SCALE_GROUPS.items())

    for ax, (group_name, param_range) in zip(axes, group_items):
        lo, hi = min(param_range), max(param_range)
        group_color = SCALE_GROUP_COLORS[group_name]

        # Filter using params_B (the canonical value, not the jitter)
        sub = model_df[(model_df["params_B"] >= lo) & (model_df["params_B"] <= hi)].copy()
        sub = sub.sort_values("params_B_plot", ignore_index=True)

        if sub.empty:
            ax.text(0.5, 0.5, "No data in this range",
                    ha="center", va="center", transform=ax.transAxes, color="#999", fontsize=12)
            ax.set_title(group_name, fontsize=14, fontweight="bold")
            continue

        x_plot  = sub["log_params_plot"].values   # guaranteed non-NaN
        y_mean  = sub["mean_stage"].values
        y_lo    = np.clip(sub["ci_lower"].values, 1, 6)
        y_hi    = np.clip(sub["ci_upper"].values, 1, 6)

        # CI shading
        if len(x_plot) > 1:
            _shade_ci(ax, x_plot, y_lo, y_hi, color=group_color, alpha=0.22)

        # Mean line
        ax.plot(x_plot, y_mean, color=group_color, linewidth=2.0, zorder=3,
                marker="o", markersize=0)

        # Scatter points (provider-coloured)
        for xi, yi, row in zip(x_plot, y_mean, sub.itertuples()):
            color = PROVIDER_COLORS.get(row.provider, "#888")
            ax.scatter(xi, yi, color=color, s=70, zorder=5,
                       edgecolors="white", linewidth=0.6)
            # Model label (stagger alternately above/below to reduce overlap)
            offset_y = 0.10 if sub.index.get_loc(row.Index) % 2 == 0 else -0.18
            ax.annotate(
                row.display_name,
                xy=(xi, yi),
                xytext=(0, int(offset_y * 100)),
                textcoords="offset points",
                fontsize=13, ha="center", color="#333",
                rotation=20, zorder=6,
            )

        # Stage reference lines
        for stage_y, stage_lbl in [(4, "Stage 4"), (5, "Stage 5"), (6, "Stage 6")]:
            ax.axhline(stage_y, color=STAGE_COLORS[stage_y],
                       linestyle=":", linewidth=0.9, alpha=0.6, zorder=0)
            ax.text(x_plot.max(), stage_y + 0.03, stage_lbl,
                    color=STAGE_COLORS[stage_y], fontsize=12, ha="right",
                    va="bottom", zorder=0)

        # Changepoints that fall within this panel's x range
        x_lo_lim = x_plot.min() - 0.2
        x_hi_lim = x_plot.max() + 0.2
        for bkp_i in changepoints:
            if 0 <= bkp_i < len(all_log_params_plot):
                bkp_x = all_log_params_plot[bkp_i]
                if x_lo_lim <= bkp_x <= x_hi_lim:
                    ax.axvline(bkp_x, color="#e63946", linestyle="--",
                               linewidth=1.4, alpha=0.85, zorder=4)
                    ax.text(bkp_x + 0.01, y_lo.min() + 0.05,
                            "CP", color="#e63946", fontsize=13, zorder=4)

        # Emergence marker
        if not np.isnan(emrg_p) and lo <= emrg_p <= hi:
            ax.axvline(np.log10(emrg_p), color="#2a9d8f", linestyle="-.",
                       linewidth=1.5, alpha=0.9, zorder=4)
            ax.text(np.log10(emrg_p) + 0.01, y_hi.max() - 0.2,
                    f"Stage 5+\nEmergence\n({_format_params(emrg_p)})",
                    color="#2a9d8f", fontsize=12, zorder=4)

        # Axes
        ax.set_title(group_name, fontsize=16, fontweight="bold", pad=10)
        ax.set_xlabel("Model Parameters (log scale)", fontsize=15)
        if ax is axes[0]:
            ax.set_ylabel("Mean Moral Reasoning Stage", fontsize=15)
        ax.set_ylim(4.8, 6.3)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
        ax.tick_params(axis='both', which='major', labelsize=13)
        _log_param_xticks(ax, sub["params_B_plot"].values, fontsize=13)
        if len(x_plot) > 1:
            ax.set_xlim(x_plot.min() - 0.1, x_plot.max() + 0.15)

    # Scenario annotation
    scenario = analysis_results.get("scenario", "")
    fig.text(0.5, -0.01, f"Detected pattern: {scenario}",
             ha="center", fontsize=13, style="italic", color="#444")

    # Shared legend
    handles = _provider_legend_handles()
    handles += [
        Line2D([0], [0], color="#e63946", linestyle="--", lw=1.4, label="Changepoint"),
        Line2D([0], [0], color="#2a9d8f", linestyle="-.", lw=1.5, label="Stage 5+ Emergence"),
        mpatches.Patch(color="#457b9d", alpha=0.3, label="95% CI"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.06), fontsize=12, framealpha=0.9)

    plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=1.0)
    fig.subplots_adjust(bottom=0.15)
    out = OUT_DIR / "fig1_emergence_curves.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out.name}")
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2: Mean Stage vs. Log-Params with Segmented Regression
# ─────────────────────────────────────────────────────────────────────────────

def plot_emergence_vs_params(
    model_df: pd.DataFrame,
    analysis_results: dict,
) -> str:
    """
    Scatter of all 13 models on a single panel.
    Uses log_params_plot for x (non-NaN for all models).
    Overlays linear + segmented regression on log_params (canonical, not jittered).
    """
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Use canonical log_params for regression, log_params_plot for scatter display
    x_canon = model_df["log_params"].values         # canonical (regression)
    x_plot  = model_df["log_params_plot"].values    # with tiny jitter (scatter)
    y_all   = model_df["mean_stage"].values

    # ── Scatter: each model ──────────────────────────────────────────────────
    for _, row in model_df.iterrows():
        color = PROVIDER_COLORS.get(row["provider"], "#888")
        ax.errorbar(
            row["log_params_plot"], row["mean_stage"],
            yerr=[[row["mean_stage"] - row["ci_lower"]],
                  [row["ci_upper"] - row["mean_stage"]]],
            fmt="none", color=color, alpha=0.50, capsize=3, linewidth=1.2, zorder=3,
        )
        ax.scatter(
            row["log_params_plot"], row["mean_stage"],
            color=color, s=90, zorder=5,
            edgecolors="white", linewidth=0.7,
        )
        # Label each point
        ax.annotate(
            row["display_name"],
            xy=(row["log_params_plot"], row["mean_stage"]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=7.5, color="#333", zorder=6,
        )

    # ── Linear regression fit ────────────────────────────────────────────────
    sl, ic, r_val, *_ = sp_stats.linregress(x_canon, y_all)
    x_fit = np.linspace(x_canon.min() - 0.05, x_canon.max() + 0.05, 300)
    ax.plot(x_fit, sl * x_fit + ic,
            color="#888", linestyle="--", linewidth=1.3, alpha=0.75,
            label=f"Linear fit  (r = {r_val:.2f})", zorder=2)

    # ── Segmented regression fit ─────────────────────────────────────────────
    seg     = analysis_results.get("segmented_regression", {})
    bkp_idx = analysis_results.get("primary_changepoint_idx")

    if (bkp_idx is not None and 0 < bkp_idx < len(x_canon)
            and not np.isnan(seg.get("slope_pre", np.nan))):
        bkp_x     = x_canon[bkp_idx]
        s1        = seg["slope_pre"]
        i1        = seg["intercept_pre"]
        s2        = seg["slope_post"]
        i2        = seg["intercept_post"]
        junction_y = s1 * bkp_x + i1

        x_pre  = x_fit[x_fit <= bkp_x]
        x_post = x_fit[x_fit >  bkp_x]

        ax.plot(x_pre,  s1 * x_pre + i1,
                color="#e63946", linewidth=1.8,
                label=f"Seg. pre  (slope = {s1:+.3f})", zorder=3)
        ax.plot(x_post, s2 * (x_post - bkp_x) + junction_y,
                color="#2a9d8f", linewidth=1.8,
                label=f"Seg. post (slope = {s2:+.3f})", zorder=3)
        ax.axvline(bkp_x, color="#e63946", linestyle="--",
                   linewidth=1.1, alpha=0.70, label="Changepoint", zorder=2)

    # ── Emergence threshold line ─────────────────────────────────────────────
    emrg   = analysis_results.get("emergence_threshold", {})
    emrg_p = emrg.get("emergence_params_B", np.nan)
    if not np.isnan(emrg_p):
        ax.axvline(np.log10(emrg_p), color="#f4a261", linestyle="-.",
                   linewidth=1.6, label=f"Stage 5+ emergence ({_format_params(emrg_p)})", zorder=2)

    # Stage 5 reference
    ax.axhline(5.0, color=STAGE_COLORS[5], linestyle=":", linewidth=0.9,
               alpha=0.55, label="Stage 5 threshold", zorder=1)

    # ── Post-conv shading ─────────────────────────────────────────────────────
    y_top = 6.4
    ax.axhspan(5.0, y_top, color="#4575b4", alpha=0.05, zorder=0)

    # ── Axes & metadata ──────────────────────────────────────────────────────
    _log_param_xticks(ax, model_df["params_B_plot"].values)
    y_min = max(4.7, y_all.min() - 0.4)
    ax.set_ylim(y_min, y_top)
    ax.set_xlim(x_plot.min() - 0.12, x_plot.max() + 0.14)
    ax.set_xlabel("Model Parameters (log scale)", fontsize=11)
    ax.set_ylabel("Mean Moral Reasoning Stage", fontsize=11)
    ax.set_title(
        "Figure 2: Emergence vs. Parameter Count\n"
        "Linear and Segmented Regression Fits",
        fontsize=12, fontweight="bold",
    )

    # Correlation box
    corr = analysis_results.get("cross_scale_correlation", {})
    rho  = corr.get("spearman_rho", np.nan)
    pval = corr.get("spearman_pval", np.nan)
    ax.text(
        0.02, 0.97,
        f"Spearman ρ = {rho:.3f}  (p = {pval:.4f})",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(fc="white", ec="#aaa", alpha=0.85, boxstyle="round,pad=0.3"),
    )

    # Two separate legends
    prov_handles = [mpatches.Patch(color=v, label=k)
                    for k, v in PROVIDER_COLORS.items()
                    if k in model_df["provider"].values]
    leg1 = ax.legend(handles=prov_handles, loc="lower right",
                     fontsize=8, title="Provider", title_fontsize=8,
                     framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out = OUT_DIR / "fig2_emergence_vs_params.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out.name}")
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3: Stage Distribution Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_stage_heatmap(model_df: pd.DataFrame) -> str:
    """
    Heatmap: models × Kohlberg stages, cell = fraction of responses at that stage.
    Models sorted by parameter count (bottom → smallerest at top).
    """
    apply_publication_style()

    stage_cols = [f"stage_{s}_pct" for s in STAGES]
    missing = [c for c in stage_cols if c not in model_df.columns]
    if missing:
        warnings.warn(f"Missing stage columns: {missing}")
        return ""

    # Build matrix: rows = models (sorted by params), columns = stages
    heat = model_df.set_index("display_name")[stage_cols].values  # (n_models, 6)
    model_labels = model_df["display_name"].tolist()
    stage_labels = [f"Stage {s}" for s in STAGES]
    n_models, n_stages = heat.shape

    fig_h = max(5.5, n_models * 0.52 + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1, zorder=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Fraction of Responses", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Cell text
    for r in range(n_models):
        for c in range(n_stages):
            val = heat[r, c]
            txt_color = "white" if val > 0.58 else "black"
            weight    = "bold"  if val > 0.20 else "normal"
            ax.text(c, r, f"{val:.0%}",
                    ha="center", va="center",
                    fontsize=8.5, color=txt_color, fontweight=weight, zorder=3)

    # Highlight Stage 5+ columns (columns 4 and 5 → Stage 5, Stage 6)
    for c_idx in [4, 5]:
        rect = plt.Rectangle(
            (c_idx - 0.5, -0.5), 1, n_models,
            fill=False, edgecolor="#2a9d8f",
            linewidth=2.2, linestyle="--", zorder=4,
        )
        ax.add_patch(rect)

    # Highlight rows of emerged models
    for r, row in enumerate(model_df.itertuples()):
        if row.emerged:
            rect = plt.Rectangle(
                (-0.5, r - 0.5), n_stages, 1,
                fill=False, edgecolor="#f4a261",
                linewidth=1.6, linestyle="-", zorder=4,
            )
            ax.add_patch(rect)

    # Axes formatting
    ax.set_xticks(range(n_stages))
    ax.set_xticklabels(stage_labels, fontsize=10)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_labels, fontsize=9)
    ax.set_xlabel("Kohlberg Stage", fontsize=11)
    ax.set_ylabel("Model (sorted by parameter count ↑)", fontsize=10)
    ax.set_title(
        "Figure 3: Stage Distribution Heatmap by Model\n"
        f"(Dashed teal = Stage 5+  |  Orange border = met ≥{POST_CONV_THRESHOLD:.0%} post-conventional threshold)",
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout()
    out = OUT_DIR / "fig3_stage_heatmap.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out.name}")
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 4: Slope Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_slope_analysis(model_df: pd.DataFrame, analysis_results: dict) -> str:
    """
    Panel A: pre vs post changepoint slope bars.
    Panel B: R² comparison, linear vs segmented.
    """
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    seg    = analysis_results.get("segmented_regression", {})
    s_pre  = seg.get("slope_pre",  np.nan)
    s_post = seg.get("slope_post", np.nan)
    p_val  = seg.get("p_value",    np.nan)
    r2_lin = seg.get("r2_linear",  np.nan)
    r2_seg = seg.get("r2_segmented", np.nan)

    def _safe(v):
        return 0.0 if (v is None or np.isnan(v)) else v

    # ── Panel A: slope comparison ─────────────────────────────────────────
    ax1 = axes[0]
    labels = ["Pre-changepoint\nslope", "Post-changepoint\nslope"]
    vals   = [_safe(s_pre), _safe(s_post)]
    colors = ["#457b9d", "#e63946"]
    bars   = ax1.bar(labels, vals, color=colors, width=0.45,
                     edgecolor="white", linewidth=0.8)

    bar_top = max(abs(v) for v in vals) * 0.08
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 v + (bar_top if v >= 0 else -bar_top - 0.005),
                 f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=10, fontweight="bold")

    ax1.axhline(0, color="#555", linewidth=0.8)
    ax1.set_ylabel("Slope  (delta stage / delta log10 params)", fontsize=10)
    ax1.set_title("Pre- vs. Post-Changepoint Slope", fontsize=11, fontweight="bold")

    sig_str = f"F-test: p = {_safe(p_val):.4f}"
    sig_col = "#2a9d8f" if (not np.isnan(p_val) and p_val < 0.05) else "#e63946"
    ax1.text(0.5, 0.06, sig_str, transform=ax1.transAxes,
             ha="center", color=sig_col, fontsize=9,
             bbox=dict(fc="white", ec=sig_col, alpha=0.85, boxstyle="round,pad=0.4"))

    # ── Panel B: R² comparison ────────────────────────────────────────────
    ax2 = axes[1]
    r2_labels = ["Linear\nR²", "Segmented\nR²"]
    r2_vals   = [_safe(r2_lin), _safe(r2_seg)]
    r2_colors = ["#888888", "#e76f51"]
    bars2 = ax2.bar(r2_labels, r2_vals, color=r2_colors, width=0.45,
                    edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars2, r2_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 v + 0.008,
                 f"{v:.4f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("R²  (goodness of fit)", fontsize=10)
    ax2.set_title("Linear vs. Segmented Model Fit", fontsize=11, fontweight="bold")
    ax2.axhline(1.0, color="#ccc", linestyle=":", linewidth=0.8)

    scenario = analysis_results.get("scenario", "")
    effect   = analysis_results.get("effect_size", np.nan)
    fig.suptitle(
        f"Figure 4: Slope Analysis\n{scenario}  |  Effect size: {_safe(effect):.2f} stages",
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout()
    out = OUT_DIR / "fig4_slope_analysis.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out.name}")
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
#  Master runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_visualizations(model_df: pd.DataFrame, analysis_results: dict) -> list[str]:
    """Produce all four figures and return list of output paths."""
    paths = []
    paths.append(plot_emergence_curves(model_df, analysis_results))
    paths.append(plot_emergence_vs_params(model_df, analysis_results))
    paths.append(plot_stage_heatmap(model_df))
    paths.append(plot_slope_analysis(model_df, analysis_results))
    return [p for p in paths if p]
