"""
visualizations.py — Publication-quality figures for Analysis 9.

Figure A: Capability Correlation Heatmap (Pearson r)
Figure B: Threshold Detection — log(params) vs. % Stage 5+ with sigmoid fit
Figure C: Multi-Capability Regression Coefficients (standardised, horizontal bar)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import (
    OUT_DIR, SCALE_COLORS, SCALE_MARKERS, TRAINING_COLORS,
    CAPABILITY_COLS, apply_publication_style, ALPHA, POST_CONV_THRESH,
)

warnings.filterwarnings("ignore")
apply_publication_style()

# ── Axis label mapping ────────────────────────────────────────────────────────
LABEL_MAP = {
    "coherence":            "Coherence\n(Conf.)",
    "response_length":      "Response\nLength",
    "sentence_count":       "Sentence\nCount",
    "avg_sentence_length":  "Avg Sent.\nLength",
    "lexical_diversity":    "Lexical\nDiversity",
    "syntactic_complexity": "Syntactic\nComplexity",
    "semantic_density":     "Semantic\nDensity",
    "mean_stage":           "Mean\nStage",
    "post_conv_pct":        "Post-Conv\n%",
    "log_params":           "log(Params)",
}


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Figure A: Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(corr_results: dict) -> str:
    """Figure A: Pearson r heatmap annotated with values + significance stars."""
    apply_publication_style()
    variables = corr_results["variables"]
    r_mat  = corr_results["pearson_r"]
    p_mat  = corr_results["corrected_p_pearson"]  # FDR-corrected

    display_labels = [LABEL_MAP.get(v, v) for v in variables]
    n = len(variables)

    fig, ax = plt.subplots(figsize=(8.5, 7))

    # Mask upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    sns.heatmap(
        r_mat.values,
        ax=ax,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        annot=False,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )

    # Annotate each cell manually with r value + stars
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue
            r_val = float(r_mat.iloc[i, j])
            p_val = float(p_mat.iloc[i, j])
            stars = _sig_stars(p_val) if i != j else ""
            txt   = f"{r_val:.2f}{stars}"
            fc    = "white" if abs(r_val) > 0.55 else "black"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                    fontsize=7.5, color=fc, fontweight="bold" if stars else "normal")

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(display_labels, rotation=0, fontsize=8)

    ax.set_title(
        "Figure A: Capability Correlation Matrix (Pearson r, FDR-corrected)\n"
        f"n = {corr_results['n_models']} models  |  *p<0.05  **p<0.01  ***p<0.001",
        fontsize=10, pad=8,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "fig_A_correlation_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure B: Threshold Detection
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_detection(model_df: pd.DataFrame,
                             threshold_results: dict) -> str:
    """
    Figure B: log(params) vs. mean_stage (coloured by scale group),
    with sigmoid fit, and AUC comparison panel.
    """
    apply_publication_style()

    # Use log_params as primary threshold predictor
    metric  = "log_params"
    t_res   = threshold_results.get(metric, {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: mean_stage vs log_params with sigmoid ──
    ax = axes[0]
    for sg, grp in model_df.groupby("scale_group"):
        ax.scatter(
            grp["log_params"], grp["mean_stage"],
            color=SCALE_COLORS.get(sg, "grey"),
            marker=SCALE_MARKERS.get(sg, "o"),
            s=90, zorder=5, label=sg,
            edgecolors="white", linewidths=0.6,
        )

    # Annotate model names
    for _, row in model_df.iterrows():
        ax.annotate(
            row["display_name"],
            (row["log_params"], row["mean_stage"]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=10.0, color="#333333",
        )

    if t_res.get("x_fine") is not None:
        ax.plot(t_res["x_fine"], t_res["y_fine"],
                color="#e15759", lw=2, label="Sigmoid fit", zorder=4)

    # Stage 5 reference line
    ax.axhline(5.0, color="#457b9d", ls="--", lw=1.2, label="Stage 5 threshold")

    # Sigmoid inflection
    sif = t_res.get("sigmoid_inflection", np.nan)
    if isinstance(sif, float) and not np.isnan(sif):
        ax.axvline(sif, color="#e15759", ls=":", lw=1.5, alpha=0.8)
        ax.text(sif + 0.02, ax.get_ylim()[1] * 0.97,
                f"Inflection\nx={sif:.2f}", color="#e15759",
                fontsize=11.0, va="top")

    ax.set_xlabel("log₁₀(Parameters, B)", fontsize=14)
    ax.set_ylabel("Mean Kohlberg Stage", fontsize=14)
    ax.set_title("Figure B: Threshold Detection\n(log Params vs. Mean Stage)", fontsize=16)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, ls="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Right: AUC comparison across metrics ──
    ax2 = axes[1]
    rows = []
    for m, tres in threshold_results.items():
        if isinstance(tres, dict) and "logistic_auc" in tres:
            auc_val = tres.get("logistic_auc", np.nan)
            if isinstance(auc_val, float) and not np.isnan(auc_val):
                rows.append({"metric": LABEL_MAP.get(m, m), "AUC": auc_val, "r2": tres.get("linear_r2", np.nan)})
    if rows:
        auc_df = pd.DataFrame(rows).sort_values("AUC", ascending=True)
        colors_bar = ["#457b9d" if v > 0.7 else "#aec6cf" for v in auc_df["AUC"]]
        bars = ax2.barh(auc_df["metric"], auc_df["AUC"],
                        color=colors_bar, edgecolor="white", height=0.6)
        ax2.axvline(0.5, color="#666", ls="--", lw=1, label="Chance (0.5)")
        ax2.axvline(0.7, color="#e15759", ls=":", lw=1.2, alpha=0.7, label="Good (0.7)")
        for bar, val in zip(bars, auc_df["AUC"]):
            ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}", va="center", fontsize=11)
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel("AUC (High vs. Low Mean Stage)", fontsize=14)
        ax2.set_title("Metric AUC: Predicting\nHigh vs. Low Mean Stage", fontsize=16)
        ax2.legend(fontsize=12)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
    else:
        ax2.text(0.5, 0.5, "AUC data unavailable", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=14, color="grey")
        ax2.set_title("Metric AUC", fontsize=16)

    plt.tight_layout()
    out_path = OUT_DIR / "fig_B_threshold_detection.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure C: Multi-Capability Regression Coefficients
# ─────────────────────────────────────────────────────────────────────────────

def plot_regression_coefficients(reg_results: dict) -> str:
    """
    Figure C: Horizontal bar chart of standardised regression coefficients.
    Error bars = 95% CI. Colours: positive = blue, negative = red.
    """
    apply_publication_style()

    if "error" in reg_results:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Insufficient data for regression",
                ha="center", va="center", transform=ax.transAxes)
        out_path = OUT_DIR / "fig_C_regression_coefficients.png"
        fig.savefig(out_path)
        plt.close(fig)
        return str(out_path)

    coef_df  = reg_results["coef_df"].copy()
    r2       = reg_results["r_squared"]
    adj_r2   = reg_results["adj_r2"]
    f_stat   = reg_results["f_stat"]
    f_p      = reg_results["f_p"]

    coef_df["label"] = coef_df["predictor"].map(lambda x: LABEL_MAP.get(x, x))
    coef_df = coef_df.sort_values("std_coef", ascending=True)

    colors_bar = ["#e15759" if v < 0 else "#457b9d" for v in coef_df["std_coef"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(coef_df) * 0.65)))

    y_pos = np.arange(len(coef_df))
    xerr_lo = coef_df["std_coef"].values - coef_df["ci_lo"].values
    xerr_hi = coef_df["ci_hi"].values - coef_df["std_coef"].values

    bars = ax.barh(y_pos, coef_df["std_coef"].values,
                   xerr=[xerr_lo, xerr_hi],
                   color=colors_bar, edgecolor="white", height=0.6,
                   error_kw={"ecolor": "#444", "capsize": 3, "lw": 1.2})

    # Significance indicators on bars
    for i, (_, row) in enumerate(coef_df.iterrows()):
        stars = _sig_stars(row["p_value"])
        if stars:
            x_annot = row["ci_hi"] + 0.015 if row["std_coef"] >= 0 else row["ci_lo"] - 0.015
            ha       = "left" if row["std_coef"] >= 0 else "right"
            ax.text(x_annot, i, stars, va="center", ha=ha,
                    fontsize=9, color="#333")

    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df["label"].values, fontsize=9)
    ax.set_xlabel("Standardised Regression Coefficient (β)", fontsize=10)
    ax.set_title(
        f"Figure C: Multi-Capability Predictors of Mean Moral Reasoning Stage\n"
        f"R² = {r2:.3f}   Adj-R² = {adj_r2:.3f}   F = {f_stat:.2f}   p = {f_p:.3f}",
        fontsize=10, pad=8,
    )

    # Legend patches
    pos_patch = mpatches.Patch(color="#457b9d", label="Positive predictor")
    neg_patch = mpatches.Patch(color="#e15759", label="Negative predictor")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", ls="--", alpha=0.35)

    plt.tight_layout()
    out_path = OUT_DIR / "fig_C_regression_coefficients.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure D: Partial Correlations
# ─────────────────────────────────────────────────────────────────────────────

def plot_partial_correlations(partial_df: pd.DataFrame) -> str:
    """
    Figure D: Side-by-side bar chart — raw Pearson r vs partial r
    (controlling for log_params). Highlights change due to scale.
    """
    apply_publication_style()

    df = partial_df.copy()
    df["label"] = df["metric"].map(lambda x: LABEL_MAP.get(x, x))
    df = df.sort_values("raw_r", key=abs, ascending=True)

    y_pos = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.65)))

    bars1 = ax.barh(y_pos - width / 2, df["raw_r"].values,
                    height=width, color="#4e79a7", label="Raw r", alpha=0.85)
    bars2 = ax.barh(y_pos + width / 2, df["partial_r"].values,
                    height=width, color="#f28e2b", label="Partial r\n(controlling scale)", alpha=0.85)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"].values, fontsize=9)
    ax.set_xlabel("Correlation with Mean Stage", fontsize=10)
    ax.set_title(
        "Figure D: Raw vs. Partial Correlation with Mean Moral Reasoning Stage\n"
        "(Partial = controlling for log₁₀(Parameters))",
        fontsize=10, pad=8,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", ls="--", alpha=0.35)

    plt.tight_layout()
    out_path = OUT_DIR / "fig_D_partial_correlations.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure E: Summary scatter panel (capability vs. stage, one per metric)
# ─────────────────────────────────────────────────────────────────────────────

def plot_capability_scatter_panel(model_df: pd.DataFrame,
                                  corr_results: dict) -> str:
    """
    Figure E: 2×4 grid of scatter plots — each capability metric vs. mean_stage.
    Points coloured by scale_group.
    """
    apply_publication_style()

    metrics_to_plot = [m for m in CAPABILITY_COLS if m in model_df.columns]
    n_metrics = len(metrics_to_plot)
    ncols = 4
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), squeeze=False)
    fig.suptitle(
        "Figure E: Capability Metrics vs. Mean Moral Reasoning Stage\n"
        "(each point = one model; coloured by scale group)",
        fontsize=11, y=1.01,
    )

    r_mat = corr_results["pearson_r"]
    p_mat = corr_results["corrected_p_pearson"]

    for idx, metric in enumerate(metrics_to_plot):
        row_i, col_i = divmod(idx, ncols)
        ax = axes[row_i][col_i]

        for sg, grp in model_df.groupby("scale_group"):
            ax.scatter(
                grp[metric], grp["mean_stage"],
                color=SCALE_COLORS.get(sg, "grey"),
                marker=SCALE_MARKERS.get(sg, "o"),
                s=70, label=sg, zorder=5,
                edgecolors="white", linewidths=0.5,
            )

        # Trend line
        valid = model_df[[metric, "mean_stage"]].dropna()
        if len(valid) >= 3:
            z = np.polyfit(valid[metric], valid["mean_stage"], 1)
            xf = np.linspace(valid[metric].min(), valid[metric].max(), 100)
            ax.plot(xf, np.polyval(z, xf), color="#666", ls="--", lw=1)

        # r and p annotation
        if metric in r_mat.columns and "mean_stage" in r_mat.index:
            r_val = r_mat.loc["mean_stage", metric]
            p_val = p_mat.loc["mean_stage", metric]
            stars = _sig_stars(float(p_val))
            ax.set_title(f"{LABEL_MAP.get(metric, metric)}\nr = {r_val:.2f}{stars}",
                         fontsize=8, pad=3)
        else:
            ax.set_title(LABEL_MAP.get(metric, metric), fontsize=8)

        ax.set_xlabel(LABEL_MAP.get(metric, metric), fontsize=7.5)
        ax.set_ylabel("Mean Stage" if col_i == 0 else "", fontsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

    # Hide any unused subplots
    for idx in range(n_metrics, nrows * ncols):
        row_i, col_i = divmod(idx, ncols)
        axes[row_i][col_i].set_visible(False)

    # Common legend
    patches = [mpatches.Patch(color=c, label=sg)
               for sg, c in SCALE_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_path = OUT_DIR / "fig_E_capability_scatter_panel.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return str(out_path)
