"""
Scale vs. Moral Reasoning Analysis
====================================
Research Question: Do larger models show higher moral reasoning stages?

This script:
  1. Loads all evaluation_data/*.xlsx files (Kohlberg stage labels)
  2. Loads matching data/*.xlsx files (model metadata)
  3. Maps each model to an approximate parameter count
  4. Computes per-model stage statistics (mean, median, mode, SD, distribution)
  5. Runs Spearman correlation (log-parameters vs mean stage)
  6. Runs Kruskal-Wallis + Dunn post-hoc tests
  7. Produces box plots, scatter plot, and stage-distribution heat-map
  8. Saves all results to analysis/results/
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import os
import glob
import warnings
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp   # Dunn's test (pip install scikit-posthocs)

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
EVAL_DIR  = ROOT / "evaluation_data"
OUT_DIR   = ROOT / "analysis" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── model metadata ────────────────────────────────────────────────────────
# (file_stem, display_name, approx_params_in_billions, category)
MODEL_META = {
    "claude_haiku_3_5":          ("Claude 3.5 Haiku",        20,    "Anthropic"),
    "claude_sonnet_4_5":         ("Claude Sonnet 4.5",        175,   "Anthropic"),
    "deepseek_r1":               ("DeepSeek-R1",              671,   "DeepSeek"),
    "deepseek_v3_1":             ("DeepSeek-V3.1",            671,   "DeepSeek"),
    "gpt-4o":                    ("GPT-4o",                   200,   "OpenAI"),
    "gpt-oss-120B":              ("GPT-OSS-120B",             120,   "OpenAI"),
    "llama3_70B":                ("Llama 3.3 70B",            70,    "Meta"),
    "llama4_scout":              ("Llama 4 Scout 17B×16E",    109,   "Meta"),
    "ministral_8b_instruct":     ("Ministral 8B",             8,     "Mistral"),
    "mistral_tiny":              ("Mistral Tiny",             7,     "Mistral"),
    "qwen3_235b_thinking":       ("Qwen3-235B (Think)",       235,   "Alibaba"),
    "qwen3_30B_Coder_Instruct":  ("Qwen3-30B Coder",         30,    "Alibaba"),
    "qwen3_32b":                 ("Qwen3-32B",                32,    "Alibaba"),
    "qwen3_80b_instruct":        ("Qwen3-80B",                80,    "Alibaba"),
}

KOHLBERG_LABELS = {
    1: "Stage 1\n(Obedience)",
    2: "Stage 2\n(Self-Interest)",
    3: "Stage 3\n(Conformity)",
    4: "Stage 4\n(Law & Order)",
    5: "Stage 5\n(Social Contract)",
    6: "Stage 6\n(Universal Ethics)",
}

PALETTE = sns.color_palette("husl", len(MODEL_META))

# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_data() -> pd.DataFrame:
    """
    Join evaluation_data (stage labels) with data files (model metadata).
    Returns a single long-format DataFrame with columns:
        model_key, display_name, params_B, log_params, category,
        kohlberg_stage, kohlberg_confidence, dilemma_type, prompt_type
    """
    frames = []
    eval_files = sorted(EVAL_DIR.glob("*_evaluation.xlsx"))

    for eval_path in eval_files:
        stem = eval_path.stem.replace("_evaluation", "")
        if stem not in MODEL_META:
            print(f"  [SKIP] {eval_path.name} – no metadata entry")
            continue

        display, params, cat = MODEL_META[stem]

        # --- evaluation data (stage labels) --------------------------------
        edf = pd.read_excel(eval_path)
        required_eval = ["kohlberg_stage", "kohlberg_confidence", "dilemma_type"]
        missing = [c for c in required_eval if c not in edf.columns]
        if missing:
            print(f"  [WARN] {eval_path.name} missing columns: {missing}")
            continue

        # --- matching data file (prompt_type etc.) -------------------------
        data_path = DATA_DIR / f"{stem}.xlsx"
        if data_path.exists():
            ddf = pd.read_excel(data_path)[["dilemma_type", "prompt_type"]]
            # merge on dilemma_type (best effort; duplicate rows get first match)
            edf = edf.merge(ddf.drop_duplicates("dilemma_type"),
                            on="dilemma_type", how="left")
        else:
            edf["prompt_type"] = np.nan

        edf["model_key"]    = stem
        edf["display_name"] = display
        edf["params_B"]     = params
        edf["log_params"]   = np.log10(params)
        edf["category"]     = cat

        frames.append(edf[["model_key", "display_name", "params_B", "log_params",
                            "category", "kohlberg_stage", "kohlberg_confidence",
                            "dilemma_type", "prompt_type"]])

    df = pd.concat(frames, ignore_index=True)

    # ensure stage is numeric
    df["kohlberg_stage"] = pd.to_numeric(df["kohlberg_stage"], errors="coerce")
    df.dropna(subset=["kohlberg_stage"], inplace=True)
    df["kohlberg_stage"] = df["kohlberg_stage"].astype(int)

    print(f"\nLoaded {len(df):,} rows from {df['model_key'].nunique()} models.\n")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PER-MODEL STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a model-level summary DataFrame with:
        display_name, params_B, log_params, category,
        mean_stage, median_stage, mode_stage, std_stage,
        n_samples, stage_1..6_pct
    """
    rows = []
    for key, grp in df.groupby("model_key"):
        stages = grp["kohlberg_stage"]
        mode_val = int(stages.mode().iloc[0]) if not stages.empty else np.nan

        row = {
            "model_key":    key,
            "display_name": grp["display_name"].iloc[0],
            "params_B":     grp["params_B"].iloc[0],
            "log_params":   grp["log_params"].iloc[0],
            "category":     grp["category"].iloc[0],
            "n_samples":    len(stages),
            "mean_stage":   stages.mean(),
            "median_stage": stages.median(),
            "mode_stage":   mode_val,
            "std_stage":    stages.std(),
        }
        # stage distribution percentages
        vcounts = stages.value_counts(normalize=True) * 100
        for s in range(1, 7):
            row[f"stage_{s}_pct"] = vcounts.get(s, 0.0)

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("params_B").reset_index(drop=True)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# 3.  BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(data: np.ndarray, stat_fn=np.mean,
                 n_boot: int = 5000, ci: float = 0.95) -> tuple[float, float]:
    boot_stats = [stat_fn(np.random.choice(data, size=len(data), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return lo, hi


def add_bootstrap_ci(df_long: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    ci_lo, ci_hi = [], []
    for key in summary["model_key"]:
        stages = df_long[df_long["model_key"] == key]["kohlberg_stage"].values
        lo, hi = bootstrap_ci(stages)
        ci_lo.append(lo)
        ci_hi.append(hi)
    summary["ci_lo"] = ci_lo
    summary["ci_hi"] = ci_hi
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# 4.  CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def spearman_with_ci(x: np.ndarray, y: np.ndarray,
                     n_boot: int = 5000) -> dict:
    rho, p = spearmanr(x, y)
    boot_rhos = []
    n = len(x)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        boot_rhos.append(r)
    lo = np.percentile(boot_rhos, 2.5)
    hi = np.percentile(boot_rhos, 97.5)
    r2 = rho ** 2
    # effect size label
    abs_rho = abs(rho)
    if abs_rho >= 0.5:
        effect = "large"
    elif abs_rho >= 0.3:
        effect = "medium"
    elif abs_rho >= 0.1:
        effect = "small"
    else:
        effect = "negligible"

    return dict(rho=rho, p=p, ci_lo=lo, ci_hi=hi, r2=r2, effect=effect)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════

def run_nonparametric_tests(df: pd.DataFrame) -> dict:
    """Kruskal-Wallis + Dunn post-hoc (Bonferroni)."""
    groups   = [g["kohlberg_stage"].values
                for _, g in df.groupby("model_key")]
    kw_stat, kw_p = kruskal(*groups)

    # Dunn pairwise
    dunn_df = sp.posthoc_dunn(df, val_col="kohlberg_stage",
                               group_col="model_key", p_adjust="bonferroni")
    return dict(kw_stat=kw_stat, kw_p=kw_p, dunn=dunn_df)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════

# ── colour helpers
CAT_COLORS = {
    "Anthropic": "#E07A5F",
    "OpenAI":    "#3D405B",
    "Meta":      "#81B29A",
    "Mistral":   "#F2CC8F",
    "Alibaba":   "#9B89C4",
    "DeepSeek":  "#52B2CF",
}

def model_color(cat):
    return CAT_COLORS.get(cat, "#888888")


def fig_box_plots(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Box plot of stage distribution per model, ordered by params_B."""
    order = summary.sort_values("params_B")["display_name"].tolist()
    colours = [model_color(summary.loc[summary["display_name"] == m, "category"].iloc[0])
               for m in order]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    bp = ax.boxplot(
        [df[df["display_name"] == m]["kohlberg_stage"].values for m in order],
        labels=order,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        flierprops=dict(marker="o", markersize=4, alpha=0.4),
        widths=0.55,
    )
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)

    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([f"Stage {i}" for i in range(1, 7)], fontsize=9)
    ax.set_ylabel("Kohlberg Stage", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model  (left → right = smaller → larger)", fontsize=11)
    ax.set_title("Kohlberg Stage Distribution by Model\n(ordered by parameter count)",
                 fontsize=14, fontweight="bold", pad=14)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # legend for categories
    handles = [plt.Rectangle((0, 0), 1, 1, color=v, alpha=0.85)
               for v in CAT_COLORS.values()]
    ax.legend(handles, CAT_COLORS.keys(), title="Provider",
              loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "box_stage_by_model.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: box_stage_by_model.png")


def fig_scatter_scale_stage(summary: pd.DataFrame, corr: dict) -> None:
    """Scatter: log10(params) vs mean stage, with 95 % CI error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    for _, row in summary.iterrows():
        col = model_color(row["category"])
        ax.errorbar(
            row["log_params"], row["mean_stage"],
            yerr=[[row["mean_stage"] - row["ci_lo"]],
                  [row["ci_hi"] - row["mean_stage"]]],
            fmt="o", color=col, ecolor=col, elinewidth=1.5,
            markersize=10, capsize=4, alpha=0.9,
        )
        ax.annotate(
            row["display_name"],
            (row["log_params"], row["mean_stage"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=7.5, alpha=0.85,
        )

    # trend line (Spearman — use linear fit for visualization)
    x = summary["log_params"].values
    y = summary["mean_stage"].values
    m, b = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min() - 0.1, x.max() + 0.1, 200)
    ax.plot(xfit, m * xfit + b, "--", color="#555555", linewidth=1.4,
            label="Linear trend")

    ax.set_xlabel("Log₁₀(Parameter Count in Billions)", fontsize=12)
    ax.set_ylabel("Mean Kohlberg Stage  ± 95 % CI", fontsize=12)
    ax.set_title(
        f"Model Scale vs. Moral Reasoning Stage\n"
        f"Spearman ρ = {corr['rho']:.3f}  "
        f"(95 % CI [{corr['ci_lo']:.3f}, {corr['ci_hi']:.3f}]),  "
        f"p = {corr['p']:.4f},  Effect: {corr['effect']}",
        fontsize=12, fontweight="bold", pad=14,
    )
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([f"Stage {i}" for i in range(1, 7)], fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)

    # custom x-tick labels
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"10^{v:.1f}\n({10**v:.0f}B)")
    )

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=v, markersize=10, alpha=0.9)
               for v in CAT_COLORS.values()]
    ax.legend(handles, CAT_COLORS.keys(), title="Provider",
              loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "scatter_scale_vs_stage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: scatter_scale_vs_stage.png")


def fig_stage_heatmap(summary: pd.DataFrame) -> None:
    """Heat-map of stage % distribution per model."""
    stage_cols = [f"stage_{i}_pct" for i in range(1, 7)]
    heat = summary.set_index("display_name")[stage_cols].copy()
    heat.columns = [f"Stage {i}" for i in range(1, 7)]
    heat.index.name = "Model"

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        heat, ax=ax,
        cmap="YlOrRd", annot=True, fmt=".1f",
        linewidths=0.5, linecolor="#cccccc",
        cbar_kws={"label": "% of responses"},
        vmin=0, vmax=100,
    )
    ax.set_title("Stage Distribution (%) per Model\n(rows ordered by parameter count)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Kohlberg Stage", fontsize=11)
    ax.set_ylabel("")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=8, rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "heatmap_stage_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: heatmap_stage_distribution.png")


def fig_mean_stage_bar(summary: pd.DataFrame) -> None:
    """Horizontal bar chart of mean stage per model with error bars."""
    s = summary.sort_values("params_B")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    colours = [model_color(c) for c in s["category"]]
    bars = ax.barh(s["display_name"], s["mean_stage"],
                   xerr=[s["mean_stage"] - s["ci_lo"],
                         s["ci_hi"] - s["mean_stage"]],
                   color=colours, alpha=0.85,
                   error_kw={"ecolor": "#333333", "capsize": 4, "elinewidth": 1.5})

    ax.set_xlim(0, 6.8)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"Stage {i}" for i in range(1, 7)], fontsize=9.5)
    ax.set_xlabel("Mean Kohlberg Stage  ± 95 % Bootstrap CI", fontsize=11)
    ax.set_title("Mean Moral Reasoning Stage per Model\n(ordered by scale: small → large)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # annotate values
    for bar, val in zip(bars, s["mean_stage"]):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=v, alpha=0.85)
               for v in CAT_COLORS.values()]
    ax.legend(handles, CAT_COLORS.keys(), title="Provider",
              loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bar_mean_stage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: bar_mean_stage.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7.  RESULTS REPORT
# ═══════════════════════════════════════════════════════════════════════════

def save_csv_report(summary: pd.DataFrame, corr: dict, tests: dict) -> None:
    # model-level stats
    summary.to_csv(OUT_DIR / "model_stats.csv", index=False)
    print("  Saved: model_stats.csv")

    # Dunn post-hoc table
    tests["dunn"].to_csv(OUT_DIR / "dunn_posthoc.csv")
    print("  Saved: dunn_posthoc.csv")

    # correlation summary
    corr_row = {k: v for k, v in corr.items() if k != "dunn"}
    pd.DataFrame([corr_row]).to_csv(OUT_DIR / "spearman_correlation.csv", index=False)
    print("  Saved: spearman_correlation.csv")


def print_report(summary: pd.DataFrame, corr: dict, tests: dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  SCALE vs. MORAL REASONING ANALYSIS — RESULTS SUMMARY")
    print(sep)

    print("\n── Per-Model Statistics (ordered by scale) ──────────────────────")
    cols = ["display_name", "params_B", "n_samples",
            "mean_stage", "median_stage", "mode_stage", "std_stage"]
    print(summary[cols].to_string(index=False, float_format="%.3f"))

    print("\n── Stage Distribution (%) ────────────────────────────────────────")
    dist_cols = ["display_name"] + [f"stage_{i}_pct" for i in range(1, 7)]
    print(summary[dist_cols].to_string(index=False, float_format="%.1f"))

    print("\n── Spearman Correlation (log-params vs. mean stage) ─────────────")
    print(f"  ρ           = {corr['rho']:.4f}")
    print(f"  95 % CI     = [{corr['ci_lo']:.4f}, {corr['ci_hi']:.4f}]")
    print(f"  p-value     = {corr['p']:.6f}")
    print(f"  R²          = {corr['r2']:.4f}")
    print(f"  Effect size = {corr['effect']}")

    print("\n── Kruskal-Wallis Test ───────────────────────────────────────────")
    print(f"  H-stat = {tests['kw_stat']:.4f},  p = {tests['kw_p']:.6f}")
    sig = "SIGNIFICANT" if tests["kw_p"] < 0.05 else "not significant"
    print(f"  Result : {sig} at α = 0.05")

    print("\n── Interpretation ───────────────────────────────────────────────")
    direction = "positive" if corr["rho"] > 0 else "negative"
    print(f"  There is a {direction} {corr['effect']} correlation (ρ={corr['rho']:.3f}) between")
    print("  log-scale model size and mean Kohlberg stage.")
    if corr['p'] < 0.05:
        print("  This correlation is statistically significant (p < 0.05).")
    else:
        print("  This correlation is NOT statistically significant (p ≥ 0.05).")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)

    print("Loading data …")
    df = load_all_data()

    print("Computing per-model statistics …")
    summary = compute_model_stats(df)

    print("Bootstrapping 95 % CIs …")
    summary = add_bootstrap_ci(df, summary)

    print("Running Spearman correlation …")
    corr = spearman_with_ci(summary["log_params"].values,
                             summary["mean_stage"].values)

    print("Running Kruskal-Wallis + Dunn tests …")
    tests = run_nonparametric_tests(df)

    print("\nGenerating plots …")
    fig_box_plots(df, summary)
    fig_scatter_scale_stage(summary, corr)
    fig_stage_heatmap(summary)
    fig_mean_stage_bar(summary)

    print("\nSaving CSV reports …")
    save_csv_report(summary, corr, tests)

    print_report(summary, corr, tests)

    print(f"\nAll outputs written to:  {OUT_DIR}\n")


if __name__ == "__main__":
    main()
