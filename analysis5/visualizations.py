"""
visualizations.py — 6 publication-grade figures for Analysis 5.
"""

from __future__ import annotations
import warnings
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
    STAGES, STAGE_COLORS, PROVIDER_COLORS, 
    ACTION_COLORS, CONSISTENCY_COLORS, EXPECTED_ACTION_BY_STAGE,
    apply_publication_style
)

apply_publication_style()
MM = 1 / 25.4

def _save(fig, path: Path, name: str) -> None:
    fig.savefig(path / name, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Action distribution by dilemma
# ═══════════════════════════════════════════════════════════════════════════

def plot_action_by_dilemma(dist_df: pd.DataFrame, out_dir: Path) -> None:
    """Stacked bar chart showing % Rule-Following vs % Rule-Breaking per dilemma."""
    df = dist_df.set_index("dilemma_type")
    
    # Fill missing if any
    for col in ["Rule-Following", "Rule-Breaking"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df[["Rule-Following", "Rule-Breaking"]].sort_values("Rule-Following", ascending=False)
    
    fig, ax = plt.subplots(figsize=(150 * MM, 90 * MM))
    
    y = np.arange(len(df))
    ax.barh(y, df["Rule-Following"], color=ACTION_COLORS["Rule-Following"], alpha=0.85, label="Rule-Following")
    ax.barh(y, df["Rule-Breaking"], left=df["Rule-Following"], color=ACTION_COLORS["Rule-Breaking"], alpha=0.85, label="Rule-Breaking (Principled)")

    ax.set_yticks(y)
    ax.set_yticklabels([d.replace('_DILEMMA', '').replace('_DILLEMA', '').title() for d in df.index], fontsize=9)
    ax.set_xlabel("Percentage of Actions (%)", fontsize=9.5)
    ax.set_xlim(0, 100)
    
    ax.set_title("Overall Action Distributions by Dilemma\n(Aggregated across all models)", fontsize=10, fontweight="bold", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, framealpha=0.9)
    
    _save(fig, out_dir, "fig1_action_by_dilemma.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Stage × Action heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_stage_action_heatmap(ct: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap showing counts of Stage vs Action Category."""
    fig, ax = plt.subplots(figsize=(110 * MM, 100 * MM))
    
    # Highlight expected cells with subtle border
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={'label': 'Number of Responses'})
    
    for i, s in enumerate(ct.index):
        for j, act in enumerate(ct.columns):
            if EXPECTED_ACTION_BY_STAGE.get(s) == act:
                rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="#d73027", lw=2.5, clip_on=False)
                ax.add_patch(rect)
                
    # Legend for the highlight
    red_box = mpatches.Patch(facecolor='none', edgecolor="#d73027", lw=2, label="Theoretically expected alignment")
    ax.legend(handles=[red_box], loc="upper right", bbox_to_anchor=(1.05, 1.15))
    
    ax.set_ylabel("Kohlberg Stage (Reasoning)", fontsize=9.5)
    ax.set_xlabel("Action Category", fontsize=9.5)
    ax.set_yticklabels([f"S{idx}" for idx in ct.index], rotation=0)
    ax.set_title("Moral Reasoning Stage vs. Action Endorsed\n(Red outlines indicate theoretically consistent mappings)", fontsize=10, fontweight="bold", pad=20)
    
    _save(fig, out_dir, "fig2_stage_action_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Consistency score bar chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_consistency_bar(consist_df: pd.DataFrame, out_dir: Path) -> None:
    """Horizontal bar chart for Consistency % by model."""
    df = consist_df.sort_values("consistency_pct", ascending=True).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(140 * MM, max(100, len(df)*8) * MM))
    
    y = np.arange(len(df))
    colors = [PROVIDER_COLORS.get(p, "#888") for p in df["provider"]]
    
    ax.barh(y, df["consistency_pct"], color=colors, alpha=0.85, height=0.6)
    
    for i, row in df.iterrows():
        ax.text(row["consistency_pct"] + 1, i, f"{row['consistency_pct']:.1f}%", va='center', fontsize=8)
        
    ax.set_yticks(y)
    ax.set_yticklabels(df["display_name"], fontsize=8)
    ax.set_xlabel("Action-Reasoning Consistency (%)", fontsize=9.5)
    ax.set_xlim(0, max(105, df["consistency_pct"].max() + 15))
    
    ax.set_title("Reasoning-Action Consistency by Model\n(% of responses where reasoning stage aligns with action)", fontsize=10, fontweight="bold")
    
    handles = [mpatches.Patch(color=c, label=p) for p, c in PROVIDER_COLORS.items()]
    ax.legend(handles=handles, title="Provider", loc="lower right", fontsize=8)
    
    _save(fig, out_dir, "fig3_consistency_score_bar.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Action by Stage facet grid 
# ═══════════════════════════════════════════════════════════════════════════

def plot_action_by_stage_model(valid_df: pd.DataFrame, out_dir: Path) -> None:
    """Line/scatter plots of % Rule-Breaking per stage, faceted by model."""
    # Compute % Rule-Breaking per model per stage
    grouped = valid_df.groupby(["model_key", "display_name", "params_B", "kohlberg_stage", "action_category"]).size().unstack(fill_value=0)
    if "Rule-Breaking" not in grouped: grouped["Rule-Breaking"] = 0
    if "Rule-Following" not in grouped: grouped["Rule-Following"] = 0
    grouped["total"] = grouped["Rule-Breaking"] + grouped["Rule-Following"]
    grouped["pct_breaking"] = (grouped["Rule-Breaking"] / grouped["total"]) * 100
    grouped = grouped.reset_index()
    
    models = grouped.drop_duplicates("model_key").sort_values("params_B")["display_name"].tolist()
    n_models = len(models)
    
    n_cols = 4
    n_rows = int(np.ceil(n_models / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(210 * MM, max(60, n_rows * 40) * MM), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Expected alignment line
    exp_x = [1, 2, 3, 4, 5, 6]
    exp_y = [0, 0, 0, 0, 100, 100]  # S1-4=RF (0% RB), S5-6=RB (100% RB)
    
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        sub = grouped[grouped["display_name"] == model_name].sort_values("kohlberg_stage")
        
        # Plot expected theoretical line
        ax.plot(exp_x, exp_y, color="#aaaaaa", ls="--", lw=1.0, zorder=0, label="Expected Alignment" if idx==0 else "")
        
        # Plot model actual mapping
        ax.plot(sub["kohlberg_stage"], sub["pct_breaking"], marker='o', color="#1a237e", lw=1.5, markersize=4, label="Actual Actions" if idx==0 else "")
        
        ax.set_title(model_name, fontsize=8, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.set_xticks(STAGES)
        if idx % n_cols == 0:
            ax.set_ylabel("% Rule-Breaking", fontsize=8)
            
    for ax in axes[n_models:]:
        ax.set_visible(False)
        
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    fig.suptitle("Percentage of Principled (Rule-Breaking) Actions by Reasoning Stage\n(Do models shift action preferences when reasoning post-conventionally at Stages 5–6?)", fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_action_by_stage_model.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Inconsistency network (Sankey-lite visualization)
# ═══════════════════════════════════════════════════════════════════════════

def plot_stage_action_sankey(ct: pd.DataFrame, out_dir: Path) -> None:
    """Sankey-like plot connecting stages to actions to visualize flow."""
    fig, ax = plt.subplots(figsize=(150 * MM, 110 * MM))
    
    stages = ct.index.tolist()[::-1] # 6 top, 1 bottom
    actions = ["Rule-Following", "Rule-Breaking"]
    
    stage_y = np.linspace(0.1, 0.9, len(stages))
    act_y = [0.7, 0.3]
    
    stage_x = 0.2
    act_x = 0.8
    
    # Draw points
    for i, _ in enumerate(stages):
        ax.plot(stage_x, stage_y[i], 'o', color="#444", markersize=8)
        ax.text(stage_x - 0.03, stage_y[i], f"Stage {stages[i]}", ha='right', va='center', fontsize=9, fontweight="bold")
        
    for j, act in enumerate(actions):
        c = ACTION_COLORS[act]
        ax.plot(act_x, act_y[j], 's', color=c, markersize=10)
        ax.text(act_x + 0.03, act_y[j], act, ha='left', va='center', fontsize=10, fontweight="bold", color=c)
        
    # Draw flows
    max_count = ct.values.max()
    for i, s in enumerate(stages):
        for j, act in enumerate(actions):
            count = ct.loc[s, act]
            if count > 0:
                is_expected = EXPECTED_ACTION_BY_STAGE.get(s) == act
                lw = (count / max_count) * 15
                alpha = 0.6 if is_expected else 0.8
                color = ACTION_COLORS[act] if is_expected else "#dd2222" # red for inconsistent flow
                ls = "-" if is_expected else "--"
                
                # Bezier curve approximation
                x_vals = np.linspace(stage_x, act_x, 100)
                y_vals = np.interp(x_vals, [stage_x, (stage_x+act_x)/2, act_x], [stage_y[i], stage_y[i], act_y[j]])
                # A smoother sigmoid curve
                t = (x_vals - stage_x) / (act_x - stage_x)
                smooth_y = stage_y[i] + (act_y[j] - stage_y[i]) * (3*t**2 - 2*t**3)
                
                ax.plot(x_vals, smooth_y, color=color, alpha=alpha, lw=lw, ls=ls, solid_capstyle="round")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    # Legend
    leg_handles = [
        mlines.Line2D([], [], color="#009E73", lw=4, label="Consistent (Rule-Following)"),
        mlines.Line2D([], [], color="#D55E00", lw=4, label="Consistent (Rule-Breaking)"),
        mlines.Line2D([], [], color="#dd2222", lw=3, ls="--", label="Inconsistent Path"),
    ]
    ax.legend(handles=leg_handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False, fontsize=8)
    
    ax.set_title("Flow from Moral Reasoning Stage to Endorsed Action\n(Line thickness ∝ number of responses)", fontsize=10, fontweight="bold")
    
    _save(fig, out_dir, "fig5_inconsistency_network.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 — 3D Stage-Action landscape
# ═══════════════════════════════════════════════════════════════════════════

def plot_3d_stage_action_landscape(ct: pd.DataFrame, out_dir: Path) -> None:
    """3D Bar chart: Stage vs Action vs Count."""
    from mpl_toolkits.mplot3d import Axes3D # noqa: F401
    
    fig = plt.figure(figsize=(160 * MM, 120 * MM))
    ax = fig.add_subplot(111, projection="3d")
    
    stages = ct.index.tolist()
    actions = ct.columns.tolist()
    
    X, Y = np.meshgrid(np.arange(len(actions)), np.arange(len(stages)))
    x_pos = X.flatten()
    y_pos = Y.flatten()
    z_pos = np.zeros_like(x_pos)
    
    counts = ct.values.flatten()
    
    dx = 0.5 * np.ones_like(z_pos)
    dy = 0.5 * np.ones_like(z_pos)
    dz = counts
    
    colors = []
    for i in range(len(y_pos)):
        s = stages[y_pos[i]]
        a = actions[x_pos[i]]
        is_exp = EXPECTED_ACTION_BY_STAGE.get(s) == a
        colors.append(STAGE_COLORS[s] if is_exp else "#999999") # grey if inconsistent
        
    ax.bar3d(x_pos - dx/2, y_pos - dy/2, z_pos, dx, dy, dz, color=colors, alpha=0.9, shade=True)
    
    ax.set_xticks(np.arange(len(actions)))
    ax.set_xticklabels(actions, fontsize=8)
    ax.set_yticks(np.arange(len(stages)))
    ax.set_yticklabels([f"S{s}" for s in stages], fontsize=8)
    
    ax.set_xlabel("Action Endorsed", labelpad=10, fontsize=9)
    ax.set_ylabel("Kohlberg Stage", labelpad=10, fontsize=9)
    ax.set_zlabel("Count", labelpad=5, fontsize=9)
    
    ax.view_init(elev=30, azim=-45)
    
    # Legend
    handles = [
        mpatches.Patch(color=c, label=f"S{s} Consistent") for s,c in STAGE_COLORS.items()
    ] + [mpatches.Patch(color="#999999", label="Inconsistent Mapping")]
    
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=7, title="Colour Key")
    
    ax.set_title("3D Distribution of Stages vs Endorsed Actions", fontsize=10, fontweight="bold", pad=15)
    
    _save(fig, out_dir, "fig6_3d_stage_action_landscape.png")

