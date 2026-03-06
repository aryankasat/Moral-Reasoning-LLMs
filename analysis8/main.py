"""
main.py — Orchestration for Analysis 8: Scale vs. Training Decomposition.

Run:
    python analysis8/main.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUT_DIR
from data_loader import load_raw_data, build_cell_summary, build_model_summary
from stat_analysis import (
    check_assumptions,
    run_two_way_anova,
    compute_effect_sizes,
    variance_partition,
    run_posthoc,
    cohens_d_scale,
    classify_hypothesis,
)
from visualizations import (
    plot_interaction,
    plot_bar_with_jitter,
    plot_box_violin,
    plot_posthoc_matrix,
    plot_variance_bars,
    plot_summary_panel,
)
from reporting import generate_report


def main() -> None:
    print("=" * 70)
    print("  Analysis 8: Scale vs. Training Decomposition")
    print("  Research Q: Does scale or training drive moral reasoning stage?")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading evaluation data …")
    raw_df   = load_raw_data()
    cell_df  = build_cell_summary(raw_df)
    model_df = build_model_summary(raw_df)

    print("\n  Cell-level means:")
    print(cell_df[["scale_group", "training_type", "n_obs", "n_models",
                    "mean_stage", "ci_lower", "ci_upper"]].to_string(index=False))

    # ── 2. Assumption checks ──────────────────────────────────────────────────
    print("\n[2/6] Checking ANOVA assumptions …")
    assumptions = check_assumptions(raw_df)

    print("  Shapiro-Wilk results:")
    for cell, W, p, status in assumptions["shapiro_results"]:
        W_s = f"{W:.4f}" if W is not None else "  —  "
        p_s = f"{p:.4f}" if p is not None else "  —  "
        print(f"    {cell:30s}  W={W_s}  p={p_s}  [{status}]")

    print(f"\n  Levene:  stat={assumptions.get('levene_stat')}  "
          f"p={assumptions.get('levene_p')}")
    print(f"  Kruskal-Wallis (Scale):    H={assumptions['kruskal_scale']['H']}  "
          f"p={assumptions['kruskal_scale']['p']}")
    print(f"  Kruskal-Wallis (Training): H={assumptions['kruskal_training']['H']}  "
          f"p={assumptions['kruskal_training']['p']}")
    print(f"  Welch ANOVA (Scale):    F={assumptions['welch_scale'].get('F')}  "
          f"p={assumptions['welch_scale'].get('p')}")
    print(f"  Welch ANOVA (Training): F={assumptions['welch_training'].get('F')}  "
          f"p={assumptions['welch_training'].get('p')}")

    # ── 3. Two-way ANOVA ──────────────────────────────────────────────────────
    print("\n[3/6] Running two-way factorial ANOVA …")
    anova_table, model = run_two_way_anova(raw_df)

    print("\n  ANOVA table:")
    print(anova_table.to_string())

    effect_df  = compute_effect_sizes(anova_table)
    partition  = variance_partition(anova_table, effect_df)
    hypothesis = classify_hypothesis(effect_df)
    cd_df      = cohens_d_scale(raw_df)

    r_squared   = model.rsquared
    residual_se = model.mse_resid ** 0.5

    print("\n  Effect sizes:")
    print(effect_df[["effect", "F", "p_value", "partial_eta2", "omega2",
                       "magnitude", "significant"]].to_string(index=False))
    print("\n  Variance partitioning:")
    for source, pct in partition.items():
        print(f"    {source:35s}  {pct:5.1f}%")
    print(f"\n  R² = {r_squared:.4f}   Residual SE = {residual_se:.4f}")
    print(f"\n  Cohen's d (Scale pairwise):")
    print(cd_df.to_string(index=False))
    print(f"\n  Supported hypothesis: {hypothesis}")

    # ── 4. Post-hoc tests ─────────────────────────────────────────────────────
    print("\n[4/6] Running post-hoc comparisons …")
    posthoc = run_posthoc(raw_df)
    for title, ph_df in posthoc.items():
        print(f"\n  ─ {title} ─")
        print(ph_df.to_string(index=False))

    # ── 5. Visualizations ─────────────────────────────────────────────────────
    print("\n[5/6] Generating figures …")
    plot_interaction(cell_df, posthoc)
    plot_bar_with_jitter(raw_df, cell_df)
    plot_box_violin(raw_df)
    plot_posthoc_matrix(posthoc)
    plot_variance_bars(partition, effect_df)
    plot_summary_panel(raw_df, cell_df, effect_df, partition, posthoc)

    # ── 6. Report ─────────────────────────────────────────────────────────────
    print("\n[6/6] Writing Markdown report …")
    generate_report(
        assumptions  = assumptions,
        anova_table  = anova_table,
        effect_df    = effect_df,
        partition    = partition,
        posthoc      = posthoc,
        hypothesis   = hypothesis,
        model_df     = model_df,
        r_squared    = r_squared,
        residual_se  = residual_se,
    )

    print("\n" + "=" * 70)
    print(f"  ✅ Analysis complete. Outputs in: {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.png")) + sorted(OUT_DIR.glob("*.md")):
        print(f"     {f.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
