"""
main.py — Orchestration for Analysis 10: Stage Transition Dynamics.

Run from project root:
    python analysis10/main.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUT_DIR, ACTIVE_STAGES
from data_loader import load_raw_data, build_stage_distribution
from metrics import (
    compute_model_metrics,
    build_transition_matrix,
    compute_residence_times,
    detect_transition_windows,
    summarize_metrics,
    extract_qualitative_samples,
)
from stat_tests import run_all_tests
from visualizations import generate_all_visualizations
from reporting import generate_report


def main() -> None:
    print("=" * 70)
    print("  Analysis 10: Stage Transition Dynamics")
    print("  Research Q: How do models transition between stages as scale increases?")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading evaluation data & building distributions …")
    obs_df = load_raw_data()
    base_model_df = build_stage_distribution(obs_df)

    # ── 2. Compute metrics ────────────────────────────────────────────────────
    print("\n[2/6] Computing capability & transition metrics …")
    model_df = compute_model_metrics(base_model_df)
    
    T_norm, T_labels = build_transition_matrix(model_df)
    residence = compute_residence_times(model_df)
    windows = detect_transition_windows(model_df)
    metrics_summary = summarize_metrics(model_df)

    print(f"  Overall Consolidation Index:  {metrics_summary['consolidation_index']*100:.1f}%")
    print(f"  Detected Transition Windows:  {len(windows)}")

    # ── 3. Statistical tests ──────────────────────────────────────────────────
    print("\n[3/6] Running statistical tests …")
    stats_results = run_all_tests(obs_df, model_df, T_norm, ACTIVE_STAGES, windows)
    
    fri = stats_results.get("friedman", {})
    if "p_value" in fri:
        print(f"  • Friedman Test (Scale Progression): p={fri['p_value']:.4e} "
              f"[{'Sig' if fri.get('significant') else 'ns'}]")

    seq = stats_results.get("chisq_transitions", {})
    if "p_value" in seq:
        print(f"  • Chi-square (Transition Sequence): p={seq['p_value']:.4e} "
              f"[{'Sig' if seq.get('significant') else 'ns'}]")
        print(f"    Sequential transition proportion: {seq.get('sequential_proportion', 0)*100:.1f}%")

    rho = stats_results.get("spearman_scale", {})
    if "p_value" in rho:
        print(f"  • Spearman Rank (Scale vs Mean Stage): rho={rho.get('spearman_rho'):.3f}, "
              f"p={rho['p_value']:.4e} [{'Sig' if rho.get('significant') else 'ns'}]")

    # ── 4. Visualizations ─────────────────────────────────────────────────────
    print("\n[4/6] Generating visualizations …")
    generate_all_visualizations(model_df, residence, T_norm, T_labels)

    # ── 5. Reporting ──────────────────────────────────────────────────────────
    print("\n[5/6] Generating Markdown report …")
    qualitative_samples = extract_qualitative_samples(obs_df, windows, max_samples_per_window=1)
    generate_report(model_df, metrics_summary, stats_results, windows, qualitative_samples)

    # ── 6. Done ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  ✅  Analysis 10 complete. Outputs → {OUT_DIR.relative_to(Path.cwd())}")
    for f in sorted(OUT_DIR.glob("*")):
        if f.suffix in [".png", ".md"]:
            print(f"       {f.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
