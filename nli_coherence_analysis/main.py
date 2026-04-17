"""
main.py — Pipeline orchestrator for NLI-Based Coherence Analysis.

Stages:
  --score-only     : Run NLI scoring only (requires evaluation data)
  --analyze        : Run correlation + visualizations + report (requires scores)
  --use-main-data  : Use main project evaluation_data/ instead of RLHF data
  (default)        : Run full pipeline: score → correlate → visualize → report

Run from project root:
  python nli_coherence_analysis/main.py                          # full pipeline
  python nli_coherence_analysis/main.py --use-main-data          # use 13-model main data
  python nli_coherence_analysis/main.py --score-only             # scoring only
  python nli_coherence_analysis/main.py --analyze                # analysis only
  python nli_coherence_analysis/main.py --analyze --use-main-data
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nli_coherence_analysis"))

from config import OUT_DIR, SCORES_DIR
from data_loader import load_data, compute_model_consistency
from nli_scorer import score_dataframe, save_scores, load_scores
from correlation import build_model_nli_summary, run_all_correlations
from visualizations import generate_all_visualizations
from reporting import generate_report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


# ── Stage 1: NLI Scoring ─────────────────────────────────────────────────────

def stage_score(use_main_data: bool) -> tuple:
    """Load data, run NLI scoring, save results."""
    _banner("Stage 1: NLI Scoring with DeBERTa-v3-large")

    # Load evaluation data
    print("\n[1/3] Loading evaluation data…")
    df, data_source = load_data(use_main_data=use_main_data)
    print(f"  Data source: {data_source}")
    print(f"  Observations: {len(df):,}")

    # Check for required columns
    if "response" not in df.columns:
        print("  ⚠️  'response' column not found — checking for alternative…")
        # Try to load raw response data and merge
        # For now, we need the response column
        if "kohlberg_reasoning" in df.columns:
            print("  Using 'kohlberg_reasoning' as surrogate for response text")
            df["response"] = df["kohlberg_reasoning"].fillna("")
        else:
            print("  ❌ No response text available. Cannot score NLI.")
            sys.exit(1)

    # Score with NLI
    print("\n[2/3] Running NLI scoring…")
    scored_df = score_dataframe(df)

    # Save
    print("\n[3/3] Saving scores…")
    filename = f"nli_scores_{data_source}.xlsx"
    save_scores(scored_df, filename=filename)

    return scored_df, data_source


# ── Stage 2: Analysis ────────────────────────────────────────────────────────

def stage_analyze(scored_df=None, data_source=None, use_main_data: bool = False) -> None:
    """Run correlation analysis, generate visualizations, and produce report."""
    _banner("Stage 2: Correlation Analysis & Visualization")

    # Load scores if not provided
    if scored_df is None:
        src = "main" if use_main_data else "rlhf"
        filename = f"nli_scores_{src}.xlsx"
        scored_df = load_scores(filename=filename)

        if scored_df is None:
            # Try the other source
            other_src = "rlhf" if src == "main" else "main"
            filename = f"nli_scores_{other_src}.xlsx"
            scored_df = load_scores(filename=filename)
            if scored_df is not None:
                src = other_src

        if scored_df is None:
            print("  ❌ No NLI scores found. Run scoring first:")
            print("     python nli_coherence_analysis/main.py --score-only")
            sys.exit(1)

        data_source = src

    print(f"\n  Data source: {data_source}")
    print(f"  Scored observations: {len(scored_df):,}")

    # Build model-level summary
    print("\n[1/4] Building model-level NLI summary…")
    model_summary = build_model_nli_summary(scored_df)

    print("\n  Model Summary:")
    display_cols = ["display_name", "mean_nli_entailment", "consistency_pct", "n_scored"]
    available = [c for c in display_cols if c in model_summary.columns]
    print(model_summary[available].to_string(index=False))

    # Run correlations
    print("\n[2/4] Running correlation analyses…")
    corr_results = run_all_correlations(scored_df, model_summary, data_source)

    # Print key results
    pb = corr_results.get("pointbiserial", {})
    if "error" not in pb:
        sig = "✅" if pb.get("significant") else "❌"
        print(f"\n  • Point-biserial: r={pb.get('r', 0):.3f}  p={pb.get('p_value', 1):.4f} {sig}")

    ml = corr_results.get("model_level", {})
    if "error" not in ml:
        sig = "✅" if ml.get("pearson_sig") else "❌"
        print(f"  • Model-level Pearson: r={ml.get('pearson_r', 0):.3f}  p={ml.get('pearson_p', 1):.4f} {sig}")

    pc = corr_results.get("partial_corr", {})
    if "error" not in pc:
        sig = "✅" if pc.get("partial_sig") else "❌"
        print(f"  • Partial (|stage): r={pc.get('r_partial', 0):.3f}  p={pc.get('p_partial', 1):.4f} {sig}")

    rlhf = corr_results.get("rlhf_pairs", {})
    if "error" not in rlhf and not rlhf.get("skipped"):
        sig = "✅" if rlhf.get("significant") else "❌"
        print(f"  • RLHF Δ NLI: {rlhf.get('overall_delta', 0):+.3f}  p={rlhf.get('mann_whitney_p', 1):.4f} {sig}")

    # Visualizations
    print("\n[3/4] Generating visualizations…")
    generate_all_visualizations(scored_df, model_summary, data_source)

    # Report
    print("\n[4/4] Generating report…")
    generate_report(scored_df, model_summary, corr_results, data_source)

    _banner("NLI Coherence Analysis Complete")
    print(f"  Outputs → nli_coherence_analysis/results/")
    for f in sorted(OUT_DIR.glob("*")):
        if f.suffix in (".png", ".md"):
            print(f"    {f.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NLI-Based Coherence Measure: DeBERTa-v3-large entailment scoring"
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="Run NLI scoring only (no analysis)",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run analysis only (requires pre-computed scores)",
    )
    parser.add_argument(
        "--use-main-data", action="store_true",
        help="Use main project evaluation_data/ (13 models) instead of RLHF data",
    )
    args = parser.parse_args()

    _banner("NLI-Based Coherence Measure — DeBERTa-v3-large")

    if args.score_only:
        stage_score(args.use_main_data)
        print("\n✅ Scoring complete. Run with --analyze to generate correlations and figures.")
        return

    if args.analyze:
        stage_analyze(use_main_data=args.use_main_data)
        return

    # Default: full pipeline
    scored_df, data_source = stage_score(args.use_main_data)
    stage_analyze(scored_df, data_source, args.use_main_data)


if __name__ == "__main__":
    main()
