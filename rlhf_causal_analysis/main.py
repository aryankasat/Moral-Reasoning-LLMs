"""
main.py — Orchestration for Analysis 11: RLHF Causal Analysis.

Pipeline stages (can be run independently via flags):
  --collect   : Run data_collector.py to collect LLM responses
  --evaluate  : Run evaluator.py to score responses with Kohlberg template
  --analyze   : Load scored data → metrics → stats → visualizations → report
  --all       : Run all three stages end-to-end (default)
  --dry-run   : Test API connectivity without making LLM calls

Run from project root:
  python rlhf_causal_analysis/main.py                   # full pipeline
  python rlhf_causal_analysis/main.py --analyze          # analysis only (data already collected)
  python rlhf_causal_analysis/main.py --dry-run          # connectivity test
"""

from __future__ import annotations

import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rlhf_causal_analysis"))

from config import OUT_DIR, PAIR_ORDER, MODEL_PAIRS
from data_loader import load_scored_data, build_pair_distributions
from metrics import compute_pair_metrics, cross_pair_consistency, build_summary_table
from stat_tests import run_all_tests
from visualizations import generate_all_visualizations
from reporting import generate_report


# ── Helpers ────────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


def _run_stage(script_args: list[str]) -> None:
    """Invoke a sub-script via subprocess (preserves its own arg parsing)."""
    cmd = [sys.executable] + script_args
    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n  ❌ Sub-script exited with code {result.returncode}")
        sys.exit(result.returncode)


# ── Stages ─────────────────────────────────────────────────────────────────────

def stage_collect(pair: str | None, dry_run: bool) -> None:
    _banner("Stage 1: Data Collection")
    args = [str(ROOT / "rlhf_causal_analysis" / "data_collector.py")]
    if pair:
        args += ["--pair", pair]
    if dry_run:
        args += ["--dry-run"]
    _run_stage(args)


def stage_evaluate(pair: str | None, use_groq_evaluator: bool) -> None:
    _banner("Stage 2: Kohlberg Evaluation")
    args = [str(ROOT / "rlhf_causal_analysis" / "evaluator.py")]
    if pair:
        args += ["--pair", pair]
    if use_groq_evaluator:
        args += ["--use-groq-evaluator"]
    _run_stage(args)


def stage_analyze() -> None:
    _banner("Stage 3: Metrics, Statistics & Visualizations")

    # 3a. Load data
    print("\n[1/5] Loading scored evaluation data…")
    obs_df  = load_scored_data()
    dist_df = build_pair_distributions(obs_df)

    # 3b. Compute pair-level metrics
    print("\n[2/5] Computing pair metrics (KL divergence, Cohen's d, bootstrap CI)…")
    pair_metrics = compute_pair_metrics(obs_df, dist_df)
    summary_tbl  = build_summary_table(pair_metrics)

    print("\n  Summary Table:")
    print(summary_tbl.to_string(index=False))

    consistency = cross_pair_consistency(pair_metrics)
    print(f"\n  Cross-pair consistency: {consistency['interpretation']}")

    # 3c. Statistical tests
    print("\n[3/5] Running statistical tests…")
    stats_results = run_all_tests(obs_df, pair_metrics)

    mw = stats_results.get("mann_whitney_per_pair", {})
    for pair_id in PAIR_ORDER:
        arch  = MODEL_PAIRS[pair_id]["architecture"]
        mw_r  = mw.get(pair_id, {})
        if "error" not in mw_r:
            sig = "✅" if mw_r.get("significant") else "❌"
            print(f"  • Mann-Whitney [{arch}]: U={mw_r.get('statistic',0):.1f}  "
                  f"p={mw_r.get('p_value',1):.4f} {sig}  "
                  f"d={pair_metrics[pair_metrics['pair_id']==pair_id]['cohens_d'].values[0]:.2f}")

    pt = stats_results.get("paired_ttest", {})
    if "p_value_onetail" in pt:
        sig = "✅" if pt.get("significant") else "❌"
        print(f"\n  • Paired t-test (cross-pair): Δ={pt.get('mean_delta',0):+.3f}  "
              f"p={pt.get('p_value_onetail',1):.4f} {sig}")

    sign = stats_results.get("sign_test", {})
    print(f"  • Sign test: {sign.get('n_positive','?')}/{sign.get('n_pairs','?')} positive "
          f"  p={sign.get('p_value',1):.4f}")

    # 3d. Visualizations
    print("\n[4/5] Generating visualizations…")
    generate_all_visualizations(obs_df, dist_df, pair_metrics)

    # 3e. Report
    print("\n[5/5] Generating report…")
    generate_report(pair_metrics, stats_results, obs_df, dist_df)

    _banner("Analysis 11 Complete")
    print(f"  Outputs → rlhf_causal_analysis/results/")
    for f in sorted(OUT_DIR.glob("*")):
        if f.suffix in (".png", ".md"):
            print(f"    {f.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analysis 11: RLHF Causal Analysis"
    )
    parser.add_argument("--collect",  action="store_true", help="Run data collection only")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation only")
    parser.add_argument("--analyze",  action="store_true", help="Run analysis only")
    parser.add_argument("--all",      action="store_true", help="Run full pipeline (default)")
    parser.add_argument("--pair",     default=None, help="Restrict collect/evaluate to one pair_id")
    parser.add_argument("--dry-run",  action="store_true", help="Test connectivity without API calls")
    parser.add_argument("--use-groq-evaluator", action="store_true",
                        help="Use Groq Llama as Kohlberg judge instead of Puter GPT")
    args = parser.parse_args()

    # Default to --all if no stage flag given
    run_all = args.all or not (args.collect or args.evaluate or args.analyze)

    _banner("Analysis 11: RLHF as Causal Driver of Moral Stage Distribution Shift")
    print(f"  Pairs: {', '.join(PAIR_ORDER)}")
    print(f"  Output: {OUT_DIR.relative_to(ROOT)}")

    if args.dry_run:
        stage_collect(args.pair, dry_run=True)
        print("\n✅ Dry-run complete — all imports and API keys validated.")
        return

    if run_all or args.collect:
        stage_collect(args.pair, dry_run=False)

    if run_all or args.evaluate:
        stage_evaluate(args.pair, args.use_groq_evaluator)

    if run_all or args.analyze:
        stage_analyze()


if __name__ == "__main__":
    main()
