"""
reporting.py — Export results and print console summary for Analysis 7.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd

from config import OUT_DIR, POST_CONV_THRESHOLD


class NumpyEncoder(json.JSONEncoder):
    """Serialize numpy types to native Python for JSON output."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, (np.ndarray,)):   return obj.tolist()
        if isinstance(obj, (np.bool_,)):     return bool(obj)
        return super().default(obj)


def export_model_summary(model_df: pd.DataFrame) -> str:
    """Export per-model summary statistics to CSV."""
    cols = [
        "model_key", "display_name", "params_B", "provider",
        "n_obs", "mean_stage", "median_stage", "std_stage",
        "ci_lower", "ci_upper", "post_conv_pct", "emerged",
        "stage_1_pct", "stage_2_pct", "stage_3_pct",
        "stage_4_pct", "stage_5_pct", "stage_6_pct",
    ]
    out_cols = [c for c in cols if c in model_df.columns]
    out = OUT_DIR / "model_summary.csv"
    model_df[out_cols].to_csv(out, index=False, float_format="%.4f")
    print(f"  Saved: {out.name}")
    return str(out)


def export_emergence_metrics(analysis_results: dict, model_df: pd.DataFrame) -> str:
    """Export key emergence metrics to CSV."""
    seg    = analysis_results.get("segmented_regression", {})
    emrg   = analysis_results.get("emergence_threshold", {})
    corr   = analysis_results.get("cross_scale_correlation", {})
    cps    = analysis_results.get("changepoints", [])
    log_p  = model_df["log_params"].values

    records = [{
        "metric":             "changepoint_indices",
        "value":              str(cps),
        "unit":               "index",
        "description":        "Indices in model_df (sorted by params) where changepoints were detected",
    }, {
        "metric":             "changepoint_params_B",
        "value":              str([round(10 ** log_p[i], 1) for i in cps if 0 <= i < len(log_p)]),
        "unit":               "B parameters",
        "description":        "Approximate parameter count at each detected changepoint",
    }, {
        "metric":             "changepoint_ci_lower",
        "value":              round(analysis_results.get("changepoint_ci_lower", float("nan")), 2),
        "unit":               "index",
        "description":        "Bootstrap 95% CI lower bound on primary changepoint index",
    }, {
        "metric":             "changepoint_ci_upper",
        "value":              round(analysis_results.get("changepoint_ci_upper", float("nan")), 2),
        "unit":               "index",
        "description":        "Bootstrap 95% CI upper bound on primary changepoint index",
    }, {
        "metric":             "slope_pre_changepoint",
        "value":              round(seg.get("slope_pre", float("nan")), 5),
        "unit":               "Δ stage / Δ log10(params)",
        "description":        "Regression slope before primary changepoint",
    }, {
        "metric":             "slope_post_changepoint",
        "value":              round(seg.get("slope_post", float("nan")), 5),
        "unit":               "Δ stage / Δ log10(params)",
        "description":        "Regression slope after primary changepoint",
    }, {
        "metric":             "f_stat",
        "value":              round(seg.get("f_stat", float("nan")), 4),
        "unit":               "F",
        "description":        "F-statistic for segmented vs linear model comparison",
    }, {
        "metric":             "f_test_p_value",
        "value":              round(seg.get("p_value", float("nan")), 6),
        "unit":               "p",
        "description":        "p-value for F-test (segmented vs linear); <0.05 = segmented significantly better",
    }, {
        "metric":             "r2_linear",
        "value":              round(seg.get("r2_linear", float("nan")), 4),
        "unit":               "R²",
        "description":        "R² for simple linear regression",
    }, {
        "metric":             "r2_segmented",
        "value":              round(seg.get("r2_segmented", float("nan")), 4),
        "unit":               "R²",
        "description":        "R² for segmented (two-piece) regression",
    }, {
        "metric":             "emergence_params_B",
        "value":              emrg.get("emergence_params_B", float("nan")),
        "unit":               "B parameters",
        "description":        f"Smallest model where post_conv_pct >= {POST_CONV_THRESHOLD:.0%}",
    }, {
        "metric":             "emergence_model",
        "value":              emrg.get("emergence_model", "N/A"),
        "unit":               "",
        "description":        "Name of first model meeting emergence threshold",
    }, {
        "metric":             "log_emergence_params",
        "value":              round(analysis_results.get("log_emergence_params", float("nan")), 4),
        "unit":               "log10(B params)",
        "description":        "Log-scale efficiency metric for emergence",
    }, {
        "metric":             "effect_size_stage_range",
        "value":              round(analysis_results.get("effect_size", float("nan")), 4),
        "unit":               "stages",
        "description":        "Range of mean_stage across all models (max - min)",
    }, {
        "metric":             "spearman_rho",
        "value":              round(corr.get("spearman_rho", float("nan")), 4),
        "unit":               "ρ",
        "description":        "Spearman correlation: log(params) × mean_stage",
    }, {
        "metric":             "spearman_p_value",
        "value":              round(corr.get("spearman_pval", float("nan")), 6),
        "unit":               "p",
        "description":        "p-value for Spearman correlation",
    }, {
        "metric":             "scenario",
        "value":              analysis_results.get("scenario", ""),
        "unit":               "",
        "description":        "Classified emergence pattern",
    }]

    out = OUT_DIR / "emergence_metrics.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"  Saved: {out.name}")

    # Also save as JSON for downstream use
    out_json = OUT_DIR / "analysis_results.json"
    with open(out_json, "w") as f:
        json.dump(analysis_results, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {out_json.name}")

    return str(out)


def print_console_report(model_df: pd.DataFrame, analysis_results: dict) -> None:
    """Pretty-print the final summary to console."""
    seg    = analysis_results.get("segmented_regression", {})
    emrg   = analysis_results.get("emergence_threshold", {})
    corr   = analysis_results.get("cross_scale_correlation", {})
    cps    = analysis_results.get("changepoints", [])
    log_p  = model_df["log_params"].values

    print("\n" + "=" * 65)
    print("  ANALYSIS 7: EMERGENCE THRESHOLD DETECTION — FINAL REPORT")
    print("=" * 65)
    print(f"\n  Models analysed   : {len(model_df)}")
    print(f"  Scale range       : {model_df['params_B'].min():.0f}B → "
          f"{model_df['params_B'].max():.0f}B parameters")
    print(f"  Stage range       : {model_df['mean_stage'].min():.2f} → "
          f"{model_df['mean_stage'].max():.2f}")

    print(f"\n  ── Changepoint Detection ──")
    if cps:
        cp_params = [round(10 ** log_p[i], 1) for i in cps if i < len(log_p)]
        cp_params_clean = [float(round(10 ** log_p[i], 1)) for i in cps if i < len(log_p)]
        print(f"  Changepoints at   : {cp_params_clean} B parameters  (indices {list(cps)})")
        print(f"  Bootstrap 95% CI  : [{analysis_results.get('changepoint_ci_lower', 'n/a'):.1f}, "
              f"{analysis_results.get('changepoint_ci_upper', 'n/a'):.1f}] index units")
    else:
        print("  No significant changepoints detected.")

    print(f"\n  ── Segmented Regression ──")
    print(f"  Slope pre-CP      : {seg.get('slope_pre', float('nan')):+.5f}")
    print(f"  Slope post-CP     : {seg.get('slope_post', float('nan')):+.5f}")
    print(f"  F-test p-value    : {seg.get('p_value', float('nan')):.4f}  "
          f"({'significant' if seg.get('is_better') else 'not significant'})")
    print(f"  R² linear         : {seg.get('r2_linear', float('nan')):.4f}")
    print(f"  R² segmented      : {seg.get('r2_segmented', float('nan')):.4f}")

    print(f"\n  ── Emergence Threshold (≥{POST_CONV_THRESHOLD:.0%} Stage 5+) ──")
    if emrg.get("emergence_model"):
        print(f"  First emergence   : {emrg['emergence_model']} ({int(emrg['emergence_params_B'])}B)")
        print(f"  Post-conv %       : {emrg['emergence_pct']:.0%}")
    else:
        print("  Emergence threshold not met by any model.")

    print(f"\n  ── Cross-Scale Correlation ──")
    print(f"  Spearman ρ        : {corr.get('spearman_rho', float('nan')):.4f}")
    print(f"  p-value           : {corr.get('spearman_pval', float('nan')):.4f}")
    print(f"  Interpretation    : {corr.get('interpretation', '')}")

    print(f"\n  ── Scenario Classification ──")
    print(f"  {analysis_results.get('scenario', 'Unknown')}")
    print(f"  Effect size       : {analysis_results.get('effect_size', float('nan')):.2f} stages")

    print("\n" + "=" * 65)
