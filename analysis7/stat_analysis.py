"""
stat_analysis.py — Statistical analysis for Analysis 7: Emergence Threshold Detection.

Methods:
  1. Changepoint detection  (PELT via ruptures, BIC criterion)
  2. Segmented regression   (F-test: two-segment vs linear)
  3. Slope analysis         (pre/post changepoint slopes + comparison)
  4. Binary emergence threshold (smallest params where post_conv_pct >= threshold)
  5. Bootstrap CIs on changepoint location
  6. Cross-scale Spearman correlation
  7. Scenario classification: Gradual / Sharp / Multi-stage
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from scipy.optimize import curve_fit

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    warnings.warn(
        "ruptures library not found. Install with: pip install ruptures\n"
        "Changepoint detection will use a fallback variance-based method.",
        stacklevel=2,
    )

from config import POST_CONV_THRESHOLD, BOOTSTRAP_ITERS, CI_LEVEL


# ─────────────────────────────────────────────────────────────────────────────
#  Changepoint detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_changepoints(
    signal: np.ndarray,
    n_bkps_max: int = 3,
    penalty: str = "bic",
) -> list[int]:
    """
    Detect changepoints in *signal* (1-D array ordered by model scale).

    Returns list of changepoint indices (positions in signal where a break occurs).
    Uses PELT with BIC if ruptures is available, otherwise falls back to a simple
    sliding-window variance method.
    """
    n = len(signal)
    if n < 4:
        return []

    if HAS_RUPTURES:
        # Reshape to (n, 1) for ruptures
        sig_2d = signal.reshape(-1, 1)
        algo   = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(sig_2d)
        try:
            # pen parameter controls sensitivity; use BIC-like automatic selection
            pen_val = np.log(n) * sig_2d.var()
            bkps    = algo.predict(pen=pen_val)
            # ruptures returns [bkp1, bkp2, …, n]; strip the terminal n
            bkps = [b for b in bkps if b < n]
        except Exception:
            bkps = []
        return bkps
    else:
        # Fallback: find the index that maximises the between-segment variance
        best_i, best_v = 1, -1.0
        for i in range(2, n - 1):
            left  = signal[:i]
            right = signal[i:]
            between = (left.mean() - right.mean()) ** 2
            if between > best_v:
                best_v, best_i = between, i
        return [best_i] if best_v > 0 else []


def bootstrap_changepoint_ci(
    signal: np.ndarray,
    n_iter: int = BOOTSTRAP_ITERS,
    ci_level: float = CI_LEVEL,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI on the primary (first) changepoint location index.

    Returns (ci_lower_idx, ci_upper_idx) as fractional indices.
    """
    n = len(signal)
    if n < 4:
        return (np.nan, np.nan)

    bkp_samples: list[int] = []
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx       = rng.integers(0, n, size=n)
        boot_sig  = signal[idx]
        boot_sig  = np.sort(boot_sig)   # maintain ordered structure
        bkps      = detect_changepoints(boot_sig)
        if bkps:
            bkp_samples.append(bkps[0])

    if not bkp_samples:
        return (np.nan, np.nan)

    alpha = 1 - ci_level
    lo = np.percentile(bkp_samples, 100 * alpha / 2)
    hi = np.percentile(bkp_samples, 100 * (1 - alpha / 2))
    return (lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
#  Segmented regression (two-segment vs linear F-test)
# ─────────────────────────────────────────────────────────────────────────────

def _piecewise_linear(x: np.ndarray, x_brk: float, a1: float, b1: float, a2: float) -> np.ndarray:
    """Piecewise linear (two-segment) model with shared junction at x_brk."""
    y = np.where(x <= x_brk,
                 a1 * x + b1,
                 a1 * x_brk + b1 + a2 * (x - x_brk))
    return y


def segmented_regression(
    x: np.ndarray,
    y: np.ndarray,
    breakpoint_idx: int,
) -> dict:
    """
    Fit a two-segment linear model at *breakpoint_idx* and compare to a
    simple linear model via F-test.

    Returns dict with:
      slope_pre, slope_post, intercept_pre, intercept_post,
      f_stat, p_value, r2_linear, r2_segmented,
      is_better (True if segmented significantly better, p < 0.05)
    """
    n = len(x)
    if n < 6 or breakpoint_idx < 2 or breakpoint_idx > n - 2:
        return {k: np.nan for k in [
            "slope_pre", "slope_post", "intercept_pre", "f_stat", "p_value",
            "r2_linear", "r2_segmented", "is_better"]}

    # --- Linear model
    slope_lin, intercept_lin, r_lin, *_ = sp_stats.linregress(x, y)
    y_hat_lin = slope_lin * x + intercept_lin
    ss_res_lin = np.sum((y - y_hat_lin) ** 2)
    ss_tot     = np.sum((y - y.mean()) ** 2)
    r2_lin     = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0.0

    # --- Two segments independently
    x1, y1 = x[:breakpoint_idx], y[:breakpoint_idx]
    x2, y2 = x[breakpoint_idx:], y[breakpoint_idx:]

    if len(x1) < 2 or len(x2) < 2:
        return {k: np.nan for k in [
            "slope_pre", "slope_post", "intercept_pre", "f_stat", "p_value",
            "r2_linear", "r2_segmented", "is_better"]}

    s1, i1, *_ = sp_stats.linregress(x1, y1)
    s2, i2, *_ = sp_stats.linregress(x2, y2)
    y_hat_seg  = np.concatenate([s1 * x1 + i1, s2 * x2 + i2])
    ss_res_seg = np.sum((y - y_hat_seg) ** 2)
    r2_seg     = 1 - ss_res_seg / ss_tot if ss_tot > 0 else 0.0

    # F-test: does the segmented model significantly reduce residual SS?
    # df1 = extra params (2), df2 = n - 4 (4 params in segmented model)
    df1 = 2
    df2 = max(n - 4, 1)
    if ss_res_seg == 0:
        f_stat, p_val = np.inf, 0.0
    else:
        f_stat = ((ss_res_lin - ss_res_seg) / df1) / (ss_res_seg / df2)
        p_val  = 1 - sp_stats.f.cdf(f_stat, df1, df2)

    return {
        "slope_pre":      s1,
        "slope_post":     s2,
        "intercept_pre":  i1,
        "intercept_post": i2,
        "f_stat":         f_stat,
        "p_value":        p_val,
        "r2_linear":      r2_lin,
        "r2_segmented":   r2_seg,
        "is_better":      bool(p_val < 0.05),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Emergence threshold
# ─────────────────────────────────────────────────────────────────────────────

def find_emergence_threshold(model_df: pd.DataFrame) -> dict:
    """
    Identify the first (smallest params_B) model where post_conv_pct >= threshold.

    Returns dict with emergence info, or NaN fields if threshold never reached.
    """
    emerged = model_df[model_df["post_conv_pct"] >= POST_CONV_THRESHOLD].sort_values("params_B")
    if emerged.empty:
        return {
            "emergence_params_B": np.nan,
            "emergence_model":    None,
            "emergence_pct":      np.nan,
            "threshold":          POST_CONV_THRESHOLD,
        }
    first = emerged.iloc[0]
    return {
        "emergence_params_B": first["params_B"],
        "emergence_model":    first["display_name"],
        "emergence_pct":      first["post_conv_pct"],
        "threshold":          POST_CONV_THRESHOLD,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Scenario classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_emergence_scenario(
    changepoints: list[int],
    seg_result: dict,
    model_df: pd.DataFrame,
) -> str:
    """
    Classify the emergence pattern as:
      Scenario A — Gradual (no clear changepoints, smooth monotonic)
      Scenario B — Sharp   (single changepoint, large slope change)
      Scenario C — Multi-stage (multiple changepoints, stepwise)
    """
    n_bkps = len(changepoints)

    if n_bkps == 0 or not seg_result.get("is_better", False):
        return "Scenario A — Gradual Emergence"

    if n_bkps >= 2:
        return "Scenario C — Multi-Stage Emergence"

    # Single changepoint: check if slope change is large
    slope_pre  = seg_result.get("slope_pre", 0)
    slope_post = seg_result.get("slope_post", 0)
    slope_jump = abs(slope_post - slope_pre)

    if slope_jump > 0.3:
        return "Scenario B — Sharp Emergence (Phase Transition)"
    return "Scenario A — Gradual Emergence"


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-scale correlation
# ─────────────────────────────────────────────────────────────────────────────

def cross_scale_correlation(model_df: pd.DataFrame) -> dict:
    """
    Spearman correlation between log(params) and mean_stage.
    Returns dict with rho, p_value, interpretation.
    """
    x = model_df["log_params"].values
    y = model_df["mean_stage"].values
    rho, pval = sp_stats.spearmanr(x, y)
    return {
        "spearman_rho":   rho,
        "spearman_pval":  pval,
        "significant":    bool(pval < 0.05),
        "interpretation": (
            "Larger models tend toward higher moral reasoning stages."
            if rho > 0 else
            "No clear positive correlation between model size and stage."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Master analysis runner
# ─────────────────────────────────────────────────────────────────────────────

def run_full_analysis(model_df: pd.DataFrame) -> dict:
    """
    Orchestrate all statistical analyses and return a results bundle dict.
    """
    signal        = model_df["mean_stage"].values
    log_params    = model_df["log_params"].values

    # 1. Changepoint detection
    changepoints  = detect_changepoints(signal)
    print(f"  Changepoints detected at indices: {changepoints}")

    # 2. Bootstrap CI on first changepoint
    bkp_ci_lo, bkp_ci_hi = bootstrap_changepoint_ci(signal)

    # 3. Segmented regression at primary changepoint
    bkp_idx = changepoints[0] if changepoints else len(signal) // 2
    seg_res = segmented_regression(log_params, signal, bkp_idx)

    # 4. Emergence threshold
    emrg = find_emergence_threshold(model_df)

    # 5. Scenario classification
    scenario = classify_emergence_scenario(changepoints, seg_res, model_df)

    # 6. Cross-scale correlation
    corr = cross_scale_correlation(model_df)

    # 7. Effect size: magnitude of stage jump across full scale range
    effect_size = signal.max() - signal.min()

    # 8. Efficiency metric: params per emergence point (log scale)
    efficiency = (
        np.log10(emrg["emergence_params_B"]) if not np.isnan(emrg["emergence_params_B"]) else np.nan
    )

    results = {
        "changepoints":          changepoints,
        "changepoint_ci_lower":  bkp_ci_lo,
        "changepoint_ci_upper":  bkp_ci_hi,
        "primary_changepoint_idx": bkp_idx,
        "segmented_regression":  seg_res,
        "emergence_threshold":   emrg,
        "scenario":              scenario,
        "cross_scale_correlation": corr,
        "effect_size":           effect_size,
        "log_emergence_params":  efficiency,
        "n_models":              len(model_df),
        "params_range":          (model_df["params_B"].min(), model_df["params_B"].max()),
        "stage_range":           (signal.min(), signal.max()),
    }

    print(f"\n{'='*60}")
    print(f"  SCENARIO:       {scenario}")
    print(f"  Effect size     (stage range): {effect_size:.2f}")
    print(f"  Spearman ρ:     {corr['spearman_rho']:.3f}  (p={corr['spearman_pval']:.4f})")
    print(f"  Slope pre:      {seg_res.get('slope_pre', np.nan):.3f}")
    print(f"  Slope post:     {seg_res.get('slope_post', np.nan):.3f}")
    print(f"  F-test p-val:   {seg_res.get('p_value', np.nan):.4f}")
    print(f"  Emergence at:   {emrg['emergence_model']} ({emrg['emergence_params_B']}B)")
    print(f"{'='*60}\n")

    return results
