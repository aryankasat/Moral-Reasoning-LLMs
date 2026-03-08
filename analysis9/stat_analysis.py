"""
stat_analysis.py — Statistical analyses for Analysis 9: Capability Correlation.

Implements:
  1. Correlation matrix (Pearson + Spearman) with bootstrap 95% CI and
     Bonferroni/FDR correction.
  2. Threshold detection via logistic regression (binary: capable of Stage 5+)
     and linear regression (continuous: % Stage 5+).
  3. Multi-capability regression with standardised coefficients.
  4. Partial correlations controlling for model scale (log_params).
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from config import (
    CAPABILITY_COLS, N_BOOTSTRAP, CI_LEVEL, ALPHA,
    POST_CONV_THRESH,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Correlation matrix
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson",
                    n: int = N_BOOTSTRAP, level: float = CI_LEVEL) -> tuple[float, float]:
    """Bootstrap CI for Pearson or Spearman r."""
    rng = np.random.default_rng(42)
    boot_rs = []
    idx = np.arange(len(x))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        xs, ys = x[s], y[s]
        if xs.std() == 0 or ys.std() == 0:
            continue
        if method == "pearson":
            r, _ = stats.pearsonr(xs, ys)
        else:
            r, _ = stats.spearmanr(xs, ys)
        boot_rs.append(r)
    alpha_half = (1 - level) / 2
    if not boot_rs:
        return np.nan, np.nan
    return float(np.percentile(boot_rs, 100 * alpha_half)), \
           float(np.percentile(boot_rs, 100 * (1 - alpha_half)))


def compute_correlation_matrix(model_df: pd.DataFrame) -> dict:
    """
    Compute pairwise Pearson and Spearman correlations among
    [capability cols, mean_stage, post_conv_pct].

    Returns dict with keys:
      pearson_r, pearson_p, spearman_r, spearman_p,
      pearson_ci_lo, pearson_ci_hi, spearman_ci_lo, spearman_ci_hi,
      corrected_p_pearson, corrected_p_spearman, variables
    """
    vars_of_interest = CAPABILITY_COLS + ["mean_stage", "post_conv_pct", "log_params"]
    # Only keep columns that actually exist in model_df
    vars_of_interest = [v for v in vars_of_interest if v in model_df.columns]

    n_vars = len(vars_of_interest)
    data   = model_df[vars_of_interest].dropna()

    pearson_r   = pd.DataFrame(np.eye(n_vars), index=vars_of_interest, columns=vars_of_interest)
    pearson_p   = pd.DataFrame(np.zeros((n_vars, n_vars)), index=vars_of_interest, columns=vars_of_interest)
    spearman_r  = pd.DataFrame(np.eye(n_vars), index=vars_of_interest, columns=vars_of_interest)
    spearman_p  = pd.DataFrame(np.zeros((n_vars, n_vars)), index=vars_of_interest, columns=vars_of_interest)
    pearson_cilo  = pd.DataFrame(np.eye(n_vars), index=vars_of_interest, columns=vars_of_interest)
    pearson_cihi  = pd.DataFrame(np.eye(n_vars), index=vars_of_interest, columns=vars_of_interest)
    spearman_cilo = pd.DataFrame(np.eye(n_vars), index=vars_of_interest, columns=vars_of_interest)
    spearman_cihi = pd.DataFrame(np.eye(n_vars), index=vars_of_interest, columns=vars_of_interest)

    pair_p_pearson  = []
    pair_p_spearman = []
    pair_idx        = []

    for v1, v2 in combinations(vars_of_interest, 2):
        x = data[v1].values.astype(float)
        y = data[v2].values.astype(float)
        if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
            continue

        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)

        pearson_r.loc[v1, v2]  = pearson_r.loc[v2, v1]  = pr
        pearson_p.loc[v1, v2]  = pearson_p.loc[v2, v1]  = pp
        spearman_r.loc[v1, v2] = spearman_r.loc[v2, v1] = sr
        spearman_p.loc[v1, v2] = spearman_p.loc[v2, v1] = sp

        pci_lo, pci_hi = _bootstrap_corr(x, y, "pearson")
        sci_lo, sci_hi = _bootstrap_corr(x, y, "spearman")

        pearson_cilo.loc[v1, v2]  = pearson_cilo.loc[v2, v1]  = pci_lo
        pearson_cihi.loc[v1, v2]  = pearson_cihi.loc[v2, v1]  = pci_hi
        spearman_cilo.loc[v1, v2] = spearman_cilo.loc[v2, v1] = sci_lo
        spearman_cihi.loc[v1, v2] = spearman_cihi.loc[v2, v1] = sci_hi

        pair_p_pearson.append(pp)
        pair_p_spearman.append(sp)
        pair_idx.append((v1, v2))

    # Multiple comparison correction (FDR, Benjamini-Hochberg)
    if pair_p_pearson:
        _, corrected_pp, _, _ = multipletests(pair_p_pearson, method="fdr_bh")
        _, corrected_sp, _, _ = multipletests(pair_p_spearman, method="fdr_bh")
    else:
        corrected_pp = np.array([])
        corrected_sp = np.array([])

    corrected_pearson  = pd.DataFrame(np.zeros((n_vars, n_vars)),
                                      index=vars_of_interest, columns=vars_of_interest)
    corrected_spearman = pd.DataFrame(np.zeros((n_vars, n_vars)),
                                      index=vars_of_interest, columns=vars_of_interest)
    for (v1, v2), cp, cs in zip(pair_idx, corrected_pp, corrected_sp):
        corrected_pearson.loc[v1, v2]  = corrected_pearson.loc[v2, v1]  = cp
        corrected_spearman.loc[v1, v2] = corrected_spearman.loc[v2, v1] = cs

    return {
        "variables":          vars_of_interest,
        "n_models":           len(data),
        "pearson_r":          pearson_r,
        "pearson_p":          pearson_p,
        "spearman_r":         spearman_r,
        "spearman_p":         spearman_p,
        "pearson_ci_lo":      pearson_cilo,
        "pearson_ci_hi":      pearson_cihi,
        "spearman_ci_lo":     spearman_cilo,
        "spearman_ci_hi":     spearman_cihi,
        "corrected_p_pearson":  corrected_pearson,
        "corrected_p_spearman": corrected_spearman,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Threshold detection
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    return L / (1 + np.exp(-k * (x - x0)))


def threshold_detection(model_df: pd.DataFrame) -> dict:
    """
    For each capability metric, assess its ability to predict moral sophistication.

    Since all 13/13 models are post-conventionally capable (≥20% Stage 5+),
    the binary "capable" target is constant. We therefore use:
      - Continuous target: mean_stage
      - AUC on high/low mean stage (split at median) as capability proxy
      - Linear regression: mean_stage ~ metric
      - Sigmoid fit on mean_stage as a function of the metric
    """
    results = {}

    # Use mean_stage as continuous outcome (has real variance)
    target_cont = model_df["mean_stage"].values
    # Binary proxy: above-median mean_stage
    median_stage = np.nanmedian(target_cont)
    target_binary = (target_cont > median_stage).astype(float)

    for metric in CAPABILITY_COLS + ["log_params"]:
        if metric not in model_df.columns:
            continue

        x_raw = model_df[metric].values.astype(float)
        valid  = ~(np.isnan(x_raw) | np.isnan(target_cont))
        if valid.sum() < 4:
            results[metric] = {"error": "insufficient data"}
            continue

        x  = x_raw[valid]
        yc = target_cont[valid]         # continuous: mean_stage
        yb = target_binary[valid]       # binary: above-median stage

        if np.nanstd(x) == 0:
            results[metric] = {"error": "zero variance in metric"}
            continue

        # ── Logistic regression for AUC (high vs low stage predictor) ──
        try:
            if len(np.unique(yb)) > 1:
                scaler = StandardScaler()
                x_sc   = scaler.fit_transform(x.reshape(-1, 1))
                clf    = LogisticRegression(max_iter=500)
                clf.fit(x_sc, yb)
                x_pred  = clf.predict_proba(x_sc)[:, 1]
                auc_val = float(roc_auc_score(yb, x_pred))
                coef    = float(clf.coef_[0][0])
                intercept_lg = float(clf.intercept_[0])
                if abs(coef) > 1e-8:
                    thresh_scaled = -intercept_lg / coef
                    thresh_metric = float(scaler.inverse_transform([[thresh_scaled]])[0][0])
                else:
                    thresh_metric = np.nan
            else:
                auc_val     = np.nan
                thresh_metric = np.nan
        except Exception:
            thresh_metric = np.nan
            auc_val       = np.nan

        # ── Linear regression: mean_stage ~ metric ──
        try:
            lm   = LinearRegression()
            lm.fit(x.reshape(-1, 1), yc)
            slope, intercept_l = float(lm.coef_[0]), float(lm.intercept_)
            # Threshold where mean_stage = POST_CONV_STAGE (5.0)
            target_val = 5.0
            if abs(slope) > 1e-8:
                linear_thresh = (target_val - intercept_l) / slope
            else:
                linear_thresh = np.nan
            r2 = float(lm.score(x.reshape(-1, 1), yc))
        except Exception:
            slope, intercept_l, linear_thresh, r2 = np.nan, np.nan, np.nan, np.nan

        # ── Sigmoid fit on mean_stage ──
        try:
            p0 = [float(np.max(yc)), 0.1, float(np.median(x))]
            popt, _ = curve_fit(_sigmoid, x, yc, p0=p0, maxfev=8000)
            x_fine   = np.linspace(x.min(), x.max(), 300)
            y_fine   = _sigmoid(x_fine, *popt)
            sigmoid_thresh = float(popt[2])   # inflection point
        except Exception:
            popt, x_fine, y_fine, sigmoid_thresh = None, None, None, np.nan

        results[metric] = {
            "logistic_50pct_threshold": thresh_metric,
            "logistic_auc":             auc_val,
            "linear_threshold_stage5":  float(linear_thresh) if not (isinstance(linear_thresh, float) and np.isnan(linear_thresh)) else np.nan,
            "linear_r2":                r2,
            "sigmoid_inflection":       sigmoid_thresh,
            "sigmoid_params":           popt,
            "x_for_plot":               x,
            "y_cont_for_plot":          yc,
            "y_binary_for_plot":        yb,
            "x_fine":                   x_fine,
            "y_fine":                   y_fine,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-capability regression
# ─────────────────────────────────────────────────────────────────────────────

def multi_capability_regression(model_df: pd.DataFrame) -> dict:
    """
    OLS regression: mean_stage ~ log_params + capability_cols
    Returns standardised coefficients with 95% CI and significance.
    """
    predictors = [c for c in CAPABILITY_COLS + ["log_params"] if c in model_df.columns]
    target      = "mean_stage"

    data = model_df[[target] + predictors].dropna()
    if len(data) < max(5, len(predictors) + 2):
        return {"error": "insufficient data for regression"}

    y = data[target].values

    # Standardise predictors for comparable coefficients
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(data[predictors].values)

    # OLS via statsmodels for CIs and p-values
    X_with_const = sm.add_constant(X_sc)
    model   = sm.OLS(y, X_with_const).fit()

    rows = []
    for i, pred in enumerate(predictors):
        idx = i + 1   # 0 is intercept
        rows.append({
            "predictor":    pred,
            "std_coef":     float(model.params[idx]),
            "ci_lo":        float(model.conf_int()[idx][0]),
            "ci_hi":        float(model.conf_int()[idx][1]),
            "t_stat":       float(model.tvalues[idx]),
            "p_value":      float(model.pvalues[idx]),
            "significant":  bool(model.pvalues[idx] < ALPHA),
        })

    coef_df = pd.DataFrame(rows).sort_values("std_coef", key=abs, ascending=False,
                                             ignore_index=True)

    return {
        "coef_df":      coef_df,
        "r_squared":    float(model.rsquared),
        "adj_r2":       float(model.rsquared_adj),
        "aic":          float(model.aic),
        "bic":          float(model.bic),
        "f_stat":       float(model.fvalue),
        "f_p":          float(model.f_pvalue),
        "n":            len(data),
        "model_summary": model.summary().as_text(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Partial correlations
# ─────────────────────────────────────────────────────────────────────────────

def partial_correlation(x: np.ndarray, y: np.ndarray,
                        z: np.ndarray) -> tuple[float, float]:
    """
    Partial correlation of x and y controlling for z.
    Returns (r_partial, p_value).
    """
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]
    n = len(x)
    if n < 4:
        return np.nan, np.nan

    # Residualise x and y on z
    def _resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        lm = LinearRegression()
        lm.fit(b.reshape(-1, 1), a)
        return a - lm.predict(b.reshape(-1, 1))

    x_res = _resid(x, z)
    y_res = _resid(y, z)

    r, p = stats.pearsonr(x_res, y_res)
    return float(r), float(p)


def compute_partial_correlations(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each capability metric, compute partial correlation with mean_stage,
    controlling for log_params (model scale).
    """
    rows = []
    z = model_df["log_params"].values

    for metric in CAPABILITY_COLS:
        if metric not in model_df.columns:
            continue
        x = model_df[metric].values
        y = model_df["mean_stage"].values

        # Raw Pearson
        valid_raw = ~(np.isnan(x) | np.isnan(y))
        if valid_raw.sum() >= 3:
            r_raw, p_raw = stats.pearsonr(x[valid_raw], y[valid_raw])
        else:
            r_raw, p_raw = np.nan, np.nan

        # Partial (controlling for scale)
        r_part, p_part = partial_correlation(x, y, z)

        rows.append({
            "metric":         metric,
            "raw_r":          r_raw,
            "raw_p":          p_raw,
            "partial_r":      r_part,
            "partial_p":      p_part,
            "scale_controlled": True,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        _, df["raw_p_fdr"],    _, _ = multipletests(df["raw_p"].fillna(1),    method="fdr_bh")
        _, df["partial_p_fdr"], _, _ = multipletests(df["partial_p"].fillna(1), method="fdr_bh")
    return df
