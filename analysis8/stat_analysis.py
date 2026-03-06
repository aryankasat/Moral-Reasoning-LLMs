"""
stat_analysis.py — Two-way factorial ANOVA for Analysis 8: Scale vs. Training Decomposition.

Corrected statistical pipeline for an INCOMPLETE factorial design (5 of 9 cells populated):

  1. Assumption checks
       - Shapiro-Wilk normality test per cell
       - Levene's test for homoscedasticity
       - Kruskal-Wallis (non-parametric) for factor-level tests
       - Welch ANOVA (robustgiven heteroscedasticity)

  2. Two-way ANOVA — sequential (Type-I) SS via nested model comparison
       Interaction df is taken directly from the fitted full-model residual df
       to avoid over-stating df_int in an incomplete design.

  3. Effect sizes — Partial η² and ω² (both parametric)
       Cohen's d for significant pairwise differences

  4. Variance partitioning — proportional SS decomposition

  5. Post-hoc pairwise comparisons
       - Tukey HSD (parametric)
       - Mann-Whitney U with Bonferroni correction (non-parametric)

  6. Hypothesis classification H1–H4
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats as scipy_stats

import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# ─────────────────────────────────────────────────────────────────────────────
# 1. Assumption checks
# ─────────────────────────────────────────────────────────────────────────────

def check_assumptions(raw_df: pd.DataFrame) -> dict:
    """
    Full battery of assumption checks:
      - Shapiro-Wilk per cell (normality)
      - Levene's test (homoscedasticity)
      - Kruskal-Wallis for Scale and Training (non-parametric alternative)
      - Welch ANOVA for Scale and Training (variance-robust parametric)
    """
    results: dict = {}
    shapiro_results = []

    cell_groups: dict[tuple, np.ndarray] = {}
    for (sg, tt), grp in raw_df.groupby(["scale_group", "training_type"], observed=True):
        stages = grp["kohlberg_stage"].values
        cell_groups[(sg, tt)] = stages
        if len(stages) < 3:
            shapiro_results.append((f"{sg} × {tt}", None, None, "n<3 (skip)"))
            continue
        W, p = scipy_stats.shapiro(stages)
        shapiro_results.append((f"{sg} × {tt}", round(W, 4), round(p, 4),
                                 "PASS" if p > 0.05 else "FAIL"))

    results["shapiro_results"] = shapiro_results
    results["normality_ok"] = all(r[3] in ("PASS",) or "skip" in r[3] for r in shapiro_results)

    # Levene across all populated cells
    cell_arrays = [v for v in cell_groups.values() if len(v) >= 2]
    if len(cell_arrays) >= 2:
        lev_stat, lev_p = scipy_stats.levene(*cell_arrays)
        results["levene_stat"] = round(float(lev_stat), 4)
        results["levene_p"]    = round(float(lev_p), 4)
        results["homoscedasticity_ok"] = lev_p > 0.05
    else:
        results["levene_stat"] = results["levene_p"] = None
        results["homoscedasticity_ok"] = None

    # ── Kruskal-Wallis (non-parametric one-way tests) ──────────────────────────
    sg_groups = [grp["kohlberg_stage"].values
                 for _, grp in raw_df.groupby("scale_group", observed=True)]
    tt_groups = [grp["kohlberg_stage"].values
                 for _, grp in raw_df.groupby("training_type", observed=True)]

    kw_sg_H, kw_sg_p = scipy_stats.kruskal(*sg_groups) if len(sg_groups) >= 2 else (np.nan, np.nan)
    kw_tt_H, kw_tt_p = scipy_stats.kruskal(*tt_groups) if len(tt_groups) >= 2 else (np.nan, np.nan)
    results["kruskal_scale"]    = {"H": round(float(kw_sg_H), 4), "p": round(float(kw_sg_p), 4)}
    results["kruskal_training"] = {"H": round(float(kw_tt_H), 4), "p": round(float(kw_tt_p), 4)}

    # ── Welch ANOVA (one-way, variance-robust) ─────────────────────────────────
    def welch_anova(groups_list):
        """Welch's one-way ANOVA (Browne-Forsythe / Welch F)."""
        k = len(groups_list)
        ns   = np.array([len(g) for g in groups_list], dtype=float)
        vars_ = np.array([np.var(g, ddof=1) for g in groups_list])
        means = np.array([np.mean(g) for g in groups_list])
        ws    = ns / vars_
        W_sum = ws.sum()
        grand_mean = (ws * means).sum() / W_sum
        numer = (ws * (means - grand_mean)**2).sum() / (k - 1)
        lambda_ = (1 / (k**2 - 1)) * ((1 - ws / W_sum)**2 / (ns - 1)).sum()
        F_welch = numer / (1 + 2 * (k - 2) * lambda_)
        df1 = k - 1
        df2 = 1 / (3 * lambda_)
        p   = float(scipy_stats.f.sf(F_welch, df1, df2))
        return round(float(F_welch), 4), df1, round(float(df2), 2), round(p, 4)

    try:
        w_sg = welch_anova(sg_groups)
        results["welch_scale"] = {"F": w_sg[0], "df1": w_sg[1], "df2": w_sg[2], "p": w_sg[3]}
    except Exception:
        results["welch_scale"] = {}

    try:
        w_tt = welch_anova(tt_groups)
        results["welch_training"] = {"F": w_tt[0], "df1": w_tt[1], "df2": w_tt[2], "p": w_tt[3]}
    except Exception:
        results["welch_training"] = {}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Two-way ANOVA — sequential SS, correct df for incomplete design
# ─────────────────────────────────────────────────────────────────────────────

def _fit_ols(df: pd.DataFrame, formula: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.ols(formula, data=df).fit()


def run_two_way_anova(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """
    Sequential (Type-I) SS decomposition via nested OLS model comparisons:

      SS_Scale     = SSR(null)  − SSR(Scale)
      SS_Training  = SSR(Scale) − SSR(Scale + Training)
      SS_Interact  = SSR(additive) − SSR(full)
      SS_Residual  = SSR(full)

    Degrees of freedom for the interaction and residual are taken directly
    from the OLS model objects (actual rank-based df), which correctly handles
    aliased columns that arise in incomplete factorial designs.
    """
    df = raw_df.copy()
    df["scale_group"]   = df["scale_group"].astype(str)
    df["training_type"] = df["training_type"].astype(str)

    m_null = _fit_ols(df, "kohlberg_stage ~ 1")
    m_sg   = _fit_ols(df, "kohlberg_stage ~ C(scale_group)")
    m_add  = _fit_ols(df, "kohlberg_stage ~ C(scale_group) + C(training_type)")
    m_full = _fit_ols(df, "kohlberg_stage ~ C(scale_group) + C(training_type)"
                          " + C(scale_group):C(training_type)")

    ss_scale    = max(float(m_null.ssr - m_sg.ssr),   0.0)
    ss_training = max(float(m_sg.ssr   - m_add.ssr),  0.0)
    ss_interact = max(float(m_add.ssr  - m_full.ssr), 0.0)
    ss_resid    = float(m_full.ssr)

    # ── Correct df: use actual model df (rank-based) ───────────────────────────
    n_sg = df["scale_group"].nunique()
    n_tt = df["training_type"].nunique()
    df_sg  = n_sg - 1
    df_tt  = n_tt - 1

    # Interaction df = additional params gained going from additive → full model
    # = (params full) - (params additive) — handles missing cells correctly
    df_int = int(round(m_add.df_resid - m_full.df_resid))
    df_int = max(df_int, 0)
    df_res = int(m_full.df_resid)

    ms_res = ss_resid / df_res if df_res > 0 else np.nan

    def _f_p(ss, df1):
        if df1 == 0 or np.isnan(ms_res) or ms_res == 0:
            return np.nan, np.nan
        ms = ss / df1
        F  = ms / ms_res
        p  = float(scipy_stats.f.sf(F, df1, df_res))
        return round(F, 6), round(p, 6)

    f_sg,  p_sg  = _f_p(ss_scale,    df_sg)
    f_tt,  p_tt  = _f_p(ss_training, df_tt)
    f_int, p_int = _f_p(ss_interact, df_int) if df_int > 0 else (np.nan, np.nan)

    anova_table = pd.DataFrame({
        "sum_sq":  [ss_scale,  ss_training, ss_interact, ss_resid],
        "df":      [df_sg,     df_tt,       df_int,      df_res],
        "F":       [f_sg,      f_tt,        f_int,       np.nan],
        "PR(>F)":  [p_sg,      p_tt,        p_int,       np.nan],
    }, index=["Scale", "Training_Type", "Scale × Training_Type", "Residual"])

    return anova_table, m_full


# ─────────────────────────────────────────────────────────────────────────────
# 3. Effect sizes
# ─────────────────────────────────────────────────────────────────────────────

def compute_effect_sizes(anova_table: pd.DataFrame) -> pd.DataFrame:
    """
    Partial η² = SS_effect / (SS_effect + SS_residual)
    ω²          = (SS_effect − df_effect × MS_resid) / (SS_total + MS_resid)
    Both are unbiased; ω² is preferred for reporting but η² is more familiar.
    """
    ss_res = float(anova_table.loc["Residual", "sum_sq"])
    df_res = float(anova_table.loc["Residual", "df"])
    ms_res = ss_res / df_res if df_res > 0 else np.nan
    ss_tot = float(anova_table["sum_sq"].sum())

    def mag(v):
        if np.isnan(v): return "—"
        if v < 0.01:    return "Negligible"
        if v < 0.06:    return "Small"
        if v < 0.14:    return "Medium"
        return "Large"

    rows = []
    for effect in anova_table.index:
        if effect == "Residual":
            continue
        ss_e  = float(anova_table.loc[effect, "sum_sq"])
        df_e  = float(anova_table.loc[effect, "df"])
        f_val = float(anova_table.loc[effect, "F"])
        p_val = float(anova_table.loc[effect, "PR(>F)"])

        p_eta2 = ss_e / (ss_e + ss_res) if (ss_e + ss_res) > 0 else np.nan
        om2    = (ss_e - df_e * ms_res) / (ss_tot + ms_res) if not np.isnan(ms_res) else np.nan
        om2    = max(om2, 0.0) if not np.isnan(om2) else np.nan

        rows.append({
            "effect":       effect,
            "SS":           round(ss_e, 4),
            "df":           int(df_e),
            "F":            round(f_val, 4) if not np.isnan(f_val) else np.nan,
            "p_value":      round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "partial_eta2": round(p_eta2, 4) if not np.isnan(p_eta2) else np.nan,
            "omega2":       round(om2, 4)   if not np.isnan(om2)   else np.nan,
            "magnitude":    mag(p_eta2),
            "significant":  (p_val < 0.05) if not np.isnan(p_val) else False,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Variance partitioning
# ─────────────────────────────────────────────────────────────────────────────

def variance_partition(anova_table: pd.DataFrame, effect_df: pd.DataFrame) -> dict:
    """% of total SS per source (η²-type partition)."""
    ss_tot = float(anova_table["sum_sq"].sum())
    out: dict[str, float] = {}
    for _, row in effect_df.iterrows():
        ss_e = row["SS"]
        out[row["effect"]] = round(100 * float(ss_e) / ss_tot, 2) if not np.isnan(float(ss_e)) else 0.0
    ss_r = float(anova_table.loc["Residual", "sum_sq"])
    out["Residual"] = round(100 * ss_r / ss_tot, 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Post-hoc comparisons — both parametric & non-parametric
# ─────────────────────────────────────────────────────────────────────────────

def run_posthoc(raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Two sets of comparisons:
      A) Scale pairwise at each Training Type level  (Tukey HSD)
      B) Training pairwise at each Scale level        (Tukey HSD)
    Plus overall non-parametric Mann-Whitney U with Bonferroni correction.
    """
    results: dict[str, pd.DataFrame] = {}

    # A: Scale pairwise within each Training Type
    for tt, grp in raw_df.groupby("training_type", observed=True):
        if grp["scale_group"].nunique() < 2:
            continue
        try:
            tukey = pairwise_tukeyhsd(
                endog  = grp["kohlberg_stage"].values,
                groups = grp["scale_group"].astype(str).values,
                alpha  = 0.05,
            )
            results[f"Scale_within_{tt}"] = pd.DataFrame(
                data    = tukey._results_table.data[1:],
                columns = tukey._results_table.data[0],
            )
        except Exception as e:
            print(f"  [WARN] Tukey scale within {tt}: {e}")

    # B: Training pairwise within each Scale level
    for sg, grp in raw_df.groupby("scale_group", observed=True):
        if grp["training_type"].nunique() < 2:
            continue
        try:
            tukey = pairwise_tukeyhsd(
                endog  = grp["kohlberg_stage"].values,
                groups = grp["training_type"].astype(str).values,
                alpha  = 0.05,
            )
            results[f"Training_within_{sg}"] = pd.DataFrame(
                data    = tukey._results_table.data[1:],
                columns = tukey._results_table.data[0],
            )
        except Exception as e:
            print(f"  [WARN] Tukey training within {sg}: {e}")

    # C: Overall non-parametric pairwise (Scale factor) — Mann-Whitney U + Bonferroni
    scale_groups = {sg: grp["kohlberg_stage"].values
                    for sg, grp in raw_df.groupby("scale_group", observed=True)}
    mw_rows = []
    pairs    = list(combinations(sorted(scale_groups.keys()), 2))
    n_tests  = len(pairs)
    for g1, g2 in pairs:
        U, p_raw = scipy_stats.mannwhitneyu(scale_groups[g1], scale_groups[g2], alternative="two-sided")
        p_bonf   = min(p_raw * n_tests, 1.0)
        r_effect = 1 - 2 * U / (len(scale_groups[g1]) * len(scale_groups[g2]))
        mw_rows.append({
            "group1": g1, "group2": g2,
            "U": round(U, 0), "p_raw": round(p_raw, 4),
            "p_bonferroni": round(p_bonf, 4),
            "r_effect_size": round(r_effect, 3),
            "significant": p_bonf < 0.05,
        })
    results["MannWhitney_Scale_Overall"] = pd.DataFrame(mw_rows)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cohen's d for model-level pairwise scale comparisons
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d_scale(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Cohen's d for each pairwise Scale group comparison."""
    groups = {sg: grp["kohlberg_stage"].values
              for sg, grp in raw_df.groupby("scale_group", observed=True)}
    rows = []
    for g1, g2 in combinations(sorted(groups.keys()), 2):
        a, b = groups[g1], groups[g2]
        pooled_sd = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
        d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append({"group1": g1, "group2": g2, "cohen_d": round(d, 3),
                     "magnitude": "small" if abs(d) < 0.5 else ("medium" if abs(d) < 0.8 else "large")})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Hypothesis classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_hypothesis(effect_df: pd.DataFrame) -> str:
    sig = {row["effect"]: row["significant"] for _, row in effect_df.iterrows()}
    scale_sig = sig.get("Scale", False)
    train_sig = sig.get("Training_Type", False)
    inter_sig = sig.get("Scale × Training_Type", False)

    if inter_sig:
        if scale_sig and train_sig:
            return "H4 — Synergistic interaction (both main effects + interaction significant)"
        return "H4 — Interaction present (non-additive effects)"
    elif scale_sig and train_sig:
        return "H3 — Both matter (additive effects, no significant interaction)"
    elif train_sig:
        return "H1 — Training dominates (training main effect; scale NS)"
    elif scale_sig:
        return "H2 — Scale dominates (scale main effect; training NS)"
    return "No dominant factor at α = 0.05"
