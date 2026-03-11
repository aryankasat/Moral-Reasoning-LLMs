"""
stat_analysis.py — Statistical computations for Action-Reasoning Consistency.

Public API
----------
compute_overall_consistency(df)      -> (pd.DataFrame, pd.DataFrame)
compute_action_distributions(df)     -> pd.DataFrame
compute_stage_action_crosstab(valid) -> pd.DataFrame
run_chi_square(valid_df)             -> dict
run_mcnemar_tests(valid_df)          -> dict
    Global McNemar test on pooled 2×2 table (expected vs actual action)
    + per-model McNemar tests with Bonferroni correction.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from config import EXPECTED_ACTION_BY_STAGE

def compute_overall_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes overall consistency % across all valid actions.
    Expected mapping:
      Stage 1-4 -> Rule-Following
      Stage 5-6 -> Rule-Breaking
    """
    valid = df[df["action_category"].isin(["Rule-Following", "Rule-Breaking"])].copy()
    valid["expected_action"] = valid["kohlberg_stage"].map(EXPECTED_ACTION_BY_STAGE)
    valid["is_consistent"] = valid["expected_action"] == valid["action_category"]

    records = []
    for mk, grp in valid.groupby("model_key", sort=False):
        n_total = len(grp)
        n_consist = grp["is_consistent"].sum()
        pct = (n_consist / n_total) * 100 if n_total > 0 else 0.0
        
        meta = grp.iloc[0]
        records.append({
            "model_key": mk,
            "display_name": meta["display_name"],
            "params_B": meta["params_B"],
            "log_params": meta["log_params"],
            "provider": meta["provider"],
            "n_valid_actions": n_total,
            "n_consistent": n_consist,
            "consistency_pct": round(pct, 2)
        })
        
    res_df = pd.DataFrame(records).sort_values("params_B", ascending=False)
    return res_df, valid

def compute_action_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes percentage of Rule-Following vs Rule-Breaking per dilemma.
    """
    valid = df[df["action_category"].isin(["Rule-Following", "Rule-Breaking"])]
    dist = valid.groupby("dilemma_type")["action_category"].value_counts(normalize=True).unstack(fill_value=0) * 100
    dist = dist.reset_index()
    dist.columns.name = None
    return dist

def compute_stage_action_crosstab(valid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a cross-tabulation of Kohlberg Stage vs Action Category.
    """
    ct = pd.crosstab(valid_df["kohlberg_stage"], valid_df["action_category"])
    
    # Ensure all stages 1-6 are present
    for s in range(1, 7):
        if s not in ct.index:
            ct.loc[s] = 0
            
    # Ensure both action columns exist
    for act in ["Rule-Following", "Rule-Breaking"]:
        if act not in ct.columns:
            ct[act] = 0
            
    return ct.sort_index()[["Rule-Following", "Rule-Breaking"]]

def run_chi_square(valid_df: pd.DataFrame) -> dict:
    """
    Chi-square test of independence between Stage and Action overall.
    """
    ct = pd.crosstab(valid_df["kohlberg_stage"], valid_df["action_category"])
    
    # Drop rows/cols with structural zeros strictly
    ct = ct.loc[(ct.sum(axis=1) > 0), (ct.sum(axis=0) > 0)]
    
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"chi2": np.nan, "p_value": np.nan, "dof": np.nan, "significant": False}
        
    chi2, p, dof, _ = chi2_contingency(ct)
    return {
        "chi2": round(chi2, 3),
        "p_value": p,
        "dof": dof,
        "significant": p < 0.05
    }


def _mcnemar_2x2(expected: np.ndarray, actual: np.ndarray) -> dict:
    """
    Build the 2×2 paired-consistency contingency table and run McNemar's test.

    Cells:
        a = both expected AND actual are Rule-Following
        b = expected Rule-Following but actual Rule-Breaking  (off-diagonal)
        c = expected Rule-Breaking but actual Rule-Following  (off-diagonal)
        d = both expected AND actual are Rule-Breaking

    McNemar's test is on the off-diagonal discordant pairs (b, c).
    Uses exact binomial when b+c < 25, chi² with continuity otherwise.
    Returns dict with keys: a, b, c, d, statistic, p_value, significant, method.
    """
    FOLLOW = "Rule-Following"
    BREAK  = "Rule-Breaking"

    a = int(np.sum((expected == FOLLOW) & (actual == FOLLOW)))
    b = int(np.sum((expected == FOLLOW) & (actual == BREAK)))
    c = int(np.sum((expected == BREAK)  & (actual == FOLLOW)))
    d = int(np.sum((expected == BREAK)  & (actual == BREAK)))

    table = np.array([[a, b], [c, d]])
    n_discordant = b + c

    if n_discordant == 0:
        # Perfect agreement on discordant cells → p = 1
        return dict(a=a, b=b, c=c, d=d,
                    statistic=0.0, p_value=1.0, significant=False,
                    method="exact", n_discordant=0)

    # exact=True uses binomial; exact=False uses chi² with continuity correction
    exact = n_discordant < 25
    result = mcnemar(table, exact=exact, correction=True)
    return dict(
        a=a, b=b, c=c, d=d,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        significant=bool(result.pvalue < 0.05),
        method="exact_binomial" if exact else "chi2_continuity",
        n_discordant=n_discordant,
    )


def run_mcnemar_tests(valid_df: pd.DataFrame) -> dict:
    """
    McNemar's test for paired action-reasoning consistency.

    For each observation the 'expected' action is derived from the model's
    assigned Kohlberg stage (via EXPECTED_ACTION_BY_STAGE) and the 'actual'
    action is the categorised action_category.  The 2×2 table of
    (expected, actual) ∈ {Rule-Following, Rule-Breaking}² is a natural
    McNemar setup because the same dilemma drives both measurements.

    Tests run:
    1. Global   — pooled across all models and dilemmas.
    2. Per-model — one test per model, Bonferroni-corrected p-values.
    3. Per-dilemma — one test per dilemma, Bonferroni-corrected.

    Returns
    -------
    dict with keys:
        global          – dict (McNemar result for pooled data)
        per_model       – pd.DataFrame  (one row per model, with p_adj)
        per_dilemma     – pd.DataFrame  (one row per dilemma, with p_adj)
    """
    # Work on a copy with expected action filled in
    df = valid_df.copy()
    df["expected_action"] = df["kohlberg_stage"].map(EXPECTED_ACTION_BY_STAGE)

    # ── 1. Global test ──────────────────────────────────────────────────────
    global_result = _mcnemar_2x2(
        df["expected_action"].values,
        df["action_category"].values,
    )

    # ── 2. Per-model tests ──────────────────────────────────────────────────
    model_rows = []
    for mk, grp in df.groupby("model_key"):
        res = _mcnemar_2x2(
            grp["expected_action"].values,
            grp["action_category"].values,
        )
        model_rows.append({
            "model_key":    mk,
            "display_name": grp["display_name"].iloc[0],
            "params_B":     grp["params_B"].iloc[0],
            "n_discordant": res["n_discordant"],
            "a": res["a"], "b": res["b"], "c": res["c"], "d": res["d"],
            "statistic":    res["statistic"],
            "p_value":      res["p_value"],
            "method":       res["method"],
        })
    per_model_df = pd.DataFrame(model_rows).sort_values("params_B")
    n_models = len(per_model_df)
    per_model_df["p_adj_bonferroni"] = np.minimum(
        per_model_df["p_value"] * n_models, 1.0
    )
    per_model_df["significant_adj"] = per_model_df["p_adj_bonferroni"] < 0.05

    # ── 3. Per-dilemma tests ────────────────────────────────────────────────
    dilemma_rows = []
    for dilemma, grp in df.groupby("dilemma_type"):
        res = _mcnemar_2x2(
            grp["expected_action"].values,
            grp["action_category"].values,
        )
        dilemma_rows.append({
            "dilemma_type":  dilemma,
            "n_discordant":  res["n_discordant"],
            "a": res["a"], "b": res["b"], "c": res["c"], "d": res["d"],
            "statistic":     res["statistic"],
            "p_value":       res["p_value"],
            "method":        res["method"],
        })
    per_dilemma_df = pd.DataFrame(dilemma_rows)
    n_dilemmas = len(per_dilemma_df)
    per_dilemma_df["p_adj_bonferroni"] = np.minimum(
        per_dilemma_df["p_value"] * n_dilemmas, 1.0
    )
    per_dilemma_df["significant_adj"] = per_dilemma_df["p_adj_bonferroni"] < 0.05

    return dict(
        global_test=global_result,
        per_model=per_model_df,
        per_dilemma=per_dilemma_df,
    )
