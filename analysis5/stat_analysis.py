"""
stat_analysis.py — Statistical computations for Action-Reasoning Consistency.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
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

