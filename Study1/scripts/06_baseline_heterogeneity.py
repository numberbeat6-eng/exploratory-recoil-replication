# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from config import *

from scipy import stats
BASELINE_VARS = {"base_gap_is_click": "preference_gap_is_click"}
def make_user_baseline(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only pre-period observations to the user level."""
    pre = panel.loc[panel["post"].eq(0)].copy()
    out = pre.groupby("user_id").size().rename("pre_days").to_frame()
    for new, old in BASELINE_VARS.items():
        out[new] = pre.groupby("user_id")[old].mean()
    return out.reset_index()


def add_quartiles(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Create fixed quartile labels from the eligible baseline distribution."""
    z = df.dropna(subset=[variable]).copy()
    z["baseline_variable"] = variable
    # Rank breaks ties deterministically while preserving equal-sized groups.
    z["baseline_rank"] = z[variable].rank(method="first")
    z["baseline_q"] = pd.qcut(z["baseline_rank"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return z


def bootstrap_group_stats(z: pd.DataFrame, n_boot: int = 2000, seed: int = 20260904):
    rng = np.random.default_rng(seed)
    rows = []
    for outcome in OUTCOMES:
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            g = z.loc[z.baseline_q.eq(q), outcome].dropna().to_numpy(float)
            boots = np.empty(n_boot)
            for b in range(n_boot):
                boots[b] = rng.choice(g, size=len(g), replace=True).mean()
            rows.append({
                "baseline_variable": z.baseline_variable.iloc[0],
                "outcome": outcome,
                "group": q,
                "n_users": int(len(g)),
                "baseline_mean": float(z.loc[z.baseline_q.eq(q), z.baseline_variable.iloc[0]].mean()),
                "baseline_median": float(z.loc[z.baseline_q.eq(q), z.baseline_variable.iloc[0]].median()),
                "cate_mean": float(g.mean()),
                "cate_median": float(np.median(g)),
                "cate_sd": float(g.std(ddof=1)),
                "ci_low": float(np.quantile(boots, 0.025)),
                "ci_high": float(np.quantile(boots, 0.975)),
            })
        q1 = z.loc[z.baseline_q.eq("Q1"), outcome].dropna().to_numpy(float)
        q4 = z.loc[z.baseline_q.eq("Q4"), outcome].dropna().to_numpy(float)
        boot_diff = np.empty(n_boot)
        for b in range(n_boot):
            boot_diff[b] = rng.choice(q4, size=len(q4), replace=True).mean() - rng.choice(q1, size=len(q1), replace=True).mean()
        observed = q4.mean() - q1.mean()
        # Welch test is reported alongside the bootstrap interval for the
        # descriptive Q4-Q1 contrast.
        welch = stats.ttest_ind(q4, q1, equal_var=False)
        rows.append({
            "baseline_variable": z.baseline_variable.iloc[0],
            "outcome": outcome,
            "group": "Q4-Q1",
            "n_users": int(len(q1) + len(q4)),
            "baseline_mean": float(z.loc[z.baseline_q.eq("Q4"), z.baseline_variable.iloc[0]].mean() - z.loc[z.baseline_q.eq("Q1"), z.baseline_variable.iloc[0]].mean()),
            "baseline_median": np.nan,
            "cate_mean": float(observed),
            "cate_median": float(np.median(q4) - np.median(q1)),
            "cate_sd": np.nan,
            "ci_low": float(np.quantile(boot_diff, 0.025)),
            "ci_high": float(np.quantile(boot_diff, 0.975)),
            "welch_t": float(welch.statistic),
            "welch_p": float(welch.pvalue),
        })
    return pd.DataFrame(rows)


def main():
    baseline = make_user_baseline(pd.read_csv(PANEL_DIR / "selected_behavioral_panel.csv"))
    cate_file = RESULT_DIR / "causal_forest_individual_effects.csv"
    cate = pd.read_csv(cate_file)
    users = baseline.merge(cate.pivot(index="user_id", columns="outcome", values="cate").reset_index(), on="user_id", how="inner")
    summary = bootstrap_group_stats(add_quartiles(users, "base_gap_is_click"))
    summary.to_csv(RESULT_DIR / "readi_style_heterogeneity_bootstrap.csv", index=False)

if __name__ == "__main__":
    main()
