# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from config import *

PANEL_FILE = PANEL_DIR / "selected_behavioral_panel.csv"
REFERENCE_DATE = pd.Timestamp("2022-04-21")
INTERVENTION_DATE = pd.Timestamp("2022-04-22")
PSEUDO_DATE = pd.Timestamp("2022-04-15")
def load_panel():
    d = pd.read_csv(PANEL_FILE, parse_dates=["date"])
    users = pd.Index(d.user_id.unique(), name="user_id")
    counts = []
    for chunk in pd.read_csv(RAND_FILE, usecols=["user_id", "tab"], chunksize=250_000):
        counts.append(chunk.loc[chunk.tab.eq(1)].groupby("user_id").size())
    total_random = pd.concat(counts, axis=1).fillna(0).sum(axis=1).reindex(users, fill_value=0)
    log_total = np.log1p(total_random)
    dose = (log_total - log_total.mean()) / log_total.std(ddof=1)
    d["future_dose_z"] = d.user_id.map(dose)

    # Recover time-invariant baseline traits from their post-period interactions.
    d["baseline_hhi"] = d.groupby("user_id")["baseline_hhi_post"].transform("max")
    d["baseline_activity"] = d.groupby("user_id")["baseline_activity_post"].transform("max")
    return d, total_random, dose


def panel_fit(data, outcome, regressors, *, cluster_time=False):
    required = [outcome, *regressors]
    x = data.dropna(subset=required).copy()
    x["entity"] = x.user_id.astype(str)
    x = x.set_index(["entity", "date"])
    fit = PanelOLS(
        x[outcome],
        x[regressors],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        cluster_entity=True,
        cluster_time=cluster_time,
    )
    return fit, x


def estimate_event_study(d, outcome):
    dates = sorted(d.date.dropna().unique())
    interaction_cols = []
    date_by_col = {}
    for raw_date in dates:
        date = pd.Timestamp(raw_date)
        if date == REFERENCE_DATE:
            continue
        col = f"dose_x_{date:%Y%m%d}"
        d[col] = d.future_dose_z * d.date.eq(date).astype(float)
        interaction_cols.append(col)
        date_by_col[col] = date

    fit, sample = panel_fit(d, outcome, [*interaction_cols, *CONTROLS])
    rows = []
    for col in interaction_cols:
        coef = float(fit.params[col])
        se = float(fit.std_errors[col])
        date = date_by_col[col]
        rows.append(
            {
                "outcome": outcome,
                "date": date,
                "event_day": int((date - INTERVENTION_DATE).days),
                "period": "pre" if date < INTERVENTION_DATE else "post",
                "reference_date": REFERENCE_DATE.date().isoformat(),
                "coef": coef,
                "std_error": se,
                "ci95_low": coef - 1.96 * se,
                "ci95_high": coef + 1.96 * se,
                "p_value": float(fit.pvalues[col]),
                "n_obs": int(fit.nobs),
                "n_users": int(sample.index.get_level_values(0).nunique()),
            }
        )

    pre_cols = [c for c in interaction_cols if date_by_col[c] < REFERENCE_DATE]
    restriction = np.zeros((len(pre_cols), len(fit.params)))
    positions = {name: j for j, name in enumerate(fit.params.index)}
    for i, col in enumerate(pre_cols):
        restriction[i, positions[col]] = 1
    test = fit.wald_test(restriction=restriction)
    joint = {
        "outcome": outcome,
        "test": "all_pre_event_coefficients_equal_zero",
        "statistic": float(test.stat),
        "df": int(test.df),
        "p_value": float(test.pval),
        "n_pre_coefficients": len(pre_cols),
        "reference_date": REFERENCE_DATE.date().isoformat(),
        "n_obs": int(fit.nobs),
        "n_users": int(sample.index.get_level_values(0).nunique()),
    }
    return rows, joint, str(fit.summary)


def estimate_pseudo_did(d, outcome):
    pre = d.loc[d.date < INTERVENTION_DATE].copy()
    pre["pseudo_post"] = pre.date.ge(PSEUDO_DATE).astype(float)
    pre["pseudo_dose"] = pre.future_dose_z * pre.pseudo_post
    pre["baseline_hhi_pseudo"] = pre.baseline_hhi * pre.pseudo_post
    pre["baseline_activity_pseudo"] = pre.baseline_activity * pre.pseudo_post
    regressors = [
        "pseudo_dose",
        "log_opp",
        "lag_activity",
        "baseline_hhi_pseudo",
        "baseline_activity_pseudo",
    ]
    fit, sample = panel_fit(pre, outcome, regressors)
    return {
        "outcome": outcome,
        "pseudo_date": PSEUDO_DATE.date().isoformat(),
        "coef": float(fit.params["pseudo_dose"]),
        "std_error": float(fit.std_errors["pseudo_dose"]),
        "t_stat": float(fit.tstats["pseudo_dose"]),
        "p_value": float(fit.pvalues["pseudo_dose"]),
        "n_obs": int(fit.nobs),
        "n_users": int(sample.index.get_level_values(0).nunique()),
    }, str(fit.summary)

def main():
    d, total_random, dose = load_panel()
    event_rows, joint_rows, pseudo_rows = [], [], []
    for outcome in OUTCOMES:
        rows, joint, _ = estimate_event_study(d.copy(), outcome)
        event_rows.extend(rows); joint_rows.append(joint)
        pseudo, _ = estimate_pseudo_did(d, outcome)
        pseudo_rows.append(pseudo)
    for name, rows in [("event_study", event_rows), ("joint_tests", joint_rows), ("pseudo_did", pseudo_rows)]:
        pd.DataFrame(rows).to_csv(RESULT_DIR / f"parallel_trends_{name}.csv", index=False)

if __name__ == "__main__":
    main()
