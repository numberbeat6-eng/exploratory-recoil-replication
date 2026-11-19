"""Reproduce Table A2 using the daily insertion rate."""
import pandas as pd
from linearmodels.panel import PanelOLS

from config import CONTROLS, PANEL_DIR, RESULT_DIR

OUTCOMES = [
    "nov_rate_is_click", "fam_rate_is_click", "preference_gap_is_click",
    "nov_rate_long_view", "nov_rate_watch_ratio",
]


def main():
    panel = pd.read_csv(PANEL_DIR / "selected_behavioral_panel.csv", parse_dates=["date"])
    insertion_rate = panel.all_random_n / panel.opp_n
    positive_post = panel.post.eq(1) & panel.all_random_n.gt(0)
    panel["rate_sd_positive"] = insertion_rate / insertion_rate.loc[positive_post].std(ddof=1)

    lagged = panel[["user_id", "date", "dose_sd", "rate_sd_positive"]].copy()
    lagged["date"] += pd.Timedelta(days=1)
    lagged = lagged.rename(columns={"dose_sd": "lag_dose", "rate_sd_positive": "lag_rate"})
    panel = panel.merge(lagged, on=["user_id", "date"], how="left", validate="one_to_one")

    current = "rate_sd_positive"
    regressors = [current, "lag_rate", *CONTROLS]
    common = panel.dropna(subset=[*OUTCOMES, *regressors, "dose_sd", "lag_dose"])
    rows = []
    for sample, threshold in [("all_eligible_days", 0), ("opportunities_ge_50", 50)]:
        sample_panel = common.loc[common.opp_n.ge(threshold)].set_index(["user_id", "date"])
        for outcome in OUTCOMES:
            fit = PanelOLS(
                sample_panel[outcome], sample_panel[regressors],
                entity_effects=True, time_effects=True, drop_absorbed=True,
            ).fit(cov_type="clustered", cluster_entity=True)
            ci = fit.conf_int().loc[current]
            rows.append({
                "sample": sample,
                "outcome": outcome,
                "coef": fit.params[current],
                "std_error": fit.std_errors[current],
                "p_value": fit.pvalues[current],
                "ci95_low": ci["lower"],
                "ci95_high": ci["upper"],
                "n_obs": fit.nobs,
                "n_users": sample_panel.index.get_level_values("user_id").nunique(),
            })
    pd.DataFrame(rows).to_csv(RESULT_DIR / "tableA2_insertion_rate.csv", index=False)


if __name__ == "__main__":
    main()
