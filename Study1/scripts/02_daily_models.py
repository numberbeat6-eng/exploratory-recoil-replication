# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from config import *

PRECISION_WEIGHTS = {y: ("gap_precision_w" if y == "preference_gap_is_click" else "nov_w") for y in OUTCOMES}
def add_calendar_value(data, source, offset, target):
    shifted = data[["user_id", "date", source]].copy()
    shifted["date"] = shifted["date"] + pd.Timedelta(days=offset)
    shifted = shifted.rename(columns={source: target})
    return data.merge(shifted, on=["user_id", "date"], how="left")


def estimate(
    data,
    outcome,
    focal,
    regressors,
    specification,
    *,
    weight=None,
    opportunity_bins=None,
    cluster_time=False,
):
    required = [outcome, focal, *regressors]
    if weight:
        required.append(weight)
    if opportunity_bins:
        required.append(opportunity_bins)
    x = data.dropna(subset=list(dict.fromkeys(required))).copy()
    x["entity"] = x.user_id.astype(str)
    x = x.set_index(["entity", "date"])
    kwargs = {
        "entity_effects": True,
        "drop_absorbed": True,
    }
    if opportunity_bins:
        kwargs["other_effects"] = x[[opportunity_bins]]
    else:
        kwargs["time_effects"] = True
    if weight:
        kwargs["weights"] = x[weight]
    fit = PanelOLS(x[outcome], x[regressors], **kwargs).fit(
        cov_type="clustered", cluster_entity=True, cluster_time=cluster_time
    )
    row = {
        "specification": specification,
        "outcome": outcome,
        "focal": focal,
        "coef": fit.params[focal],
        "std_error": fit.std_errors[focal],
        "t_stat": fit.tstats[focal],
        "p_value": fit.pvalues[focal],
        "n_obs": fit.nobs,
        "n_users": x.index.get_level_values(0).nunique(),
        "r2_within": fit.rsquared_within,
    }
    if "lead_dose" in fit.params.index:
        row["lead_coef"] = fit.params["lead_dose"]
        row["lead_p_value"] = fit.pvalues["lead_dose"]
    if "lag_dose" in fit.params.index:
        row["lag_coef"] = fit.params["lag_dose"]
        row["lag_p_value"] = fit.pvalues["lag_dose"]
    if "dose_sd" in fit.params.index:
        row["current_dose_coef"] = fit.params["dose_sd"]
        row["current_dose_p_value"] = fit.pvalues["dose_sd"]
    return row, str(fit.summary)

def main():
    d = pd.read_csv(PANEL_DIR / "selected_behavioral_panel.csv", parse_dates=["date"])
    d = add_calendar_value(d, "dose_sd", 1, "lag_dose")
    d = add_calendar_value(d, "dose_sd", -1, "lead_dose")
    for outcome in OUTCOMES:
        previous = f"previous_{outcome}"
        if previous not in d.columns:
            d = add_calendar_value(d, outcome, 1, previous)
    d["gap_precision_w"] = 1 / (1 / d.fam_w + 1 / d.nov_w)

    # Date-by-opportunity cells compare users browsing at similar depth on the same day.
    within_date_rank = d.groupby("date")["opp_n"].rank(method="average", pct=True)
    d["opp_decile"] = np.minimum((within_date_rank * 10).astype(int), 9)
    d["opp_ventile"] = np.minimum((within_date_rank * 20).astype(int), 19)
    date_label = d.date.dt.strftime("%Y%m%d")
    d["date_opp_decile"] = pd.Categorical(date_label + "_" + d.opp_decile.astype(str)).codes
    d["date_opp_ventile"] = pd.Categorical(date_label + "_" + d.opp_ventile.astype(str)).codes
    d["log_opp_cu"] = d.log_opp ** 3

    rows = []
    summaries = []
    for outcome in OUTCOMES:
        d[f"lag_{outcome}"] = d[f"previous_{outcome}"]
        d[f"change_{outcome}"] = d[outcome] - d[f"previous_{outcome}"]
        specifications = [
            ("R3_lagged_outcome", outcome, "dose_sd", ["dose_sd", f"lag_{outcome}", *CONTROLS], {}),
            ("R4_distributed_lag", outcome, "dose_sd", ["dose_sd", "lag_dose", *CONTROLS], {}),
            ("R8_weighted_by_novel_exposures", outcome, "dose_sd", ["dose_sd", *CONTROLS], {"weight": PRECISION_WEIGHTS[outcome]}),
            ("R9_date_by_opportunity_decile_FE", outcome, "dose_sd", ["dose_sd", *CONTROLS], {"opportunity_bins": "date_opp_decile"}),
            ("R11_distributed_lag_two_way_cluster", outcome, "dose_sd", ["dose_sd", "lag_dose", *CONTROLS], {"cluster_time": True}),
        ]
        if outcome != "preference_gap_is_click":
            specifications.append(("R5_conditional_lead_placebo", outcome, "lead_dose", ["dose_sd", "lead_dose", *CONTROLS], {}))
        for spec, dep, focal, regressors, options in specifications:
            row, summary = estimate(
                d, dep, focal, regressors, spec, **options
            )
            rows.append(row)
            summaries.append(f"{spec}: {dep}; focal={focal}\n{summary}")

    out = pd.DataFrame(rows)
    out = out.sort_values(["outcome", "specification"]).reset_index(drop=True)
    out.to_csv(RESULT_DIR / "behavioral_robustness_results.csv", index=False, encoding="utf-8-sig")
    selected_specs = [
        "R1_full_controls",
        "R3_lagged_outcome",
        "R4_distributed_lag",
        "R5_conditional_lead_placebo",
        "R6_dynamic_lead_lag",
        "R9_date_by_opportunity_decile_FE",
        "R10_date_by_opportunity_ventile_FE",
        "R11_distributed_lag_two_way_cluster",
    ]
    out.loc[out.specification.isin(selected_specs)].to_csv(
        RESULT_DIR / "behavioral_selected_results.csv", index=False, encoding="utf-8-sig"
    )

    effects = []
    main_rows = out.loc[out.specification.eq("R4_distributed_lag")]
    for _, result in main_rows.iterrows():
        outcome = result.outcome
        sample = d.dropna(subset=[outcome, "dose_sd", "lag_dose", *CONTROLS]).copy()
        post_sample = sample.loc[sample.post.eq(1)]
        q10, q90 = post_sample.dose_sd.quantile([0.1, 0.9])
        contrast = result.coef * (q90 - q10)
        effects.append(
            {
                "outcome": outcome,
                "coef_per_dose_sd": result.coef,
                "ci95_low": result.coef - 1.96 * result.std_error,
                "ci95_high": result.coef + 1.96 * result.std_error,
                "p_value": result.p_value,
                "post_dose_p10": q10,
                "post_dose_p90": q90,
                "p90_minus_p10_effect": contrast,
                "overall_outcome_mean": sample[outcome].mean(),
                "post_outcome_mean": post_sample[outcome].mean(),
                "contrast_relative_to_post_mean": (
                    contrast / post_sample[outcome].mean()
                    if outcome != "preference_gap_is_click"
                    else np.nan
                ),
                "n_obs": len(sample),
                "n_users": sample.user_id.nunique(),
            }
        )
    pd.DataFrame(effects).to_csv(
        RESULT_DIR / "behavioral_effect_sizes.csv", index=False, encoding="utf-8-sig"
    )
    (RESULT_DIR / "behavioral_robustness_summaries.txt").write_text(
        "\n\n".join(summaries), encoding="utf-8"
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
