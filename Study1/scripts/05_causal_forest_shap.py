# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from config import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from econml.dml import CausalForestDML
import shap
PRE_FEATURES = ["nov_rate_is_click", "nov_rate_long_view", "nov_rate_watch_ratio", "preference_gap_is_click", "preference_gap_long_view", "preference_gap_watch_ratio", "standard_n", "valid_tag_n", "log_opp", "lag_activity"]
def user_features(panel):
    pre = panel.loc[panel.post.eq(0)].copy()
    post = panel.loc[panel.post.eq(1)].copy()
    agg = {}
    for col in PRE_FEATURES:
        if col in pre:
            agg[col] = (col, "mean")
    u_pre = pre.groupby("user_id").agg(**{f"pre_{k}": v for k, v in agg.items()})
    u_pre["pre_days"] = pre.groupby("user_id").size()
    u_post = post.groupby("user_id").agg(
        post_dose=("dose_sd", "mean"),
        post_days=("post", "size"),
    )
    for y in OUTCOMES:
        u_pre[f"pre_{y}"] = pre.groupby("user_id")[y].mean()
        u_post[f"post_{y}"] = post.groupby("user_id")[y].mean()
        u_post[f"change_{y}"] = u_post[f"post_{y}"] - u_pre[f"pre_{y}"]
    users = u_pre.join(u_post, how="inner").reset_index()
    return users


def causal_forest(users):
    feature_cols = [c for c in users.columns if c.startswith("pre_")]
    feature_cols = [c for c in feature_cols if users[c].notna().all() and users[c].nunique() > 1]
    x = users.dropna(subset=feature_cols + ["post_dose"]).copy()
    X = x[feature_cols].to_numpy(float)
    T = x.post_dose.to_numpy(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rows = []
    shap_rows = []
    cate_rows = []
    shap_detail_rows = []
    model_summaries = []
    for y in OUTCOMES:
        Y = x[f"change_{y}"].to_numpy(float)
        forest = CausalForestDML(
            n_estimators=400,
            min_samples_leaf=20,
            max_samples=0.45,
            honest=True,
            inference="blb",
            discrete_treatment=False,
            random_state=20260904,
            model_y=RandomForestRegressor(n_estimators=200, min_samples_leaf=10, random_state=20260904, n_jobs=-1),
            model_t=RandomForestRegressor(n_estimators=200, min_samples_leaf=10, random_state=20260904, n_jobs=-1),
        )
        forest.fit(Y, T, X=Xs)
        tau = np.asarray(forest.effect(Xs)).reshape(-1)
        lo, hi = np.nanpercentile(tau, [2.5, 97.5])
        rows.append({
            "model_id": "CausalForestDML_user_change",
            "outcome": y,
            "treatment": "post_mean_dose_sd",
            "n_users": int(len(x)),
            "mean_cate": float(np.mean(tau)),
            "median_cate": float(np.median(tau)),
            "cate_p25": float(np.percentile(tau, 25)),
            "cate_p75": float(np.percentile(tau, 75)),
            "cate_p2_5": float(lo),
            "cate_p97_5": float(hi),
            "cate_sd": float(np.std(tau, ddof=1)),
        })
        cate_rows.extend(
            {"outcome": y, "user_id": int(uid), "cate": float(c)}
            for uid, c in zip(x.user_id, tau)
        )
        model_summaries.append(f"{y}: n={len(x)}, mean CATE={np.mean(tau):.6f}, CATE SD={np.std(tau, ddof=1):.6f}")
        # RF-SHAP stage: fit a random-forest surrogate to the estimated CATE surface.
        # This follows the paper's Figure-6 logic; SHAP explains heterogeneity in CATE,
        # not a second causal effect.
        surrogate = RandomForestRegressor(
            n_estimators=500, min_samples_leaf=15, max_features=0.8,
            random_state=20260904, n_jobs=-1,
        )
        surrogate.fit(Xs, tau)
        explainer = shap.TreeExplainer(surrogate)
        sv = explainer.shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[0]
        importance = np.mean(np.abs(np.asarray(sv)), axis=0)
        for f, imp in sorted(zip(feature_cols, importance), key=lambda z: -z[1]):
            shap_rows.append({
                "outcome": y,
                "feature": f,
                "mean_abs_shap": float(imp),
                "rank": int(np.argsort(-importance).tolist().index(feature_cols.index(f)) + 1),
            })
        for row_i, uid in enumerate(x.user_id):
            for col_i, f in enumerate(feature_cols):
                shap_detail_rows.append({
                    "outcome": y,
                    "user_id": int(uid),
                    "feature": f,
                    "feature_value": float(X[row_i, col_i]),
                    "feature_value_scaled": float(Xs[row_i, col_i]),
                    "shap_value": float(np.asarray(sv)[row_i, col_i]),
                    "cate": float(tau[row_i]),
                })
    return (pd.DataFrame(rows), pd.DataFrame(shap_rows), pd.DataFrame(cate_rows),
            pd.DataFrame(shap_detail_rows), "\n".join(model_summaries), {
        "n_users": int(len(x)), "features": feature_cols,
    })


def main():
    users = user_features(pd.read_csv(PANEL_DIR / "selected_behavioral_panel.csv", parse_dates=["date"]))
    summary, importance, cate, detail, text, meta = causal_forest(users)
    for name, data in [("cate_results", summary), ("shap_importance", importance), ("individual_effects", cate), ("shap_detail", detail)]:
        data.to_csv(RESULT_DIR / f"causal_forest_{name}.csv", index=False)
    (RESULT_DIR / "causal_forest_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
