from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from config import *
HERE = FIGURE_DIR
RESULTS = RESULT_DIR
PANEL = PANEL_DIR / "selected_behavioral_panel.csv"
RANDOM_LOG = RAND_FILE

OUTCOMES = [
    "nov_rate_is_click",
    "nov_rate_long_view",
    "nov_rate_watch_ratio",
    "preference_gap_is_click",
]
OUTCOME_LABELS = {
    "nov_rate_is_click": "Out-of-profile effective play",
    "nov_rate_long_view": "Out-of-profile long viewing",
    "nov_rate_watch_ratio": "Out-of-profile watch ratio",
    "preference_gap_is_click": "Familiar-content effective-play advantage",
    "is_click": "Effective play",
    "long_view": "Long viewing",
    "watch_ratio": "Watch ratio",
}
FEATURE_LABELS = {
    "pre_nov_rate_is_click": "Out-of-profile effective play",
    "pre_nov_rate_long_view": "Out-of-profile long viewing",
    "pre_nov_rate_watch_ratio": "Out-of-profile watch ratio",
    "pre_preference_gap_is_click": "Familiar-content effective-play advantage",
    "pre_preference_gap_long_view": "Familiar-content long-view advantage",
    "pre_preference_gap_watch_ratio": "Familiar-content watch-ratio advantage",
    "pre_standard_n": "Standard recommendations",
    "pre_valid_tag_n": "Tagged standard recommendations",
    "pre_log_opp": "Log recommendation opportunities",
    "pre_lag_activity": "Lagged standard-stream activity",
    "pre_days": "Pretreatment days observed",
}
BLUE = "#315B7D"
ORANGE = "#B36A2E"
GREEN = "#487A5B"
RED = "#9A4D4D"
GREY = "#777777"


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    })


def finish(fig, filename):
    fig.savefig(HERE / filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_parallel_trends():
    data = pd.read_csv(RESULTS / "parallel_trends_event_study.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), sharex=True)
    for ax, outcome in zip(axes.flat, OUTCOMES):
        z = data.loc[data.outcome.eq(outcome)].sort_values("event_day")
        for period, color, label in [("pre", BLUE, "Pretreatment"), ("post", ORANGE, "Post-period")]:
            q = z.loc[z.period.eq(period)]
            ax.errorbar(q.event_day, q.coef, yerr=1.96 * q.std_error, fmt="o", ms=3,
                        color=color, ecolor=color, elinewidth=0.8, capsize=1.8, label=label)
        ax.scatter([-1], [0], marker="D", s=18, color="#222222", label="Reference date", zorder=3)
        ax.axhline(0, color=GREY, linewidth=0.7)
        ax.axvline(-0.5, color="#999999", linestyle="--", linewidth=0.7)
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.set_ylabel("Date-by-dose coefficient")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axes[1, 0].set_xlabel("Days relative to random-exposure start")
    axes[1, 1].set_xlabel("Days relative to random-exposure start")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    finish(fig, "parallel_trends.pdf")


def plot_preperiod_quartiles():
    panel = pd.read_csv(PANEL, parse_dates=["date"])
    users = pd.Index(panel.user_id.unique(), name="user_id")
    random = pd.read_csv(RANDOM_LOG, usecols=["user_id", "tab"])
    total = random.loc[random.tab.eq(1)].groupby("user_id").size().reindex(users, fill_value=0)
    groups = total.rank(method="first").to_frame("rank")
    groups["dose_quartile"] = pd.qcut(groups["rank"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    pre = panel.loc[panel.date.lt(pd.Timestamp("2022-04-22"))].merge(
        groups[["dose_quartile"]], left_on="user_id", right_index=True, how="left"
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), sharex=True)
    colours = dict(zip(["Q1", "Q2", "Q3", "Q4"], [BLUE, GREEN, "#A97800", RED]))
    for ax, outcome in zip(axes.flat, OUTCOMES):
        means = pre.groupby(["date", "dose_quartile"], observed=True)[outcome].mean().reset_index()
        first = means.date.min()
        base = means.loc[means.date.eq(first)].set_index("dose_quartile")[outcome]
        means["change"] = means[outcome] - means.dose_quartile.map(base).astype(float)
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            z = means.loc[means.dose_quartile.eq(q)]
            ax.plot(z.date, z.change, marker="o", ms=2.2, linewidth=0.9, color=colours[q], label=q)
        ax.axhline(0, color=GREY, linewidth=0.7)
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.set_ylabel("Change from April 8")
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=35)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               title="Future exposure-dose quartile")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    finish(fig, "preperiod_quartiles.pdf")


def plot_event_attenuation():
    data = pd.read_csv(RESULTS / "mechanism_event_results.csv")
    ids = [
        "W20events_subwindow5_baseline",
        "W20events_subwindow10_baseline",
        "W20_baseline_event_did",
    ]
    z = data.loc[data.model_id.isin(ids) & data.moderator.eq("none")].copy()
    order = {ids[0]: 5, ids[1]: 10, ids[2]: 20}
    z["distance"] = z.model_id.map(order)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
    for ax, outcome, color in zip(axes, ["is_click", "long_view", "watch_ratio"], [BLUE, GREEN, RED]):
        q = z.loc[z.outcome.eq(outcome)].sort_values("distance")
        ax.errorbar(q.distance, q.coef, yerr=1.96 * q.std_error, fmt="o-", ms=4,
                    color=color, ecolor=color, linewidth=1.0, capsize=2.5)
        ax.axhline(0, color=GREY, linewidth=0.7)
        ax.set_xticks([5, 10, 20])
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.set_xlabel("Recommendations on each side")
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.5)
    axes[0].set_ylabel("Post-insertion change in\nout-of-profile acceptance gradient")
    fig.tight_layout()
    finish(fig, "event_window_attenuation.pdf")


def plot_cate_distribution():
    data = pd.read_csv(RESULTS / "causal_forest_individual_effects.csv")
    values = [data.loc[data.outcome.eq(o), "cate"].to_numpy() for o in OUTCOMES]
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    boxes = ax.boxplot(values, vert=True, patch_artist=True, showfliers=False,
                       whis=(2.5, 97.5), widths=0.58,
                       medianprops={"color": "#7A1E1E", "linewidth": 1.2})
    for box in boxes["boxes"]:
        box.set_facecolor("#AFC7DA")
        box.set_edgecolor(BLUE)
    ax.axhline(0, color=GREY, linewidth=0.7)
    ax.set_xticks(range(1, 5), ["Effective\nplay", "Long\nviewing", "Watch\nratio", "Familiar-content\nadvantage"])
    ax.set_ylabel("Estimated response to a one-unit\nincrease in mean exposure")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.5)
    fig.tight_layout()
    finish(fig, "causal_forest_distribution.pdf")


def plot_baseline_heterogeneity():
    data = pd.read_csv(RESULTS / "readi_style_heterogeneity_bootstrap.csv")
    data = data.loc[data.group.isin(["Q1", "Q2", "Q3", "Q4"])].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharex=True)
    for ax, outcome in zip(axes.flat, OUTCOMES):
        q = data.loc[data.outcome.eq(outcome)].set_index("group").loc[["Q1", "Q2", "Q3", "Q4"]]
        x = np.arange(4)
        ax.errorbar(x, q.cate_mean, yerr=[q.cate_mean - q.ci_low, q.ci_high - q.cate_mean],
                    fmt="o-", color=BLUE, ecolor=GREY, capsize=2.5, linewidth=1.0, ms=4)
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.set_ylabel("Mean estimated response")
        ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.5)
    axes[1, 0].set_xlabel("Pretreatment familiar-content advantage")
    axes[1, 1].set_xlabel("Pretreatment familiar-content advantage")
    fig.tight_layout()
    finish(fig, "baseline_heterogeneity.pdf")


def plot_shap_importance():
    data = pd.read_csv(RESULTS / "causal_forest_shap_importance.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.3))
    for ax, outcome in zip(axes.flat, OUTCOMES):
        q = data.loc[data.outcome.eq(outcome)].nsmallest(6, "rank").sort_values("mean_abs_shap")
        ax.barh([FEATURE_LABELS.get(f, f.replace("pre_", "")) for f in q.feature],
                q.mean_abs_shap, color=BLUE)
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.set_xlabel("Mean absolute SHAP value")
        ax.grid(axis="x", color="#E0E0E0", linewidth=0.5)
    fig.tight_layout(h_pad=1.2, w_pad=1.4)
    finish(fig, "shap_importance.pdf")


def plot_shap_detail():
    data = pd.read_csv(RESULTS / "causal_forest_shap_detail.csv")
    z = data.loc[data.outcome.eq("nov_rate_is_click")].copy()
    importance = z.groupby("feature").shap_value.apply(lambda x: np.mean(np.abs(x)))
    features = list(reversed(importance.nlargest(6).index.tolist()))
    fig = plt.figure(figsize=(7.4, 7.0))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.45, 1, 1], hspace=0.60, wspace=0.48)
    ax = fig.add_subplot(grid[0, :])
    rng = np.random.default_rng(20260904)
    for y, feature in enumerate(features):
        q = z.loc[z.feature.eq(feature)]
        vals = q.feature_value_scaled.to_numpy(float)
        lo, hi = np.nanpercentile(vals, [5, 95])
        colours = np.clip((vals - lo) / (hi - lo + 1e-12), 0, 1)
        ax.scatter(q.shap_value, y + rng.uniform(-0.15, 0.15, len(q)), c=colours,
                   cmap="coolwarm", s=5, alpha=0.65, linewidths=0, rasterized=True)
    ax.axvline(0, color=GREY, linewidth=0.7)
    ax.set_yticks(range(len(features)), [FEATURE_LABELS.get(f, f) for f in features])
    ax.set_xlabel("SHAP contribution to predicted effective-play response")
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.5)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax, pad=0.012, fraction=0.025)
    bar.set_ticks([0.05, 0.95], labels=["Low", "High"])
    bar.set_label("Feature value")
    for j, feature in enumerate(features):
        ad = fig.add_subplot(grid[1 + j // 3, j % 3])
        q = z.loc[z.feature.eq(feature)]
        ad.scatter(q.feature_value, q.shap_value, c=q.feature_value_scaled,
                   cmap="coolwarm", s=5, alpha=0.55, linewidths=0, rasterized=True)
        ad.axhline(0, color=GREY, linewidth=0.6)
        ad.set_xlabel(FEATURE_LABELS.get(feature, feature), fontsize=7.3)
        ad.set_ylabel("SHAP value", fontsize=7.3)
        ad.grid(color="#E5E5E5", linewidth=0.4)
    fig.tight_layout()
    finish(fig, "shap_effective_play.pdf")


def main():
    style()
    plot_parallel_trends()
    plot_preperiod_quartiles()
    plot_event_attenuation()
    plot_cate_distribution()
    plot_baseline_heterogeneity()
    plot_shap_importance()
    plot_shap_detail()


if __name__ == "__main__":
    main()
