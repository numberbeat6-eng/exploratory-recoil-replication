# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from config import *

from profile_common import *
from statsmodels.stats.multitest import multipletests
OUTCOMES = ["is_click", "long_view", "watch_ratio"]

def build_pre_profile(tag_series, eligible_users):
    parts = []
    usecols = ["user_id", "video_id", "date", "tab", "long_view", "play_time_ms", "duration_ms"]
    for chunk in pd.read_csv(PRE_FILE, usecols=usecols, chunksize=CHUNK_SIZE):
        chunk = chunk.loc[chunk.tab.eq(1) & chunk.user_id.isin(eligible_users)].copy()
        if chunk.empty:
            continue
        tagged = explode_tags(chunk, tag_series)
        if not tagged.empty:
            parts.append(tagged.groupby(["user_id", "tag"], as_index=False)[["rec_weight", "cons_weight"]].sum())
    raw = pd.concat(parts, ignore_index=True).groupby(["user_id", "tag"], as_index=False).sum()
    profile = build_profile(raw)
    top5 = (
        profile.sort_values(["user_id", "h"], ascending=[True, False])
        .groupby("user_id", as_index=False)
        .head(5)[["user_id", "tag"]]
        .assign(familiar=1.0)
    )
    return profile, top5


def add_novel_share(chunk, tag_series, top5):
    chunk = chunk.copy()
    chunk["tag"] = chunk.video_id.map(tag_series)
    chunk = chunk.loc[chunk.tag.notna() & chunk.tag.ne("")].copy()
    if chunk.empty:
        return chunk
    chunk["_row"] = np.arange(len(chunk), dtype=np.int64)
    chunk["tags"] = chunk.tag.str.split(",")
    chunk["n_tags"] = chunk.tags.str.len()
    long = chunk[["_row", "user_id", "tags"]].explode("tags").rename(columns={"tags": "tag_id"})
    long["tag_id"] = pd.to_numeric(long.tag_id, errors="coerce")
    long = long.dropna(subset=["tag_id"])
    long["tag_id"] = long.tag_id.astype(np.int64)
    long = long.merge(top5, left_on=["user_id", "tag_id"], right_on=["user_id", "tag"], how="left")
    long["familiar"] = long.familiar.fillna(0.0)
    familiar_share = long.groupby("_row").familiar.mean()
    chunk["novel_share"] = 1.0 - chunk._row.map(familiar_share)
    return chunk.drop(columns=["tag", "tags", "n_tags", "_row"])


def load_standard_events(tag_series, top5, eligible_users):
    parts = []
    usecols = [
        "user_id", "video_id", "date", "time_ms", "tab", "is_click",
        "long_view", "play_time_ms", "duration_ms",
    ]
    dtypes = {
        "user_id": "int32", "video_id": "int64", "date": "int32", "time_ms": "int64",
        "tab": "int8", "is_click": "int8", "long_view": "int8",
        "play_time_ms": "float32", "duration_ms": "float32",
    }
    for chunk in pd.read_csv(POST_FILE, usecols=usecols, dtype=dtypes, chunksize=CHUNK_SIZE):
        chunk = chunk.loc[chunk.tab.eq(1) & chunk.user_id.isin(eligible_users)].copy()
        if chunk.empty:
            continue
        chunk = add_novel_share(chunk, tag_series, top5)
        duration = chunk.duration_ms.to_numpy(float)
        play = chunk.play_time_ms.fillna(0).to_numpy(float)
        chunk["watch_ratio"] = np.clip(
            np.divide(play, duration, out=np.zeros(len(chunk)), where=duration > 0), 0, 1
        ).astype("float32")
        parts.append(chunk[["user_id", "date", "time_ms", "is_click", "long_view", "watch_ratio", "novel_share"]])
    standard = pd.concat(parts, ignore_index=True)
    standard = standard.sort_values(["user_id", "date", "time_ms"]).reset_index(drop=True)
    return standard


def load_random_events(tag_series, profile, top5, eligible_users):
    usecols = [
        "user_id", "video_id", "date", "time_ms", "tab", "is_click",
        "long_view", "play_time_ms", "duration_ms",
    ]
    random = pd.read_csv(RAND_FILE, usecols=usecols)
    random = random.loc[random.tab.eq(1) & random.user_id.isin(eligible_users)].copy()
    random = random.sort_values(["user_id", "date", "time_ms"]).reset_index(drop=True)
    random["random_event_row"] = np.arange(len(random), dtype=np.int64)
    random["tag"] = random.video_id.map(tag_series)
    random["tags"] = random.tag.str.split(",")
    random["n_tags"] = pd.to_numeric(random.tags.str.len(), errors="coerce").astype(float)

    long = random.loc[random.tags.notna(), ["random_event_row", "user_id", "tags"]].explode("tags")
    long = long.rename(columns={"tags": "tag_id"})
    long["tag_id"] = pd.to_numeric(long.tag_id, errors="coerce")
    long = long.dropna(subset=["tag_id"])
    long["tag_id"] = long.tag_id.astype(np.int64)
    long = long.merge(
        profile[["user_id", "tag", "h"]],
        left_on=["user_id", "tag_id"], right_on=["user_id", "tag"], how="left",
    )
    long["h"] = long.h.fillna(0.0)
    sum_h = long.groupby("random_event_row").h.sum()
    random["sum_profile_weight"] = random.random_event_row.map(sum_h)
    profile_norm = profile.drop_duplicates("user_id").set_index("user_id").profile_norm
    random["profile_norm"] = pd.to_numeric(random.user_id.map(profile_norm), errors="coerce")
    random["sum_profile_weight"] = pd.to_numeric(random.sum_profile_weight, errors="coerce")
    random["random_similarity"] = np.divide(
        random.sum_profile_weight,
        np.sqrt(random.n_tags) * random.profile_norm,
        out=np.full(len(random), np.nan),
        where=(random.n_tags > 0) & (random.profile_norm > 0),
    )
    random["random_distance"] = 1.0 - random.random_similarity

    familiar = long[["random_event_row", "user_id", "tag_id"]].merge(
        top5, left_on=["user_id", "tag_id"], right_on=["user_id", "tag"], how="left"
    )
    familiar["familiar"] = familiar.familiar.fillna(0.0)
    random["random_novel_share"] = 1.0 - random.random_event_row.map(
        familiar.groupby("random_event_row").familiar.mean()
    )
    duration = random.duration_ms.to_numpy(float)
    play = random.play_time_ms.fillna(0).to_numpy(float)
    random["random_watch_ratio"] = np.clip(
        np.divide(play, duration, out=np.zeros(len(random)), where=duration > 0), 0, 1
    )
    random["random_rejected"] = 1.0 - random.is_click.astype(float)
    baseline_hhi = profile.drop_duplicates("user_id").set_index("user_id").baseline_hhi
    random["baseline_hhi"] = random.user_id.map(baseline_hhi)
    return random


def build_window(standard, random, width):
    std_groups = standard.groupby(["user_id", "date"], sort=False).indices
    selected_blocks = []
    event_blocks = []
    position_blocks = []
    event_rows = []
    event_id = 0
    event_columns = [
        "user_id", "date", "time_ms", "is_click", "long_view", "random_watch_ratio",
        "random_distance", "random_similarity", "random_novel_share", "random_rejected",
        "baseline_hhi",
    ]
    for key, group in random.groupby(["user_id", "date"], sort=False):
        std_idx = std_groups.get(key)
        if std_idx is None or len(std_idx) < 2 * width:
            continue
        std_idx = np.asarray(std_idx)
        std_times = standard.loc[std_idx, "time_ms"].to_numpy()
        group = group.sort_values("time_ms")
        positions = np.searchsorted(std_times, group.time_ms.to_numpy(), side="left")
        group_rows = group.reset_index(drop=True)
        for j, position in enumerate(positions):
            if position < width or len(std_idx) - position < width:
                continue
            left_gap = np.inf if j == 0 else position - positions[j - 1]
            right_gap = np.inf if j == len(positions) - 1 else positions[j + 1] - position
            if left_gap < 2 * width or right_gap < 2 * width:
                continue
            event = group_rows.iloc[j]
            if pd.isna(event.random_distance) or pd.isna(event.baseline_hhi):
                continue
            chosen = np.concatenate([std_idx[position - width:position], std_idx[position:position + width]])
            relpos = np.concatenate([np.arange(-width, 0), np.arange(1, width + 1)])
            selected_blocks.append(chosen)
            event_blocks.append(np.full(2 * width, event_id, dtype=np.int64))
            position_blocks.append(relpos.astype(np.int16))
            record = {"event_id": event_id}
            record.update({col: event[col] for col in event_columns})
            event_rows.append(record)
            event_id += 1

    selected = np.concatenate(selected_blocks)
    window = standard.iloc[selected].copy().reset_index(drop=True)
    window["event_id"] = np.concatenate(event_blocks)
    window["relative_position"] = np.concatenate(position_blocks)
    events = pd.DataFrame(event_rows)
    window = window.merge(events, on="event_id", suffixes=("", "_event"), validate="many_to_one")
    window["after"] = window.relative_position.gt(0).astype(float)
    assert not window[["event_id", "relative_position"]].duplicated().any()
    return window, events


def estimate(window, outcome, model_id, moderator=None):
    x = window.copy()
    x["after_novel"] = x.after * x.novel_share
    regressors = ["novel_share", "after_novel"]
    focal = "after_novel"
    if moderator:
        source, suffix = moderator
        x[f"novel_{suffix}"] = x.novel_share * x[source]
        x[f"after_{suffix}"] = x.after * x[source]
        x[f"after_novel_{suffix}"] = x.after_novel * x[source]
        regressors.extend([f"novel_{suffix}", f"after_{suffix}", f"after_novel_{suffix}"])
        focal = f"after_novel_{suffix}"
    required = [outcome, "user_id", "event_id", "relative_position", *regressors]
    x = x.dropna(subset=required).copy()
    x = x.set_index(["event_id", "relative_position"])
    clusters = pd.DataFrame({"user_id": x.user_id.astype(np.int64)}, index=x.index)
    fit = PanelOLS(
        x[outcome], x[regressors], entity_effects=True, time_effects=True, drop_absorbed=True
    ).fit(cov_type="clustered", clusters=clusters)
    row = {
        "model_id": model_id,
        "outcome": outcome,
        "moderator": moderator[0] if moderator else "none",
        "focal": focal,
        "coef": float(fit.params[focal]),
        "std_error": float(fit.std_errors[focal]),
        "t_stat": float(fit.tstats[focal]),
        "p_value": float(fit.pvalues[focal]),
        "baseline_after_novel_coef": float(fit.params["after_novel"]),
        "baseline_after_novel_p": float(fit.pvalues["after_novel"]),
        "n_obs": int(fit.nobs),
        "n_events": int(x.index.get_level_values(0).nunique()),
        "n_users": int(x.user_id.nunique()),
    }
    return row, fit, str(fit.summary)

def main():
    eligible = set(pd.read_csv(PANEL_DIR / "selected_behavioral_panel.csv", usecols=["user_id"]).user_id)
    tags = load_video_tags()
    profile, top5 = build_pre_profile(tags, eligible)
    standard = load_standard_events(tags, top5, eligible)
    random = load_random_events(tags, profile, top5, eligible)
    windows = {}
    for width in [10, 20, 50]:
        windows[width], _ = build_window(standard, random, width)
        if width == 10:
            windows[width].to_parquet(PANEL_DIR / "mechanism_event_window_w10.parquet", index=False)
    rows = []
    for width, window in windows.items():
        for y in OUTCOMES:
            row, _, _ = estimate(window, y, f"W{width}_baseline_event_did")
            row["window"] = width; row["family"] = "window_baselines"
            rows.append(row)
    for width in [5, 10]:
        window = windows[20].loc[windows[20].relative_position.abs().le(width)]
        for y in OUTCOMES:
            row, _, _ = estimate(window, y, f"W20events_subwindow{width}_baseline")
            row["window"] = width; row["family"] = "common_event_decay"
            rows.append(row)
    out = pd.DataFrame(rows)
    # Table 3 reports BH q values from all nine baseline window-outcome tests.
    family = out.loc[out.family.eq("window_baselines")].copy()
    family["fdr_q_within_family"] = multipletests(family.p_value, method="fdr_bh")[1]
    out = out.merge(family[["model_id", "outcome", "fdr_q_within_family"]], on=["model_id", "outcome"], how="left")
    idx = out.family.eq("common_event_decay")
    out.loc[idx, "fdr_q_within_family"] = multipletests(out.loc[idx, "p_value"], method="fdr_bh")[1]
    out = out.loc[~out.model_id.eq("W50_baseline_event_did")]
    out.to_csv(RESULT_DIR / "mechanism_event_results.csv", index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
