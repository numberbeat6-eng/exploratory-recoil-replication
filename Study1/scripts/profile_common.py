# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from config import *

def load_video_tags() -> pd.Series:
    """Load only the video-to-tag mapping needed by the log aggregation."""
    video = pd.read_csv(
        VIDEO_FILE,
        usecols=["video_id", "tag"],
        dtype={"video_id": "int64", "tag": "string"},
    )
    video = video.drop_duplicates("video_id", keep="first").set_index("video_id")
    return video["tag"]


def feedback_weight(frame: pd.DataFrame) -> np.ndarray:
    """Use capped watch ratio; fall back to long_view when duration is unusable."""
    duration = pd.to_numeric(frame["duration_ms"], errors="coerce").to_numpy(dtype=float)
    play = pd.to_numeric(frame["play_time_ms"], errors="coerce").fillna(0).to_numpy(dtype=float)
    ratio = np.divide(play, duration, out=np.zeros_like(play), where=duration > 0)
    ratio = np.clip(ratio, 0.0, 1.0)
    long_view = pd.to_numeric(frame["long_view"], errors="coerce").fillna(0).to_numpy(dtype=float)
    return np.where((duration > 0) & np.isfinite(ratio), ratio, long_view)


def explode_tags(frame: pd.DataFrame, tag_series: pd.Series) -> pd.DataFrame:
    """Return one row per user/date/tag with record and feedback weights."""
    frame = frame.copy()
    frame["tag"] = frame["video_id"].map(tag_series)
    frame["valid_tag"] = frame["tag"].notna() & frame["tag"].ne("")
    frame = frame.loc[frame["valid_tag"]].copy()
    if frame.empty:
        return pd.DataFrame(columns=["user_id", "date", "tag", "rec_weight", "cons_weight"])
    frame["tag"] = frame["tag"].str.split(",")
    frame["n_tags"] = frame["tag"].str.len()
    frame["rec_weight"] = 1.0 / frame["n_tags"]
    frame["cons_weight"] = feedback_weight(frame) / frame["n_tags"]
    long = frame[["user_id", "date", "tag", "rec_weight", "cons_weight"]].explode("tag")
    long["tag"] = pd.to_numeric(long["tag"], errors="coerce")
    long = long.dropna(subset=["tag"])
    long["tag"] = long["tag"].astype(int)
    return long


def process_standard(path: Path, tag_series: pd.Series, collect_profile: bool = False):
    """Aggregate standard logs into user-day totals and tag distributions."""
    totals = []
    tag_parts = []
    profile_parts = []
    for chunk in pd.read_csv(
        path,
        usecols=["user_id", "video_id", "date", "tab", "long_view", "play_time_ms", "duration_ms"],
        dtype={"user_id": "int64", "video_id": "int64", "date": "int64", "tab": "int8", "long_view": "int8"},
        chunksize=CHUNK_SIZE,
    ):
        chunk = chunk.loc[chunk["tab"].eq(1)].copy()
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["date"].astype(str), format="%Y%m%d")
        chunk["has_tag"] = chunk["video_id"].map(tag_series).notna()
        total = chunk.groupby(["user_id", "date"], as_index=False).agg(
            standard_n=("video_id", "size"), valid_tag_n=("has_tag", "sum")
        )
        totals.append(total)
        tags = explode_tags(chunk, tag_series)
        if not tags.empty:
            tag_parts.append(tags.groupby(["user_id", "date", "tag"], as_index=False).sum())
        if collect_profile:
            profile = explode_tags(chunk, tag_series)
            if not profile.empty:
                profile_parts.append(
                    profile.groupby(["user_id", "tag"], as_index=False)[["rec_weight", "cons_weight"]].sum()
                )

    totals = pd.concat(totals, ignore_index=True).groupby(["user_id", "date"], as_index=False).sum()
    tags = pd.concat(tag_parts, ignore_index=True).groupby(["user_id", "date", "tag"], as_index=False).sum()
    if collect_profile:
        profile = pd.concat(profile_parts, ignore_index=True).groupby(["user_id", "tag"], as_index=False).sum()
        return totals, tags, profile
    return totals, tags


def build_profile(profile: pd.DataFrame) -> pd.DataFrame:
    profile = profile.copy()
    profile["profile_weight"] = profile["cons_weight"]
    profile = profile.loc[profile["profile_weight"] > 0].copy()
    if profile.empty:
        raise RuntimeError("No positive pre-period profile weight was found.")
    totals = profile.groupby("user_id")["profile_weight"].sum().rename("profile_total")
    profile = profile.join(totals, on="user_id")
    profile["h"] = profile["profile_weight"] / profile["profile_total"]
    norms = profile.assign(h2=profile["h"] ** 2).groupby("user_id").agg(
        profile_norm=("h", lambda x: float(np.sqrt(np.sum(x * x)))),
        baseline_hhi=("h", lambda x: float(np.sum(x * x))),
        baseline_tags=("h", "size"),
    )
    return profile[["user_id", "tag", "h"]].merge(norms, on="user_id", how="left")

