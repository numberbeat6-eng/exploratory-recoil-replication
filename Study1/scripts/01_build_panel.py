# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from config import *

from profile_common import *

def aggregate_log(path, tags, top5):
    parts=[]; totals=[]
    for chunk in pd.read_csv(path,usecols=["user_id","video_id","date","tab","is_click","long_view","play_time_ms","duration_ms"],chunksize=CHUNK_SIZE):
        chunk=chunk.loc[chunk.tab.eq(1)].copy()
        if chunk.empty: continue
        chunk["date"]=pd.to_datetime(chunk.date.astype(str),format="%Y%m%d")
        chunk["tag"]=chunk.video_id.map(tags)
        totals.append(chunk.groupby(["user_id","date"],as_index=False).agg(standard_n=("video_id","size"),valid_tag_n=("tag",lambda x:x.notna().sum())))
        chunk=chunk.loc[chunk.tag.notna() & chunk.tag.ne("")].copy()
        chunk["tag"]=chunk.tag.str.split(","); chunk["n_tags"]=chunk.tag.str.len()
        duration=chunk.duration_ms.to_numpy(float); play=chunk.play_time_ms.fillna(0).to_numpy(float)
        chunk["watch_ratio"]=np.clip(np.divide(play,duration,out=np.zeros(len(chunk)),where=duration>0),0,1)
        long=chunk[["user_id","date","tag","n_tags","is_click","long_view","watch_ratio"]].explode("tag")
        long["tag"]=pd.to_numeric(long.tag,errors="coerce"); long=long.dropna(subset=["tag"]); long["tag"]=long.tag.astype(int)
        long=long.merge(top5.assign(familiar=1),on=["user_id","tag"],how="left"); long["familiar"]=long.familiar.fillna(0)
        long["w"]=1/long.n_tags
        long["fam_w"]=long.w*long.familiar; long["nov_w"]=long.w*(1-long.familiar)
        for beh in ["is_click","long_view","watch_ratio"]:
            long[f"fam_{beh}"]=long.fam_w*long[beh]
            long[f"nov_{beh}"]=long.nov_w*long[beh]
        aggcols=["fam_w","nov_w","fam_is_click","nov_is_click","fam_long_view","nov_long_view","fam_watch_ratio","nov_watch_ratio"]
        parts.append(long.groupby(["user_id","date"],as_index=False)[aggcols].sum())
    a=pd.concat(parts,ignore_index=True).groupby(["user_id","date"],as_index=False).sum()
    t=pd.concat(totals,ignore_index=True).groupby(["user_id","date"],as_index=False).sum()
    a=a.merge(t,on=["user_id","date"])
    for beh in ["is_click","long_view","watch_ratio"]:
        a[f"fam_rate_{beh}"]=a[f"fam_{beh}"]/a.fam_w
        a[f"nov_rate_{beh}"]=a[f"nov_{beh}"]/a.nov_w
        a[f"preference_gap_{beh}"]=a[f"fam_rate_{beh}"]-a[f"nov_rate_{beh}"]
    return a

def calendar_shift(d, outcome, days, name):
    z=d[["user_id","date",outcome]].copy(); z["date"]=z.date-pd.Timedelta(days=days); z=z.rename(columns={outcome:name})
    return d.merge(z,on=["user_id","date"],how="left")

def rebuild_behavioral_panel():
    tags = load_video_tags()
    pre_total, _, raw = process_standard(PRE_FILE, tags, collect_profile=True)
    profile = build_profile(raw)
    top5 = profile.sort_values(["user_id", "h"], ascending=[True, False]).groupby("user_id").head(5)[["user_id", "tag"]]
    pre = aggregate_log(PRE_FILE, tags, top5)
    post = aggregate_log(POST_FILE, tags, top5)
    pre["post"] = 0
    post["post"] = 1
    # The lag is built before filtering user-days on activity or usable tags.
    post_total, _ = process_standard(POST_FILE, tags)
    lag = pd.concat([pre_total, post_total], ignore_index=True)[["user_id", "date", "standard_n"]]
    lag["date"] += pd.Timedelta(days=1)
    lag = lag.rename(columns={"standard_n": "lag_activity"})
    d = pd.concat([pre, post], ignore_index=True)
    d = d.loc[(d.standard_n >= 10) & (d.valid_tag_n / d.standard_n >= .9)].copy()
    d = d.merge(lag, on=["user_id", "date"], how="left", validate="one_to_one")
    d["lag_activity"] = d.lag_activity.fillna(0)
    hhi = profile.drop_duplicates("user_id").set_index("user_id").baseline_hhi
    d["baseline_hhi_post"] = d.user_id.map(hhi) * d.post
    d["baseline_activity_post"] = d.user_id.map(pre_total.groupby("user_id").standard_n.mean()) * d.post
    both = d.groupby("user_id").post.nunique()
    d = d.loc[d.user_id.isin(both[both.eq(2)].index)]
    return d.sort_values(["user_id", "date"]).reset_index(drop=True)


def main():
    d=rebuild_behavioral_panel()
    rand=pd.read_csv(RAND_FILE,usecols=["user_id","date","tab"]); rand=rand.loc[rand.tab.eq(1)].copy(); rand["date"]=pd.to_datetime(rand.date.astype(str),format="%Y%m%d")
    rand=rand.groupby(["user_id","date"],as_index=False).size().rename(columns={"size":"all_random_n"})
    d=d.merge(rand,on=["user_id","date"],how="left"); d["all_random_n"]=d.all_random_n.fillna(0)
    d["opp_n"] = d["standard_n"] + d["all_random_n"]
    d["log_opp"] = np.log1p(d["opp_n"])
    d["log_random"]=np.log1p(d.all_random_n); sd=d.loc[d.all_random_n>0,"log_random"].std(); d["dose_sd"]=d.log_random/sd
    d["log_opp_sq"]=d.log_opp**2
    outcomes=["nov_rate_is_click","nov_rate_long_view","nov_rate_watch_ratio"]
    for y in outcomes:
        d=calendar_shift(d,y,-1,f"previous_{y}")
    d.to_csv(PANEL_DIR / "selected_behavioral_panel.csv", index=False, encoding="utf-8-sig")
    summary = {"user_days": len(d), "users": int(d.user_id.nunique()), "treatment_sd_log_units": sd}
    (RESULT_DIR / "selected_behavioral_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
