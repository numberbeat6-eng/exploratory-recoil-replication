"""Compare inserted videos with standard recommendations in the primary event windows."""
import json

import numpy as np
import pandas as pd

from config import PANEL_DIR, RESULT_DIR

PANEL = PANEL_DIR / "mechanism_event_window_w10.parquet"


def main():
    d = pd.read_parquet(PANEL)
    ev = d.drop_duplicates('event_id')
    std = d[d.relative_position != 0].copy()
    res = {}

    # ---- quality of the insertion versus the standard stream in the same windows
    res['n_events'] = int(len(ev))
    res['n_users'] = int(ev.user_id.nunique())
    res['insert_effective_play'] = float(ev.is_click_event.mean())
    res['insert_long_view'] = float(ev.long_view_event.mean())
    res['insert_watch_ratio'] = float(ev.random_watch_ratio.mean())
    res['std_effective_play'] = float(std.is_click.mean())
    res['std_long_view'] = float(std.long_view.mean())
    res['std_watch_ratio'] = float(std.watch_ratio.mean())
    fam = std[std.novel_share == 0]
    out = std[std.novel_share == 1]
    res['std_familiar_effective_play'] = float(fam.is_click.mean())
    res['std_outside_effective_play'] = float(out.is_click.mean())
    res['std_familiar_long_view'] = float(fam.long_view.mean())
    res['std_outside_long_view'] = float(out.long_view.mean())
    res['insert_outside_share'] = float((ev.random_novel_share == 1).mean())
    res['insert_rejected_share'] = float(ev.random_rejected.mean())

    # ---- user-level paired comparison (insertion vs the user's own standard recs)
    u = pd.DataFrame({
        'ins': ev.groupby('user_id').is_click_event.mean(),
        'std': std.groupby('user_id').is_click.mean()}).dropna()
    diff = u['ins'] - u['std']
    res['user_level_diff_effective_play'] = float(diff.mean())
    res['user_level_diff_se'] = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    res['user_level_n'] = int(len(diff))

    with (RESULT_DIR / "s1_insertion_quality.json").open("w", encoding="utf-8") as handle:
        json.dump(res, handle, indent=2)


if __name__ == "__main__":
    main()
