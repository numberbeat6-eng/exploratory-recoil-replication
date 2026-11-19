"""Reproduce the Study 2 analyses from the public KuaiRec 2.0 small matrix."""
import argparse
import ast
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from figures import fig_deciles, fig_rules
from phases import clustered_multivariate_fe, phase_pairs
from sampling import METHODS, build_profiles, run_all_methods, summarize_user_values


def load_data(data):
    small = pd.read_csv(data / 'small_matrix.csv', usecols=[
        'user_id', 'video_id', 'watch_ratio', 'video_duration', 'play_duration', 'timestamp',
    ])
    users = np.sort(small.user_id.unique())
    items = np.sort(small.video_id.unique())
    if (len(users), len(items), len(small)) != (1411, 3327, 4676570):
        raise ValueError('Expected the KuaiRec 2.0 small matrix: 1,411 users, 3,327 videos, and 4,676,570 rows.')
    u = np.searchsorted(users, small.user_id)
    v = np.searchsorted(items, small.video_id)
    small['u_idx'], small['v_idx'] = u, v
    small['index_raw'] = np.arange(len(small))
    label_lists = pd.read_csv(data / 'item_categories.csv').set_index('video_id').feat.map(ast.literal_eval)
    label_lists = label_lists.loc[items]
    item_labels = [set(labels) for labels in label_lists]
    category_codes, cats = np.unique(
        [labels[0] if labels else -999 for labels in label_lists], return_inverse=True
    )
    duration = small.groupby('video_id').video_duration.median().reindex(items).to_numpy()
    truth = np.full((len(users), len(items)), np.nan)
    truth[u, v] = small.watch_ratio.to_numpy()
    if np.isfinite(truth).sum() != len(small):
        raise ValueError('The small matrix contains duplicated user–video pairs or invalid watch ratios.')
    flags = np.where(small.video_duration <= 7000,
                     small.play_duration >= small.video_duration,
                     small.play_duration > 7000).astype(float)
    effective = np.full(truth.shape, np.nan)
    effective[u, v] = flags
    return small, users, items, cats, category_codes, item_labels, duration, truth, flags, effective


def profile_gap_out_of_sample(complete, cats):
    """Original split-sample profile check underlying Table 5."""
    rng = np.random.default_rng(0)
    train = rng.random(complete.shape[1]) < 0.5
    test = ~train
    category_masks = [cats == c for c in range(int(cats.max()) + 1)]
    training_rates = np.full((len(complete), len(category_masks)), np.nan)
    for c, mask in enumerate(category_masks):
        if mask.sum() >= 5 and (mask & train).sum() >= 3:
            training_rates[:, c] = np.nanmean(complete[:, mask & train], axis=1)

    gaps = []
    for u, values in enumerate(complete):
        rates = training_rates[u]
        usable = np.flatnonzero(np.isfinite(rates))
        if len(usable) < 5:
            continue
        familiar_cats = usable[np.argsort(-rates[usable])[:3]]
        familiar = np.isin(cats, familiar_cats)
        familiar_rate = np.nanmean(values[familiar & test])
        unfamiliar_rate = np.nanmean(values[~familiar & test])
        if np.isfinite(familiar_rate) and np.isfinite(unfamiliar_rate):
            gaps.append(familiar_rate - unfamiliar_rate)
    return float(np.mean(gaps))


def completion_summaries(truth, users, items, cats, profile, output):
    """Table 5, Table B1, Figure 6, and the random-batch results in Section 4.2."""
    _, familiar_cats, complete, valid, _ = profile
    totals = np.nansum(complete, axis=0)
    counts = valid.sum(axis=0)
    user_rows, batch_rows, five_video_rows = [], [], []
    doses = (1, 2, 3, 5, 10, 20, 50)
    for u, user_id in enumerate(users):
        familiar = np.isin(cats, familiar_cats[u]) & valid[u]
        unfamiliar = (~np.isin(cats, familiar_cats[u])) & valid[u]
        familiar_values, unfamiliar_values = complete[u, familiar], complete[u, unfamiliar]
        n, k = len(unfamiliar_values), int(unfamiliar_values.sum())
        familiar_rate, unfamiliar_rate = familiar_values.mean(), unfamiliar_values.mean()
        crowd_rates = (totals[unfamiliar] - unfamiliar_values) / np.maximum(counts[unfamiliar] - 1, 1)
        deciles = np.array_split(np.argsort(-crowd_rates, kind='stable'), 10)
        user_rows.append(dict(user_id=int(user_id), fam=familiar_rate, out=unfamiliar_rate,
                              n_out=n, n_pos=k,
                              **{f'dec{i}': unfamiliar_values[part].mean()
                                 for i, part in enumerate(deciles, 1)}))

        rng = np.random.default_rng(42 + u)
        draws = np.array([unfamiliar_values[rng.choice(n, 5, replace=False)].sum()
                          for _ in range(500)])
        five_video_rows.append(dict(p_no_completion=float(np.mean(draws == 0)),
                                    p_majority_unfinished=float(np.mean(draws <= 2)),
                                    p_all_completed=hypergeom.pmf(5, n, k, 5),
                                    expected_posterior=float(np.mean((3.2 + draws) / 15))))

        rng = np.random.default_rng(100 + u)
        for dose in doses:
            draws = np.array([unfamiliar_values[rng.choice(n, dose, replace=False)].sum()
                              for _ in range(200)])
            expected_posterior = np.mean((3.2 + draws) / (10 + dose))
            batch_rows.append(dict(dose=dose, expected_completions=draws.mean(),
                                   no_completion=np.mean(draws == 0), posterior=expected_posterior,
                                   below_familiar=float(expected_posterior < familiar_rate)))

    deciles = pd.DataFrame(user_rows)
    deciles.to_csv(output / 'figure_6_data.csv', index=False)
    fig_deciles(deciles, output)
    table_5 = dict(users=len(users), videos=len(items), observed_share=float(valid.mean()),
                   familiar=float(deciles.fam.mean()), unfamiliar=float(deciles.out.mean()),
                   in_sample_gap=float((deciles.fam - deciles.out).mean()),
                   out_of_sample_gap=profile_gap_out_of_sample(complete, cats),
                   completable_unfamiliar_mean=float(deciles.n_pos.mean()),
                   completable_unfamiliar_min=int(deciles.n_pos.min()),
                   completable_unfamiliar_share=float((deciles.n_pos / deciles.n_out).mean()))
    (output / 'table_5.json').write_text(json.dumps(table_5, indent=2), encoding='utf-8')

    summary = pd.DataFrame(batch_rows).groupby('dose', as_index=False).mean()
    summary.to_csv(output / 'table_B1.csv', index=False)
    five_video = pd.DataFrame(five_video_rows)
    rng = np.random.default_rng(42)
    stats = {}
    for col in ['p_no_completion', 'p_majority_unfinished', 'expected_posterior']:
        values = five_video[col].to_numpy()
        samples = rng.integers(0, len(values), size=(2000, len(values)))
        ci = np.percentile(values[samples].mean(axis=1), [2.5, 97.5])
        stats[col] = dict(mean=float(values.mean()), ci_low=float(ci[0]), ci_high=float(ci[1]))
    stats['p_all_completed'] = float(five_video.p_all_completed.mean())
    (output / 'five_video_batch.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')


def sequential_results(small, users, items, item_labels, flags, category_codes, profile, label, output):
    pairs = phase_pairs(small, users, items, profile[1], item_labels, flags, category_codes)
    available = ['cur_out_quality', 'cur_fam_quality']
    outcomes = ['next_out_acceptance']
    baseline = clustered_multivariate_fe(pairs, available[:1], outcomes)
    adjusted = clustered_multivariate_fe(pairs, available, outcomes)
    quintiles = (pairs.assign(quintile=pd.qcut(pairs.cur_out_quality, 5, labels=False) + 1)
                 .groupby('quintile')[['cur_out_quality', 'next_out_acceptance']].mean())
    summary = dict(pairs=len(pairs), users=int(pairs.user_id.nunique()),
                   pooled_correlation=float(pairs.cur_out_quality.corr(pairs.next_out_acceptance)),
                   baseline=baseline['coefficients'], adjusted=adjusted['coefficients'],
                   quintiles=quintiles.reset_index().to_dict('records'))
    pairs.to_csv(output / f'phase_pairs_{label}.csv', index=False)
    (output / f'temporal_{label}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='KuaiRec 2.0 data folder')
    parser.add_argument('--output', required=True, type=Path, help='Directory for Study 2 results')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    small, users, items, cats, codes, labels, duration, truth, flags, effective = load_data(args.data)

    print('Completion: profiles and sequential phases', flush=True)
    profile = build_profiles(truth, users, cats)
    sequential_results(small, users, items, labels, (small.watch_ratio >= 1).to_numpy(float),
                       codes, profile, 'completion', output)
    print('Seven-second effective play: profiles and sequential phases', flush=True)
    effective_profile = build_profiles(effective, users, cats)
    sequential_results(small, users, items, labels, flags, codes, effective_profile, 'effective_play', output)
    del small, flags

    print('Table 5, Table B1, and Figure 6', flush=True)
    completion_summaries(truth, users, items, cats, profile, output)

    print('Table 6 and Figure 7', flush=True)
    rules = run_all_methods(truth, users, cats, duration, profile[1], profile[0], *profile[2:])
    rules.to_csv(output / 'figure_7_data.csv', index=False)
    pd.DataFrame(summarize_user_values(rules, METHODS, completion_protocol=True)).to_csv(
        output / 'table_6.csv', index=False
    )
    fig_rules(rules, output)

    print('Table B2: exact row-wise seven-second effective play', flush=True)
    rules = run_all_methods(effective, users, cats, duration,
                            effective_profile[1], effective_profile[0], *effective_profile[2:])
    rules['unfamiliar_pool_mean'] = rules.fam_ref - rules.gap
    pd.DataFrame(summarize_user_values(rules, ['fam_ref', 'unfamiliar_pool_mean', *METHODS])).to_csv(
        output / 'table_B2.csv', index=False
    )


if __name__ == '__main__':
    main()
