"""Familiar profiles and the eight counterfactual selection rules."""
import copy
import numpy as np
import pandas as pd

N_SIM, DOSE = 200, 5
METHODS = [
    'random', 'duration_matched', 'category_adjacent', 'category_screened',
    'collaborative_filtering', 'pop_plus_screened', 'popularity', 'quality_matched_oracle',
]


def build_profiles(truth, users, cats, n_familiar=3, min_cat_items=5):
    n_users, n_items = truth.shape
    ncat = int(cats.max()) + 1
    valid = np.isfinite(truth)
    complete = (truth >= 1).astype(float)
    complete[~valid] = np.nan
    cat_accept = np.full((n_users, ncat), np.nan)
    cat_count = np.zeros(ncat, dtype=int)
    for c in range(ncat):
        mask = (cats == c)
        if mask.sum() >= min_cat_items:
            cat_accept[:, c] = np.nanmean(complete[:, mask], axis=1)
            cat_count[c] = mask.sum()
    familiar_cats = np.full((n_users, n_familiar), -1, dtype=int)
    for u in range(n_users):
        vc = np.where(np.isfinite(cat_accept[u]) & (cat_count >= min_cat_items))[0]
        if len(vc) >= n_familiar + 2:
            familiar_cats[u] = vc[np.argsort(-cat_accept[u, vc])[:n_familiar]]
    return cat_accept, familiar_cats, complete, valid, cat_count


def duration_fixed(rng,pools,probabilities):
    if any(p>0 and len(pool)<DOSE for p,pool in zip(probabilities,pools)):
        raise ValueError('Duration stratum too small for distinct sampling')
    bins=rng.choice(len(probabilities),size=(N_SIM,DOSE),p=probabilities)
    selections=[]
    for batch in bins:
        positions=[]
        for b,pool in enumerate(pools):
            count=int((batch==b).sum())
            if count:
                positions.extend(rng.choice(pool,count,replace=False))
        assert len(positions)==DOSE and len(set(positions))==DOSE
        selections.append(positions)
    return np.asarray(selections)


def run_all_methods(truth, users, cats, duration, familiar_cats, cat_accept, complete, valid, cat_count):
    n_users, n_items = truth.shape
    ncat = int(cats.max()) + 1
    valid_all = np.isfinite(truth)
    global_complete = (truth >= 1).astype(float)
    global_complete[~valid_all] = np.nan

    # Category correlation matrix
    cat_vectors = np.full((n_users, ncat), np.nan)
    for c in range(ncat):
        if cat_count[c] >= 5:
            mask = (cats == c)
            cat_vectors[:, c] = np.nanmean(complete[:, mask], axis=1)
    cat_vectors_filled = np.nan_to_num(cat_vectors)
    # Constant columns have undefined correlations and are excluded by the
    # category-selection conditions below.
    with np.errstate(invalid='ignore', divide='ignore'):
        cat_corr = np.corrcoef(cat_vectors_filled.T)

    n_neighbours = 50

    results = []
    for u in range(n_users):
        if u % 200 == 0:
            print(f'Selection rules: {u}/{n_users} users', flush=True)
        fc = familiar_cats[u]
        if fc[0] < 0:
            continue
        fam_mask = np.zeros(n_items, dtype=bool)
        out_mask = np.zeros(n_items, dtype=bool)
        for c in range(ncat):
            if c in fc:
                fam_mask |= (cats == c)
            else:
                out_mask |= (cats == c)
        fam_valid = fam_mask & valid[u]
        out_valid = out_mask & valid[u]
        if fam_valid.sum() < 10 or out_valid.sum() < 20:
            continue
        fam_cv = (truth[u, fam_valid] >= 1).astype(float)
        out_cv = (truth[u, out_valid] >= 1).astype(float)
        out_idx = np.where(out_valid)[0]
        fam_ref = np.mean(fam_cv)
        gap = fam_ref - np.mean(out_cv)
        n_out = len(out_idx)
        rng = np.random.default_rng(42 + u)
        n_sim = 200
        dose = 5

        # 1. Random
        rand_rates = [np.mean(out_cv[rng.choice(n_out, dose, replace=False)]) for _ in range(n_sim)]

        # 2. Duration-matched
        dur_bo = np.digitize(duration[out_valid], [10000, 20000, 40000])
        dur_bf = np.digitize(duration[fam_valid], [10000, 20000, 40000])
        fdd = np.bincount(dur_bf, minlength=5); fdd = fdd / max(fdd.sum(), 1)
        fixed_rng = np.random.default_rng()
        fixed_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        pools = [np.where(dur_bo == b)[0] for b in range(len(fdd))]
        positions = duration_fixed(fixed_rng, pools, fdd)
        dur_rate = float(out_cv[positions].mean())
        # Consume the historical duration draws to preserve the random streams
        # of the downstream rules when reproducing the reported results.
        for _ in range(n_sim):
            for pool, fraction in zip(pools, fdd):
                if fraction > 0 and len(pool):
                    count = min(max(1, int(round(fraction * dose))), len(pool))
                    rng.choice(pool, count, replace=False)

        # 3. Category-screened (4th-6th best)
        uca = cat_accept[u]
        elig = np.where(np.isfinite(uca) & (uca > 0) & np.array([c not in fc for c in range(ncat)]))[0]
        if len(elig) >= 3:
            te = elig[np.argsort(-uca[elig])[:3]]
            sp = np.zeros(n_items, dtype=bool)
            for c in te: sp |= (cats == c)
            sv = sp & out_valid
            si = np.where(sv)[0]
            if len(si) >= 5:
                sr = [np.mean((truth[u, si[rng.choice(len(si), dose, replace=False)]] >= 1).astype(float)) for _ in range(n_sim)]
                scr_rate = np.mean(sr)
            else: scr_rate = np.nan
        else: scr_rate = np.nan

        # Leave-one-out popularity: completion rate among all OTHER users,
        # so the target user's own response never enters the ranking.
        others = np.ones(n_users, dtype=bool); others[u] = False
        loo_pop = np.nanmean(global_complete[others][:, out_idx], axis=0)

        # 4. Popularity-based (top-5 outside videos by LOO popularity)
        tp = np.argsort(-loo_pop)[:dose]
        pop_rate = np.mean(out_cv[tp])

        # 5. Popularity + screened (hybrid)
        if len(elig) >= 3:
            te = elig[np.argsort(-uca[elig])[:3]]
            sp = np.zeros(n_items, dtype=bool)
            for c in te: sp |= (cats == c)
            # Positions within the outside pool (index into out_cv / loo_pop).
            # Fix: the earlier version ranked item indices but then indexed
            # out_cv with positions of the sub-pool, mismatching videos.
            pos = np.where(sp[out_idx])[0]
            if len(pos) >= 5:
                th = pos[np.argsort(-loo_pop[pos])[:dose]]
                hyb_rate = np.mean(out_cv[th])
            else: hyb_rate = np.nan
        else: hyb_rate = np.nan

        # 6. Category-adjacent (correlated with familiar)
        fcs = np.zeros(ncat)
        for fcc in fc:
            if fcc >= 0:
                cv = np.abs(cat_corr[fcc])
                for c in range(ncat):
                    if c not in fc and cat_count[c] >= 5:
                        fcs[c] += cv[c]
        adj_elig = np.where((fcs > 0) & np.array([c not in fc for c in range(ncat)]) & (cat_count >= 5))[0]
        if len(adj_elig) >= 3:
            ta = adj_elig[np.argsort(-fcs[adj_elig])[:3]]
            ap = np.zeros(n_items, dtype=bool)
            for c in ta: ap |= (cats == c)
            # Fix: sample positions inside the adjacent sub-pool, not the first
            # len(ai) positions of the whole outside pool.
            ai = np.where(ap[out_idx])[0]
            if len(ai) >= 5:
                ar = [np.mean(out_cv[ai[rng.choice(len(ai), dose, replace=False)]]) for _ in range(n_sim)]
                adj_rate = np.mean(ar)
            else: adj_rate = np.nan
        else: adj_rate = np.nan

        # 7. User-based collaborative filtering.
        # Similarity uses only the user's familiar-category responses (what a
        # platform would observe before exploring), so the outside outcomes
        # being evaluated never enter the similarity. Score each outside video
        # by its completion rate among the 50 most similar other users.
        fi = np.where(fam_valid)[0]
        focal = global_complete[u, fi]
        candidates = global_complete[others][:, fi]
        a = np.nan_to_num(focal - np.nanmean(focal))
        B = np.nan_to_num(candidates - np.nanmean(candidates, axis=1, keepdims=True))
        sim = B @ a / (np.linalg.norm(B, axis=1) * np.linalg.norm(a) + 1e-12)
        nb = np.argsort(-sim)[:n_neighbours]
        cf_score = np.nanmean(global_complete[others][nb][:, out_idx], axis=0)
        tc = np.argsort(-cf_score)[:dose]
        cf_rate = np.mean(out_cv[tc])

        # 8. Oracle
        pos_idx = np.where(out_cv >= 1)[0]
        if len(pos_idx) >= dose:
            ora = [np.mean(out_cv[pos_idx[rng.choice(len(pos_idx), dose, replace=False)]]) for _ in range(n_sim)]
            ora_rate = np.mean(ora)
        else: ora_rate = np.nan

        results.append({
            'user_id': int(users[u]), 'u_idx': u, 'fam_ref': fam_ref, 'gap': gap,
            'n_pos': int(out_cv.sum()), 'n_total': int(out_valid.sum()),
            'random': np.mean(rand_rates), 'duration_matched': dur_rate,
            'category_screened': scr_rate, 'popularity': pop_rate,
            'pop_plus_screened': hyb_rate, 'category_adjacent': adj_rate,
            'collaborative_filtering': cf_rate, 'quality_matched_oracle': ora_rate,
        })
    return pd.DataFrame(results)


def summarize_user_values(df,columns,seed=42,completion_protocol=False):
    rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(df),size=(2000,len(df)))
    draws={}
    if completion_protocol:
        # Table 6 retains the original method-specific intervals for unchanged
        # rules. The corrected duration and CF rules use the same seed-42 draws.
        legacy_order=['random','duration_matched','category_adjacent','category_screened',
                      'popularity','pop_plus_screened','collaborative_filtering','quality_matched_oracle']
        legacy_rng=np.random.default_rng(seed)
        for method in legacy_order:
            legacy_samples=legacy_rng.integers(0,len(df),size=samples.shape)
            if method not in {'duration_matched','collaborative_filtering'}:
                draws[method]=legacy_samples
    rows=[]
    for col in columns:
        vals=df[col].to_numpy()
        if not np.isfinite(vals).all():
            raise ValueError(f'Unexpected missing values in {col}')
        method_samples=draws.get(col,samples)
        ci=np.percentile(vals[method_samples].mean(axis=1),[2.5,97.5])
        delta=vals-df.fam_ref.to_numpy()
        dci=np.percentile(delta[method_samples].mean(axis=1),[2.5,97.5])
        rows.append(dict(method=col,n=len(vals),mean=float(vals.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),
                         minus_familiar=float(delta.mean()),minus_familiar_ci_low=float(dci[0]),
                         minus_familiar_ci_high=float(dci[1]),above_familiar_share=float((delta>0).mean())))
    return rows
