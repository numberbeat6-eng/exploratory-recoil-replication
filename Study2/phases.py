"""Sequential phases and user-clustered two-way fixed-effects regressions."""
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def phase_pairs(small,users,items,fc,item_labels,flags,category_codes):
    rows=[]
    for u, recs in small.groupby('u_idx',sort=False):
        u=int(u)
        if fc[u,0]<0 or len(recs)<200:
            continue
        recs=recs.sort_values('timestamp').reset_index(drop=True)
        v=recs.v_idx.to_numpy()
        # Translate dense matrix category indices back to original tag codes.
        familiar=set(category_codes[fc[u]].tolist())
        fam=np.array([bool(item_labels[idx]&familiar) for idx in v])
        y=flags[recs.index_raw.to_numpy()]
        n=len(recs)
        phase_size=n//5
        pdata=[]
        for p in range(5):
            start=p*phase_size;end=(p+1)*phase_size if p<4 else n
            f=fam[start:end];out=~f
            if out.sum()<5 or f.sum()<5:
                pdata.append(None)
                continue
            pdata.append(dict(out_quality=float(y[start:end][out].mean()),
                              fam_quality=float(y[start:end][f].mean()),out_fraction=float(out.mean())))
        for p in range(4):
            a,b=pdata[p],pdata[p+1]
            if a is None or b is None:
                continue
            gap=a['fam_quality']-a['out_quality'];ngap=b['fam_quality']-b['out_quality']
            rows.append(dict(user_id=int(users[u]),u_idx=u,phase=p,
                             cur_out_quality=a['out_quality'],cur_fam_quality=a['fam_quality'],
                             cur_out_fraction=a['out_fraction'],cur_gap=gap,
                             next_out_acceptance=b['out_quality'],next_fam_acceptance=b['fam_quality'],
                             next_out_fraction=b['out_fraction'],next_gap=ngap,
                             acceptance_change=b['out_quality']-a['out_quality'],gap_change=ngap-gap))
    return pd.DataFrame(rows).sort_values(['u_idx','phase']).reset_index(drop=True)


def demean(values, user, phase):
    out = np.array(values, dtype=float, copy=True)
    if out.ndim == 1:
        out = out[:, None]
    for _ in range(200):
        old = out.copy()
        for group in (user, phase):
            counts = np.bincount(group)
            means = np.column_stack([np.bincount(group, weights=out[:, j]) / counts
                                     for j in range(out.shape[1])])
            out -= means[group]
        if np.max(np.abs(out-old)) < 1e-13:
            break
    else:
        raise RuntimeError('Fixed-effect projection failed to converge')
    return out


def clustered_multivariate_fe(df, xnames, ynames):
    """FWL equivalent to separate OLS with user/phase dummies; joint cluster meat.

    The finite-sample factor G/(G-1)*(N-1)/(N-K) matches statsmodels OLS
    cluster correction with explicit intercept and user/phase dummy columns.
    """
    d = df.dropna(subset=xnames+ynames+['user_id','phase']).copy()
    user = pd.factorize(d.user_id)[0]
    phase = pd.factorize(d.phase)[0]
    g, n = len(np.unique(user)), len(d)
    k = g + len(np.unique(phase)) - 1 + len(xnames)
    residualized = demean(d[xnames+ynames].to_numpy(), user, phase)
    x, y = residualized[:, :len(xnames)], residualized[:, len(xnames):]
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ y
    errors = y-x@beta
    scores = []
    for j in range(len(ynames)):
        score = np.column_stack([np.bincount(user, weights=x[:, q]*errors[:, j], minlength=g)
                                 for q in range(len(xnames))])
        scores.append(score)
    joint_scores = np.column_stack(scores)
    joint_bread = np.kron(np.eye(len(ynames)), bread)
    covariance = joint_bread @ (joint_scores.T@joint_scores) @ joint_bread
    covariance *= g/(g-1)*(n-1)/(n-k)
    flat = beta.T.ravel()
    rows = []
    for j, yn in enumerate(ynames):
        for q, xn in enumerate(xnames):
            idx = j*len(xnames)+q
            se = float(np.sqrt(covariance[idx,idx]))
            rows.append(dict(outcome=yn,predictor=xn,coef=float(flat[idx]),se=se,
                             p=float(2*student_t.sf(abs(flat[idx]/se),g-1)),
                             ci_low=float(flat[idx]-student_t.ppf(.975,g-1)*se),
                             ci_high=float(flat[idx]+student_t.ppf(.975,g-1)*se),
                             n=n,n_users=g,df_resid=n-k))
    return dict(coefficients=rows,beta=flat.tolist(),covariance=covariance.tolist(),
                n=n,n_users=g,df_resid=n-k,cluster_df=g-1)
