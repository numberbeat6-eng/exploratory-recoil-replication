"""Draw the decile heatmap and selection-rule distributions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.stats import gaussian_kde

BLUE, GREY = '#315B7D', '#777777'
CMAP = LinearSegmentedColormap.from_list('s2', ['#FFFFFF', '#C9D9E6', '#6F97B8', '#315B7D', '#13283B'])
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight',
})

RULES = [('Random sampling', 'random', 'set'), ('Category-adjacent', 'category_adjacent', 'set'),
         ('Duration-matched', 'duration_matched', 'set'), ('Category-screened', 'category_screened', 'set'),
         ('Collaborative filtering', 'collaborative_filtering', 'rank'),
         ('Popularity, screened categories', 'pop_plus_screened', 'rank'), ('Popularity', 'popularity', 'rank')]


def save(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_deciles(d, output):
    dec = [f'dec{i}' for i in range(1, 11)]
    d = d.dropna(subset=dec).sort_values('fam', ascending=False)
    blocks = [d[dec].values, d[['fam']].values]
    gap = 0.6
    xs, x = [], 0.0
    for blk in blocks:
        xs.append([x + j for j in range(blk.shape[1])])
        x += blk.shape[1] + gap
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    n = len(d)
    for bi, blk in enumerate(blocks):
        x0 = xs[bi][0] - 0.5
        im = ax.imshow(blk * 100, aspect='auto', cmap=CMAP, vmin=0, vmax=100, interpolation='nearest',
                       extent=(x0, x0 + blk.shape[1], n, 0))
        for j, xx in enumerate(xs[bi]):
            ax.text(xx, -18, f'{blk[:, j].mean() * 100:.0f}', ha='center', va='bottom', fontsize=7.2)
    ax.set_xlim(-0.5, x - gap - 0.5)
    ax.set_ylim(n, -140)
    ax.set_xticks(sum(xs, []), [str(i) for i in range(1, 11)] + ['Familiar'])
    ax.tick_params(axis='x', length=0)
    ax.set_yticks([])
    for s in ('left', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.set_ylabel(f'{n:,} users, sorted by familiar completion')
    a, b = xs[0][0] - 0.45, xs[0][-1] + 0.45
    ax.plot([a, b], [-95, -95], color='black', lw=0.7, clip_on=False)
    ax.text((a + b) / 2, -107, "Unfamiliar videos, by decile of other users' completion", ha='center',
            va='bottom', fontsize=8.2)
    ax.text(-0.5, -18, 'Mean', ha='right', va='bottom', fontsize=7.2, style='italic')
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_ticks([0, 25, 50, 75, 100])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)
    cb.set_label('Completion rate of the user (%)', fontsize=7.8)
    print('decile means', np.round(d[dec].mean().values * 100, 1), ' familiar', round(d.fam.mean() * 100, 1))
    save(fig, output / 'Figure_6.pdf')


def fig_rules(d, output):
    fam = d.fam_ref.values * 100
    rows = RULES[::-1]
    xs = np.linspace(-50, 90, 600)
    step = 1.0
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.axvline(0, color=GREY, lw=0.8, zorder=1)
    ax.axvspan(-50, 0, color='#F1F1F1', zorder=0, lw=0)
    for i, (lab, col, kind) in enumerate(rows):
        v = d[col].values * 100 - fam
        v = v[~np.isnan(v)]
        k = gaussian_kde(v, bw_method=0.25)(xs)
        k = k / k.max() * 0.85 * step
        base = (len(rows) - 1 - i) * step
        c = BLUE if kind == 'rank' else GREY
        ax.fill_between(xs, base, base + k, color=c, alpha=0.35, lw=0, zorder=2 + i)
        ax.plot(xs, base + k, color=c, lw=1.0, zorder=2 + i)
        ax.plot([xs[0], xs[-1]], [base, base], color=c, lw=0.6, zorder=2 + i)
        med = np.median(v)
        ax.plot([med, med], [base, base + np.interp(med, xs, k)], color='black', lw=0.9, zorder=3 + i)
        ax.text(92, base + 0.22, f'{(v > 0).mean() * 100:.1f}%', ha='left', va='center', fontsize=7.8)
        ax.text(-52, base + 0.22, lab, ha='right', va='center', fontsize=8)
        print(f'{lab:32s} median {med:6.1f}  share above familiar {(v > 0).mean() * 100:5.1f}  n {len(v)}')
    top = len(rows) * step
    ax.text(92, top - 0.05, 'Better off\nthan familiar', ha='left', va='bottom', fontsize=7.6, style='italic')
    ax.text(-25, top - 0.05, 'Below familiar', ha='center', va='bottom', fontsize=7.6, style='italic', color=GREY)
    ax.text(45, top - 0.05, 'Above familiar', ha='center', va='bottom', fontsize=7.6, style='italic', color=GREY)
    ax.plot([-50, 90], [4 * step - 0.05] * 2, color='#BBBBBB', lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(-50, 90)
    ax.set_ylim(-0.1, top + 0.35)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel("Completion of the five chosen videos minus the user's familiar completion (percentage points)")
    save(fig, output / 'Figure_7.pdf')
