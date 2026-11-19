"""Render the appendix sample and sequence diagrams."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

from config import FIGURE_DIR
HERE = FIGURE_DIR
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'text.color': 'black', 'figure.facecolor': 'white'})


def canvas(width, height, xlim, ylim):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set(xlim=(0, xlim), ylim=(0, ylim))
    ax.axis('off')
    fig.subplots_adjust(left=.025, right=.975, bottom=.025, top=.975)
    return fig, ax


def box(ax, x, y, w, h, label, size=9, weight='normal'):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor='black',linewidth=1.05,zorder=2))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,
            weight=weight,linespacing=1.45,zorder=3)


def arrow(ax, start, end, dashed=False):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=10,
        linewidth=1,color='black',linestyle='--' if dashed else '-',shrinkA=0,shrinkB=0,zorder=1))


def route(ax, points):
    ax.plot(*zip(*points[:-1]),color='black',lw=1,zorder=1)
    arrow(ax,points[-2],points[-1])


def save(fig, name):
    for ext in ('pdf','svg'):
        fig.savefig(HERE/f'{name}.{ext}',facecolor='white',bbox_inches='tight',pad_inches=.06)
    plt.close(fig)


def samples():
    fig,ax=canvas(7.3,5.5,14.6,11)
    box(ax,.2,9.85,14.2,.95,'KuaiRand-1K: 1,000 users\nSingle-column feed, April 8 to May 8, 2022',weight='bold')
    route(ax,[(7.3,9.85),(7.3,9.45),(3.5,9.45),(3.5,8.85)])
    route(ax,[(7.3,9.85),(7.3,9.45),(10.9,9.45),(10.9,8.85)])
    box(ax,.2,6.7,6.6,2.15,'Behavioural panel\nAt least 10 standard recommendations per day\nUsable tags for at least 90% of recommendations\n24,528 user-days from 955 users',size=8.7)
    box(ax,7.4,6.7,7,2.15,'Within-day event selection\n10 tagged standard recommendations on each side\nAdjacent insertions at least 20 positions away\nUsable profile and random-video tags',size=8.7)
    route(ax,[(3.5,6.7),(3.5,6.2),(1.75,6.2),(1.75,5.65)])
    route(ax,[(3.5,6.7),(3.5,6.2),(5.25,6.2),(5.25,5.65)])
    box(ax,.2,2.6,3.1,3.05,'Daily model\n\nRequired covariates\nand lagged exposure\navailable\n\n21,756 user-days\n953 users',size=8.7)
    box(ax,3.7,2.6,3.1,3.05,'User-level forest\n\nAcceptance averaged\nwithin each period\nBoth periods observed\n\n955 users',size=8.7)
    arrow(ax,(10.9,6.7),(10.9,5.65))
    box(ax,7.4,3.95,7,1.7,'Primary event sample\n12,172 insertions from 917 users\n243,440 standard recommendations',size=9)
    arrow(ax,(10.9,3.95),(10.9,2.95))
    box(ax,7.4,.35,7,2.6,'Common sample for window comparisons\n20 tagged recommendations on each side\nAdjacent insertions at least 40 positions away\n\n6,192 insertions from 783 users\nCompare windows of 5, 10 and 20 positions',size=8.7)
    save(fig,'sample_construction')


def sequence():
    fig,ax=canvas(7.3,4.6,14.6,9.2)
    ax.text(.2,8.95,'A. Profile construction and observation periods',weight='bold',va='top')
    box(ax,.2,6.55,6.2,1.7,"April 8 to April 21\nPretreatment standard recommendations\nFix each user's five familiar tags",size=9)
    box(ax,8,6.55,6.4,1.7,'April 22 to May 8\nStandard recommendations with random insertions\nUse the fixed familiar-tag profile',size=8)
    arrow(ax,(6.4,7.4),(8,7.4))
    ax.text(.2,5.85,'B. Positions around one random insertion',weight='bold',va='top')
    box(ax,.2,4.2,5.55,1.05,'Standard recommendations\nBefore insertion: negative positions',size=8.7)
    box(ax,6.3,4.2,2,1.05,'Random video\nPosition 0',size=8.7)
    box(ax,8.85,4.2,5.55,1.05,'Standard recommendations\nAfter insertion: positive positions',size=8.7)
    arrow(ax,(5.75,4.72),(6.3,4.72));arrow(ax,(8.3,4.72),(8.85,4.72))
    ax.plot([7.3,7.3],[1.05,4.2],linestyle=':',color='black',lw=.8)
    for halfwidth,y,label in [(2.35,3.25,'5 on each side'),(4.45,2.35,'10 on each side'),(6.75,1.45,'20 on each side')]:
        ax.plot([7.3-halfwidth,7.3+halfwidth],[y,y],color='black',lw=1)
        for x in (7.3-halfwidth,7.3+halfwidth):
            ax.plot([x,x],[y-.12,y+.12],color='black',lw=1)
        ax.text(7.3,y,label,ha='center',va='center',size=9,
            bbox={'facecolor':'white','edgecolor':'none','pad':3})
    ax.text(7.3,.5,'Viewing outcomes use standard recommendations on both sides.\nThe random video marks the insertion and is excluded from the outcomes.',
        ha='center',va='center',size=8.5,linespacing=1.4)
    save(fig,'exposure_sequence')


if __name__=='__main__':
    samples()
    sequence()
