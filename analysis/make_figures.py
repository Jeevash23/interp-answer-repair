"""Draw the two data figures of the manuscript from saved results.

Inputs : studies/bridge-005/results/analysis/summary.json (original analysis)
         analysis/results/self-vs-wording.json             (run self_vs_wording.py first)
         studies/bridge-005/results/test/*.npz              (per-example scatter)
Outputs: paper/figures/recovery-and-margins.pdf, paper/figures/signature-map.pdf

Run from the project root:  .venv/bin/python analysis/make_figures.py
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, ROOT, geometry, load, read, study_rows

OUT = ROOT / 'paper/figures'
BLUE, ORANGE, GREY, INK, MUTED = '#2a78d6', '#eb6834', '#8a8985', '#0b0b0b', '#52514e'
CELLS = ['w0_o0', 'w0_o1', 'w1_o0', 'w1_o1']
LABELS = ['F6 / order 0', 'F6 / order 1', 'F7 / order 0', 'F7 / order 1']

plt.rcParams.update({
    # STIXGeneral ships with matplotlib, so the figures look the same on every machine.
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42})


def interval(ax, y, d, key, colour, marker, filled, label=None):
    lo, hi = d[key]
    ax.plot([lo, hi], [y, y], color=colour, lw=1.4, solid_capstyle='round', zorder=2)
    ax.plot(d['mean'], y, marker=marker, ms=5.5, color=colour, mfc=colour if filled else 'white',
            mew=1.3, ls='none', zorder=3, label=label)


def recovery_and_margins(summary):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2), sharey=True, gridspec_kw=dict(wspace=0.16))
    ys = np.arange(4)[::-1]
    off = 0.14
    panels = [('(a) Answer recovery', 'increase in logit gap', None),
              ('(b) Margin, zero repl.', r'$\leftarrow$ closer to recipient', 'zero'),
              ('(c) Margin, mean repl.', r'$\leftarrow$ closer to recipient', 'fit_mean')]
    for ax, (title, xlabel, ref) in zip(axes, panels):
        for y, cell in zip(ys, CELLS):
            if ref is None:
                w = summary['primary']['recovery'][cell]
                s = summary['conditions']['self']['cells'][cell]['recovery']
            else:
                w = summary['primary']['margin'][ref][cell]
                s = summary['conditions']['self']['geometry'][ref]['margin']['cells'][cell]
            interval(ax, y + off, w, 'adjusted_ci', BLUE, 'o', True,
                     'Wording transfer (adjusted interval)' if y == ys[0] else None)
            interval(ax, y - off, s, 'ci95', ORANGE, 'D', False,
                     'Self transfer (pointwise interval)' if y == ys[0] else None)
        ax.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.set_title(title, color=INK, pad=4, fontsize=8)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=7.5)
        ax.grid(axis='x', color='#e6e5e0', lw=0.5, zorder=0)
    axes[0].set_yticks(ys, LABELS)
    axes[0].set_xlim(0, 4.3)
    for ax in axes[1:]:
        ax.set_xlim(-0.6, 0.05)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, handletextpad=0.4)
    fig.savefig(OUT / 'recovery-and-margins.pdf', bbox_inches='tight')
    plt.close(fig)


def signature_map(sw):
    s = study_rows('bridge-005')
    test = s['test']
    R = load(test / 'clean.npz')['raw'][..., 0].astype(float)
    D = R[:, s['donor']]
    Sw = geometry(load(test / 'wording.npz')['raw'][..., 0].astype(float), R, D, s['scale'])['S'][0]
    Ss = geometry(load(test / 'self.npz')['raw'][..., 0].astype(float), R, D, s['scale'])['S'][0]
    fig = plt.figure(figsize=(6.6, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.42)
    for k, (ref, title) in enumerate([('zero', '(a) Zero replacement'), ('mean', '(b) Mean replacement')]):
        ax = fig.add_subplot(gs[0, k])
        r = sw['f6_f7']['references'][ref]
        pts = {name: (r[name]['S']['mean'], r[name]['O']['mean']) for name in ['corrupt', 'self', 'wording']}
        ax.plot([0, 1], [0, 0], color=GREY, lw=1.0, zorder=1)
        for x, lab in [(0, 'R'), (1, 'D')]:
            ax.plot(x, 0, 'o', ms=5.5, color=INK, zorder=4)
            ax.text(x, -0.13, lab, ha='center', va='top', fontsize=8, color=INK)
        cx, cy = pts['corrupt']
        ax.plot([cx, 0], [cy, 0], color=GREY, lw=0.8, ls=(0, (1.5, 2)), zorder=1)
        ang = np.degrees(np.arctan2(cy, cx))
        if k == 0:
            ax.text(cx * 0.22 + 0.05, cy * 0.22, 'full repair', rotation=ang, rotation_mode='anchor',
                    fontsize=6.5, color=MUTED, ha='left', va='top')
        ax.plot(cx, cy, 'o', ms=5.5, mfc='white', mec=MUTED, mew=1.2, zorder=4, label='corrupted $\\mathbf{C}$')
        ax.text(cx + 0.06, cy + 0.02, 'C', fontsize=8, color=MUTED, va='bottom')
        for name, colour, marker, ms, z, lab in [('self', ORANGE, 'D', 7.5, 5, 'self transfer'),
                                                 ('wording', BLUE, 'o', 4.2, 6, 'wording transfer')]:
            x, y = pts[name]
            ax.annotate('', xy=(x, y), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='-|>', color=colour, lw=1.0, shrinkA=4, shrinkB=5, mutation_scale=8))
            ax.plot(x, y, marker, ms=ms, color=colour, mfc=colour if name == 'wording' else 'white', mew=1.3,
                    zorder=z, ls='none', label=lab)
        ex, ey = pts['self']
        ax.text(ex - 0.09, ey, 'E', fontsize=8, color=INK, ha='right', va='center')
        if k == 1:
            handles, labels = ax.get_legend_handles_labels()
        ax.set_xlim(-0.35, 1.15)
        ax.set_ylim(-0.3, 1.65)
        ax.set_aspect('equal')
        ax.set_xlabel('position along clean line, $S$')
        if k == 0:
            ax.set_ylabel('distance from line, $O$')
        ax.set_title(title, color=INK, pad=4)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.36, -0.09), ncol=3, frameon=False,
               fontsize=7, handletextpad=0.3, columnspacing=1.2)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot([-0.6, 1.0], [-0.6, 1.0], color=GREY, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.scatter(Ss, Sw, s=3, color=BLUE, alpha=0.35, lw=0, zorder=2, rasterized=True)
    ax.set_xlim(-0.6, 1.0)
    ax.set_ylim(-0.6, 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel('$S$ after self transfer')
    ax.set_ylabel('$S$ after wording transfer')
    ax.set_title('(c) Per configuration', color=INK, pad=8)
    r = np.corrcoef(Ss, Sw)[0, 1]
    ax.text(0.97, 0.04, f'r = {r:.2f}', transform=ax.transAxes, ha='right', fontsize=7, color=MUTED)
    fig.savefig(OUT / 'signature-map.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    return float(r)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = read(ROOT / 'studies/bridge-005/results/analysis/summary.json')
    sw = read(RESULTS / 'self-vs-wording.json')
    recovery_and_margins(summary)
    r = signature_map(sw)
    print('figures written to', OUT, '| per-example S correlation (zero):', round(r, 3))


if __name__ == '__main__':
    main()
