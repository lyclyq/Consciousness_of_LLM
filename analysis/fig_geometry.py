"""Figure self_index_geometry (fig:self_geometry), readout version: the main geometry figure follows the readout position.
Template default, non-thinking mode, 75 held-out bodies.
Top row: the response onset (turn 'reply', last prompt token) of each held-out body under the three roles, centred per body
and projected on the self-index direction (x) and the User-System direction (y), both fitted on discovery bodies, at five layers.
Bottom row: distance ratios across all 64 layers at the response onset and on the message content, and the direction
consistency of the offsets at the response onset.
Inputs: summaries/J_onset_geometry.json (two_level_geometry.py; the archived copy if that script did not run) and
summaries/L_message_content_geometry.json (message content geometry; from the results package unless message_geometry.py ran).
Outputs: paper/source/figures/self_index_geometry.pdf/.png.

usage: fig_geometry.py --results RES --out OUT
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, TL, Roots, arguments, prefer  # noqa: E402

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
ROLE = {'A': ('#2a78d6', '-', 'Assistant', 'assistant', 'o'), 'U': ('#eb6834', '--', 'User', 'user', 's'),
        'S': ('#1baf7a', '-.', 'System', 'system', '^')}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.2, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 8.8, 'axes.titleweight': 'bold',
                     'axes.titlecolor': INK, 'legend.frameon': False, 'pdf.fonttype': 42})
L = list(range(64))


def curve_style(ax, ylabel):
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(0, 63)
    ax.set_xticks([0, 16, 32, 48, 63])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('layer')


def main_(onset_path, l_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    P = json.load(open(onset_path, encoding='utf-8'))
    R = json.load(open(l_path, encoding='utf-8'))['results']['native']['confirmation']
    fig = plt.figure(figsize=(10.4, 5.6))
    top = fig.add_gridspec(1, 5, top=0.86, bottom=0.53, wspace=0.42)
    for i, l in enumerate(P['layers']):
        ax = fig.add_subplot(top[0, i])
        pts = P['points'][str(l)]
        scale = max(abs(np.array(pts[r])).max() for r in pts)
        for k, (col, _, name, key, mk) in ROLE.items():
            a = np.array(pts[key]) / scale
            ax.scatter(a[:, 0], a[:, 1], s=9, color=col, marker=mk, alpha=0.75, linewidths=0, label=name)
        ax.axvline(0, color=GRID, linewidth=0.8, zorder=0)
        ax.axhline(0, color=GRID, linewidth=0.8, zorder=0)
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title(f'L{l}', loc='center')
        if i == 0:
            ax.set_ylabel('User-System direction')
        ax.set_xlabel('self-index direction')
    fig.text(0.01, 0.985, f"(a) Response onset of {P['n_heldout']} held-out texts under the three roles "
             '(each text\'s mean removed; two directions fitted on discovery texts)',
             ha='left', va='top', fontsize=8.8, fontweight='bold', color=INK)
    h, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.955), markerscale=1.6, fontsize=8)
    bottom = fig.add_gridspec(1, 3, top=0.40, bottom=0.07, wspace=0.32, width_ratios=[1.25, 1.25, 1.0])
    onset = {k: [P['heldout']['raw'][l][f'R_{k}'] for l in L] for k in 'AUS'}
    content = {k: [R[f'residual__L{l:02d}']['raw'][f'R_{k}']['mean'] for l in L] for k in 'AUS'}
    lo = min(min(v) for c in (onset, content) for v in c.values())
    hi = max(max(v) for c in (onset, content) for v in c.values())
    ratio = 'distance ratio R\n(lower = more separated)'
    for j, (curves, title) in enumerate(((onset, '(b) Response onset'), (content, '(c) Message content'))):
        ax = fig.add_subplot(bottom[0, j])
        ax.axhline(1, color=INK2, linewidth=0.8, linestyle=':')
        for k, (col, ls, name, key, mk) in ROLE.items():
            ax.plot(L, curves[k], color=col, linestyle=ls, linewidth=1.5)
        ax.set_ylim(np.floor(lo * 10) / 10, np.ceil(hi * 10) / 10)
        ax.set_title(title, loc='left')
        curve_style(ax, ratio)
    ax = fig.add_subplot(bottom[0, 2])
    for k, (col, ls, name, key, mk) in ROLE.items():
        ax.plot(L, P['direction_consistency'][key]['mean_of_category_means'], color=col, linestyle=ls, linewidth=1.5)
    ax.set_ylim(0, 1.02)
    ax.set_title('(d) Direction consistency, response onset', loc='left')
    curve_style(ax, 'held-out cosine')
    fig.savefig(f'{out_dir}/self_index_geometry.pdf', bbox_inches='tight')
    fig.savefig(f'{out_dir}/self_index_geometry.png', dpi=150, bbox_inches='tight')
    print('saved')


def main(argv=None):
    args = arguments(__doc__).parse_args(argv)
    r = Roots(args)
    onset, lsum = prefer(r, TL + '/summaries/J_onset_geometry.json'), prefer(r, TL + '/summaries/L_message_content_geometry.json')
    print('inputs:', onset, lsum)
    main_(onset, lsum, str(r.out(FIGURES)))


if __name__ == '__main__':
    main()
