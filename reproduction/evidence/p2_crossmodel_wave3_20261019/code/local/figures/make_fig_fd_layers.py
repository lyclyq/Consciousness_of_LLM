"""Figure fig:self_fd (appendix, 11-01): first-order sensitivity of the attribution answer by layer in Qwen3.8-27B without thinking,
from runs/pro/singlepass_swap/qwen38_27b_r03fd (R03 Part B; crossmodel_readout_fd_r03.py, every second layer). Per computed layer,
the mean over items of the central-difference slope of f = logit(you letter) - logit(user letter) at the response onset when the
states of that layer (aligned positions) are moved along the part of the own-User difference in the eight fitted coordinates and
along the rest of it, with 95% bootstrap intervals over items (2000 draws, seed 1); left at the User message, right at the model's
own message; shaded, the last third of the layers. Style as the layer-window figure of make_v4_mechanism.py (fonts, colours, the
"last third" band): the coordinates in the colour of the index edits, the rest dark and dashed.
usage: make_fig_fd_layers.py ARCHIVE OUT_DIR   (OUT_DIR = paper/source/figures)"""
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'pdf.fonttype': 42, 'axes.titlesize': 9, 'axes.titleweight': 'bold',
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False})
C_SWAP, C_INK, C_GRID, C_SHADE = '#2a78d6', '#2b2f36', '#e3e6ea', '#f3f4f6'
NL = 64


def ci(vals, draws=2000):
    rng = random.Random(1)
    bs = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals) for _ in range(draws))
    return bs[int(0.025 * draws)], bs[int(0.975 * draws) - 1]


def main():
    A, out = Path(sys.argv[1]), Path(sys.argv[2])
    per = {'U': {}, 'A': {}}
    layers = None
    for p in sorted((A / 'runs/pro/singlepass_swap/qwen38_27b_r03fd/items').glob('*.json')):
        j = json.load(open(p, encoding='utf-8'))
        layers = j['layers']
        for ep in ('U', 'A'):
            for k in ('V', 'perp'):
                for l, v in zip(layers, j['endpoints'][ep]['D'][k]):
                    per[ep].setdefault(k, {}).setdefault(l, []).append(v)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.5), sharey=True, gridspec_kw={'wspace': 0.08})
    for ax, ep, title in zip(axes, ('U', 'A'), ('User message', "model's own message")):
        ax.axvspan(math.ceil(2 * NL / 3) - 0.5, NL - 0.5, color=C_SHADE, zorder=0, lw=0)
        ax.axhline(0, color='#9aa1a9', lw=0.7, zorder=1)
        for k, col, ls, mk in (('V', C_SWAP, '-', 'o'), ('perp', C_INK, (0, (3, 2)), None)):
            m = [sum(per[ep][k][l]) / len(per[ep][k][l]) for l in layers]
            lo, hi = zip(*(ci(per[ep][k][l]) for l in layers))
            ax.fill_between(layers, lo, hi, color=col, alpha=0.15, lw=0, zorder=2)
            ax.plot(layers, m, color=col, lw=1.5, ls=ls, marker=mk, ms=3.0, mec='white', mew=0.5, zorder=3)
        ax.set_xlim(-0.5, NL - 0.5)
        ax.set_xticks(range(0, NL + 1, 16))
        ax.set_title(title)
        ax.grid(axis='y', color=C_GRID, lw=0.6, zorder=0)
    fig.text(0.5, -0.02, 'layer whose states are moved (one layer at a time, every second layer)', ha='center', fontsize=7.2)
    ylo, yhi = axes[0].get_ylim()
    for ax in axes:
        ax.text(NL - 1, yhi - 0.03 * (yhi - ylo), 'last third', ha='right', va='top', fontsize=6.2, color='#6b7280')
    axes[0].set_ylabel('slope of logit("you") - logit("user")\nper layer, mean over items', fontsize=7.4)
    handles = [plt.Line2D([], [], color=C_SWAP, lw=1.5, marker='o', ms=3.0, mec='white', mew=0.5),
               plt.Line2D([], [], color=C_INK, lw=1.5, ls=(0, (3, 2)))]
    fig.legend(handles, ['along the eight coordinates', 'along the rest of the own-User difference'], loc='lower center', ncol=2,
               frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.16))
    fig.savefig(out / 'self_fd_layers.pdf', bbox_inches='tight')
    fig.savefig(out / 'self_fd_layers.png', dpi=200, bbox_inches='tight')
    print('written', out / 'self_fd_layers.pdf')


if __name__ == '__main__':
    main()
