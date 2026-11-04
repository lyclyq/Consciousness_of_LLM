"""Figure self_reference (fig:self_reference): template default, non-thinking mode, 150 bodies, both option orders: probability
of 'you (the AI assistant)', 'the user', 'the system' in answer to 'who wrote the message before this question?' for Assistant,
User and System messages carrying the same text.
Input: summaries/J_native_geometry_default.json (two_level_geometry.py; the archived copy if that script did not run).
Outputs: paper/source/figures/self_reference.pdf/.png.

usage: fig_selfref.py --results RES --out OUT
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

ANS = [('you (AI assistant)', '#2a78d6'), ('the user', '#eb6834'), ('the system', '#1baf7a')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.2, 'pdf.fonttype': 42})


def main_(j_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    B = json.load(open(j_path, encoding='utf-8'))['attr_baseline']
    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    roles = [('assistant', 'Assistant\nmessage'), ('user', 'User\nmessage'), ('system', 'System\nmessage')]
    w = 0.26
    for k, (name, col) in enumerate(ANS):
        xs = np.arange(len(roles)) + (k - 1) * w
        vals = [B[f'default__{r}'][k] for r, _ in roles]
        ax.bar(xs, vals, width=w - 0.03, color=col, label=f'answer: {name}')
        for x, v in zip(xs, vals):
            if v >= 0.03:
                ax.text(x, v + 0.015, f'{v:.2f}', ha='center', va='bottom', fontsize=6.8)
    ax.set_xticks(range(len(roles)), [n for _, n in roles])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel('probability (150 bodies x 2 orders)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7.0, frameon=False, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.9)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/self_reference.pdf')
    fig.savefig(f'{out_dir}/self_reference.png', dpi=150)
    print('saved', {r: [round(v, 3) for v in B[f'default__{r}']] for r, _ in roles})


def main(argv=None):
    args = arguments(__doc__).parse_args(argv)
    r = Roots(args)
    j = prefer(r, TL + '/summaries/J_native_geometry_default.json')
    print('input:', j)
    main_(j, str(r.out(FIGURES)))


if __name__ == '__main__':
    main()
