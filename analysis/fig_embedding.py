"""Figure self_embedding (fig:self_embedding): the self-index is produced by the network, not inherited from the role words.
(a) Triangles of the three roles drawn from raw Euclidean distances (User-System side at the bottom, Assistant at the apex,
    each triangle scaled to its longest side): the role words in the input embedding and the output unembedding (model
    weights only), and message content (mean distances over 75 held-out bodies) after the first block, at L31 and at L63.
(b) Distance ratio R_A (raw and cosine) for the role-word embeddings and for message content at the first-block sites.
Inputs (two-level summaries; the archived copies unless message_geometry.py ran): role_word_embeddings.json,
L_message_content_geometry.json, L_paper_numbers.json.
Outputs: paper/source/figures/self_embedding.pdf/.png.

usage: fig_embedding.py --results RES --out OUT
"""
import json
import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, TL, Roots, arguments, prefer  # noqa: E402

C = {'A': '#2a78d6', 'U': '#eb6834', 'S': '#1baf7a'}
INK2, GRID, BLUE, GRAY = '#52514e', '#e6e6e6', '#2a78d6', '#9a9a9a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'axes.titlesize': 10.5,
                     'axes.titleweight': 'bold', 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5})


def triangle(ax, d, title, ratio):
    au, as_, us = d['AU'], d['AS'], d['US']
    m = max(au, as_, us)
    au, as_, us = au / m, as_ / m, us / m
    U, S = np.array([-us / 2, 0.0]), np.array([us / 2, 0.0])
    x = (au ** 2 - as_ ** 2) / (2 * us)
    A = np.array([x, math.sqrt(max(au ** 2 - (x + us / 2) ** 2, 0.0))])
    for p, q, ls, col in ((A, U, '--', '#b5b3ad'), (A, S, '--', '#b5b3ad'), (U, S, '-', INK2)):
        ax.plot(*zip(p, q), ls=ls, color=col, lw=1.6 if ls == '-' else 1.1)
    for r, p in (('A', A), ('U', U), ('S', S)):
        ax.add_patch(Circle(p, 0.09, fc=C[r], ec='white', lw=1.2, zorder=3))
        ax.text(*p, r, ha='center', va='center', fontsize=8.5, color='white', fontweight='bold', zorder=4)
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.15, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=8.8, fontweight='normal')
    ax.text(0, -0.2, f'$R_A$ = {ratio:.2f}', ha='center', va='top', fontsize=9.5)


def main_(emb_json, l_json, l_numbers, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    E = json.load(open(emb_json, encoding='utf-8'))
    L = json.load(open(l_json, encoding='utf-8'))['results']['native']['confirmation']
    T = json.load(open(l_numbers, encoding='utf-8'))['table']
    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.15)
    cases = [('role words,\ninput emb.', E['embed_tokens.weight']['D_raw'], E['embed_tokens.weight']['R_A_raw']),
             ('role words,\noutput unemb.', E['lm_head.weight']['D_raw'], E['lm_head.weight']['R_A_raw'])]
    for l in (0, 31, 63):
        r = L[f'residual__L{l:02d}']['raw']
        cases.append((f'message content,\nL{l}', {p: r[f'D_{p}']['mean'] for p in ('AU', 'AS', 'US')}, r['R_A']['mean']))
    for i, (title, d, ratio) in enumerate(cases):
        triangle(fig.add_subplot(gs[0, i]), d, title, ratio)
    fig.text(0.01, 0.985, "(a) The three roles drawn to scale: role words in the model's weights vs. message content",
             ha='left', va='top', fontsize=10.5, fontweight='bold')
    ax = fig.add_subplot(gs[1, :])
    sites = [('input\nembedding', E['embed_tokens.weight']['R_A_raw'], E['embed_tokens.weight']['R_A_cos']),
             ('output\nunembedding', E['lm_head.weight']['R_A_raw'], E['lm_head.weight']['R_A_cos']),
             ('content\nFFN L0', T['FFN__L00']['raw_R_A'], T['FFN__L00']['cos_R_A']),
             ('content\nquery L1', T['Q__L01']['raw_R_A'], T['Q__L01']['cos_R_A']),
             ('content\nresidual L0', T['residual__L00']['raw_R_A'], T['residual__L00']['cos_R_A'])]
    w = 0.36
    for k, (label, col, idx) in enumerate((('raw distance', BLUE, 1), ('cosine distance', GRAY, 2))):
        xs = np.arange(len(sites)) + (k - 0.5) * w
        vals = [s[idx] for s in sites]
        ax.bar(xs, vals, width=w - 0.04, color=col, label=label)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle=':')
    ax.axhline(2 / 3, color=INK2, linewidth=0.8, linestyle='--')
    ax.text(len(sites) - 0.45, 1.0, 'no role singled out (1)', ha='right', va='bottom', fontsize=8.5, color=INK2)
    ax.text(len(sites) - 0.45, 2 / 3, 'sufficient bound (2/3)', ha='right', va='bottom', fontsize=8.5, color=INK2)
    ax.axvline(1.5, color=GRID, linewidth=1.2)
    ax.set_xticks(range(len(sites)), [s[0] for s in sites])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('$R_A$ (lower = Assistant\nmore apart)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9.5, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.12))
    ax.set_title('(b) Distance ratio: role-word weights vs. message content after the first block', loc='left', pad=22)
    fig.savefig(f'{out_dir}/self_embedding.pdf', bbox_inches='tight')
    fig.savefig(f'{out_dir}/self_embedding.png', dpi=150, bbox_inches='tight')
    print('saved')


def main(argv=None):
    args = arguments(__doc__).parse_args(argv)
    r = Roots(args)
    ins = [prefer(r, TL + f'/summaries/{n}') for n in ('role_word_embeddings.json', 'L_message_content_geometry.json', 'L_paper_numbers.json')]
    print('inputs:', ins)
    main_(*ins, str(r.out(FIGURES)))


if __name__ == '__main__':
    main()
