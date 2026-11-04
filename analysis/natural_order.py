"""Natural conversation order, Figure self_natural (fig:self_natural): the two levels of the models of the natural-order
appendix, single pass, on the axes of Figure self_crossmodel (fig_crossmodel.py):
x = self-index: share of decoder layers at which the own message is singled out (smallest distance ratio R_A) in at least 80% of the
    held-out bodies, at message content or at the response onset, a layer counted once (DeepSeek-V4.1-Flash: mean of the distance
    ratios of its four streams, as in Table tab:self_natural3);
y = self-reference: mean P("you" | own) minus the larger of P("you" | User) and P("you" | System), over the 150 bodies and the three
    option orders, as in Table tab:self_natural (the run summaries average the three orders).
Arrows: the six models whose templates keep three roles in natural order (Table tab:self_natural), from the main probe (same
quantities from the main suite summaries, as in fig_crossmodel.py) to the natural order. Squares: the models whose templates do not
accept the main probe, natural order only (their User condition is merged into one user turn): DeepSeek-V4.1-Flash (chat mode)
and Ministral-3-14B-Instruct (Ministral-3-14B-Reasoning has geometry only). Marker fill: the layers singled out cover every layer
of the last third of the depth.
Outputs: paper/source/figures/self_natural.pdf/.png and stats/self_natural_points_v1.json (per model and order: x, y, and whether
the last third is covered).

usage: natural_order.py --results RES --out OUT
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, M8, W3, Roots, arguments, run_path  # noqa: E402
from newdev import NEWDEV  # noqa: E402

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'pdf.fonttype': 42, 'xtick.labelsize': 9, 'ytick.labelsize': 9})
NAMES = {'exaone4_32b': 'EXAONE-4.0.1-32B', 'granite41_30b': 'Granite-4.1-30B', 'glm46v_flash': 'GLM-4.6V-Flash',
         'seed_oss_36b': 'Seed-OSS-36B', 'gemma4_31b': 'Gemma-4-31B', 'olmo2_7b': 'OLMo-2-7B',
         'dsv41_flash': 'DeepSeek-V4.1-Flash', 'ministral3_14b': 'Ministral-3-14B-Instruct'}
# the new developers' three-role natural-order runs with a single-pass reading (ERNIE is thinking only: geometry in the table only)
NEW3 = [k for k, v in NEWDEV.items() if v['natural'] and v['natural'][0] == 'natural3' and v['attr']]
NAMES.update({k: NEWDEV[k]['short'] for k in NEW3})
# colors of Figure self_crossmodel; the two natural-order-only models get two hues not used there
COL = {'gemma4_31b': '#c49a00', 'exaone4_32b': '#1098ad', 'seed_oss_36b': '#8c564b', 'glm46v_flash': '#1baf7a',
       'granite41_30b': '#7383c9', 'olmo2_7b': '#b8b85a', 'dsv41_flash': '#d9480f', 'ministral3_14b': '#5c5f66'}
COL.update({k: NEWDEV[k]['color'] for k in NEW3})
# label positions (data coordinates) of the natural-order points
LABEL = {'gemma4_31b': (0.60, 0.94, 'right'), 'exaone4_32b': (0.92, 0.32, 'right'), 'dsv41_flash': (0.92, 0.42, 'right'),
         'glm46v_flash': (0.80, 0.10, 'right'), 'granite41_30b': (0.80, -0.08, 'right'), 'seed_oss_36b': (0.30, 0.10, 'right'),
         'olmo2_7b': (0.43, -0.13, 'left'), 'ministral3_14b': (0.03, -0.17, 'left'),
         'moonlight16b': (0.62, 0.20, 'right'), 'mimo7b_rl': (0.97, -0.13, 'right'), 'phi4': (0.97, -0.18, 'right')}


def union_of(content, onset):
    """share of layers singled out at content or onset (>= 80% of held-out bodies), and whether that covers the last third"""
    u = (np.asarray(content, float) >= 0.8) | (np.asarray(onset, float) >= 0.8)
    L = len(u)
    return float(u.sum()) / L, bool(u[int(np.ceil(2 * L / 3)):].all())


def claim(attr):
    return attr['assistant']['self'] - max(attr['user']['self'], attr['system']['self'])


def main_(A3, A8, out_dir, points_json):
    main_runs = {'exaone4_32b': run_path(A3, 'suite/exaone4_32b_en/summary.json'), 'seed_oss_36b': run_path(A3, 'suite/seed_oss_36b/summary.json'),
                 'granite41_30b': run_path(A3, 'suite/granite41_30b_en/summary.json'), 'olmo2_7b': run_path(A3, 'suite/olmo2_7b_en/summary.json'),
                 'gemma4_31b': A8 / 'models/gemma4_31b/non_thinking/geo_attr/_run_summary.json',
                 'glm46v_flash': A8 / 'models/glm46v_flash/non_thinking/geo_attr/_run_summary.json'}
    nat_runs = {'exaone4_32b': run_path(A3, 'suite_nat3/exaone4_32b_en/summary.json'), 'seed_oss_36b': run_path(A3, 'suite_nat3/seed_oss_36b/summary.json'),
                'granite41_30b': run_path(A3, 'suite_nat3/granite41_30b_en/summary.json'), 'olmo2_7b': run_path(A3, 'suite_nat3/olmo2_7b_en/summary.json'),
                'gemma4_31b': run_path(A3, 'suite_nat3/gemma4_31b_en/summary.json'), 'glm46v_flash': run_path(A3, 'suite_nat3/glm46v_flash/summary.json'),
                'ministral3_14b': run_path(A3, 'suite_nat3/ministral3_14b_instruct_en/summary.json'),
                'dsv41_flash': run_path(A3, 'suite_nat3/dsv41_flash_1017/summary.json')}
    main_runs.update({k: run_path(A3, NEWDEV[k]['attr'] + '/summary.json') for k in NEW3})
    nat_runs.update({k: run_path(A3, NEWDEV[k]['natural'][1] + '/summary.json') for k in NEW3})
    P = {}
    for m, f in main_runs.items():
        s = json.load(open(f, encoding='utf-8'))
        x, lf = union_of(s['content']['share_A_lowest'], s['onset']['share_A_lowest'])
        P[m, 'main'] = (x, claim(s['attr']), lf)
    for m, f in nat_runs.items():
        s = json.load(open(f, encoding='utf-8'))
        sfx = '_mean_of_streams' if m == 'dsv41_flash' else ''
        x, lf = union_of(s['separate_content' + sfx]['share_A_lowest'], s['separate_onset' + sfx]['share_A_lowest'])
        P[m, 'natural'] = (x, claim(s['attr']), lf)
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.axhline(0, color='#bbbbbb', lw=0.8, zorder=1)
    drawn = []
    for m in NAMES:
        x, y, lf = P[m, 'natural']
        n_under = sum(abs(x - x0) < 0.02 and abs(y - y0) < 0.02 for x0, y0 in drawn)
        drawn.append((x, y))
        if (m, 'main') in P:
            x0, y0, lf0 = P[m, 'main']
            ax.scatter(x0, y0, s=30, facecolor=COL[m] if lf0 else 'white', edgecolor=COL[m], linewidth=1.2, alpha=0.55, zorder=3)
            ax.annotate('', xy=(x, y), xytext=(x0, y0), zorder=2,
                        arrowprops=dict(arrowstyle='-|>', color=COL[m], lw=1.3, shrinkA=4, shrinkB=5.5, mutation_scale=10))
            mk = 'o'
        else:
            mk = 's'
        if n_under:
            # a point (nearly) on top of an earlier one is drawn as a ring around it, as in Figure self_crossmodel
            ax.scatter(x, y, s=58 + 90 * n_under, marker=mk, facecolor='none', edgecolor=COL[m], linewidth=1.5, zorder=5)
        else:
            ax.scatter(x, y, s=58, marker=mk, facecolor=COL[m] if lf else 'white', edgecolor=COL[m] if not lf else 'white',
                       linewidth=1.6 if not lf else 0.8, zorder=4)
        tx, ty, ha = LABEL.get(m, (x + 0.03, y + 0.06, 'left'))
        ax.annotate(NAMES[m], (x, y), xytext=(tx, ty), textcoords='data', ha=ha, va='center', fontsize=7.8, color='#333333',
                    arrowprops=dict(arrowstyle='-', color='#9a9a9a', lw=0.6, shrinkA=1.5, shrinkB=4))
    hs = [plt.Line2D([], [], ls='', marker='o', ms=5, mfc='white', mec='#777777', alpha=0.6, label='main probe'),
          plt.Line2D([], [], ls='', marker='o', ms=7, mfc='#777777', mec='white', label='natural order'),
          plt.Line2D([], [], ls='', marker='s', ms=7, mfc='#777777', mec='white', label='natural order only'),
          plt.Line2D([], [], ls='', marker='o', ms=7, mfc='white', mec='#777777', mew=1.5, label='last third not covered')]
    ax.legend(handles=hs, loc='upper left', fontsize=7.6, frameon=False, handletextpad=0.3, borderaxespad=0.2)
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.215, 1.05)
    ax.set_xlabel('self-index: share of layers singled out at message content or response onset')
    ax.set_ylabel('self-reference\n(own minus other roles, "you")')
    ax.spines[['top', 'right']].set_visible(False)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f'{out_dir}/self_natural.pdf', bbox_inches='tight')
    fig.savefig(f'{out_dir}/self_natural.png', dpi=170, bbox_inches='tight')
    pts = {f'{m}/{k}': [round(v[0], 3), round(v[1], 3), v[2]] for (m, k), v in P.items()}
    json.dump(pts, open(points_json, 'w'), indent=1)
    for k, v in pts.items():
        print(k, v)


def main(argv=None):
    args = arguments(__doc__).parse_args(argv)
    r = Roots(args)
    main_(r.res(W3), r.res(M8), str(r.out(FIGURES)), str(r.out(W3 + '/stats') / 'self_natural_points_v1.json'))


if __name__ == '__main__':
    main()
