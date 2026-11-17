"""Figure self_wording (fig:self_wording, Appendix "How the Question Names the Model"): the edit of the causal tests under three
wordings of the answer option that names the model, in the seven readouts of question_wording.py. Style as Figure
self_mechanism (causal_tests.py: fonts, readout colours, bar widths, error bars, grid, legend).
Two panels: (a) removal (own message <- User coordinates, change from the unedited own message) and (b) installation (User
message <- own coordinates, change from the unedited User message). x: the three wordings (main "You (the AI assistant)" /
你（AI 助手）, pronoun "You" / 你, role name "The AI assistant" / AI 助手); bars: the readouts; height: the item-paired change of the
option naming the model, error bars: its 95% interval over the 15 wording families. Random edits are not drawn. Qwen3.8-27B with
the English items: the blues of the Chinese items, hatched.
Input: evidence/p2_wording_swap_20261109/results/wording_summary_all.json from question_wording.py (in OUT, else the archived copy).
Outputs: paper/source/figures/self_wording.pdf/.png.

usage: fig_wording.py --results RES --out OUT
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, WS, Roots, arguments, prefer  # noqa: E402

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'pdf.fonttype': 42, 'axes.titlesize': 9, 'axes.titleweight': 'bold',
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'hatch.linewidth': 0.7})
C_INK, C_GRID, C_ZERO = '#2b2f36', '#e3e6ea', '#9aa1a9'
READOUTS = [  # key in the summary, legend label, colour (as causal_tests.C_RO), hatch
    ('qsp', 'Qwen3.8-27B, no thinking, Chinese items', '#8dbbef', None),
    ('qth', 'Qwen3.8-27B, thinking, Chinese items', '#2a78d6', None),
    ('qsp_en', 'Qwen3.8-27B, no thinking, English items', '#8dbbef', '////'),
    ('qth_en', 'Qwen3.8-27B, thinking, English items', '#2a78d6', '////'),
    ('g20', 'gpt-oss-20b, thinking', '#a61e4d', None),
    ('g120', 'gpt-oss-120b, thinking', '#e64980', None),
    ('m26', 'MiMo-V2.6-Distill-9B, default call', '#f08c00', None)]
WORDINGS = [('paper', 'main\n"You (the AI assistant)"'), ('pron', 'pronoun\n"You"'), ('role', 'role name\n"The AI assistant"')]
# the names of the edits in the summary: 去除 = removal, 植入 = installation
PANELS = [('去除', '(a) removal: own message ← User', (-1.06, 0.06), [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0]),
          ('植入', '(b) installation: User message ← own', (-0.2, 0.92), [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])]


def draw(S, F):
    F.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.75), gridspec_kw={'wspace': 0.2})
    w = 0.118
    missing = []
    for ax, (edit, title, ylim, yt) in zip(axes, PANELS):
        for j, (wd, _) in enumerate(WORDINGS):
            for i, (ro, _, col, hatch) in enumerate(READOUTS):
                s = S.get(ro, {}).get('wordings', {}).get(wd, {}).get(edit, {}).get('self')
                if s is None:
                    missing.append(f'{ro}/{wd}/{edit}')
                    continue
                m, (lo, hi) = s['mean'], s['ci_families']
                x = j + (i - (len(READOUTS) - 1) / 2) * w
                ax.bar(x, m, width=w * 0.92, color=col, edgecolor='white', lw=0.4, hatch=hatch, zorder=3)
                ax.errorbar(x, m, yerr=[[max(0.0, m - lo)], [max(0.0, hi - m)]], fmt='none', ecolor=C_INK, lw=0.7, capsize=1.2, zorder=4)
                print(f'{title[:3]} {wd:5s} {ro:6s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]')
        ax.axhline(0, color=C_ZERO, lw=0.7, zorder=2)
        ax.set_xticks(range(len(WORDINGS)))
        ax.set_xticklabels([lab for _, lab in WORDINGS], fontsize=7.2)
        ax.set_xlim(-0.55, len(WORDINGS) - 0.45)
        ax.set_ylim(*ylim)
        ax.set_yticks(yt)
        ax.set_title(title)
        ax.grid(axis='y', color=C_GRID, lw=0.6, zorder=0)
        ax.tick_params(axis='x', length=0)
        ax.spines['bottom'].set_visible(False)
    axes[0].set_ylabel('change in claim of the message\n(option naming the model)')
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=col, edgecolor='white', lw=0.4, hatch=hatch) for _, _, col, hatch in READOUTS]
    fig.legend(handles, [t for _, t, _, _ in READOUTS], loc='lower center', ncol=4, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.145), handlelength=1.3, handleheight=1.1, columnspacing=0.8, handletextpad=0.4)
    fig.savefig(F / 'self_wording.pdf', bbox_inches='tight')
    fig.savefig(F / 'self_wording.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('missing:', ', '.join(missing) or 'none')


def main(argv=None):
    args = arguments(__doc__).parse_args(argv)
    r = Roots(args)
    S = json.loads(prefer(r, WS + '/results/wording_summary_all.json').read_text(encoding='utf-8'))['by_readout']
    draw(S, r.out(FIGURES))


if __name__ == '__main__':
    main()
