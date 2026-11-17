"""Figure self_inducer (paper v5, section "What Induces the Self-Index", user 10-28/10-29): the self-index and self-reference with
and without the Assistant-only format, in the four readouts of the causal tests.
Top row: per layer, the share of held-out bodies in which the model's own message has the smallest distance ratio, at message
content (solid) and at the response onset (dashed), default rendering (dark) against the format ablation (orange): Qwen3.8-27B,
non-thinking and thinking renderings, template switch preserve_thinking=False (runs/pro/qwen38_nothink/suite_mask/qwen38_27b_{default,
switch,think_default,think_switch}); gpt-oss-20b and gpt-oss-120b, thinking rendering, default (runs/pro/qwen38_nothink/suite_think/)
against the channel tag masked with recomputed positions (suite_mask/*_maskpos). The dotted line marks 80%.
Bottom row: self-reference of the same readouts, unedited (150 held-out items): probability of "you" without thinking, share of
finished items answered "you" with thinking, for the model's own and for User messages; default from stats/v4_mechanism.json
(make_v4_mechanism.py), ablation from runs/pro/qwen38_nothink/{singlepass_swap,thinking_swap}/qwen38_27b_nothink_all and
thinking_swap/gpt_oss_{20b,120b}_mask (base_A, base_U).
usage: make_fig_inducer.py ARCHIVE OUT_DIR"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'axes.titlesize': 8.4, 'axes.titleweight': 'bold',
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False})
C_DEF, C_ABL, C_GRID = '#2a4d8f', '#e07b39', '#e3e6ea'
G = 'runs/pro/qwen38_nothink/suite_mask/'
COLS = [('qwen_sp', 'Qwen3.8-27B\nno thinking', G + 'qwen38_27b_default', G + 'qwen38_27b_switch',
         'runs/pro/qwen38_nothink/singlepass_swap/qwen38_27b_nothink_all', 'template switch'),
        ('qwen_th', 'Qwen3.8-27B\nthinking', G + 'qwen38_27b_think_default', G + 'qwen38_27b_think_switch',
         'runs/pro/qwen38_nothink/thinking_swap/qwen38_27b_nothink_all', 'template switch'),
        ('oss_th', 'gpt-oss-20b\nthinking', 'runs/pro/qwen38_nothink/suite_think/gpt_oss_20b_mxfp4_en', G + 'gpt_oss_20b_mxfp4_en_maskpos',
         'runs/pro/qwen38_nothink/thinking_swap/gpt_oss_20b_mask', 'tag masked'),
        ('oss120_th', 'gpt-oss-120b\nthinking', 'runs/pro/qwen38_nothink/suite_think/gpt_oss_120b_en', G + 'gpt_oss_120b_en_maskpos',
         'runs/pro/qwen38_nothink/thinking_swap/gpt_oss_120b_mask', 'tag masked')]


def share(d):
    s = json.load(open(Path(d) / 'summary.json', encoding='utf-8'))
    return s['content']['share_A_lowest'], s['onset']['share_A_lowest']


def claim(run, cond):
    s = json.load(open(Path(run) / 'summary.json', encoding='utf-8'))[cond]
    return s['p_self'] if 'p_self' in s else s['you_share']


def main(a, out):
    os.chdir(a)
    M = json.load(open('stats/v4_mechanism.json', encoding='utf-8'))['results']
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.1), gridspec_kw={'height_ratios': [1.25, 1], 'hspace': 0.5, 'wspace': 0.12})
    rows = {}
    for j, (ro, title, dd, da, abl_run, abl_name) in enumerate(COLS):
        ax = axes[0][j]
        for d, col, lab in ((dd, C_DEF, 'default'), (da, C_ABL, abl_name)):
            c, o = share(d)
            L = len(c)
            x = [i / (L - 1) for i in range(L)]
            ax.plot(x, c, color=col, lw=1.3, label=f'{lab}, message content')
            ax.plot(x, o, color=col, lw=1.1, ls=(0, (3, 1.5)), label=f'{lab}, response onset')
            rows[f'{ro}/{lab}'] = (sum(v >= 0.8 for v in c), sum(v >= 0.8 for v in o), L)
        ax.axhline(0.8, color='#9aa1a9', lw=0.6, ls=':')
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 1 / 3, 2 / 3, 1])
        ax.set_xticklabels(['0', '1/3', '2/3', '1'])
        ax.set_title(title)
        ax.grid(axis='y', color=C_GRID, lw=0.5)
        if j:
            ax.set_yticklabels([])
        ax = axes[1][j]
        d_own, d_user = M[ro]['all']['base_A']['stat'], M[ro]['all']['base_U']['stat']
        a_own, a_user = claim(abl_run, 'base_A'), claim(abl_run, 'base_U')
        rows[f'{ro}/claim'] = (round(d_own, 3), round(d_user, 3), round(a_own, 3), round(a_user, 3))
        for k, (v, col, x) in enumerate(((d_own, C_DEF, 0), (a_own, C_ABL, 1), (d_user, C_DEF, 2.4), (a_user, C_ABL, 3.4))):
            ax.bar(x, v, width=0.85, color=col, alpha=1.0 if x < 2 else 0.55, zorder=3)
            ax.text(x, v + 0.03, f'{v:.2f}', ha='center', va='bottom', fontsize=6.4, color='#2b2f36')
        ax.set_xticks([0.5, 2.9])
        ax.set_xticklabels(["own", "User"], fontsize=7.2)
        ax.set_ylim(0, 1.12)
        ax.tick_params(axis='x', length=0)
        ax.grid(axis='y', color=C_GRID, lw=0.5, zorder=0)
        if j:
            ax.set_yticklabels([])
    axes[0][0].set_ylabel('share of held-out bodies\nown message singled out', fontsize=7.2)
    axes[1][0].set_ylabel('claimed as own\n(P("you") / share "you")', fontsize=7.2)
    fig.text(0.5, 0.455, 'relative depth (layer / number of layers)', ha='center', fontsize=7.2)
    h = [plt.Line2D([], [], color=C_DEF, lw=1.3), plt.Line2D([], [], color=C_ABL, lw=1.3),
         plt.Line2D([], [], color='#555555', lw=1.3), plt.Line2D([], [], color='#555555', lw=1.1, ls=(0, (3, 1.5)))]
    fig.legend(h, ['default rendering', 'Assistant-only format removed (template switch) or masked (gpt-oss)', 'message content',
                   'response onset'], loc='lower center', ncol=4, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, -0.05))
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, 'self_inducer.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(out, 'self_inducer.png'), dpi=200, bbox_inches='tight')
    for k, v in rows.items():
        print(k, v)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
