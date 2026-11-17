"""Draft of the evidence figure of the final paper (11-14): the three lines of evidence for the self-index in one aligned figure,
one row per model, rows grouped as in the paper (self-reference, self-index only, neither level).
  (a) geometry: share of layers at which the own message is singled out in >= 80% of the held-out bodies, on message content and
      at the response onset (main: paper Table tab:self_crossmodel; natural: appendix tables tab:self_natural, tab:self_natural3);
  (b) next-token KL (content KL) in units of the ordinary-word KL (median of 32 word pairs = 1): external roles U||S and
      self = min(A||U, A||S) (A||U alone where the order has no System message); grey bar = range of the word pairs;
  (c) retention along the message (share kept): ordinary words (median, range), external roles U-S, self = min(A-U, A-S), 95% intervals.
Inputs: results/final_20261113/final_tables_20261113.json (rows, groups, KL sources), results/stage2_20261113/analysis_b150_20261113.json.
usage: fig_evidence_draft_20261114.py main|natural OUT_PREFIX"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EV = Path(__file__).resolve().parents[3]
WA = EV / 'p2_words_along_20261110'
FT = json.loads((WA / 'results/final_20261113/final_tables_20261113.json').read_text(encoding='utf-8'))
AN = json.loads((WA / 'results/stage2_20261113/analysis_b150_20261113.json').read_text(encoding='utf-8'))['units']['b150']

# layers with the self-index: (content, onset, of). Four-stream models: mean of streams, as in the paper table.
GEO_MAIN = {'qwen38_27b': (64, 64, 64), 'gpt_oss_120b_en': (25, 33, 36), 'gpt_oss_20b_en': (20, 24, 24), 'qwen38_flashnext': (38, 48, 48),
            'glm53_flash': (18, 19, 45), 'mimo_v26_distill_qwen9b': (32, 32, 32), 'exaone4_32b_en': (64, 64, 64),
            'glm46v_flash': (40, 40, 40), 'glm47_flash': (34, 45, 47), 'r1_distill_llama70b': (0, 80, 80), 'phi4_en': (35, 35, 40),
            'granite42_30b_en': (30, 59, 64), 'granite41_30b_en': (17, 32, 64), 'olmo2_13b_en': (4, 13, 40), 'olmo2_7b_en': (5, 6, 32),
            'apertus8b_en': (0, 0, 32), 'moonlight16b_a3b_instruct': (0, 0, 27), 'mimo7b_rl_0530': (0, 0, 36),
            'mimo_vl7b_rl_2508': (0, 0, 36), 'ernie45_21b_a3b_thinking': (0, 0, 28), 'seed_oss_36b': (0, 0, 64),
            'gemma4_31b_en': (0, 0, 60), 'qwen36_27b': (6, 1, 64), 'qwen36_35b_a3b': (1, 0, 40), 'qwen35_27b': (13, 0, 64)}
GEO_NAT = {'exaone4_32b_en': (64, 64, 64), 'gemma4_31b_en': (0, 40, 60), 'glm46v_flash': (36, 16, 40), 'granite41_30b_en': (61, 54, 64),
           'seed_oss_36b': (3, 22, 64), 'olmo2_7b_en': (5, 15, 32), 'ernie45_21b_a3b_thinking': (16, 28, 28),
           'mimo7b_rl_0530': (34, 27, 36), 'moonlight16b_a3b_instruct': (27, 27, 27), 'phi4_en': (25, 37, 40),
           'dsv41_flash': (38, 40, 40), 'ministral3_14b_instruct_en': (0, 0, 40), 'ministral3_14b_reasoning_en': (0, 0, 40)}
ROW_ORDER_MAIN = ['qwen38_27b', 'gpt_oss_120b_en', 'gpt_oss_20b_en', 'qwen38_flashnext', 'glm53_flash', 'mimo_v26_distill_qwen9b',
                  'exaone4_32b_en', 'glm46v_flash', 'glm47_flash', 'r1_distill_llama70b', 'phi4_en', 'granite42_30b_en',
                  'granite41_30b_en', 'olmo2_13b_en', 'olmo2_7b_en', 'apertus8b_en', 'moonlight16b_a3b_instruct', 'mimo7b_rl_0530',
                  'mimo_vl7b_rl_2508', 'ernie45_21b_a3b_thinking', 'seed_oss_36b', 'gemma4_31b_en', 'qwen36_27b', 'qwen36_35b_a3b',
                  'qwen35_27b']
SHORT = {'DeepSeek-R1-Distill-Llama-70B': 'R1-Distill-Llama-70B', 'MiMo-V2.6-Distill-Qwen-9B': 'MiMo-V2.6-Distill-9B',
         'MiMo-VL-7B-RL-2508': 'MiMo-VL-7B', 'Ministral-3-14B-Instruct': 'Ministral-3-14B-Inst.',
         'Ministral-3-14B-Reasoning': 'Ministral-3-14B-Reas.'}
GLAB = {'self-reference': 'Self-reference', 'self-index only': 'Self-index only', 'neither': 'Neither level'}
C_SELF, C_ROLE, C_WORD, INK, MUTED, GRID, BAND = '#2a78d6', '#eb6834', '#8d8d89', '#1f1f1e', '#6b6b68', '#e4e4e0', '#f4f4f1'


def kl_row(src):
    s = json.loads((EV / src / 'summary.json').read_text(encoding='utf-8'))['kinds']['kl_content']
    tu = s.get('to_US') or {}
    if 'A-U' not in tu:   # no System message: only self against words
        w = s['word_median_to_AU_12']
        return {'self': 1 / w['ratio'], 'role': None, 'wlo': w['min'] / w['ratio'], 'whi': w['max'] / w['ratio']}
    w = s['word_median_to_US_12']
    return {'self': min(tu['A-U']['ratio'], tu['A-S']['ratio']) / w['ratio'], 'role': 1 / w['ratio'],
            'wlo': w['min'] / w['ratio'], 'whi': w['max'] / w['ratio']}


def main():
    order, out = sys.argv[1], sys.argv[2]
    rows = [r for r in FT['rows'][order] if r['group'] in GLAB]
    if order == 'main':
        rows.sort(key=lambda r: ROW_ORDER_MAIN.index(r['tag']))
    geo = GEO_MAIN if order == 'main' else GEO_NAT
    groups = [g for g in GLAB if any(r['group'] == g for r in rows)]
    # y positions with a gap between groups
    y, ypos, spans = 0.0, {}, []
    for g in groups:
        y0 = y
        for r in rows:
            if r['group'] == g:
                ypos[r['tag']] = y
                y += 1
        spans.append((g, y0 - 0.5, y - 0.5))
        y += 0.9
    n = len(rows)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 0.235 * n + 1.9), sharey=True, gridspec_kw={'width_ratios': [0.8, 1.25, 1.25], 'wspace': 0.07})
    for ax in axs:
        for i, (g, a, b) in enumerate(spans):
            ax.axhspan(a, b, color=BAND if i % 2 == 0 else 'white', zorder=0, lw=0)
        ax.set_ylim(y - 0.9 - 0.4, -0.6)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=7, colors=MUTED, length=2.5)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    axs[0].set_yticks([ypos[r['tag']] for r in rows])
    axs[0].set_yticklabels([SHORT.get(r['name'], r['name']) for r in rows], fontsize=7.2, color=INK)
    for g, a, b in spans:
        axs[0].text(-0.95, (a + b) / 2, GLAB[g], rotation=90, va='center', ha='center', fontsize=7.5, color=MUTED,
                    transform=axs[0].get_yaxis_transform(), clip_on=False)
    # (a) geometry
    ax = axs[0]
    for r in rows:
        yy = ypos[r['tag']]
        if r['tag'] not in geo:
            ax.text(0.5, yy, 'main probe only', fontsize=5.8, color=MUTED, ha='center', va='center', style='italic')
            continue
        c, o, of = geo[r['tag']]
        ax.plot([c / of, o / of], [yy, yy], color=C_SELF, lw=1.0, alpha=0.45, zorder=2)
        ax.plot(o / of, yy, 'o', ms=4.6, mfc='white', mec=C_SELF, mew=1.1, zorder=3)
        ax.plot(c / of, yy, 'o', ms=4.6, mfc=C_SELF, mec='white', mew=0.7, zorder=4)
    ax.set_xlim(-0.06, 1.06)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '0.5', '1'])
    ax.set_xlabel('share of layers\nwith the self-index', fontsize=7.4, color=INK)
    ax.set_title('a  Geometry', fontsize=8.6, color=INK, loc='left', fontweight='bold')
    # (b) next-token KL in units of the ordinary-word KL
    ax = axs[1]
    for r in rows:
        yy = ypos[r['tag']]
        k = kl_row(r['kl']['src'])
        ax.plot([k['wlo'], k['whi']], [yy, yy], color=C_WORD, lw=2.6, alpha=0.45, solid_capstyle='butt', zorder=2)
        ax.plot(1, yy, '|', ms=6, color=C_WORD, mew=1.3, zorder=3)
        if k['role'] is not None:
            ax.plot(k['role'], yy, 's', ms=4.4, mfc=C_ROLE, mec='white', mew=0.7, zorder=4)
        ax.plot(k['self'], yy, 'o', ms=5.0, mfc=C_SELF, mec='white', mew=0.7, zorder=5)
    ax.set_xscale('log')
    ax.set_xlim(0.03, 250)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(['0.1', '1', '10', '100'])
    ax.axvline(1, color=C_WORD, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xlabel('next-token KL, in units of the\nordinary-word KL (log scale)', fontsize=7.4, color=INK)
    ax.set_title('b  Next-token KL', fontsize=8.6, color=INK, loc='left', fontweight='bold')
    # (c) retention
    ax = axs[2]
    for r in rows:
        yy = ypos[r['tag']]
        u = AN[order][r['tag']]
        q = u['word_share_quantiles']
        ax.plot([q[0], q[4]], [yy, yy], color=C_WORD, lw=2.6, alpha=0.45, solid_capstyle='butt', zorder=2)
        ax.plot(q[2], yy, '|', ms=6, color=C_WORD, mew=1.3, zorder=3)
        if 'U-S' in u['rows']:
            us = u['rows']['U-S']
            ax.plot(us['share_ci'], [yy, yy], color=C_ROLE, lw=0.9, zorder=3)
            ax.plot(us['share'], yy, 's', ms=4.4, mfc=C_ROLE, mec='white', mew=0.7, zorder=4)
        ax.plot(u['self']['share_ci'], [yy, yy], color=C_SELF, lw=0.9, zorder=4)
        ax.plot(u['self']['share'], yy, 'o', ms=5.0, mfc=C_SELF, mec='white', mew=0.7, zorder=5)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '0.5', '1'])
    ax.set_xlabel('share of the difference kept from the\nfirst to the last third of the message', fontsize=7.4, color=INK)
    ax.set_title('c  Retention along message', fontsize=8.6, color=INK, loc='left', fontweight='bold')
    # legend
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', ms=5, mfc=C_SELF, mec='white', label='own message against external (self)'),
         Line2D([], [], marker='s', ls='', ms=4.6, mfc=C_ROLE, mec='white', label='User against System (external roles)'),
         Line2D([], [], marker='|', ls='-', lw=2.6, alpha=0.6, ms=6, color=C_WORD, mew=1.3, label='ordinary word pairs (median, range)'),
         Line2D([], [], marker='o', ls='', ms=4.6, mfc='white', mec=C_SELF, mew=1.1, label='a: response onset (filled: message content)')]
    fig.legend(handles=h, loc='upper center', ncol=2, fontsize=6.9, frameon=False, bbox_to_anchor=(0.6, 1.0), columnspacing=1.6,
               handletextpad=0.5, labelcolor=INK)
    fig.subplots_adjust(left=0.265, right=0.975, top=1 - 0.78 / (0.235 * n + 1.9), bottom=0.9 / (0.235 * n + 1.9))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out + '.png', dpi=220, facecolor='white')
    fig.savefig(out + '.pdf', facecolor='white')
    print('written', out, 'rows', n)


if __name__ == '__main__':
    main()
