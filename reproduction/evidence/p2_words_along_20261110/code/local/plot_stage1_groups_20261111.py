"""Stage-1 scatter plots (11-11, internal; not for the paper or the release). Main probe, 75 held-out bodies, 32 word pairs, one
setting per model (the rendering of the paper's main table). Groups follow the paper's Table 5. Two measures on the same y scale:
the self share minus the median word share, and the self share minus the share of U-S. Top: against the content-index coverage
(Table tab:self_kl). Bottom: by group. Bars: 95% family-resampling intervals from the analysis.
usage: plot_stage1_groups_20261111.py ANALYSIS_JSON COVERAGE_JSON OUT_PNG"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

A = json.load(open(sys.argv[1], encoding='utf-8'))
COV = json.load(open(sys.argv[2], encoding='utf-8'))
U = A['units']['b150']['main']
GROUPS = [('self-reference', '#2a78d6', 'o', ['qwen38_27b', 'qwen38_flashnext', 'gpt_oss_20b_en', 'gpt_oss_120b_en', 'exaone4_32b_en',
                                                'mimo_v26_distill_qwen9b']),
          ('self-index only', '#eb6834', 's', ['glm46v_flash', 'glm47_flash', 'r1_distill_llama70b', 'phi4_en', 'granite42_30b_en',
                                                'granite41_30b_en']),
          ('neither', '#1baf7a', '^', ['olmo2_13b_en', 'olmo2_7b_en', 'apertus8b_en', 'moonlight16b_a3b_instruct', 'mimo7b_rl_0530',
                                       'mimo_vl7b_rl_2508', 'ernie45_21b_a3b_thinking', 'seed_oss_36b', 'gemma4_31b_en', 'qwen36_27b',
                                       'qwen36_35b_a3b', 'qwen35_27b'])]
SHORT = {'qwen38_27b': 'Qwen3.8', 'qwen38_flashnext': 'Flash-Next', 'gpt_oss_20b_en': 'gpt-oss-20b', 'gpt_oss_120b_en': 'gpt-oss-120b',
         'exaone4_32b_en': 'EXAONE', 'mimo_v26_distill_qwen9b': 'MiMo-V2.6', 'glm46v_flash': 'GLM-4.6V', 'glm47_flash': 'GLM-4.7',
         'r1_distill_llama70b': 'R1-70B', 'phi4_en': 'phi-4', 'granite42_30b_en': 'Granite-4.2', 'granite41_30b_en': 'Granite-4.1',
         'olmo2_13b_en': 'OLMo-13B', 'olmo2_7b_en': 'OLMo-7B', 'apertus8b_en': 'Apertus', 'moonlight16b_a3b_instruct': 'Moonlight',
         'mimo7b_rl_0530': 'MiMo-7B', 'mimo_vl7b_rl_2508': 'MiMo-VL', 'ernie45_21b_a3b_thinking': 'ERNIE', 'seed_oss_36b': 'Seed',
         'gemma4_31b_en': 'Gemma', 'qwen36_27b': 'Qwen3.6-27B', 'qwen36_35b_a3b': 'Qwen3.6-35B', 'qwen35_27b': 'Qwen3.5'}
MEAS = [('advantage_vs_word_median', 'own vs external, minus the median ordinary word'),
        ('advantage_vs_US', 'own vs external, minus User vs System')]
INK, MUTED, GRID = '#1f1f1e', '#6b6b68', '#e6e6e3'


def rank(a):
    a = np.asarray(a, float)
    r = np.empty(len(a))
    r[a.argsort(kind='mergesort')] = np.arange(len(a))
    for v in np.unique(a):
        m = a == v
        r[m] = r[m].mean()
    return r


def spearman(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def perm(x, y, n=10000, seed=20261111):
    rng = np.random.default_rng(seed)
    r0 = spearman(x, y)
    null = np.array([spearman(rng.permutation(x), y) for _ in range(n)])
    return r0, float((np.abs(null) >= abs(r0) - 1e-12).mean())


plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlesize': 9.5, 'axes.titlecolor': INK, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6), sharey=True)
allv = []
for j, (key, label) in enumerate(MEAS):
    xs, ys, lv = [], [], []
    for gi, (gname, col, mk, tags) in enumerate(GROUPS):
        for k, t in enumerate(tags):
            c = COV['main'][t]
            x = c['layers'] / c['of']
            v = U[t][key]
            y, lo, hi = v['value'], v['ci'][0], v['ci'][1]
            allv += [lo, hi]
            xs.append(x); ys.append(y); lv.append(2 - gi)
            a = ax[0, j]
            a.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt='none', ecolor=col, elinewidth=1, alpha=0.55, capsize=0)
            a.scatter(x, y, s=46, marker=mk, color=col, edgecolor='#fcfcfb', linewidth=1.2, zorder=3,
                      label=gname if k == 0 else None)
            a.annotate(SHORT[t], (x, y), xytext=(5, 2), textcoords='offset points', fontsize=6.4, color=MUTED)
            b = ax[1, j]
            jx = gi + (k - (len(tags) - 1) / 2) * (0.5 / max(len(tags) - 1, 1))
            b.errorbar(jx, y, yerr=[[y - lo], [hi - y]], fmt='none', ecolor=col, elinewidth=1, alpha=0.55)
            b.scatter(jx, y, s=46, marker=mk, color=col, edgecolor='#fcfcfb', linewidth=1.2, zorder=3)
            b.annotate(SHORT[t], (jx, y), xytext=(4, -2), textcoords='offset points', fontsize=5.8, color=MUTED, rotation=0)
        gy = [U[t][key]['value'] for t in tags]
        med, q1, q3 = np.median(gy), *np.quantile(gy, [0.25, 0.75])
        ax[1, j].hlines(med, gi - 0.32, gi + 0.32, color=INK, linewidth=2, zorder=4)
        ax[1, j].text(gi + 0.34, med, f'median {med:+.2f}\nIQR {q3 - q1:.2f}', va='center', fontsize=6.6, color=INK)
    rc, pc = perm(xs, ys)
    rl, pl = perm(lv, ys)
    ax[0, j].set_title(f'{label}\nSpearman with coverage {rc:.2f} (perm p {pc:.3f}), n = {len(xs)}')
    ax[0, j].set_xlabel('content-index coverage (share of layers, paper Table tab:self_kl)')
    ax[1, j].set_title(f'by group (Table 5); Spearman with group order {rl:.2f} (perm p {pl:.3f})')
    ax[1, j].set_xticks([0, 1, 2], [g[0] for g in GROUPS])
    ax[1, j].set_xlim(-0.5, 2.9)
    for a in (ax[0, j], ax[1, j]):
        a.axhline(0, color=MUTED, linewidth=0.8, linestyle='--', zorder=1)
        a.grid(axis='y', color=GRID, linewidth=0.6)
        a.spines[['top', 'right']].set_visible(False)
for a in ax[:, 0]:
    a.set_ylabel('difference in share kept (last third vs first third)')
ax[0, 0].legend(frameon=False, loc='lower right', fontsize=7.5)
fig.suptitle('Words along the message, stage 1: main probe, 75 held-out bodies, 32 ordinary-word pairs, 24 models '
             '(GLM-5.3-Flash pending); bars 95% family-resampling intervals', fontsize=9, color=INK, y=0.995)
fig.tight_layout()
fig.savefig(sys.argv[3], dpi=170)
print('saved', sys.argv[3])
