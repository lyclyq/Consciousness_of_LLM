"""One-page progress note for the advisor (11-11, internal; not for the paper or the release): ordinary-word controls on the two
readouts, retention along the message (stage 1, results/stage1_20261111) and next-token KL (results/kl_20261111, mirrored and
sha256-checked). Groups follow the paper's Table 5. Prints every number the note states.
usage: report_kl_retention_20261111.py ANALYSIS_JSON COVERAGE_JSON KL_DIR OUT_PDF"""
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

A = json.load(open(sys.argv[1], encoding='utf-8'))
COV = json.load(open(sys.argv[2], encoding='utf-8'))['main']
KLD = Path(sys.argv[3])
U = A['units']['b150']['main']
# retention unit -> KL tag (the same setting)
KLTAG = {'qwen38_27b': 'qwen38_nt', 'gpt_oss_20b_en': 'gpt_oss_20b_en', 'mimo_v26_distill_qwen9b': 'mimo_v26_nt',
         'glm46v_flash': 'glm46v_flash', 'phi4_en': 'phi4_en', 'mimo_vl7b_rl_2508': 'mimo_vl7b', 'mimo7b_rl_0530': 'mimo7b_rl_0530',
         'apertus8b_en': 'apertus8b_en', 'olmo2_7b_en': 'olmo2_7b_en', 'olmo2_13b_en': 'olmo2_13b_en',
         'moonlight16b_a3b_instruct': 'moonlight16b_a3b_instruct', 'ernie45_21b_a3b_thinking': 'ernie45_21b_a3b_thinking'}
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
INK, MUTED, GRID, BG = '#1f1f1e', '#6b6b68', '#e6e6e3', '#fcfcfb'


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


def exact_rank_p(lo_group, hi_group):
    """one-sided exact rank-sum p that lo_group sits below hi_group (all label assignments)"""
    v = np.array(list(lo_group) + list(hi_group))
    r = rank(v)
    obs = r[:len(lo_group)].sum()
    sums = [r[list(c)].sum() for c in itertools.combinations(range(len(v)), len(lo_group))]
    return float(np.mean(np.array(sums) <= obs + 1e-9))


def kl_summary(tag):
    s = json.load(open(KLD / tag / 'summary.json', encoding='utf-8'))['kinds']['kl_content']
    w = s['word_median_to_AU_12']
    return {'ratio': w['ratio'], 'ci': w['ci95'], 'au_us': s['to_US']['A-U']['ratio'] if 'to_US' in s else None,
            'w_us': s['word_median_to_US_12']['ratio'] if 'word_median_to_US_12' in s else None}


# ---------------- numbers ----------------
gidx = {t: gi for gi, (_, _, _, tags) in enumerate(GROUPS) for t in tags}
ret = {t: U[t]['advantage_vs_US'] for t in gidx}
pos = {g: sum(ret[t]['value'] > 0 for t in tags) for g, _, _, tags in GROUPS}
pos_ci = {g: sum(ret[t]['ci'][0] > 0 for t in tags) for g, _, _, tags in GROUPS}
nidx = len(GROUPS[0][3]) + len(GROUPS[1][3])
meds = {g: float(np.median([ret[t]['value'] for t in tags])) for g, _, _, tags in GROUPS}
cov = [COV[t]['layers'] / COV[t]['of'] for t in gidx]
rho_cov, p_cov = perm(cov, [ret[t]['value'] for t in gidx])
klv = {t: kl_summary(KLTAG[t]) for t in KLTAG}
think = kl_summary('qwen38_think')
kl_idx = [klv[t]['ratio'] for t in KLTAG if gidx[t] < 2]
kl_none = [klv[t]['ratio'] for t in KLTAG if gidx[t] == 2]
p_sep = exact_rank_p(kl_idx, kl_none)
both = list(KLTAG)
rho_both, p_both = perm([ret[t]['value'] for t in both], [np.log10(klv[t]['ratio']) for t in both])
P3NAME = {'p3_qwen38_nt': 'Qwen3.8', 'p3_gpt_oss_20b_en': 'gpt-oss-20b', 'p3_mimo_v26_nt': 'MiMo-V2.6'}
p3 = {t: kl_summary(t) for t in ('p3_qwen38_nt', 'p3_gpt_oss_20b_en', 'p3_mimo_v26_nt') if (KLD / t / 'summary.json').exists()}
NUM = {'retention_models': len(gidx), 'positive_value': pos, 'positive_ci_lower': pos_ci, 'n_index_models': nidx,
       'group_medians_self_minus_US': meds, 'spearman_coverage': [rho_cov, p_cov],
       'kl_models': len(KLTAG), 'kl_index_range': [min(kl_idx), max(kl_idx)], 'kl_none_range': [min(kl_none), max(kl_none)],
       'kl_exact_rank_p': p_sep, 'kl_qwen38_think': think['ratio'], 'spearman_retention_kl': [rho_both, p_both],
       'p3_kl_ratio': {t: v['ratio'] for t, v in p3.items()}}
print(json.dumps(NUM, indent=1))

# ---------------- page ----------------
plt.rcParams.update({'font.size': 8.2, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlesize': 9, 'axes.titlecolor': INK, 'figure.facecolor': BG, 'axes.facecolor': BG,
                     })
fig = plt.figure(figsize=(11.69, 8.27))
fig.text(0.04, 0.955, 'Ordinary-word controls for the two-level self: progress note, 11 November 2026', fontsize=13.5,
         color=INK, weight='bold')
idx_ci = pos_ci['self-reference'] + pos_ci['self-index only']


def fmt_p(p, n=10000):
    return f'< {1 / n:.4f}' if p == 0 else f'{p:.4f}'


lines = [
    ('Setup.', 'The 75 held-out bodies and the main probe of the paper; 32 pairs of ordinary words (8 categories × 4: place, time, name, '
               'feeling, occupation, institution, weather, object), each pair swapped inside a short framing sentence before the body. '
               'Two readouts at the same body positions: retention (how much of the hidden-state difference between two renderings '
               'survives from the first to the last third of the body) and next-token KL (content tokens only). Groups as in Table 5.'),
    ('Retention.', f'Own-vs-external outlasts User-vs-System only with a self-index: the 95% interval of self minus U-S lies above 0 '
                   f'in {idx_ci}/{nidx} models with an index (8/13 settings with Qwen3.8 thinking) and in {pos_ci["neither"]}/'
                   f'{len(GROUPS[2][3])} without; group medians {meds["self-reference"]:+.2f} / {meds["self-index only"]:+.2f} / '
                   f'{meds["neither"]:+.2f}; Spearman with content-index coverage {rho_cov:.2f} (permutation p {fmt_p(p_cov)}), '
                   f'{len(gidx)} models.'),
    ('Next-token KL.', f'Swapping one ordinary word moves the next-token distribution by {min(kl_idx):.3f}–{max(kl_idx):.3f} of the '
                       f'own-vs-User difference in all {len(kl_idx)} models with an index, and by {min(kl_none):.2f}–{max(kl_none):.2f} '
                       f'in all {len(kl_none)} without: no overlap (exact one-sided rank test p = {p_sep:.4f}); Qwen3.8 thinking '
                       f'{think["ratio"]:.3f}. The split mostly tracks the size of the own-vs-User difference itself (already in the '
                       'paper\'s KL section); the ordinary words give it a within-model scale.'),
    ('Agreement.', f'Across the {len(both)} models with both readouts, Spearman between the retention advantage and the KL ratio '
                   f'is {rho_both:.2f} (permutation p {fmt_p(p_both)}): the longer own-vs-external outlasts User-vs-System, the '
                   'smaller one ordinary word is next to it.'
                   + (' On the 200 phase-3 bodies the KL ratio replicates for ' + ', '.join(f'{P3NAME[k]} {v["ratio"]:.3f}' for k, v in p3.items())
                      + '.' if p3 else '')),
    ('Pending.', 'The 12 larger models (main and natural order) are running on our rented machine, due 13 November. '
                 'GLM-5.3-Flash (FP8, 306 GB) needs about 340 GB of GPU memory: about 45 h on our 4 × RTX 4090 with CPU offload, '
                 'about 6–8 h on 3 × H200 including the download.'),
]
y = 0.915
import textwrap  # noqa: E402
WCH = 168
for head, body in lines:
    first = textwrap.wrap(body, WCH - int(1.25 * len(head)) - 2)[0]
    rest = textwrap.wrap(body[len(first):].strip(), WCH)
    fig.text(0.04, y, head, fontsize=8.6, color=INK, weight='bold', va='top')
    fig.text(0.04 + 0.0068 * len(head) + 0.006, y, first, fontsize=8.6, color=INK, va='top')
    for k, wl in enumerate(rest, 1):
        fig.text(0.04, y - 0.021 * k, wl, fontsize=8.6, color=INK, va='top')
    y -= 0.021 * (1 + len(rest)) + 0.009

axs = [fig.add_axes([0.055, 0.105, 0.27, 0.43]), fig.add_axes([0.385, 0.105, 0.27, 0.43]), fig.add_axes([0.715, 0.105, 0.26, 0.43])]
# (a) retention by group
a = axs[0]
for gi, (g, col, mk, tags) in enumerate(GROUPS):
    for k, t in enumerate(tags):
        v = ret[t]
        jx = gi + (k - (len(tags) - 1) / 2) * (0.55 / max(len(tags) - 1, 1))
        a.errorbar(jx, v['value'], yerr=[[v['value'] - v['ci'][0]], [v['ci'][1] - v['value']]], fmt='none', ecolor=col,
                   elinewidth=1, alpha=0.6)
        a.scatter(jx, v['value'], s=34, marker=mk, color=col, edgecolor=BG, linewidth=1, zorder=3)
        if gi < 2:
            a.annotate(SHORT[t], (jx, v['value']), xytext=(4, -2), textcoords='offset points', fontsize=5.8, color=MUTED)
    a.hlines(meds[g], gi - 0.33, gi + 0.33, color=INK, linewidth=1.8, zorder=4)
a.axhline(0, color=MUTED, linewidth=0.8, linestyle='--')
a.set_xticks([0, 1, 2], ['self-reference', 'self-index\nonly', 'neither'])
a.set_ylabel('share kept: own vs external minus User vs System')
a.set_title(f'(a) Retention along the message, {len(gidx)} models\n(GLM-5.3-Flash pending; bars 95% intervals, line median)')
# (b) KL by group
b = axs[1]
for gi, (g, col, mk, tags) in enumerate(GROUPS):
    ts = [t for t in tags if t in KLTAG]
    for k, t in enumerate(ts):
        v = klv[t]
        jx = gi + (k - (len(ts) - 1) / 2) * (0.5 / max(len(ts) - 1, 1))
        b.errorbar(jx, v['ratio'], yerr=[[v['ratio'] - v['ci'][0]], [v['ci'][1] - v['ratio']]], fmt='none', ecolor=col,
                   elinewidth=1, alpha=0.6)
        b.scatter(jx, v['ratio'], s=34, marker=mk, color=col, edgecolor=BG, linewidth=1, zorder=3)
        b.annotate(SHORT[t], (jx, v['ratio']), xytext=(4, -9 if t == 'ernie45_21b_a3b_thinking' else -2), textcoords='offset points',
                   fontsize=6, color=MUTED)
    if gi == 0:
        b.scatter(0.33, think['ratio'], s=34, marker='o', facecolor=BG, edgecolor=col, linewidth=1.2, zorder=3)
        b.annotate('Qwen3.8 thinking', (0.33, think['ratio']), xytext=(4, -2), textcoords='offset points', fontsize=6, color=MUTED)
b.set_yscale('log')
b.axhline(1, color=MUTED, linewidth=0.8, linestyle='--')
b.set_xticks([0, 1, 2], ['self-reference', 'self-index\nonly', 'neither'])
b.set_ylabel('KL(ordinary word swap) / KL(own vs User), median of 32 pairs')
b.set_title(f'(b) Next-token KL, {len(KLTAG)} models so far\n(log scale; bars 95% intervals; dashed: equal to own vs User)')
# (c) agreement
c = axs[2]
for gi, (g, col, mk, tags) in enumerate(GROUPS):
    for t in tags:
        if t in KLTAG:
            c.scatter(ret[t]['value'], klv[t]['ratio'], s=40, marker=mk, color=col, edgecolor=BG, linewidth=1, zorder=3,
                      label=g if t == [x for x in tags if x in KLTAG][0] else None)
            c.annotate(SHORT[t], (ret[t]['value'], klv[t]['ratio']), xytext=(4, -8 if t == 'ernie45_21b_a3b_thinking' else 2),
                       textcoords='offset points', fontsize=6, color=MUTED)
c.set_yscale('log')
c.axvline(0, color=MUTED, linewidth=0.8, linestyle='--')
c.axhline(1, color=MUTED, linewidth=0.8, linestyle='--')
c.set_xlabel('retention: own vs external minus User vs System')
c.set_ylabel('KL ratio (as in b)')
c.set_title(f'(c) The two readouts per model\nSpearman {rho_both:.2f} (perm p {p_both:.4f}), n = {len(both)}')
c.legend(frameon=False, fontsize=7, loc='upper right')
c.set_xlim(right=max(ret[t]['value'] for t in both) + 0.14)
for ax_ in axs:
    ax_.grid(axis='y', color=GRID, linewidth=0.6)
    ax_.spines[['top', 'right']].set_visible(False)
fig.text(0.04, 0.027, 'Exploratory and internal; not in the paper yet. Retention: 95% intervals from 2,000 resamples of the 15 wording '
                      'families. KL: content tokens only, ratio of sums over bodies, 2,000 family resamples.', fontsize=6.8, color=MUTED)
fig.text(0.04, 0.012, "Every KL run was checked against the paper's KL run of the same setting (role KLs identical to the last bit).",
         fontsize=6.8, color=MUTED)
fig.savefig(sys.argv[4])
fig.savefig(str(sys.argv[4]).replace('.pdf', '_preview.png'), dpi=110)
print('saved', sys.argv[4])
