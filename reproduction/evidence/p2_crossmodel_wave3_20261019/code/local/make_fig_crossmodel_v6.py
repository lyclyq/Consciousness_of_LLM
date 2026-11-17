"""Figure self_crossmodel (paper v5 restructure, 10-29: the union-axis script make_fig_crossmodel_v5.py of 10-26 plus OLMo-2-7B and
OLMo-2-13B, English, no thinking mode; their geometry from their suite summaries and their self-reference from
stats/v5_olmo_attribution_non_thinking.csv, first two option orders, written by stats_v5_olmo.py): the two levels across eighteen models, separately for the template's non-thinking and
thinking modes. Same quantities as v2/v3 (make_fig_crossmodel.py, make_fig_crossmodel_v3.py):
x = self-index: share of decoder layers at which the Assistant message is singled out (smallest distance ratio R_A) in at least
    80% of the held-out bodies at message content or at the response onset, a layer counted once (v5, user 10-26; v4 took the
    larger of the two positions) (four-stream models Qwen3.8-Flash-Next and GLM-5.3-Flash: mean of the per-stream shares);
y = self-reference: non-thinking, body-level mean P("you" | own) minus the larger of P("you" | User) and P("you" | System), first
    two option orders; thinking, share of finished items answered "you" for own messages minus the larger share for User and System.
Marker fill shows whether that union covers every layer of the last third of the depth: filled = yes, open = no.
v4 changes: three more models, Qwen3.5-27B (Chinese, both modes), Granite-4.1-30B (English, no thinking mode) and Granite-4.2-30B
(English, both modes), with geometry from their non-thinking suite summaries and self-reference from the wave-3 archive tables
written by stats_v4_crossmodel.py (v4_geometry_summary.csv, v4_attribution_non_thinking.csv, v4_thinking_counts.csv); the
v4 thinking table also replaces the earlier 60-item thinking counts of Qwen3.8-Flash-Next, Qwen3.6-27B and Seed-OSS-36B with their
900-item runs (300 items per source). Qwen3.5-27B and Granite-4.2-30B have no thinking-rendering geometry (it is required only for
models with thinking self-reference, and neither has any); their thinking points use the non-thinking geometry, the only geometry
measured for them (THINK_GEO_FROM_NON_THINKING).
Inputs: both archives' stats tables (plus the v4 tables of the wave-3 archive), per-layer shares from the run summaries and the
stream-geometry files, and the Qwen3.8-27B 900-item thinking summary (N).
usage: make_fig_crossmodel_v4.py ARCHIVE8 ARCHIVE3 N_SUMMARY OUT_DIR"""
import csv
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'axes.titlesize': 10.5,
                     'axes.titleweight': 'bold', 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5})
NAMES = {'qwen38_27b': 'Qwen3.8-27B', 'qwen38_flashnext': 'Qwen3.8-Flash-Next', 'gpt_oss_120b': 'gpt-oss-120b', 'gpt_oss_20b': 'gpt-oss-20b',
         'glm53_flash': 'GLM-5.3-Flash', 'exaone4_32b': 'EXAONE-4.0.1-32B', 'glm46v_flash': 'GLM-4.6V-Flash', 'glm47_flash': 'GLM-4.7-Flash',
         'r1_llama70b': 'R1-Distill-Llama-70B', 'seed_oss_36b': 'Seed-OSS-36B', 'gemma4_31b': 'Gemma-4-31B', 'qwen36_27b': 'Qwen3.6-27B',
         'qwen36_35b': 'Qwen3.6-35B-A3B', 'qwen35_27b': 'Qwen3.5-27B', 'granite41_30b': 'Granite-4.1-30B',
         'granite42_30b': 'Granite-4.2-30B', 'olmo2_7b': 'OLMo-2-7B', 'olmo2_13b': 'OLMo-2-13B'}
COL = {'qwen38_27b': '#2a78d6', 'qwen38_flashnext': '#6a3fb5', 'glm46v_flash': '#1baf7a', 'glm47_flash': '#0f7d58',
       'qwen36_35b': '#9aa0a6', 'qwen36_27b': '#6f767d', 'gemma4_31b': '#c49a00', 'r1_llama70b': '#eb6834',
       'exaone4_32b': '#1098ad', 'seed_oss_36b': '#8c564b', 'gpt_oss_20b': '#a61e4d', 'gpt_oss_120b': '#e64980', 'glm53_flash': '#74b816',
       'qwen35_27b': '#a3b3c6', 'granite42_30b': '#22305e', 'granite41_30b': '#7383c9', 'olmo2_7b': '#b8b85a', 'olmo2_13b': '#6b6b1f'}
# models without thinking-rendering geometry whose thinking point uses the non-thinking geometry (see docstring)
THINK_GEO_FROM_NON_THINKING = ('qwen35_27b', 'granite42_30b')
LABEL = {'non_thinking': {'glm47_flash': (0.90, 0.36, 'right'), 'qwen38_flashnext': (0.90, 0.29, 'right'),
                          'glm46v_flash': (0.90, 0.22, 'right'), 'exaone4_32b': (0.90, 0.15, 'right'),
                          'gemma4_31b': (0.08, 0.30, 'left'), 'seed_oss_36b': (0.08, 0.20, 'left'),
                          'qwen36_35b': (0.16, 0.06, 'left'), 'qwen36_27b': (0.04, -0.195, 'left'),
                          'qwen35_27b': (0.22, -0.150, 'left'), 'granite41_30b': (0.63, -0.06, 'left'),
                          'granite42_30b': (0.87, -0.125, 'right'), 'olmo2_7b': (0.33, 0.125, 'left'),
                          'olmo2_13b': (0.45, -0.195, 'left')},
         'thinking': {'gpt_oss_120b': (0.84, 1.02, 'right'), 'qwen38_27b': (0.94, 0.89, 'right'),
                      'exaone4_32b': (0.94, 0.30, 'right'), 'glm46v_flash': (0.94, 0.22, 'right'), 'glm47_flash': (0.66, 0.16, 'right'),
                      'gemma4_31b': (0.10, 0.34, 'left'), 'seed_oss_36b': (0.33, 0.10, 'left'),
                      'qwen36_27b': (0.13, -0.09, 'left'), 'qwen36_35b': (0.00, 0.24, 'left'),
                      'qwen35_27b': (0.00, 0.15, 'left'), 'granite42_30b': (0.88, -0.055, 'right'),
                      'r1_llama70b': (0.97, -0.135, 'right')}}


def rows(path):
    return list(csv.DictReader(open(path, encoding='utf-8')))


def late_full(share):
    sh = np.asarray(share, float)
    L = len(sh)
    return bool((sh[int(np.ceil(2 * L / 3)):] >= 0.8).all())


def union_of(content, onset):
    """Share of layers singled out at content or onset (>= 80% of held-out bodies), and whether the union covers the last third."""
    u = (np.asarray(content, float) >= 0.8) | (np.asarray(onset, float) >= 0.8)
    L = len(u)
    return float(u.sum()) / L, bool(u[int(np.ceil(2 * L / 3)):].all())


def main(a8, a3, n_json, out_dir):
    A8, A3 = Path(a8), Path(a3)
    idx, late = {}, {}
    for A in (A8, A3):
        for r in rows(A / 'stats' / 'geometry_summary.csv'):
            if r['measure'] not in ('geo_attr', 'geo') or r['model'] not in NAMES:
                continue
            if r.get('statistic', 'single_state') != 'single_state':
                continue
            key = (r['model'], 'non_thinking' if r['mode'] == 'non_thinking' else 'thinking')
            idx[key] = max(idx.get(key, 0.0), int(r['layers_A_apart_80pct']) / int(r['n_layers']))
    # per-layer shares for the last-third criterion
    summ = {}
    for f in glob.glob(str(A8 / 'models' / '*' / '*' / 'geo*' / '_run_summary.json')):
        p = Path(f)
        if p.parts[-2] in ('geo_attr', 'geo'):
            summ[(p.parts[-4], 'non_thinking' if p.parts[-3] == 'non_thinking' else 'thinking')] = f
    R = A3 / 'runs'
    summ.update({('seed_oss_36b', 'non_thinking'): R / 'server3/suite/seed_oss_36b/summary.json',
                 ('seed_oss_36b', 'thinking'): R / 'server3/suite_think/seed_oss_36b/summary.json',
                 ('exaone4_32b', 'non_thinking'): R / 'server3/suite/exaone4_32b_en/summary.json',
                 ('exaone4_32b', 'thinking'): R / 'server3/suite_think/exaone4_32b_en/summary.json',
                 ('gpt_oss_20b', 'thinking'): R / 'pro/suite_think/gpt_oss_20b_mxfp4_en/summary.json',
                 ('gpt_oss_120b', 'thinking'): R / 'pro/suite_think/gpt_oss_120b_en/summary.json',
                 ('qwen35_27b', 'non_thinking'): R / 'server3/suite/qwen35_27b/summary.json',
                 ('granite41_30b', 'non_thinking'): R / 'server3/suite/granite41_30b_en/summary.json',
                 ('granite42_30b', 'non_thinking'): R / 'server3/suite/granite42_30b_en/summary.json',
                 ('olmo2_7b', 'non_thinking'): R / 'pro/suite/olmo2_7b_en/summary.json',
                 ('olmo2_13b', 'non_thinking'): R / 'pro/suite/olmo2_13b_en/summary.json'})
    for r in rows(A3 / 'stats' / 'v4_geometry_summary.csv'):
        key = (r['model'], r['mode'])
        idx[key] = max(idx.get(key, 0.0), int(r['layers_A_apart_80pct']) / int(r['n_layers']))
    for key, f in summ.items():
        if key[0] not in NAMES or key[0] in ('qwen38_flashnext',):
            continue
        s = json.load(open(f, encoding='utf-8'))
        idx[key], late[key] = union_of(s['content']['share_A_lowest'], s['onset']['share_A_lowest'])
    streams = {('qwen38_flashnext', 'non_thinking'): R / 'pro/flashnext_stream_20261020/flashnext_stream_geometry_non_20261020.json',
               ('qwen38_flashnext', 'thinking'): R / 'pro/flashnext_stream_20261020/flashnext_stream_geometry_think_20261020.json',
               ('glm53_flash', 'thinking'): R / 'server3/suite_think/glm53_flash_stream_geometry.json'}
    for key, f in streams.items():
        s = json.load(open(f, encoding='utf-8'))
        L = len(s['content']['mean_of_streams']['share_A_lowest_by_layer'])
        idx[key], late[key] = union_of(s['content']['mean_of_streams']['share_A_lowest_by_layer'],
                                       s['onset']['mean_of_streams']['share_A_lowest_by_layer'])
    ref = {}
    for A in (A8, A3):
        for r in rows(A / 'stats' / 'attribution_non_thinking_body_level.csv'):
            if r['orders'] == '2_orders_as_paper' and r['model'] in NAMES:
                ref[(r['model'], 'non_thinking')] = float(r['own_minus_max_other'])
        by = {}
        for r in rows(A / 'stats' / 'thinking_attribution_counts.csv'):
            by.setdefault(r['model'], {})[r['role']] = float(r['you_share'])
        for m, d in by.items():
            if len(d) == 3 and m in NAMES:
                ref[(m, 'thinking')] = d['assistant'] - max(d['user'], d['system'])
    for f in ('v4_attribution_non_thinking.csv', 'v5_olmo_attribution_non_thinking.csv'):
        for r in rows(A3 / 'stats' / f):
            if r['orders'] == '2_orders_as_paper' and r['model'] in NAMES:
                ref[(r['model'], 'non_thinking')] = float(r['own_minus_max_other'])
    by = {}
    for r in rows(A3 / 'stats' / 'v4_thinking_counts.csv'):
        by.setdefault(r['model'], {})[r['role']] = float(r['you_share'])
    for m, d in by.items():
        if len(d) == 3 and m in NAMES:
            ref[(m, 'thinking')] = d['assistant'] - max(d['user'], d['system'])
    for m in THINK_GEO_FROM_NON_THINKING:
        idx[(m, 'thinking')] = idx[(m, 'non_thinking')]
        late[(m, 'thinking')] = late[(m, 'non_thinking')]
    N = json.load(open(n_json, encoding='utf-8'))
    ref[('qwen38_27b', 'thinking')] = N['assistant']['on_p_self'] - max(N['user']['on_p_self'], N['system']['on_p_self'])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), sharey=True, gridspec_kw={'wspace': 0.08})
    table = {}
    for ax, mode, title in zip(axes, ('non_thinking', 'thinking'), ('(a) without thinking', '(b) with thinking')):
        ax.axhline(0, color='#bbbbbb', lw=0.8, zorder=1)
        pts = [(m, idx[(m, mode)], ref[(m, mode)], late.get((m, mode))) for m in NAMES if (m, mode) in idx and (m, mode) in ref]
        drawn = []
        for m, x, y, lf in pts:
            table[f'{m}/{mode}'] = (round(x, 3), round(y, 3), lf)
            n_under = sum(abs(x - x0) < 0.02 and abs(y - y0) < 0.02 for x0, y0 in drawn)
            if n_under:
                # a point (nearly) on top of earlier ones is drawn as a ring around them, so all stay visible
                ax.scatter(x, y, s=46 + 80 * n_under, facecolor='none', edgecolor=COL[m], linewidth=1.5, zorder=4)
            elif lf:
                ax.scatter(x, y, s=46, color=COL[m], edgecolor='white', linewidth=0.8, zorder=3)
            else:
                ax.scatter(x, y, s=46, facecolor='white', edgecolor=COL[m], linewidth=1.6, zorder=3)
            drawn.append((x, y))
            tx, ty, ha = LABEL[mode].get(m, (x - 0.03, y, 'right'))
            far = abs(tx - x) > 0.05 or abs(ty - y) > 0.03
            ax.annotate(NAMES[m], (x, y), xytext=(tx, ty), textcoords='data', ha=ha, va='center', fontsize=7.5, color='#333333',
                        arrowprops=dict(arrowstyle='-', color='#9a9a9a', lw=0.6, shrinkA=1.5, shrinkB=3.5) if far else None)
        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(-0.215, 1.05)
        ax.set_title(title, loc='left')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('self-reference\n(own minus other roles, "you")')
    fig.supxlabel('self-index: share of layers singled out at message content or response onset', fontsize=10, y=-0.01)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f'{out_dir}/self_crossmodel.pdf', bbox_inches='tight')
    fig.savefig(f'{out_dir}/self_crossmodel.png', dpi=170, bbox_inches='tight')
    json.dump(table, open(f'{out_dir}/self_crossmodel_points.json', 'w'), indent=1)
    for k, v in table.items():
        print(k, v)


if __name__ == '__main__':
    main(*sys.argv[1:])
