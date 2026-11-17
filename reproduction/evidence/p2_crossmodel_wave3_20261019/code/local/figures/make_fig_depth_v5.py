"""Figure self_crossmodel_depth (paper v5 restructure, 10-29: make_fig_depth_v4.py plus OLMo-2-13B and OLMo-2-7B, no thinking mode,
from their suite summaries, at the head of the group without either level): where in depth each of the eighteen models singles out its own message. For every
model, mode (template non-thinking / thinking rendering) and reading position (message content / response onset), the share of
held-out bodies in which the Assistant message has the smallest distance ratio R_A, at every layer, against relative depth (layer
index divided by the number of layers); dashed lines split the depth into thirds. Four-stream models (Qwen3.8-Flash-Next,
GLM-5.3-Flash): mean of the per-stream ratios. Models are ordered by group (self-reference, self-index without self-reference,
neither). Rows without the mode are marked n/a; rows whose thinking-rendering geometry was not measured (Qwen3.5-27B and
Granite-4.2-30B: required only for models with thinking self-reference, and neither has any) are marked "not measured".
v4 changes: Qwen3.5-27B, Granite-4.1-30B (no thinking mode) and Granite-4.2-30B added from their non-thinking suite summaries.
usage: make_fig_depth_v4.py ARCHIVE8 ARCHIVE3 OUT_DIR"""
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42, 'axes.titlesize': 9.5,
                     'axes.titleweight': 'bold', 'xtick.labelsize': 8, 'ytick.labelsize': 8})
ORDER = [('qwen38_27b', 'Qwen3.8-27B'), ('gpt_oss_120b', 'gpt-oss-120b'), ('gpt_oss_20b', 'gpt-oss-20b'),
         ('qwen38_flashnext', 'Qwen3.8-Flash-Next'), ('glm53_flash', 'GLM-5.3-Flash'), ('exaone4_32b', 'EXAONE-4.0.1-32B'),
         ('glm46v_flash', 'GLM-4.6V-Flash'), ('glm47_flash', 'GLM-4.7-Flash'), ('r1_llama70b', 'R1-Distill-Llama-70B'),
         ('granite42_30b', 'Granite-4.2-30B'), ('granite41_30b', 'Granite-4.1-30B'),
         ('olmo2_13b', 'OLMo-2-13B'), ('olmo2_7b', 'OLMo-2-7B'),
         ('seed_oss_36b', 'Seed-OSS-36B'), ('gemma4_31b', 'Gemma-4-31B'), ('qwen36_27b', 'Qwen3.6-27B'), ('qwen36_35b', 'Qwen3.6-35B-A3B'),
         ('qwen35_27b', 'Qwen3.5-27B')]
GROUP_BREAKS = [6, 11]         # rows after which a group ends (self-reference | index without self-reference | neither)
NO_THINKING_MODE = ('granite41_30b', 'olmo2_13b', 'olmo2_7b')
THINK_GEO_NOT_MEASURED = ('qwen35_27b', 'granite42_30b')


def load(a8, a3):
    A8, A3 = Path(a8), Path(a3)
    R = A3 / 'runs'
    share = {}
    for f in glob.glob(str(A8 / 'models' / '*' / '*' / 'geo*' / '_run_summary.json')):
        p = Path(f)
        if p.parts[-2] not in ('geo_attr', 'geo') or p.parts[-4] == 'qwen38_flashnext':
            continue
        s = json.load(open(f, encoding='utf-8'))
        mode = 'non_thinking' if p.parts[-3] == 'non_thinking' else 'thinking'
        for k in ('content', 'onset'):
            if k in s and 'share_A_lowest' in s[k]:
                share[(p.parts[-4], mode, k)] = np.array(s[k]['share_A_lowest'], float)
    single = {('seed_oss_36b', 'non_thinking'): R / 'server3/suite/seed_oss_36b/summary.json',
              ('seed_oss_36b', 'thinking'): R / 'server3/suite_think/seed_oss_36b/summary.json',
              ('exaone4_32b', 'non_thinking'): R / 'server3/suite/exaone4_32b_en/summary.json',
              ('exaone4_32b', 'thinking'): R / 'server3/suite_think/exaone4_32b_en/summary.json',
              ('gpt_oss_20b', 'thinking'): R / 'pro/suite_think/gpt_oss_20b_mxfp4_en/summary.json',
              ('gpt_oss_120b', 'thinking'): R / 'pro/suite_think/gpt_oss_120b_en/summary.json',
              ('qwen35_27b', 'non_thinking'): R / 'server3/suite/qwen35_27b/summary.json',
              ('granite41_30b', 'non_thinking'): R / 'server3/suite/granite41_30b_en/summary.json',
              ('granite42_30b', 'non_thinking'): R / 'server3/suite/granite42_30b_en/summary.json',
              ('olmo2_13b', 'non_thinking'): R / 'pro/suite/olmo2_13b_en/summary.json',
              ('olmo2_7b', 'non_thinking'): R / 'pro/suite/olmo2_7b_en/summary.json'}
    for (m, mode), f in single.items():
        s = json.load(open(f, encoding='utf-8'))
        for k in ('content', 'onset'):
            share[(m, mode, k)] = np.array(s[k]['share_A_lowest'], float)
    streams = {('qwen38_flashnext', 'non_thinking'): R / 'pro/flashnext_stream_20261020/flashnext_stream_geometry_non_20261020.json',
               ('qwen38_flashnext', 'thinking'): R / 'pro/flashnext_stream_20261020/flashnext_stream_geometry_think_20261020.json',
               ('glm53_flash', 'thinking'): R / 'server3/suite_think/glm53_flash_stream_geometry.json'}
    for (m, mode), f in streams.items():
        s = json.load(open(f, encoding='utf-8'))
        for k in ('content', 'onset'):
            share[(m, mode, k)] = np.array(s[k]['mean_of_streams']['share_A_lowest_by_layer'], float)
    return share


def main(a8, a3, out_dir):
    share = load(a8, a3)
    cmap = LinearSegmentedColormap.from_list('seq', ['#ffffff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'])
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.7), sharex=True, sharey=True, gridspec_kw={'wspace': 0.08, 'hspace': 0.2})
    n = len(ORDER)
    for c, (mode, mtitle) in enumerate((('non_thinking', 'no thinking'), ('thinking', 'thinking'))):
        for r, (kind, ktitle) in enumerate((('content', 'content'), ('onset', 'onset'))):
            ax = axes[r, c]
            for i, (m, name) in enumerate(ORDER):
                v = share.get((m, mode, kind))
                if v is None:
                    ax.add_patch(plt.Rectangle((0, i - 0.45), 1, 0.9, facecolor='#f2f2f2', edgecolor='none'))
                    note = ('n/a (no thinking mode)' if m in NO_THINKING_MODE else
                            'not measured' if m in THINK_GEO_NOT_MEASURED else 'n/a (thinking only)')
                    ax.text(0.5, i, note, ha='center', va='center', fontsize=7, color='#777777', zorder=6,
                            bbox=dict(boxstyle='square,pad=0.15', facecolor='#f2f2f2', edgecolor='none'))
                    continue
                L = len(v)
                edges = np.linspace(0, 1, L + 1)
                ax.pcolormesh(edges, [i - 0.45, i + 0.45], v[None, :], cmap=cmap, vmin=0, vmax=1, shading='flat')
            ax.axvline(1 / 3, color='#555555', lw=0.7, ls='--', zorder=5)
            ax.axvline(2 / 3, color='#b35806', lw=1.1, ls='-', zorder=5)
            if r == 0:
                # mark the last third, the part of the depth used in the text
                ax.plot([2 / 3, 1], [-0.95, -0.95], color='#b35806', lw=2.2, clip_on=False, solid_capstyle='butt')
                ax.text(5 / 6, -1.25, 'last third', ha='center', va='bottom', fontsize=7.5, color='#b35806', clip_on=False)
            for b in GROUP_BREAKS:
                ax.axhline(b - 0.5, color='#333333', lw=0.8, zorder=5)
            ax.set_ylim(n - 0.5, -0.5)
            ax.set_xlim(0, 1)
            ax.set_yticks(range(n))
            ax.set_yticklabels([name for _, name in ORDER])
            ax.set_title(f'({"abcd"[2 * r + c]}) {ktitle}, {mtitle}', loc='left', pad=14 if r == 0 else 6)
            if r == 1:
                ax.set_xlabel('relative depth (layer / number of layers)')
            ax.tick_params(length=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label('share of held-out bodies in which\nthe own message is singled out', fontsize=8)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f'{out_dir}/self_crossmodel_depth.pdf', bbox_inches='tight')
    fig.savefig(f'{out_dir}/self_crossmodel_depth.png', dpi=170, bbox_inches='tight')
    print('rows', len(ORDER), 'entries', len(share))


if __name__ == '__main__':
    main(*sys.argv[1:])
