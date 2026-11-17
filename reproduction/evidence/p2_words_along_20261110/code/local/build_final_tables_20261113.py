"""Final summary of the two ordinary-word readouts for the final paper (11-13): retention along the message (stage 2 analysis,
results/stage2_20261113, = stage 1 plus GLM-5.3 from the H200 part1 run) and next-token KL (ordinary words / A||U, content KL),
every model of the paper, main probe and natural order. Groups follow the paper (tab:self_crossmodel): self-reference (7 models),
a self-index without self-reference (6), neither level (12). The Qwen3.8-27B thinking rendering is listed but not counted in the
groups (the paper classifies by the non-thinking rendering). KL sources follow the data source table of ../../README_zh.md:
PRO words_along/kl (mirrored in results/kl_20261113_pro), R1 and DeepSeek-V4.1 from server3 (results/kl_20261112), GLM-5.3 from
the H200 (p2_modeswitch_20261107/results/vast_h200_20261111/m/out/kl). Read-only on all inputs.
usage: build_final_tables_20261113.py OUT_PREFIX [--flashnext DIR]   (DIR = the Flash-Next KL output once mirrored)"""
import argparse
import json
from pathlib import Path

import numpy as np

EV = Path(__file__).resolve().parents[3]
WA = EV / 'p2_words_along_20261110'
RET = WA / 'results/stage2_20261113/analysis_b150_20261113.json'
COV = WA / 'items/content_index_coverage.json'
PRO = WA / 'results/kl_20261113_pro/kl'
S3 = WA / 'results/kl_20261112'
H200 = EV / 'p2_modeswitch_20261107/results/vast_h200_20261111/m/out/kl'

GROUPS = {
    'self-reference': ['qwen38_27b', 'qwen38_flashnext', 'gpt_oss_20b_en', 'gpt_oss_120b_en', 'glm53_flash', 'exaone4_32b_en',
                       'mimo_v26_distill_qwen9b'],
    'self-index only': ['glm46v_flash', 'glm47_flash', 'r1_distill_llama70b', 'phi4_en', 'granite42_30b_en', 'granite41_30b_en'],
    'neither': ['olmo2_13b_en', 'olmo2_7b_en', 'apertus8b_en', 'moonlight16b_a3b_instruct', 'mimo7b_rl_0530', 'mimo_vl7b_rl_2508',
                'ernie45_21b_a3b_thinking', 'seed_oss_36b', 'gemma4_31b_en', 'qwen36_27b', 'qwen36_35b_a3b', 'qwen35_27b'],
}
EXTRA = ['qwen38_27b_think']
# natural order: groups from the paper's appendix (tab:self_natural, tab:self_natural2, tab:self_natural3). Templates without a System
# message after a user turn (Qwen3.8, gpt-oss-20b, R1, Apertus) take the self-index from the main probe. ERNIE-4.5 (thinking only):
# self-index in this order, self-reference not measured in it.
# 11-15 (audit of the NMI version): OLMo-2-7B moved from 'neither' to 'self-index only'. Its natural-order onset carries the index at
# 0-based layers 11-15 (share 1.00) and 16 (0.89), beyond L/3 = 10.67, at every threshold of 70/80/90%; the hand-copied appendix
# table had it as 'neither'. classify_groups_by_rule_20261115.py now recomputes both orders by rule and asserts these lists.
NAT_GROUPS = {
    'self-reference': ['qwen38_27b', 'gpt_oss_20b_en', 'exaone4_32b_en', 'gemma4_31b_en', 'dsv41_flash'],
    'self-index only': ['glm46v_flash', 'granite41_30b_en', 'mimo7b_rl_0530', 'moonlight16b_a3b_instruct', 'phi4_en',
                        'ernie45_21b_a3b_thinking', 'r1_distill_llama70b', 'olmo2_7b_en'],
    'neither': ['seed_oss_36b', 'apertus8b_en', 'ministral3_14b_instruct_en', 'ministral3_14b_reasoning_en'],
}
NATURAL_ONLY = ['dsv41_flash', 'ministral3_14b_instruct_en', 'ministral3_14b_reasoning_en']   # models that accept only that order
NAMES = {'dsv41_flash': 'DeepSeek-V4.1-Flash', 'ministral3_14b_instruct_en': 'Ministral-3-14B-Instruct',
         'ministral3_14b_reasoning_en': 'Ministral-3-14B-Reasoning'}
KL_MAIN = {'qwen38_27b': PRO / 'qwen38_nt', 'qwen38_27b_think': PRO / 'qwen38_think', 'mimo_v26_distill_qwen9b': PRO / 'mimo_v26_nt',
           'mimo_vl7b_rl_2508': PRO / 'mimo_vl7b', 'r1_distill_llama70b': S3 / 'server3_r1/out/r1_distill_llama70b',
           'glm53_flash': H200 / 'glm53_flash_all'}
KL_NAT = {'qwen38_27b': PRO / 'natural/qwen38_nt', 'dsv41_flash': S3 / 'server3_dsv41/natural/dsv41_flash',
          'r1_distill_llama70b': S3 / 'server3_r1/out/natural/r1_distill_llama70b'}


def kl(path):
    f = Path(path) / 'summary.json'
    if not f.exists():
        return None
    s = json.loads(f.read_text(encoding='utf-8'))['kinds']['kl_content']
    to_us = s.get('to_US') or {}
    return {'words_AU': s['word_median_to_AU_12']['ratio'], 'words_AU_ci': s['word_median_to_AU_12']['ci95'],
            'AU_US': to_us.get('A-U', {}).get('ratio'), 'words_US': (s.get('word_median_to_US_12') or {}).get('ratio'),
            'src': str(Path(path).relative_to(EV)).replace('\\', '/')}


def ret(u):
    if u is None:
        return None
    return {'self': u['self']['share'], 'US': u['rows']['U-S']['share'] if 'U-S' in u['rows'] else None,
            'word_med': u['word_share_quantiles'][2], 'adv_US': u.get('advantage_vs_US'), 'adv_words': u['advantage_vs_word_median']}


def pos(x):
    return x is not None and x['ci'][0] > 0


def neg(x):
    return x is not None and x['ci'][1] < 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--flashnext', default='')
    a = ap.parse_args()
    if a.flashnext:
        KL_MAIN['qwen38_flashnext'] = Path(a.flashnext)
    R = json.loads(RET.read_text(encoding='utf-8'))['units']['b150']
    C = json.loads(COV.read_text(encoding='utf-8'))
    rows = {'main': [], 'natural': []}
    for order in ('main', 'natural'):
        G = GROUPS if order == 'main' else NAT_GROUPS
        grp = {t: g for g, ts in G.items() for t in ts}
        tags = [t for ts in G.values() for t in ts] + (EXTRA if order == 'main' else [])
        for t in tags:
            u = R[order].get(t)
            kp = (KL_MAIN if order == 'main' else KL_NAT).get(t, PRO / (t if order == 'main' else f'natural/{t}'))
            k = kl(kp) if (order == 'main' or u is not None or t in KL_NAT) else None
            if order == 'natural' and u is None and k is None:
                continue
            cov = C[order].get(t)
            rows[order].append({'tag': t, 'name': C['main'].get(t, {}).get('name') or NAMES.get(t, t), 'group': grp.get(t, '-'),
                                'coverage': (cov['layers'] / cov['of']) if cov else None, 'retention': ret(u), 'kl': k})
    # group summaries, one row per model (the extra rendering excluded); self - U-S only where the order has a System message
    summ = {}
    for order, G in (('main', GROUPS), ('natural', NAT_GROUPS)):
      for g, ts in G.items():
        rs = [r for r in rows[order] if r['tag'] in ts]
        au = [r['kl']['words_AU'] for r in rs if r['kl']]
        aus = [r['retention']['adv_US'] for r in rs if r['retention'] and r['retention']['adv_US']]
        awd = [r['retention']['adv_words'] for r in rs if r['retention']]
        summ[f'{order}: {g}'] = {'n': len(ts),
                   'retention_self_minus_US': {'median': float(np.median([x['value'] for x in aus])), 'ci_above_0': sum(map(pos, aus)),
                                               'ci_below_0': sum(map(neg, aus)), 'n': len(aus)},
                   'retention_self_minus_words': {'median': float(np.median([x['value'] for x in awd])), 'ci_above_0': sum(map(pos, awd)),
                                                  'ci_below_0': sum(map(neg, awd)), 'n': len(awd)},
                   'kl_words_over_AU': {'median': float(np.median(au)), 'min': float(min(au)), 'max': float(max(au)), 'n': len(au),
                                        'missing': [r['tag'] for r in rs if not r['kl']]}}
    out = {'groups': GROUPS, 'natural_groups': NAT_GROUPS, 'rows': rows, 'group_summary': summ,
           'inputs': {'retention': str(RET.relative_to(EV)), 'coverage': str(COV.relative_to(EV))}}
    Path(a.out + '.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

    def f(x, d=2):
        return '-' if x is None else f'{x:.{d}f}'

    def fci(x, d=2):
        return '-' if x is None else f"{x['value']:+.{d}f} [{x['ci'][0]:+.{d}f}, {x['ci'][1]:+.{d}f}]"

    def fkl(k):
        return '-' if k is None else f"{k['words_AU']:.3f} [{k['words_AU_ci'][0]:.3f}, {k['words_AU_ci'][1]:.3f}]"

    L = ['# Final summary of the ordinary-word readouts (11-13)', '',
         'Retention: share kept = sqrt(C_last / C_first), median over kept layers (approximate reading, see README). '
         'Self = min(A-U, A-S), A-U alone without a System message. KL: content KL, ordinary-word median / A||U. '
         'Intervals: 95%, family resampling (main and natural b150). Groups as in the paper. Qwen3.8 thinking rendering listed, not counted.', '']
    for order in ('main', 'natural'):
        L += [f'## {order}', '', '| model | group | coverage | self kept | U-S kept | words kept (median) | self - U-S | self - words | '
              'A||U / U||S | words / A||U |', '|---|---|---|---|---|---|---|---|---|---|']
        for r in rows[order]:
            rt, k = r['retention'], r['kl']
            L.append(f"| {r['name']} | {r['group']} | {f(r['coverage'])} | {f(rt and rt['self'])} | {f(rt and rt['US'])} | "
                     f"{f(rt and rt['word_med'])} | {fci(rt and rt['adv_US'])} | {fci(rt and rt['adv_words'])} | "
                     f"{f(k and k['AU_US'])} | {fkl(k)} |")
        L.append('')
    L += ['## Group summary (one rendering per model; natural-order groups from the paper appendix)', '',
          '| group | n | self - U-S: median (CI > 0 / CI < 0) | self - words: median (CI > 0 / CI < 0) | words / A||U: median [min, max] |',
          '|---|---|---|---|---|']
    for g, s in summ.items():
        a1, a2, k = s['retention_self_minus_US'], s['retention_self_minus_words'], s['kl_words_over_AU']
        miss = f" (missing: {', '.join(k['missing'])})" if k['missing'] else ''
        L.append(f"| {g} | {s['n']} | {a1['median']:+.2f} ({a1['ci_above_0']} / {a1['ci_below_0']} of {a1['n']}) | "
                 f"{a2['median']:+.2f} ({a2['ci_above_0']} / {a2['ci_below_0']} of {a2['n']}) | "
                 f"{k['median']:.3f} [{k['min']:.3f}, {k['max']:.3f}], n = {k['n']}{miss} |")
    Path(a.out + '.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
