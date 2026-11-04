"""Sensitivity of the cross-model classification: the threshold and depth band of the self-index criterion, and the option
orders of single-pass self-reference.

threshold (stats/r04_threshold_20261101.json, .md), from the per-layer shares already archived: for every model and rendering,
  the share of held-out bodies in which the model's own message has the smallest distance ratio (share_A_lowest), on message
  content and at the response onset, read from the same files as fig_crossmodel.py (four-stream models: mean of the per-stream
  ratios). Per threshold t in 0.7, 0.8 (the paper), 0.9:
    - self-index (Definition def:self): some layer l >= L/3 with share >= t at content or onset;
    - last third (l >= 2L/3) and last quarter (l >= 3L/4): layers of the band with share >= t at content or onset, and whether all are;
    - classification under the non-thinking rendering (thinking rendering for models with only a thinking mode), as in the paper;
      band coverage also in the rendering of each mode in which the model recognizes its own messages.
  Self-reference is behavioural and does not depend on t; the models and modes with self-reference are those of tab:self_crossmodel.
orders (stats/r14_orders_20261101.json, .md): single-pass self-reference with all three cyclic option orders (balanced) against
  the first two (the paper), from the body-level tables of crossmodel_classification.py. Per model with a non-thinking mode:
  P("you") for own, User and System messages, the excess own - max(User, System) (body-level means over the orders, as the paper),
  bodies with a positive excess, and whether the excess reaches the 0.10 margin of Definition def:self; the self-index from the
  threshold analysis (80%, classification rendering). Thinking items exist only in the first two orders (300 per source), so the
  balanced summary covers the single pass only.
orders_tex (stats/r14_tex_{en,zh}.tex): the body of tab:self_orders, between "%% tab:self_orders" and "%% end".

usage: thresholds_orders.py --results RES --out OUT   (needs the stats CSVs of crossmodel_classification.py in OUT)
"""
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import M8, W3, Roots, arguments, run_path  # noqa: E402
from newdev import NEWDEV  # noqa: E402

NAMES = {'qwen38_27b': 'Qwen3.8-27B', 'gpt_oss_120b': 'gpt-oss-120b', 'gpt_oss_20b': 'gpt-oss-20b',
         'qwen38_flashnext': 'Qwen3.8-Flash-Next', 'glm53_flash': 'GLM-5.3-Flash', 'exaone4_32b': 'EXAONE-4.0.1-32B',
         'glm46v_flash': 'GLM-4.6V-Flash', 'glm47_flash': 'GLM-4.7-Flash', 'r1_llama70b': 'R1-Distill-Llama-70B',
         'granite42_30b': 'Granite-4.2-30B', 'granite41_30b': 'Granite-4.1-30B', 'olmo2_13b': 'OLMo-2-13B', 'olmo2_7b': 'OLMo-2-7B',
         'seed_oss_36b': 'Seed-OSS-36B', 'gemma4_31b': 'Gemma-4-31B', 'qwen36_27b': 'Qwen3.6-27B', 'qwen36_35b': 'Qwen3.6-35B-A3B',
         'qwen35_27b': 'Qwen3.5-27B'}
NAMES.update({k: v['short'] for k, v in NEWDEV.items()})
SELFREF = [('qwen38_27b', 'non_thinking'), ('qwen38_27b', 'thinking'), ('qwen38_flashnext', 'thinking'), ('gpt_oss_20b', 'thinking'),
           ('gpt_oss_120b', 'thinking'), ('glm53_flash', 'thinking'), ('exaone4_32b', 'thinking'), ('mimo_distill9b', 'thinking')]
TS = (0.7, 0.8, 0.9)


def sources(A8, A3, E):
    """per (model, rendering): content and onset shares and the source file (relative to the evidence folder E)"""
    summ = {}
    for f in glob.glob(str(A8 / 'models' / '*' / '*' / 'geo*' / '_run_summary.json')):
        p = Path(f)
        if p.parts[-2] in ('geo_attr', 'geo'):
            summ[(p.parts[-4], 'non_thinking' if p.parts[-3] == 'non_thinking' else 'thinking')] = Path(f)
    summ.update({('seed_oss_36b', 'non_thinking'): run_path(A3, 'suite/seed_oss_36b/summary.json'),
                 ('seed_oss_36b', 'thinking'): run_path(A3, 'suite_think/seed_oss_36b/summary.json'),
                 ('exaone4_32b', 'non_thinking'): run_path(A3, 'suite/exaone4_32b_en/summary.json'),
                 ('exaone4_32b', 'thinking'): run_path(A3, 'suite_think/exaone4_32b_en/summary.json'),
                 ('gpt_oss_20b', 'thinking'): run_path(A3, 'suite_think/gpt_oss_20b_mxfp4_en/summary.json'),
                 ('gpt_oss_120b', 'thinking'): run_path(A3, 'suite_think/gpt_oss_120b_en/summary.json'),
                 ('qwen35_27b', 'non_thinking'): run_path(A3, 'suite/qwen35_27b/summary.json'),
                 ('granite41_30b', 'non_thinking'): run_path(A3, 'suite/granite41_30b_en/summary.json'),
                 ('granite42_30b', 'non_thinking'): run_path(A3, 'suite/granite42_30b_en/summary.json'),
                 ('olmo2_7b', 'non_thinking'): run_path(A3, 'suite/olmo2_7b_en/summary.json'),
                 ('olmo2_13b', 'non_thinking'): run_path(A3, 'suite/olmo2_13b_en/summary.json')})
    summ.update({(k, mode): run_path(A3, d + '/summary.json') for k, v in NEWDEV.items() for mode, d in v['geo'].items()})
    out = {}
    for key, f in summ.items():
        if key[0] not in NAMES or key[0] == 'qwen38_flashnext':
            continue
        s = json.loads(f.read_text(encoding='utf-8'))
        out[key] = (s['content']['share_A_lowest'], s['onset']['share_A_lowest'], str(f.relative_to(E)))
    streams = {('qwen38_flashnext', 'non_thinking'): run_path(A3, 'flashnext_stream_20261020/flashnext_stream_geometry_non_20261020.json'),
               ('qwen38_flashnext', 'thinking'): run_path(A3, 'flashnext_stream_20261020/flashnext_stream_geometry_think_20261020.json'),
               ('glm53_flash', 'thinking'): run_path(A3, 'suite_think/glm53_flash_stream_geometry.json')}
    per_stream = {}
    for key, f in streams.items():
        s = json.loads(f.read_text(encoding='utf-8'))
        out[key] = (s['content']['mean_of_streams']['share_A_lowest_by_layer'], s['onset']['mean_of_streams']['share_A_lowest_by_layer'],
                    str(f.relative_to(E)))
        per_stream[key] = {k: (s['content'][k]['share_A_lowest_by_layer'], s['onset'][k]['share_A_lowest_by_layer'])
                           for k in ('stream0', 'stream1', 'stream2', 'stream3')}
    return out, per_stream


def measure(content, onset, t):
    L = len(content)
    assert len(onset) == L
    hit = [content[i] >= t or onset[i] >= t for i in range(L)]
    first, third, quarter = math.ceil(L / 3), math.ceil(2 * L / 3), math.ceil(3 * L / 4)
    return {'L': L, 'content': sum(c >= t for c in content), 'onset': sum(o >= t for o in onset),
            'index': any(hit[first:]), 'last_third': [sum(hit[third:]), L - third], 'last_quarter': [sum(hit[quarter:]), L - quarter]}


def threshold(A8, A3, E, S3):
    src, per_stream = sources(A8, A3, E)
    res = {'files': {f'{m}/{mode}': v[2] for (m, mode), v in sorted(src.items())}, 'models': {}, 'summary': {}}
    for m in NAMES:
        modes = [mode for mode in ('non_thinking', 'thinking') if (m, mode) in src]
        cls = 'non_thinking' if 'non_thinking' in modes else 'thinking'
        res['models'][m] = {'classification_rendering': cls,
                            'by_rendering': {mode: {str(t): measure(src[(m, mode)][0], src[(m, mode)][1], t) for t in TS} for mode in modes}}
        streams = {mode: {st: {str(t): measure(c, o, t) for t in TS} for st, (c, o) in per_stream[(m, mode)].items()}
                   for mode in modes if (m, mode) in per_stream}
        if streams:
            res['models'][m]['streams'] = streams
    selfref_models = {m for m, _ in SELFREF}
    L = ['# R04: threshold and depth band of the self-index criterion (11-01)', '',
         'Share of held-out bodies in which the own message has the smallest distance ratio, at content or onset (a layer counted once). '
         'Self-index: some layer beyond the first third at or above the threshold. Classification under the non-thinking rendering '
         '(thinking rendering for thinking-only templates); self-reference models and modes as in tab:self_crossmodel.', '']
    for t in TS:
        k = str(t)
        cl = {m: v['by_rendering'][v['classification_rendering']][k] for m, v in res['models'].items()}
        idx = sorted(m for m in NAMES if cl[m]['index'])
        sr_modes = {f'{m}/{mode}': res['models'][m]['by_rendering'][mode][k] for m, mode in SELFREF}
        noidx_all = {f'{m}/{mode}': v[k] for m in NAMES if not cl[m]['index'] for mode, v in res['models'][m]['by_rendering'].items()}
        s = {'n_index': len(idx), 'index_models': idx, 'n_neither': sum(1 for m in NAMES if not cl[m]['index'] and m not in selfref_models),
             'nested': all(cl[m]['index'] for m in selfref_models),
             'selfref_full_last_third': all(v['last_third'][0] == v['last_third'][1] for v in sr_modes.values()),
             'selfref_full_last_quarter': all(v['last_quarter'][0] == v['last_quarter'][1] for v in sr_modes.values()),
             'selfref_min_last_third': min(((v['last_third'][0], v['last_third'][1], key) for key, v in sr_modes.items()),
                                           key=lambda x: x[0] / x[1]),
             'noindex_max_last_third': max(((v['last_third'][0], v['last_third'][1], key) for key, v in noidx_all.items()), default=None),
             'noindex_max_last_quarter': max(((v['last_quarter'][0], v['last_quarter'][1], key) for key, v in noidx_all.items()), default=None)}
        res['summary'][k] = s
    base = set(res['summary']['0.8']['index_models'])
    for t in TS:
        s = res['summary'][str(t)]
        s['changed_vs_0.8'] = sorted(set(s['index_models']) ^ base)
    L += ['| threshold | models with a self-index | neither level | changed vs 80% | nested | self-ref: full last third | self-ref: full last quarter | '
          'self-ref: least last third | no index: most last third | no index: most last quarter |', '|---|---|---|---|---|---|---|---|---|---|']
    for t in TS:
        s = res['summary'][str(t)]
        f = lambda x: 'none' if x is None else f'{x[0]}/{x[1]} ({x[2]})'  # noqa: E731
        L.append(f"| {int(t * 100)}% | {s['n_index']} | {s['n_neither']} | {', '.join(NAMES[m] for m in s['changed_vs_0.8']) or '-'} | {s['nested']} | "
                 f"{s['selfref_full_last_third']} | {s['selfref_full_last_quarter']} | {f(s['selfref_min_last_third'])} | "
                 f"{f(s['noindex_max_last_third'])} | {f(s['noindex_max_last_quarter'])} |")
    L += ['', '## Per model and rendering', '', '| model | rendering | L | ' + ' | '.join(
        f'{int(t * 100)}%: content / onset / index / last third / last quarter' for t in TS) + ' |', '|---|---|---|' + '---|' * len(TS)]
    for m, v in res['models'].items():
        for mode, d in v['by_rendering'].items():
            cells = []
            for t in TS:
                x = d[str(t)]
                cells.append(f"{x['content']} / {x['onset']} / {'yes' if x['index'] else 'no'} / {x['last_third'][0]}/{x['last_third'][1]} / "
                             f"{x['last_quarter'][0]}/{x['last_quarter'][1]}")
            mark = ' (classification)' if mode == v['classification_rendering'] else ''
            L.append(f"| {NAMES[m]} | {mode}{mark} | {d['0.8']['L']} | " + ' | '.join(cells) + ' |')
            for st, sd in v.get('streams', {}).get(mode, {}).items():
                cells = [f"{sd[str(t)]['content']} / {sd[str(t)]['onset']} / {'yes' if sd[str(t)]['index'] else 'no'} / "
                         f"{sd[str(t)]['last_third'][0]}/{sd[str(t)]['last_third'][1]} / {sd[str(t)]['last_quarter'][0]}/{sd[str(t)]['last_quarter'][1]}"
                         for t in TS]
                L.append(f"| {NAMES[m]}, {st} | {mode} | {sd['0.8']['L']} | " + ' | '.join(cells) + ' |')
    (S3 / 'r04_threshold_20261101.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    (S3 / 'r04_threshold_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


ONAMES = {'qwen38_27b': 'Qwen3.8-27B', 'qwen38_flashnext': 'Qwen3.8-Flash-Next', 'exaone4_32b': 'EXAONE-4.0.1-32B',
          'glm46v_flash': 'GLM-4.6V-Flash', 'glm47_flash': 'GLM-4.7-Flash', 'granite42_30b': 'Granite-4.2-30B',
          'granite41_30b': 'Granite-4.1-30B', 'olmo2_13b': 'OLMo-2-13B', 'olmo2_7b': 'OLMo-2-7B', 'seed_oss_36b': 'Seed-OSS-36B',
          'gemma4_31b': 'Gemma-4-31B', 'qwen36_27b': 'Qwen3.6-27B', 'qwen36_35b': 'Qwen3.6-35B-A3B', 'qwen35_27b': 'Qwen3.5-27B'}
ONAMES.update({k: v['short'] for k, v in NEWDEV.items() if v['attr']})
# row order of tab:self_orders in the paper: by group (self-reference, index only, neither), new models placed in their group
ORDER = ['qwen38_27b', 'qwen38_flashnext', 'exaone4_32b', 'mimo_distill9b', 'glm46v_flash', 'glm47_flash', 'phi4', 'granite42_30b',
         'granite41_30b', 'olmo2_13b', 'olmo2_7b', 'apertus8b', 'moonlight16b', 'mimo7b_rl', 'seed_oss_36b', 'gemma4_31b', 'qwen36_27b',
         'qwen36_35b', 'qwen35_27b']
assert sorted(ORDER) == sorted(ONAMES), set(ORDER) ^ set(ONAMES)
ONAMES = {k: ONAMES[k] for k in ORDER}
MARGIN = 0.10


def orders(S8, S3):
    files = [S8 / 'attribution_non_thinking_body_level.csv', S3 / 'attribution_non_thinking_body_level.csv',
             S3 / 'v4_attribution_non_thinking.csv', S3 / 'v5_olmo_attribution_non_thinking.csv', S3 / 'v6_newdev_attribution_non_thinking.csv']
    rows = {}
    for f in files:
        for r in csv.DictReader(open(f, encoding='utf-8', newline='')):
            if r['model'] in ONAMES:
                assert (r['model'], r['orders']) not in rows, (r['model'], r['orders'])
                rows[(r['model'], r['orders'])] = r
    absent = [k for m in ONAMES for k in ((m, '2_orders_as_paper'), (m, '3_orders')) if k not in rows]
    if absent:  # a table row whose per-body records are not in the results package
        print('orders: needs inputs not in the package (body-level rows of ' + ', '.join(sorted({m for m, _ in absent})) + '); skipped')
        return False
    r04 = json.loads((S3 / 'r04_threshold_20261101.json').read_text(encoding='utf-8'))['models']
    out, L = {}, ['# R14: single-pass self-reference, three option orders against the first two (11-01)', '',
                  'Excess = mean over bodies of P(you | own) - max(P(you | User), P(you | System)), probabilities averaged over the orders '
                  'first (paper definition). Self-reference: excess >= 0.10. Self-index: 80% criterion, classification rendering.', '',
                  '| model | self-index | 2 orders: own / User / System | excess | bodies > 0 | 3 orders: own / User / System | excess | '
                  'bodies > 0 | self-reference 2 / 3 |', '|---|---|---|---|---|---|---|---|---|']
    for m, name in ONAMES.items():
        two, three = rows[(m, '2_orders_as_paper')], rows[(m, '3_orders')]
        v = r04[m]
        idx = v['by_rendering'][v['classification_rendering']]['0.8']['index']
        f = lambda r: [float(r['p_you_own']), float(r['p_you_user']), float(r['p_you_system']), float(r['own_minus_max_other']),  # noqa: E731
                       int(r['bodies_positive'])]
        a, b = f(two), f(three)
        out[m] = {'index': idx, 'two_orders': a, 'three_orders': b, 'selfref_two': a[3] >= MARGIN, 'selfref_three': b[3] >= MARGIN}
        L.append(f"| {name} | {'yes' if idx else 'no'} | {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f} | {a[3]:+.3f} | {a[4]} | "
                 f"{b[0]:.3f} / {b[1]:.3f} / {b[2]:.3f} | {b[3]:+.3f} | {b[4]} | {'yes' if a[3] >= MARGIN else 'no'} / "
                 f"{'yes' if b[3] >= MARGIN else 'no'} |")
    nested = {k: all(v['index'] for v in out.values() if v[f'selfref_{k}']) for k in ('two', 'three')}
    changed = [ONAMES[m] for m, v in out.items() if v['selfref_two'] != v['selfref_three']]
    noidx_max = {k: max((v[f'{k}_orders'][3], ONAMES[m]) for m, v in out.items() if not v['index']) for k in ('two', 'three')}
    L += ['', f"Nesting (no self-reference without a self-index): two orders {nested['two']}, three orders {nested['three']}. "
          f"Verdict changed by the third order: {', '.join(changed) or 'none'}. Largest excess without a self-index: two orders "
          f"{noidx_max['two'][0]:+.3f} ({noidx_max['two'][1]}), three orders {noidx_max['three'][0]:+.3f} ({noidx_max['three'][1]})."]
    (S3 / 'r14_orders_20261101.json').write_text(json.dumps({'results': out, 'nested': nested, 'changed': changed,
                                                            'noindex_max': noidx_max}, indent=1), encoding='utf-8')
    (S3 / 'r14_orders_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


def orders_tex(S3):
    R = json.load(open(S3 / 'r14_orders_20261101.json', encoding='utf-8'))
    YES = {'en': ('yes', 'no'), 'zh': ('有', '无')}

    def sg(x, d=2):
        """signed number in math mode; 0.00 without sign"""
        s = f'{x:+.{d}f}'
        if s in ('+0.00', '-0.00'):
            return f'\\({abs(x):.{d}f}\\)'
        return f'\\({s}\\)'

    def block_orders(lang):
        out = []
        for m, name in ONAMES.items():
            v = R['results'][m]
            a, b = v['two_orders'], v['three_orders']
            out.append(f"{name} & {YES[lang][0] if v['index'] else YES[lang][1]} & {sg(a[3])} & {a[4]} & {sg(b[3])} & {b[4]} \\\\")
        return out

    APP = {'tab:self_orders': block_orders}
    for lang in ('en', 'zh'):
        text = []
        for lab, f in APP.items():
            text += [f'%% {lab}'] + f(lang) + ['%% end']
        (S3 / f'r14_tex_{lang}.tex').write_text('\n'.join(text) + '\n', encoding='utf-8')
        print('\n'.join(text))
    print('nested', R['nested'], 'changed', R['changed'], 'no-index max', R['noindex_max'])


def main(argv=None):
    args = arguments(__doc__).parse_args(argv)
    r = Roots(args)
    S8, S3 = r.out(M8 + '/stats'), r.out(W3 + '/stats')
    threshold(r.res(M8), r.res(W3), r.res('evidence'), S3)
    if orders(S8, S3) is not False:
        orders_tex(S3)


if __name__ == '__main__':
    main()
