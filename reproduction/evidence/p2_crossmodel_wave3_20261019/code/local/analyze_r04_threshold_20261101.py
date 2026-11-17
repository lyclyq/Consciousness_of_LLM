"""R04 of the supervisor's review (11-01): sensitivity of the cross-model self-index criterion to its threshold and to the depth band,
from the per-layer shares already archived (no new runs). For every model and rendering, the share of held-out bodies in which the
model's own message has the smallest distance ratio (share_A_lowest), on message content and at the response onset, read from the
same files as make_fig_crossmodel_v6.py (four-stream models: mean of the per-stream ratios). Per threshold t in 0.7, 0.8 (the
paper), 0.9:
  - self-index (Definition def:self): some layer l >= L/3 with share >= t at content or onset;
  - last third (l >= 2L/3) and last quarter (l >= 3L/4): layers of the band with share >= t at content or onset, and whether all are;
  - classification under the non-thinking rendering (thinking rendering for models with only a thinking mode), as in the paper;
    band coverage also in the rendering of each mode in which the model recognizes its own messages.
Self-reference is behavioural and does not depend on t; the models and modes with self-reference are those of tab:self_crossmodel.
Outputs (ARCHIVE/stats): r04_threshold_20261101.json, r04_threshold_20261101.md.
usage: analyze_r04_threshold_20261101.py ARCHIVE8 ARCHIVE3"""
import glob
import json
import math
import sys
from pathlib import Path

NAMES = {'qwen38_27b': 'Qwen3.8-27B', 'gpt_oss_120b': 'gpt-oss-120b', 'gpt_oss_20b': 'gpt-oss-20b',
         'qwen38_flashnext': 'Qwen3.8-Flash-Next', 'glm53_flash': 'GLM-5.3-Flash', 'exaone4_32b': 'EXAONE-4.0.1-32B',
         'glm46v_flash': 'GLM-4.6V-Flash', 'glm47_flash': 'GLM-4.7-Flash', 'r1_llama70b': 'R1-Distill-Llama-70B',
         'granite42_30b': 'Granite-4.2-30B', 'granite41_30b': 'Granite-4.1-30B', 'olmo2_13b': 'OLMo-2-13B', 'olmo2_7b': 'OLMo-2-7B',
         'seed_oss_36b': 'Seed-OSS-36B', 'gemma4_31b': 'Gemma-4-31B', 'qwen36_27b': 'Qwen3.6-27B', 'qwen36_35b': 'Qwen3.6-35B-A3B',
         'qwen35_27b': 'Qwen3.5-27B'}
SELFREF = [('qwen38_27b', 'non_thinking'), ('qwen38_27b', 'thinking'), ('qwen38_flashnext', 'thinking'), ('gpt_oss_20b', 'thinking'),
           ('gpt_oss_120b', 'thinking'), ('glm53_flash', 'thinking'), ('exaone4_32b', 'thinking')]
TS = (0.7, 0.8, 0.9)


def sources(A8, A3):
    summ = {}
    for f in glob.glob(str(A8 / 'models' / '*' / '*' / 'geo*' / '_run_summary.json')):
        p = Path(f)
        if p.parts[-2] in ('geo_attr', 'geo'):
            summ[(p.parts[-4], 'non_thinking' if p.parts[-3] == 'non_thinking' else 'thinking')] = Path(f)
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
    out = {}
    for key, f in summ.items():
        if key[0] not in NAMES or key[0] == 'qwen38_flashnext':
            continue
        s = json.loads(f.read_text(encoding='utf-8'))
        out[key] = (s['content']['share_A_lowest'], s['onset']['share_A_lowest'], str(f))
    streams = {('qwen38_flashnext', 'non_thinking'): R / 'pro/flashnext_stream_20261020/flashnext_stream_geometry_non_20261020.json',
               ('qwen38_flashnext', 'thinking'): R / 'pro/flashnext_stream_20261020/flashnext_stream_geometry_think_20261020.json',
               ('glm53_flash', 'thinking'): R / 'server3/suite_think/glm53_flash_stream_geometry.json'}
    per_stream = {}
    for key, f in streams.items():
        s = json.loads(f.read_text(encoding='utf-8'))
        out[key] = (s['content']['mean_of_streams']['share_A_lowest_by_layer'], s['onset']['mean_of_streams']['share_A_lowest_by_layer'],
                    str(f))
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


def main():
    A8, A3 = Path(sys.argv[1]), Path(sys.argv[2])
    src, per_stream = sources(A8, A3)
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
    (A3 / 'stats' / 'r04_threshold_20261101.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    (A3 / 'stats' / 'r04_threshold_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
