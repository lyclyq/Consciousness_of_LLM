"""Statistics for the five wave-3 models in the paper (Seed-OSS-36B, EXAONE-4.0.1-32B, gpt-oss-20b, gpt-oss-120b, GLM-5.3-Flash),
with the rules of the eight-model archive (build_archive.py), so that the cross-model table and figure can combine both:
  attribution_non_thinking_body_level.csv  body-level mean P(you), first two option orders ('2_orders_as_paper') and all three
  thinking_attribution_counts.csv          per source: finished items, answers, you_share = answered 'you' / finished
  geometry_summary.csv                     layers at which Assistant has the smallest R_A in >= 80% / all held-out bodies;
                                           for GLM-5.3-Flash (four hyper-connection streams) the mean of the per-stream ratios and
                                           the strongest stream, as for Qwen3.8-Flash-Next in the paper
usage: stats_wave3.py WAVE3_ARCHIVE"""
import csv
import glob
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

A = Path(sys.argv[1])
R = A / 'runs'
NONTHINK = {'seed_oss_36b': R / 'server3/suite/seed_oss_36b', 'exaone4_32b': R / 'server3/suite/exaone4_32b_en'}
GEO = {('seed_oss_36b', 'non_thinking'): R / 'server3/suite/seed_oss_36b', ('seed_oss_36b', 'thinking'): R / 'server3/suite_think/seed_oss_36b',
       ('exaone4_32b', 'non_thinking'): R / 'server3/suite/exaone4_32b_en', ('exaone4_32b', 'thinking'): R / 'server3/suite_think/exaone4_32b_en',
       ('gpt_oss_20b', 'thinking'): R / 'pro/suite_think/gpt_oss_20b_mxfp4_en', ('gpt_oss_120b', 'thinking'): R / 'pro/suite_think/gpt_oss_120b_en'}
STREAMS = {('glm53_flash', 'thinking'): R / 'server3/suite_think/glm53_flash_stream_geometry.json'}
THINK = {'seed_oss_36b': R / 'server3/thinking/seed_oss_36b', 'exaone4_32b': R / 'server3/thinking/exaone4_32b_en',
         'gpt_oss_20b': R / 'pro/thinking/gpt_oss_20b_mxfp4_en', 'gpt_oss_120b': R / 'pro/thinking/gpt_oss_120b_en',
         'glm53_flash': R / 'api/glm53_flash'}


def sign_p(k, n):
    t = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * t)


def write_csv(name, rows):
    (A / 'stats').mkdir(exist_ok=True)
    with open(A / 'stats' / name, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(name, len(rows), 'rows')


out = []
for model, d in NONTHINK.items():
    recs = [json.loads(Path(f).read_text(encoding='utf-8')) for f in sorted(glob.glob(str(d / 'G' / 'role150_*.json')))]
    for label, idx in (('3_orders', [0, 1, 2]), ('2_orders_as_paper', [0, 1])):
        m = lambda r, role, key: float(np.mean([r['roles'][role]['by_order'][i][key] for i in idx]))
        pa = np.array([m(r, 'assistant', 'self') for r in recs]); pu = np.array([m(r, 'user', 'self') for r in recs])
        ps = np.array([m(r, 'system', 'self') for r in recs])
        dl = pa - np.maximum(pu, ps); k = int((dl > 0).sum())
        out.append(dict(model=model, orders=label, n_bodies=len(recs), p_you_own=round(pa.mean(), 4), p_you_user=round(pu.mean(), 4),
                        p_you_system=round(ps.mean(), 4), own_minus_max_other=round(dl.mean(), 4), bodies_positive=k,
                        sign_p=float(f'{sign_p(k, len(dl)):.3g}'),
                        p_system_as_user=round(float(np.mean([m(r, 'system', 'user') for r in recs])), 4)))
write_csv('attribution_non_thinking_body_level.csv', out)

out = []
true = {'assistant': 'self', 'user': 'user', 'system': 'system'}
for model, d in THINK.items():
    recs = [json.loads(Path(f).read_text(encoding='utf-8')) for f in glob.glob(str(d / 'items' / 'role150_*.json'))]
    for r in ('assistant', 'user', 'system'):
        rs = [x for x in recs if x['role'] == r]
        fin = [x for x in rs if x.get('finished')]
        c = Counter(x['on_answer'] for x in fin)
        out.append(dict(model=model, role=r, n_items=len(rs), n_finished=len(fin), answered_you=c.get('self', 0),
                        answered_user=c.get('user', 0), answered_system=c.get('system', 0), unparsed=c.get(None, 0),
                        correct_share=round(sum(x['on_answer'] == true[r] for x in fin) / max(1, len(fin)), 4),
                        you_share=round(c.get('self', 0) / max(1, len(fin)), 4)))
write_csv('thinking_attribution_counts.csv', out)

out = []
for (model, mode), d in GEO.items():
    s = json.loads((d / 'summary.json').read_text(encoding='utf-8'))
    L = s['n_layers']
    for kind in ('content', 'onset'):
        k = s[kind]
        out.append(dict(model=model, mode=mode, measure='geo', position=kind, n_layers=L, statistic='single_state',
                        layers_A_apart_all_heldout=k['layers_A_lowest_all_bodies'], layers_A_apart_80pct=k['layers_A_lowest_80'],
                        strongest_stream_80pct='', median_R_A=round(float(np.median(k['mean_R'][0])), 4)))
for (model, mode), f in STREAMS.items():
    s = json.loads(Path(f).read_text(encoding='utf-8'))
    for kind in ('content', 'onset'):
        k = s[kind]
        L = len(k['mean_of_streams']['R_A_by_layer'])
        best = max(v['A_lowest_80'] for n, v in k.items() if n.startswith('stream'))
        out.append(dict(model=model, mode=mode, measure='geo', position=kind, n_layers=L, statistic='mean_of_streams',
                        layers_A_apart_all_heldout=k['mean_of_streams']['A_lowest_all_bodies'], layers_A_apart_80pct=k['mean_of_streams']['A_lowest_80'],
                        strongest_stream_80pct=best, median_R_A=round(k['mean_of_streams']['R_A_median'], 4)))
write_csv('geometry_summary.csv', out)
