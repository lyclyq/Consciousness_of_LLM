"""Statistics for the models added to or completed in the v4 cross-model comparison, with the rules of stats_wave3.py:
added: Qwen3.5-27B (Chinese), Granite-4.1-30B (English, no thinking mode), Granite-4.2-30B (English); completed to 900 thinking
items: Qwen3.8-Flash-Next, Qwen3.6-27B, Seed-OSS-36B. Outputs (WAVE3_ARCHIVE/stats):
  v4_geometry_summary.csv         layers at which Assistant has the smallest R_A in >= 80% / all held-out bodies, and the layers of
                                  the last third (the last floor(L/3) layers) at which it does so in >= 80%
  v4_attribution_non_thinking.csv body-level mean P(you), first two option orders, and all three
  v4_thinking_counts.csv          per source: finished items, answers, you_share = answered 'you' / finished
usage: stats_v4_crossmodel.py WAVE3_ARCHIVE"""
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
NONTHINK = {'qwen35_27b': R / 'server3/suite/qwen35_27b', 'granite41_30b': R / 'server3/suite/granite41_30b_en',
            'granite42_30b': R / 'server3/suite/granite42_30b_en'}
GEO = {(m, 'non_thinking'): d for m, d in NONTHINK.items()}
THINK = {'qwen38_flashnext': R / 'pro/thinking/flashnext', 'qwen36_27b': R / 'pro/thinking/qwen36_27b',
         'seed_oss_36b': R / 'server3/thinking/seed_oss_36b', 'qwen35_27b': R / 'server3/thinking/qwen35_27b',
         'granite42_30b': R / 'server3/thinking/granite42_30b_en'}


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
    for r in rows:
        print('  ', r)


out = []
for model, d in NONTHINK.items():
    recs = [json.loads(Path(f).read_text(encoding='utf-8')) for f in sorted(glob.glob(str(d / 'G' / 'role150_*.json')))]
    recs = [r for r in recs if 'roles' in r]
    for label, idx in (('3_orders', [0, 1, 2]), ('2_orders_as_paper', [0, 1])):
        m = lambda r, role, key: float(np.mean([r['roles'][role]['by_order'][i][key] for i in idx]))
        pa = np.array([m(r, 'assistant', 'self') for r in recs]); pu = np.array([m(r, 'user', 'self') for r in recs])
        ps = np.array([m(r, 'system', 'self') for r in recs])
        dl = pa - np.maximum(pu, ps); k = int((dl > 0).sum())
        out.append(dict(model=model, orders=label, n_bodies=len(recs), p_you_own=round(pa.mean(), 4), p_you_user=round(pu.mean(), 4),
                        p_you_system=round(ps.mean(), 4), own_minus_max_other=round(dl.mean(), 4), bodies_positive=k,
                        sign_p=float(f'{sign_p(k, len(dl)):.3g}'),
                        p_system_as_user=round(float(np.mean([m(r, 'system', 'user') for r in recs])), 4)))
write_csv('v4_attribution_non_thinking.csv', out)

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
write_csv('v4_thinking_counts.csv', out)

out = []
for (model, mode), d in GEO.items():
    s = json.loads((d / 'summary.json').read_text(encoding='utf-8'))
    L = s['n_layers']
    last = range(L - L // 3, L)
    for kind in ('content', 'onset'):
        k = s[kind]
        sh = k['share_A_lowest']
        lay80 = [i for i in range(L) if sh[i] >= 0.8]
        out.append(dict(model=model, mode=mode, position=kind, n_layers=L, layers_A_apart_all_heldout=k['layers_A_lowest_all_bodies'],
                        layers_A_apart_80pct=k['layers_A_lowest_80'], last_third_80pct=sum(i in last for i in lay80),
                        last_third_size=len(last), layers_80pct=' '.join(map(str, lay80)),
                        median_R_A=round(float(np.median(k['mean_R'][0])), 4)))
write_csv('v4_geometry_summary.csv', out)
