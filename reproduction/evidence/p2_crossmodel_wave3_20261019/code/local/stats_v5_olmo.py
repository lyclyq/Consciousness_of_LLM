"""Statistics for the two OLMo-2 models added to the cross-model comparison in paper v5 (user 10-29), with the rules of
stats_v4_crossmodel.py (from which the code is copied; only the model list and the output names differ): OLMo-2-1124-7B-Instruct
and OLMo-2-1124-13B-Instruct (English, no thinking mode). Outputs (WAVE3_ARCHIVE/stats):
  v5_olmo_geometry_summary.csv         as v4_geometry_summary.csv (last third = the last floor(L/3) layers)
  v5_olmo_attribution_non_thinking.csv as v4_attribution_non_thinking.csv (body-level mean P(you), first two option orders, and all three)
usage: stats_v5_olmo.py WAVE3_ARCHIVE"""
import csv
import glob
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

A = Path(sys.argv[1])
R = A / 'runs'
NONTHINK = {'olmo2_7b': R / 'pro/suite/olmo2_7b_en', 'olmo2_13b': R / 'pro/suite/olmo2_13b_en'}
GEO = {(m, 'non_thinking'): d for m, d in NONTHINK.items()}


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
        m = lambda r, role, key: float(np.mean([r['roles'][role]['by_order'][i][key] for i in idx]))  # noqa: E731
        pa = np.array([m(r, 'assistant', 'self') for r in recs]); pu = np.array([m(r, 'user', 'self') for r in recs])
        ps = np.array([m(r, 'system', 'self') for r in recs])
        dl = pa - np.maximum(pu, ps); k = int((dl > 0).sum())
        out.append(dict(model=model, orders=label, n_bodies=len(recs), p_you_own=round(pa.mean(), 4), p_you_user=round(pu.mean(), 4),
                        p_you_system=round(ps.mean(), 4), own_minus_max_other=round(dl.mean(), 4), bodies_positive=k,
                        sign_p=float(f'{sign_p(k, len(dl)):.3g}'),
                        p_system_as_user=round(float(np.mean([m(r, 'system', 'user') for r in recs])), 4)))
write_csv('v5_olmo_attribution_non_thinking.csv', out)

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
write_csv('v5_olmo_geometry_summary.csv', out)
