"""Dose-floor random control of the R06 follow-up (PRO runs of 10-31, code/pro/r06f_pro.sh; user 10-31: "可以先跑加幅度的"), from the
per-item results in the wave-3 archive. Each random edit of R06 (same replacement in a random 8-dim subspace, base seeds 20261022..26)
is scaled up per item and per layer to at least the realized relative size of the V edit on the same item (r05q_all; never scaled
down). Statistics as in the paper (make_v4_mechanism.py: items, summarize): the item-paired difference from the unedited condition of
the same recipient with a 95% bootstrap interval; the unedited items are those of r05q_all, copied into every run.
  frswU8_A  random, own message (floor: the V removal swU8_A)
  frswA8_U  random, User message (floor: the V installation swA8_U)
Dose check: per item, the realized relative size of the scaled random edit over all edited layers, sqrt(sum ||delta||^2 / sum ||x||^2),
divided by that of the V edit of the same item, and per layer the smallest ratio of the two relative sizes (>= 1 if the floor held).
For comparison: the V edit and the unscaled random edit of the same base seeds (stats/r05r06_20261031.json).
Outputs (ARCHIVE/stats): r06f_20261031.json, r06f_20261031.md.
usage: analyze_r06f_20261031.py ARCHIVE"""
import json
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_r05r06_20261031 import READOUTS, SEED, f  # noqa: E402
from make_v4_mechanism import items, summarize  # noqa: E402

RUNS = [f'r06f_s{j}' for j in range(5)]
PAIRS = (('frswU8_A', 'swU8_A', 'rswU8_A', 'removal (random, own message)'), ('frswA8_U', 'swA8_U', 'rswA8_U', 'installation (random, User message)'))


def rel(d, keep=None):
    ix = [i for i, l in enumerate(d['layers']) if keep is None or l in keep]
    return math.sqrt(sum(d['edit2'][i] for i in ix) / sum(d['state2'][i] for i in ix))


def dose_check(recs, vrecs, c, vc):
    V = {(r['id'], tuple(r['order'])): r['dose'] for r in vrecs if r['cond'] == vc and r.get('dose')}
    ratio, lmin, up, sc = [], [], 0, []
    for r in recs:
        if r['cond'] != c or not r.get('dose') or (r['id'], tuple(r['order'])) not in V:
            continue
        d, v = r['dose'], V[(r['id'], tuple(r['order']))]
        ratio.append(rel(d) / rel(v))
        per = [math.sqrt(d['edit2'][i] / d['state2'][i]) / math.sqrt(v['edit2'][j] / v['state2'][j])
               for i, l in enumerate(d['layers']) for j, m in enumerate(v['layers']) if l == m and v['edit2'][j] > 0]
        lmin.append(min(per))
        s = r['floor']['scale']
        up += sum(x > 1 for x in s)
        sc += s
    if not ratio:
        return None
    return {'n': len(ratio), 'ratio_median': statistics.median(ratio), 'ratio_min': min(ratio), 'layer_ratio_min': min(lmin),
            'scaled_up_share': up / len(sc), 'scale_median': statistics.median(sc)}


def main():
    A = Path(sys.argv[1])
    prev = json.load(open(A / 'stats/r05r06_20261031.json', encoding='utf-8'))['results']
    res, missing = {}, []
    for rd, _, root, _ in READOUTS:
        vrecs = items(str(A / 'runs/pro' / f'{root}_r05q_all'))
        res[rd] = {'V': {vc: prev[rd]['r05q_all']['stats'].get(vc) for _, vc, _, _ in PAIRS},
                   'unscaled': {rc: [prev[rd][r]['stats'].get(rc) for r in SEED if r in prev[rd]] for _, _, rc, _ in PAIRS}}
        for run in RUNS:
            d = A / 'runs/pro' / f'{root}_{run}'
            if not (d / 'summary.json').exists():
                missing.append(f'{root}_{run}')
                continue
            recs = items(str(d))
            res[rd][run] = {'stats': summarize(recs, None), 'dose': {c: dose_check(recs, vrecs, c, vc) for c, vc, _, _ in PAIRS},
                            'n_items': len(recs)}
    L = ['# R06 follow-up: dose-floor random control (PRO, 10-31)', '',
         'Item-paired difference from the unedited condition (95% bootstrap interval); thinking: share answering "you", '
         'no thinking: mean P("you"). k = 8, all positions. The random edit of each base seed is scaled up per item and layer to at '
         'least the realized relative size of the V edit on the same item (never down).', '',
         '## Effect', '',
         '| readout | edit | V (r05q_all) | random, unscaled: max abs over 5 seeds | ' + ' | '.join(str(v) for v in SEED.values())
         + ' | max abs | seeds with interval excluding 0 |', '|---' * 11 + '|']
    for rd, _, _, name in READOUTS:
        for c, vc, rc, lab in PAIRS:
            un = [x['diff'] for x in res[rd]['unscaled'][rc] if x and 'diff' in x]
            vals = [res[rd][r]['stats'].get(c) if r in res[rd] else None for r in RUNS]
            ds = [v for v in vals if v and 'diff' in v]
            excl = sum(v['diff_ci'][0] > 0 or v['diff_ci'][1] < 0 for v in ds)
            row = [name, lab, f(res[rd]['V'].get(vc)), f'{max(abs(x) for x in un):.3f} (n={len(un)})' if un else '—']
            row += [f(v) for v in vals]
            row += [f"{max(abs(v['diff']) for v in ds):.3f} (n={len(ds)})" if ds else '—', f'{excl} of {len(ds)}']
            L.append('| ' + ' | '.join(row) + ' |')
    L += ['', '## Dose check', '',
          'Per item: relative size of the scaled random edit over all layers / that of the V edit of the same item (median, minimum '
          'over items); the smallest per-layer ratio over all items and layers (>= 1 if the floor held); the share of item-layers '
          'scaled up and the median scale factor. Pooled over the five seeds.', '',
          '| readout | edit | items | ratio, median | ratio, min | per-layer ratio, min | layers scaled up | scale, median |',
          '|---|---|---|---|---|---|---|---|']
    for rd, _, _, name in READOUTS:
        for c, _, _, lab in PAIRS:
            D = [res[rd][r]['dose'][c] for r in RUNS if r in res[rd] and res[rd][r]['dose'].get(c)]
            if not D:
                continue
            n = sum(x['n'] for x in D)
            L.append(f"| {name} | {lab} | {n} | {statistics.median(x['ratio_median'] for x in D):.2f} | {min(x['ratio_min'] for x in D):.2f} | "
                     f"{min(x['layer_ratio_min'] for x in D):.3f} | {sum(x['scaled_up_share'] * x['n'] for x in D) / n:.2f} | "
                     f"{statistics.median(x['scale_median'] for x in D):.2f} |")
    if missing:
        L += ['', 'Not finished (left out): ' + ', '.join(missing)]
    (A / 'stats/r06f_20261031.json').write_text(json.dumps({'results': res, 'missing': missing}, indent=1), encoding='utf-8')
    (A / 'stats/r06f_20261031.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
