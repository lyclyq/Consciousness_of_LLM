"""Smoke rule of the R06 dose-floor runs (fixed before any result; execution only, no outcome is looked at): the smoke folder holds
frswU8_A and frswA8_U on 2 held-out bodies x 2 option orders, all positions, fit_q.pt, --dose-floor <dir>_r05q_all. It passes if
each condition has its 4 items; with thinking, at least 3 of the 4 items of each condition finished; in a single pass, the mean
letter mass of each condition is at least 0.9; every item records a dose and a floor over the same layers with edited positions,
every factor is at least 1, and at every layer the realized relative size edit2 / state2 is at least that of the V item of the
target condition in FLOOR_DIR (relative tolerance 1e-3). Exit status 0 = pass.
usage: r06f_smoke_check.py SMOKE_DIR FLOOR_DIR thinking|single"""
import json
import sys
from collections import defaultdict
from pathlib import Path

d, fd, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
by = defaultdict(list)
for f in sorted((d / 'items').glob('*.json')):
    r = json.loads(f.read_text(encoding='utf-8'))
    by[r['cond']].append((f.stem.split('__')[-1], r))
problems = []
for c in ('frswU8_A', 'frswA8_U'):
    rs = by.get(c, [])
    if len(rs) != 4:
        problems.append(f'{c}: {len(rs)} items')
        continue
    if mode == 'thinking' and sum(r['finished'] for _, r in rs) < 3:
        problems.append(f'{c}: {sum(r["finished"] for _, r in rs)} of 4 finished')
    if mode == 'single' and sum(r['mass'] for _, r in rs) / 4 < 0.9:
        problems.append(f'{c}: mean letter mass {sum(r["mass"] for _, r in rs) / 4:.3f}')
    for o, r in rs:
        D, F = r.get('dose'), r.get('floor')
        if not D or not F or D['n_pos'] <= 0 or D['layers'] != F['layers']:
            problems.append(f'{c} {r["id"]} {o}: dose or floor missing or not over the same layers')
            continue
        if min(F['scale']) < 1:
            problems.append(f'{c} {r["id"]} {o}: factor below 1')
        T = json.loads((fd / 'items' / f'{F["target"]}__{r["id"]}__{o}.json').read_text(encoding='utf-8'))['dose']
        tv = dict(zip(T['layers'], (e / s for e, s in zip(T['edit2'], T['state2']))))
        low = [l for l, e, s in zip(D['layers'], D['edit2'], D['state2']) if e / s < tv[l] * (1 - 1e-3)]
        if low:
            problems.append(f'{c} {r["id"]} {o}: below the V size at layers {low[:5]}')
summ = {c: (len(rs), sum(r.get('finished', True) for _, r in rs)) for c, rs in by.items()}
print(f'smoke {d.name}: ' + ('PASS' if not problems else 'FAIL ' + '; '.join(problems)) + f' | items/finished {summ}')
sys.exit(1 if problems else 0)
