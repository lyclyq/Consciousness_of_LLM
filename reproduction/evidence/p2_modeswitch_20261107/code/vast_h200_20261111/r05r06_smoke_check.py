"""Smoke rule of the R05/R06 runs (fixed before any result; execution only, no outcome is looked at): the smoke folder holds the
five conditions base_A, base_U, swU8_A, swA8_U, rswU8_A on 2 held-out bodies x 2 option orders, edited in the message region with
fit_m.pt. It passes if every condition has its 4 items; with thinking, at least 3 of the 4 items of each condition finished; in a
single pass, the mean letter mass of each condition is at least 0.9; every edited item records a dose with edited positions and a
non-zero change at some layer, and no unedited item records one. Exit status 0 = pass.
usage: r05r06_smoke_check.py SMOKE_DIR thinking|single"""
import json
import sys
from collections import defaultdict
from pathlib import Path

d, mode = Path(sys.argv[1]), sys.argv[2]
by = defaultdict(list)
for f in sorted((d / 'items').glob('*.json')):
    r = json.loads(f.read_text(encoding='utf-8'))
    by[r['cond']].append(r)
problems = []
for c in ('base_A', 'base_U', 'swU8_A', 'swA8_U', 'rswU8_A'):
    rs = by.get(c, [])
    if len(rs) != 4:
        problems.append(f'{c}: {len(rs)} items')
        continue
    if mode == 'thinking' and sum(r['finished'] for r in rs) < 3:
        problems.append(f'{c}: {sum(r["finished"] for r in rs)} of 4 finished')
    if mode == 'single' and sum(r['mass'] for r in rs) / 4 < 0.9:
        problems.append(f'{c}: mean letter mass {sum(r["mass"] for r in rs) / 4:.3f}')
    for r in rs:
        dose = r.get('dose')
        if c.startswith('base'):
            if dose is not None:
                problems.append(f'{c}: unedited item with a dose')
        elif not dose or dose['n_pos'] <= 0 or not any(e > 0 for e in dose['edit2']):
            problems.append(f'{c} {r["id"]}: no edit recorded')
summ = {c: (len(rs), sum(r.get('finished', True) for r in rs)) for c, rs in by.items()}
print(f'smoke {d.name}: ' + ('PASS' if not problems else 'FAIL ' + '; '.join(problems)) + f' | items/finished {summ}')
sys.exit(1 if problems else 0)
