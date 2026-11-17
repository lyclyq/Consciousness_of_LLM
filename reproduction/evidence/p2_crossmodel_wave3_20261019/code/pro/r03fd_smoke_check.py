"""Smoke rule of the R03 part-B runs (crossmodel_readout_fd_r03.py; fixed before any result; execution only, no outcome is looked
at): the smoke folder holds 2 held-out bodies x 2 option orders. It passes if there are 4 items; in a single pass all 4, with
thinking at least 3, have both endpoints complete (an answer letter after the reasoning at both); every complete item has, at both
endpoints and at every layer applied, a finite unperturbed f and finite derivatives for the five directions, and a non-zero norm
of the whole difference. Printed for information only (not a criterion): the median over items and layers of |D_zero| / |D_total|,
the numerical noise of a batch relative to the derivative along the whole difference. Exit status 0 = pass.
usage: r03fd_smoke_check.py SMOKE_DIR thinking|single"""
import json
import math
import statistics
import sys
from pathlib import Path

d, mode = Path(sys.argv[1]), sys.argv[2]
recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((d / 'items').glob('*.json'))]
problems = []
if len(recs) != 4:
    problems.append(f'{len(recs)} items')
full = [r for r in recs if all('D' in r['endpoints'].get(e, {}) for e in ('A', 'U'))]
if len(full) < (4 if mode == 'single' else 3):
    problems.append(f'{len(full)} of {len(recs)} items complete')
noise = []
for r in full:
    for e in ('A', 'U'):
        E = r['endpoints'][e]
        if not math.isfinite(E['f0']):
            problems.append(f'{r["id"]} {e}: f0 not finite')
        for k, xs in E['D'].items():
            if len(xs) != len(r['layers']) or not all(math.isfinite(x) for x in xs):
                problems.append(f'{r["id"]} {e}: D {k} incomplete or not finite')
        noise += [abs(z) / abs(t) for z, t in zip(E['D']['zero'], E['D']['total']) if t]
    if not all(n > 0 for n in r['norms']['total']):
        problems.append(f'{r["id"]}: zero difference at some layer')
info = f'median |D_zero|/|D_total| {statistics.median(noise):.3g}' if noise else 'no noise estimate'
print(f'smoke {d.name}: ' + ('PASS' if not problems else 'FAIL ' + '; '.join(problems[:8])) + f' | {len(full)}/{len(recs)} complete; {info}')
sys.exit(1 if problems else 0)
