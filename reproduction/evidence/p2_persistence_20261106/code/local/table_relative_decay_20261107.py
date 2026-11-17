"""Print role and word M2 side by side for every persistence analysis JSON (read-only)."""
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
rows = []
for sub in ('analysis_server3', 'analysis_pro'):
    for p in sorted((root / sub).glob('*.json')):
        d = json.loads(p.read_text(encoding='utf-8'))
        stem = p.stem
        phase = '3' if stem.startswith('p3') else '1'
        r = d['roles']
        def g(k):
            v = r.get(k, {})
            return (v.get('M2_size_kept'), v.get('M2_ci95'), v.get('C1_median')) if v.get('defined') else (None, None, None)
        au, as_, us = g('A-U'), g('A-S'), g('U-S')
        w = d.get('words', {})
        wb = {k.rsplit('_', 1)[0]: v for k, v in w.items() if k.endswith('_before')}
        wa = {k.rsplit('_', 1)[0]: v for k, v in w.items() if k.endswith('_after')}
        bt = d.get('body_tokens_full', {})
        rows.append(dict(tag=d['tag'], order=d['order'], phase=phase, group=d['group'], n=d['n_held_out'],
                         au=au, as_=as_, us=us, wb=wb, wa=wa, bt=bt))

def f(x):
    return '  -  ' if x is None else f'{x:.3f}'

def fc(t):
    m, c, _ = t
    if m is None:
        return '-'
    return f'{m:.3f} [{c[0]:.2f},{c[1]:.2f}]'

print('phase order tag | A-U | A-S | U-S | words before: place time name feeling | words after: place time name feeling | C1 ratio U-S/A-U, word(before, median)/A-U')
for x in sorted(rows, key=lambda x: (x['order'], x['phase'], x['group'], x['tag'])):
    wbs = ' '.join(f(x['wb'][k]['M2_size_kept']) if k in x['wb'] else '  -  ' for k in ('place', 'time', 'name', 'feeling'))
    was = ' '.join(f(x['wa'][k]['M2_size_kept']) if k in x['wa'] else '  -  ' for k in ('place', 'time', 'name', 'feeling'))
    c_au = x['au'][2]
    c_us = x['us'][2]
    rat_us = (np.sqrt(c_us / c_au) if (c_au and c_us and c_au > 0 and c_us > 0) else None)
    wc = [x['wb'][k]['C1_median'] for k in x['wb']]
    rat_w = (np.sqrt(np.median(wc) / c_au) if (wc and c_au and c_au > 0 and np.median(wc) > 0) else None)
    print(f"{x['phase']} {x['order'][:3]} {x['tag']:<26} {x['group'][:5]:<5} | {fc(x['au'])} | {fc(x['as_'])} | {fc(x['us'])} | {wbs} | {was} | {f(rat_us)} {f(rat_w)}")

print()
print('body tokens of the full body (phase 1 only, common ratio last/first third): A-U A-S U-S words-before')
for x in sorted(rows, key=lambda x: (x['order'], x['phase'], x['group'], x['tag'])):
    if not x['bt']:
        continue
    b = x['bt']
    ws = ' '.join(f"{k.split('_')[0]}={b[k]['common_ratio']:.2f}" for k in b if k.endswith('_before'))
    print(f"{x['phase']} {x['order'][:3]} {x['tag']:<26} A-U {b.get('A-U', {}).get('common_ratio', float('nan')):.2f} "
          f"A-S {b.get('A-S', {}).get('common_ratio', float('nan')):.2f} U-S {b.get('U-S', {}).get('common_ratio', float('nan')):.2f} | {ws}")
