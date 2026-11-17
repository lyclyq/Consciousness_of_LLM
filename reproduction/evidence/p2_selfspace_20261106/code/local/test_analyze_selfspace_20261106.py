"""Synthetic check of analyze_selfspace_20261106.py: planted structure must come out.
Planted: content c (per token, random) is shared by all conditions; A = c + u (one global offset), S = c + small noise offset,
U = c; word pairs differ only in the first third of the body tokens (local effect); A_switch = c + 0.2 u; noise pair identical
up to tiny noise. Expected: E1 own ratio near 0 and U/S ratios large; E2 A-U distance retention near 1, word retention near 0,
null d' near 0; E3 A/U near 1 (no planted expansion).
usage: test_analyze_selfspace_20261106.py ANALYZE_SCRIPT TMP_DIR"""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

script, tmp = Path(sys.argv[1]), Path(sys.argv[2])
run, work, outj = tmp / 'run', tmp / 'work', tmp / 'out.json'
run.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
names = ['A', 'U', 'S', 'A_switch', 'A_p0', 'U_p0', 'S_p0', 'U_p0alt', 'U_you', 'U_me', 'U_hospital', 'U_school', 'U_yesterday',
         'U_tomorrow', 'U_ming', 'U_hong', 'U_pleasant', 'U_painful']
KL, L, d = [0, 8, 16, 24, 32, 39, 47, 55, 63], 64, 24
u = rng.normal(size=d) * 3
sdir = rng.normal(size=d)
word = {p: rng.normal(size=d) * 3 for p in ('hospital', 'yesterday', 'ming', 'pleasant')}
json.dump({'script_sha256': 'synthetic', 'conditions': [{'name': n} for n in names], 'keep_layers': KL, 'n_layers': L,
           'n_bodies': 150}, open(run / 'identity.json', 'w'))
for b in range(150):
    n = int(rng.integers(49, 66))
    c = rng.normal(size=(n, d)) + rng.normal(size=d) * 2  # tokens + body-specific mean
    first = (np.arange(n) / n) < 1 / 3
    st = {}
    for nm in names:
        h = c.copy()
        if nm in ('A', 'A_p0'):
            h = h + u
        if nm == 'A_switch':
            h = h + 0.2 * u
        if nm in ('S', 'S_p0'):
            h = h + 0.5 * sdir
        for p, v in word.items():
            if nm == f'U_{p}':
                h = h + first[:, None] * v
        h = h + rng.normal(size=h.shape) * 0.01
        st[nm] = h
    tokens = np.stack([np.stack([st[nm]] * len(KL)) for nm in names]).astype(np.float16)
    pooled = np.stack([np.repeat(st[nm].mean(0)[None], L, 0) for nm in names]).astype(np.float16)
    np.savez(run / f'role150_x_{b:03d}.npz', pooled=pooled, pooled3=pooled[:3], tokens=tokens)
    json.dump({'id': f'role150_x_{b:03d}', 'split': 'discovery' if b % 2 else 'confirmation', 'n_body_tokens': n},
              open(run / f'role150_x_{b:03d}.json', 'w'))
r = subprocess.run([sys.executable, '-I', str(script), str(run), str(work), str(outj)], capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:])
assert r.returncode == 0
o = json.loads(outj.read_text(encoding='utf-8'))
e1 = o['E1']['plain']
assert e1['A']['pooled_median_over_layers'] < 0.1 and e1['U']['pooled_median_over_layers'] > 1, e1
assert e1['A']['token_median_over_layers'] < 0.1, e1['A']
assert o['E2']['A-U']['distance_shift_retention_median'] > 0.9, o['E2']['A-U']
assert o['E2']['A-U']['semantic_shift_retention_median'] > 0.95, o['E2']['A-U']
assert o['E2']['U_you-U_me']['source_reliable'] is False, o['E2']['U_you-U_me']
assert o['E2']['U_hospital-U_school']['distance_shift_retention_median'] < 0.3, o['E2']['U_hospital-U_school']
assert abs(o['E2']['U_p0-U_p0alt']['source_dprime_median']) < 0.5, o['E2']['U_p0-U_p0alt']
assert 0.9 < o['E3']['L47']['A/U']['value'] < 1.1, o['E3']['L47']
assert o['E2']['verdict']['shift_words_below_self_distance'] and o['E1']['plain']['verdict']['own_below_0.3']
print('SYNTHETIC TEST PASSED')
