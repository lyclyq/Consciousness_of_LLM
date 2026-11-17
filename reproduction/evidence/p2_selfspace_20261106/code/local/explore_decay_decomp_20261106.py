"""EXPLORATORY (not pre-registered; 11-06, user: "user assistant 为何不会衰减"). Splits the drop of the distance-shift retention
(E2 of analyze_selfspace_20261106.py) into two parts: does the effect shrink (smaller difference vectors in the last third of the
body) or does it turn (same size, another direction)? For each pair (c1, c2) and stored token layer beyond the first third, on the
75 confirmation bodies, with D = h_c1 - h_c2 at the same body token:
  total energy   E_seg = mean over the segment's tokens of |D|^2   (all of the effect, shared or token-specific)
  common energy  C_seg = |mean D|^2, unbiased across the 15 held-out wording families (U-statistic of the family means)
  share_seg      C_seg / E_seg   (1 = every token shifted by the same vector, a pure translation)
  total_ret = sqrt(E_last / E_first)   common_ret = sqrt(C_last / C_first)
  cos = cosine between the first- and last-third mean D (unbiased cross term over different families)
The E2 shift retention is roughly common_ret x cos. Bodies are weighted equally within a family, families equally. Summaries are
medians over the layers, with leave-one-family-out jackknife standard errors.
usage: explore_decay_decomp_20261106.py RUN_DIR OUT_JSON"""
import json
import sys
from pathlib import Path
import numpy as np

RUN, OUTJ = Path(sys.argv[1]), Path(sys.argv[2])
ident = json.loads((RUN / 'identity.json').read_text(encoding='utf-8'))
names = [c['name'] for c in ident['conditions']]
KL = list(ident['keep_layers'])
L = ident['n_layers']
metas = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(RUN.glob('role150_*.json'))]
conf = [m for m in metas if m['split'] == 'confirmation']
assert len(conf) == 75, len(conf)
fams = sorted({m['category'] for m in conf})
assert len(fams) == 15, len(fams)
tk_layers = [l for l in KL if l >= L / 3]
PAIRS = [('A', 'U'), ('A', 'S'), ('U', 'S'), ('A_p0', 'U_p0'), ('A', 'A_switch'), ('A_switch', 'U'), ('U_p0', 'U'),
         ('U_you', 'U_me'), ('U_hospital', 'U_school'), ('U_yesterday', 'U_tomorrow'), ('U_ming', 'U_hong'),
         ('U_pleasant', 'U_painful'), ('U_p0', 'U_p0alt')]
ci = {n: i for i, n in enumerate(names)}
P, J = len(PAIRS), len(tk_layers)
acc = {}
for b, m in enumerate(conf):
    z = np.load(RUN / f'{m["id"]}.npz')['tokens']
    nt = z.shape[2]
    assert nt == m['n_body_tokens'], (m['id'], nt, m['n_body_tokens'])
    d = z.shape[-1]
    pos = np.arange(nt) / nt
    F, La = pos < 1 / 3, pos >= 2 / 3
    f = m['category']
    if f not in acc:
        acc[f] = {'mF': np.zeros((P, J, d)), 'mL': np.zeros((P, J, d)), 'eF': np.zeros((P, J)), 'eL': np.zeros((P, J)), 'n': 0}
    a = acc[f]
    for p, (c1, c2) in enumerate(PAIRS):
        for j, l in enumerate(tk_layers):
            k = KL.index(l)
            D = z[ci[c1], k].astype(np.float32) - z[ci[c2], k].astype(np.float32)
            a['mF'][p, j] += D[F].mean(0)
            a['mL'][p, j] += D[La].mean(0)
            a['eF'][p, j] += (D[F].astype(np.float64) ** 2).sum(1).mean()
            a['eL'][p, j] += (D[La].astype(np.float64) ** 2).sum(1).mean()
    a['n'] += 1
    del z
    if b % 15 == 0:
        print('body', b, flush=True)
for a in acc.values():
    for k in ('mF', 'mL', 'eF', 'eL'):
        a[k] = a[k] / a['n']


def stats(S):
    MF, ML = np.stack([acc[f]['mF'] for f in S]), np.stack([acc[f]['mL'] for f in S])
    n = len(S)
    sF, sL = MF.sum(0), ML.sum(0)
    CF = ((sF ** 2).sum(-1) - (MF ** 2).sum(-1).sum(0)) / (n * (n - 1))
    CL = ((sL ** 2).sum(-1) - (ML ** 2).sum(-1).sum(0)) / (n * (n - 1))
    X = ((sF * sL).sum(-1) - (MF * ML).sum(-1).sum(0)) / (n * (n - 1))
    EF = np.mean([acc[f]['eF'] for f in S], 0)
    EL = np.mean([acc[f]['eL'] for f in S], 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'total_ret': np.sqrt(EL / EF), 'common_ret': np.sqrt(CL / CF), 'cos_first_last': X / np.sqrt(CF * CL),
                'share_first': CF / EF, 'share_last': CL / EL, 'total_energy_first': EF}


full = stats(fams)
jack = [stats([g for g in fams if g != f]) for f in fams]
nf = len(fams)
out = {'note': 'exploratory, not pre-registered', 'unit': f'held-out wording family ({nf}); jackknife over families',
       'layers': tk_layers, 'pairs': {}}
for p, (c1, c2) in enumerate(PAIRS):
    row = {}
    for k, v in full.items():
        med = float(np.nanmedian(v[p]))
        jk = np.array([np.nanmedian(s[k][p]) for s in jack])
        se = float(np.sqrt((nf - 1) / nf * np.nansum((jk - np.nanmean(jk)) ** 2)))
        row[k] = {'median_over_layers': med, 'jackknife_se': se, 'per_layer': {str(l): float(v[p, j]) for j, l in enumerate(tk_layers)}}
    out['pairs'][f'{c1}-{c2}'] = row
    print(f'{c1}-{c2:12s}', ' '.join(f'{k} {row[k]["median_over_layers"]:.3f}±{row[k]["jackknife_se"]:.3f}'
                                     for k in ('total_ret', 'common_ret', 'cos_first_last', 'share_first', 'share_last')), flush=True)
OUTJ.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('written', OUTJ)
