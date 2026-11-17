"""EXPLORATORY (not pre-registered; 11-06, user: "这个衰减问题是不是只是因为某个身份的填充一直存在"). Is the persistence of the own
offset along the message just an effect of extra tokens before the body (the empty think block of an Assistant message, which
also shifts the body positions)? Same distance-retention measure as E2 of analyze_selfspace_20261106.py (shift ratio, last third
vs first third of the body, direction from discovery first-third tokens, family bootstrap over the 15 held-out wording families),
for pairs that add tokens before the body without changing the role, next to the role pairs:
  A - A_switch   same role, empty think block present vs absent (+4 tokens, body shifted)
  U_p0 - U       User message with vs without a 4-token neutral prefix (+4 tokens, body shifted)
  S_p0 - S, A_p0 - A   the same prefix in the other two roles
  A - U, A_switch - U, U - S   role pairs (A - U also differs by the think block; A_switch - U and U - S only by the role word)
usage: explore_filler_20261106.py RUN_DIR OUT_JSON"""
import json
import sys
from pathlib import Path
import numpy as np

RUN, OUTJ = Path(sys.argv[1]), Path(sys.argv[2])
NB = 2000
rng = np.random.default_rng(20261106)
ident = json.loads((RUN / 'identity.json').read_text(encoding='utf-8'))
names = [c['name'] for c in ident['conditions']]
KL = list(ident['keep_layers'])
L = ident['n_layers']
metas = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(RUN.glob('role150_*.json'))]
B = len(metas)
conf = np.array([m['split'] == 'confirmation' for m in metas])
disc = np.array([m['split'] == 'discovery' for m in metas])
nt = np.array([m['n_body_tokens'] for m in metas])
conf_idx, disc_idx = np.where(conf)[0], np.where(disc)[0]
fam_of = [metas[b].get('category') or metas[b]['id'] for b in conf_idx]
fams = sorted(set(fam_of))
fam_pos = {f: [i for i, x in enumerate(fam_of) if x == f] for f in fams}
boots = [np.concatenate([fam_pos[fams[k]] for k in rng.integers(0, len(fams), size=len(fams))]) for _ in range(NB)]
tk_layers = [l for l in KL if l >= L / 3]
NEED = ['A', 'A_switch', 'U', 'S', 'A_p0', 'U_p0', 'S_p0']
ci = {n: names.index(n) for n in NEED}
H = {}  # (cond, layer) -> list of per-body arrays (float32)
for b, m in enumerate(metas):
    z = np.load(RUN / f'{m["id"]}.npz')['tokens']
    for c in NEED:
        for l in tk_layers:
            H.setdefault((c, l), []).append(z[ci[c], KL.index(l)].astype(np.float32))
pos = [np.arange(nt[b]) / nt[b] for b in range(B)]


def sums(p1, p2):
    return np.array([len(p1), p1.sum(), (p1 ** 2).sum(), p2.sum(), (p2 ** 2).sum()])


def shift(s):
    t = s.sum(0)
    return (t[1] - t[3]) / t[0]


def dprime(s):
    t = s.sum(0)
    n = t[0]
    m1, m2 = t[1] / n, t[3] / n
    return (m1 - m2) / np.sqrt(max((t[2] / n - m1 ** 2 + t[4] / n - m2 ** 2) / 2, 1e-12))


PAIRS = [('A', 'U'), ('A', 'A_switch'), ('U_p0', 'U'), ('S_p0', 'S'), ('A_p0', 'A'), ('A_switch', 'U'), ('U', 'S')]
out = {'note': 'exploratory, not pre-registered', 'bootstrap_unit': f'held-out wording family ({len(fams)})', 'pairs': {}}
for c1, c2 in PAIRS:
    ret, src, bs = [], [], np.zeros((len(tk_layers), NB))
    for j, l in enumerate(tk_layers):
        w = sum(((H[(c1, l)][b] - H[(c2, l)][b])[pos[b] < 1 / 3]).sum(0) for b in disc_idx)
        w /= np.linalg.norm(w)
        Ss, St = np.zeros((B, 5)), np.zeros((B, 5))
        for b in conf_idx:
            p1, p2 = H[(c1, l)][b] @ w, H[(c2, l)][b] @ w
            f, la = pos[b] < 1 / 3, pos[b] >= 2 / 3
            Ss[b], St[b] = sums(p1[f], p2[f]), sums(p1[la], p2[la])
        ret.append(shift(St[conf_idx]) / shift(Ss[conf_idx]))
        src.append(dprime(Ss[conf_idx]))
        for q, i in enumerate(boots):
            sel = conf_idx[i]
            bs[j, q] = shift(St[sel]) / shift(Ss[sel])
    med = np.median(bs, 0)
    out['pairs'][f'{c1}-{c2}'] = {'distance_shift_retention_median': float(np.median(ret)),
                                  'ci95': [float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))],
                                  'source_dprime_median': float(np.median(src)),
                                  'per_layer': {str(l): float(ret[j]) for j, l in enumerate(tk_layers)}}
    r = out['pairs'][f'{c1}-{c2}']
    print(f'{c1}-{c2:10s} retention {r["distance_shift_retention_median"]:.3f} {[round(x, 3) for x in r["ci95"]]} '
          f"src d' {r['source_dprime_median']:.2f} per layer {[round(v, 2) for v in r['per_layer'].values()]}", flush=True)
OUTJ.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('written', OUTJ)
