"""Primary endpoints E1-E3 of PREREG_zh.md (self-space experiment, Qwen3.8-27B), with the analysis clarifications in README_zh.md
(written before the data were looked at). The per-body npz files are first split into per-layer memory maps (derived, regenerable);
every token-level quantity is then accumulated body by body (two passes: directions from discovery bodies, then projections), so
memory stays at a few MB per body.

E1 inside/outside: for role r with the other two o1, o2, axis a_r = discovery mean of h_r - (h_o1 + h_o2) / 2; on each confirmation
   body (pooled) or token, |(h_o1 - h_o2).a| / |(h_r - (h_o1 + h_o2) / 2).a|; median over bodies (tokens).
E2 transfer: per pair (c1, c2), direction = discovery mean of h_c1 - h_c2 over the source tokens; d' = difference of the mean
   projections / sqrt(mean of the two within-condition variances), on confirmation tokens; retention = d'(target) / d'(source).
   (a) source = first third of the body tokens, target = last third; (b) 16 semantic regions (k-means on standardized U_p0 states at
   L32, fitted on discovery tokens), each region in turn the source and all other regions the target, mean over regions.
E3 spread: RMS distance of confirmation bodies (pooled) or tokens from their condition centroid, ratio A/U and U/S, at L47 L55 L63.
CIs: 2,000 bootstrap resamples of the 15 held-out wording families (directions, axes and regions stay fixed; token-level E1: 500). Layer
summaries are medians over the analysed layers (beyond the first third: pooled L22-L63, stored token layers L24-L63), recomputed in
every resample.
usage: analyze_selfspace_20261106.py RUN_DIR WORK_DIR|MEM OUT_JSON   (MEM: arrays in memory, nothing written but OUT_JSON)"""
import json
import sys
from pathlib import Path
import numpy as np

RUN, WORK, OUTJ = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
NB = 2000
rng = np.random.default_rng(20261106)
ident = json.loads((RUN / 'identity.json').read_text(encoding='utf-8'))
names = [c['name'] for c in ident['conditions']]
KL = list(ident['keep_layers'])
L = ident['n_layers']
metas = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(RUN.glob('role150_*.json'))]
assert len(metas) == ident['n_bodies'], (len(metas), ident['n_bodies'])
B = len(metas)
conf = np.array([m['split'] == 'confirmation' for m in metas])
disc = np.array([m['split'] == 'discovery' for m in metas])
assert conf.sum() == 75 and disc.sum() == 75
nt = np.array([m['n_body_tokens'] for m in metas])
T = int(nt.max())
C = len(names)
ci = {n: i for i, n in enumerate(names)}
conf_idx, disc_idx = np.where(conf)[0], np.where(disc)[0]
# Bootstrap unit = wording family, as in the paper (Sec. setup: the 75 held-out bodies form 15 families of five variants that
# differ only in their numbers; the variants are not independent). Each resample draws 15 families with replacement and takes
# all bodies of each drawn family; the resample is a list of positions into conf_idx. (Correction of 11-06: the first server run
# resampled the 75 bodies as if independent.) Bodies without a category (synthetic test) are their own family.
fam_of = [metas[b].get('category') or metas[b]['id'] for b in conf_idx]
fams = sorted(set(fam_of))
fam_pos = {f: [i for i, x in enumerate(fam_of) if x == f] for f in fams}
boots = [np.concatenate([fam_pos[fams[k]] for k in rng.integers(0, len(fams), size=len(fams))]) for _ in range(NB)]

# ---- per-layer arrays: in memory (WORK_DIR = MEM, e.g. on server3) or split into memory maps (once) ----
d = np.load(RUN / f'{metas[0]["id"]}.npz')['pooled'].shape[-1]
if str(WORK) == 'MEM':
    pooled = np.zeros((B, C, L, d), np.float16)
    tokfile = {l: np.zeros((B, C, T, d), np.float16) for l in KL}
    for b, m in enumerate(metas):
        z = np.load(RUN / f'{m["id"]}.npz')
        assert z['tokens'].shape == (C, len(KL), nt[b], d)
        pooled[b] = z['pooled']
        tk_ = z['tokens']
        for k, l in enumerate(KL):
            tokfile[l][b, :, :nt[b]] = tk_[:, k]
        del z, tk_
    print('loaded into memory', flush=True)
else:
    WORK.mkdir(parents=True, exist_ok=True)
if str(WORK) != 'MEM' and not (WORK / 'SPLIT_DONE').exists():
    pooled = np.lib.format.open_memmap(WORK / 'pooled.npy', 'w+', np.float16, (B, C, L, d))
    toks = {l: np.lib.format.open_memmap(WORK / f'tokens_L{l}.npy', 'w+', np.float16, (B, C, T, d)) for l in KL}
    for b, m in enumerate(metas):
        z = np.load(RUN / f'{m["id"]}.npz')
        assert z['tokens'].shape == (C, len(KL), nt[b], d)
        pooled[b] = z['pooled']
        tk = z['tokens']
        for k, l in enumerate(KL):
            toks[l][b, :, :nt[b]] = tk[:, k]
        del z, tk
        if b % 25 == 0:
            print('split', b, flush=True)
    pooled.flush()
    for t in toks.values():
        t.flush()
    del pooled, toks
    (WORK / 'SPLIT_DONE').write_text('ok\n')
if str(WORK) != 'MEM':
    pooled = np.load(WORK / 'pooled.npy', mmap_mode='r')
    tokfile = {l: np.load(WORK / f'tokens_L{l}.npy', mmap_mode='r') for l in KL}


def tk(l, c, b):
    return np.asarray(tokfile[l][b, ci[c], :nt[b]], np.float32)


def ci95(x):
    x = np.asarray(x)
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def unit(v):
    return v / np.linalg.norm(v)


out = {'identity_script_sha256': ident['script_sha256'], 'n_bodies': B, 'n_confirmation': int(conf.sum()), 'bootstrap': NB,
       'bootstrap_unit': f'held-out wording family ({len(fams)})'}
pl_layers = [l for l in range(L) if l >= L / 3]
tk_layers = [l for l in KL if l >= L / 3]
out['layers'] = {'pooled': [pl_layers[0], pl_layers[-1]], 'tokens': tk_layers}

# ---- E1 ----
ROLESETS = {'plain': ('A', 'U', 'S'), 'p0': ('A_p0', 'U_p0', 'S_p0'), 'switch': ('A_switch', 'U', 'S')}
e1 = {}
for tag, (a, u, s) in ROLESETS.items():
    per, boot = {}, {}
    for r, o1, o2 in ((a, u, s), (u, a, s), (s, a, u)):
        vals = np.zeros((len(pl_layers), len(conf_idx)))
        for j, l in enumerate(pl_layers):
            X = {c: np.asarray(pooled[:, ci[c], l], np.float32) for c in (r, o1, o2)}
            off = X[r] - (X[o1] + X[o2]) / 2
            ax = unit(off[disc].mean(0))
            vals[j] = np.abs((X[o1] - X[o2])[conf] @ ax) / np.abs(off[conf] @ ax)
        boot[r] = np.array([np.median(np.median(vals[:, i], 1)) for i in boots])
        per[r] = {'pooled_median_over_layers': float(np.median(np.median(vals, 1))), 'ci95': ci95(boot[r]),
                  'per_layer_median': {str(l): float(np.median(vals[j])) for j, l in enumerate(pl_layers)}}
        rat = {}  # token level: per layer, per confirmation body, the ratio of each token
        for l in tk_layers:
            acc, n = np.zeros(d), 0
            for b in disc_idx:
                o = tk(l, r, b) - (tk(l, o1, b) + tk(l, o2, b)) / 2
                acc += o.sum(0)
                n += len(o)
            ax = unit(acc / n)
            rat[l] = {}
            for b in conf_idx:
                h1, h2 = tk(l, o1, b), tk(l, o2, b)
                o = tk(l, r, b) - (h1 + h2) / 2
                rat[l][b] = np.abs((h1 - h2) @ ax) / np.abs(o @ ax)
        tmed = lambda sel: np.median([np.median(np.concatenate([rat[l][b] for b in sel])) for l in tk_layers])  # noqa: E731
        per[r]['token_median_over_layers'] = float(tmed(conf_idx))
        per[r]['token_ci95'] = ci95([tmed(conf_idx[i]) for i in boots[:500]])
    for r in (u, s):
        per[r]['ratio_to_own_ci95'] = ci95(boot[r] / boot[a])
    per['verdict'] = {'own_below_0.3': per[a]['ci95'][1] < 0.3,
                      'others_at_least_2x_own': all(per[r]['ratio_to_own_ci95'][0] >= 2 for r in (u, s))}
    e1[tag] = per
    print('E1', tag, {r: (round(per[r]['pooled_median_over_layers'], 3), round(per[r]['token_median_over_layers'], 3)) for r in (a, u, s)},
          per['verdict'], flush=True)
out['E1'] = e1

# ---- semantic regions: k-means on standardized U_p0 token states at L32, fitted on discovery tokens ----
from sklearn.cluster import KMeans  # noqa: E402
Hd = np.concatenate([tk(32, 'U_p0', b) for b in disc_idx])
mu, sd = Hd.mean(0), Hd.std(0) + 1e-6
km = KMeans(n_clusters=16, n_init=10, random_state=0).fit((Hd - mu) / sd)
del Hd
region = {b: km.predict((tk(32, 'U_p0', b) - mu) / sd) for b in range(B)}
G = 16
cnt_d = np.bincount(np.concatenate([region[b] for b in disc_idx]), minlength=G)
cnt_c = np.bincount(np.concatenate([region[b] for b in conf_idx]), minlength=G)
usable = [g for g in range(G) if cnt_d[g] >= 20 and cnt_c[g] >= 20]
out['regions'] = {'k': G, 'tokens_discovery': cnt_d.tolist(), 'tokens_confirmation': cnt_c.tolist(), 'usable': usable}
print('regions', cnt_c.tolist(), 'usable', len(usable), flush=True)


def sums(p1, p2):
    return np.array([len(p1), p1.sum(), (p1 ** 2).sum(), p2.sum(), (p2 ** 2).sum()])


def dprime(s):  # s: (bodies, 5) sums; nan if the set is empty
    t = s.sum(0)
    n = t[0]
    if n == 0:
        return np.nan
    m1, m2 = t[1] / n, t[3] / n
    v1, v2 = t[2] / n - m1 ** 2, t[4] / n - m2 ** 2
    return (m1 - m2) / np.sqrt(max((v1 + v2) / 2, 1e-12))


def shift(s):  # mean difference of the projections (no variance in it); nan if the set is empty
    t = s.sum(0)
    return np.nan if t[0] == 0 else (t[1] - t[3]) / t[0]


def nci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [float('nan')] * 2


# ---- E2 ----
# Two retention measures on the same token sets and directions:
#   'dprime' = d'(target) / d'(source)   (PREREG wording)
#   'shift'  = mean shift along the direction in the target / in the source   (amendment of 11-06, before the data: on synthetic
#              data a pure translation gives 0.87 with the d' ratio for the region split, because the target set (all other
#              regions) is more heterogeneous than the source region; the shift ratio gives 1.00)
# The verdicts use 'shift'; 'dprime' is reported alongside. A retention is defined only where the source effect is reliable
# (95% CI of the source d' excludes 0 and |d'| >= 0.2); otherwise it is reported as undefined.
PAIRS = [('A_p0', 'U_p0'), ('A', 'U'), ('A_p0', 'S_p0'), ('U', 'S'), ('U_you', 'U_me'), ('U_hospital', 'U_school'),
         ('U_yesterday', 'U_tomorrow'), ('U_ming', 'U_hong'), ('U_pleasant', 'U_painful'), ('U_p0', 'U_p0alt'), ('A_switch', 'U')]
pos = {b: np.arange(nt[b]) / nt[b] for b in range(B)}
e2, e2bs = {}, {}
for c1, c2 in PAIRS:
    M = {('distance', m): [] for m in ('dprime', 'shift')}
    M.update({('semantic', m): [] for m in ('dprime', 'shift')})
    dsrc = []
    BS = {k: np.zeros((len(tk_layers), NB)) for k in M}
    BSsrc = np.zeros((len(tk_layers), NB))
    for j, l in enumerate(tk_layers):
        wf, nf = np.zeros(d), 0
        wg, ng = np.zeros((G, d)), np.zeros(G)
        for b in disc_idx:
            D = tk(l, c1, b) - tk(l, c2, b)
            f = pos[b] < 1 / 3
            wf += D[f].sum(0)
            nf += f.sum()
            np.add.at(wg, region[b], D)
            ng += np.bincount(region[b], minlength=G)
        W = np.vstack([unit(wf / nf)] + [unit(wg[g] / ng[g]) for g in usable])
        Ssrc, Stgt = np.zeros((B, 5)), np.zeros((B, 5))
        Rs, Rt = np.zeros((len(usable), B, 5)), np.zeros((len(usable), B, 5))
        for b in conf_idx:
            P1, P2 = tk(l, c1, b) @ W.T, tk(l, c2, b) @ W.T
            f, la = pos[b] < 1 / 3, pos[b] >= 2 / 3
            Ssrc[b], Stgt[b] = sums(P1[f, 0], P2[f, 0]), sums(P1[la, 0], P2[la, 0])
            for k, g in enumerate(usable):
                ing = region[b] == g
                Rs[k, b], Rt[k, b] = sums(P1[ing, k + 1], P2[ing, k + 1]), sums(P1[~ing, k + 1], P2[~ing, k + 1])
        def measures(sel):
            out_ = {('distance', 'dprime'): dprime(Stgt[sel]) / dprime(Ssrc[sel]),
                    ('distance', 'shift'): shift(Stgt[sel]) / shift(Ssrc[sel])}
            with np.errstate(invalid='ignore', divide='ignore'):
                out_[('semantic', 'dprime')] = np.nanmean([dprime(Rt[k][sel]) / dprime(Rs[k][sel]) for k in range(len(usable))])
                out_[('semantic', 'shift')] = np.nanmean([shift(Rt[k][sel]) / shift(Rs[k][sel]) for k in range(len(usable))])
            return out_
        for k, v in measures(conf_idx).items():
            M[k].append(v)
        dsrc.append(dprime(Ssrc[conf_idx]))
        for q, i in enumerate(boots):
            sel = conf_idx[i]
            for k, v in measures(sel).items():
                BS[k][j, q] = v
            BSsrc[j, q] = dprime(Ssrc[sel])
    key = f'{c1}-{c2}'
    src_ci = nci95(np.nanmedian(BSsrc, 0))
    reliable = bool((src_ci[0] > 0 or src_ci[1] < 0) and abs(np.median(dsrc)) >= 0.2)  # 0.2 = the PREREG 'near 0' bound
    e2bs[key] = {k: np.nanmedian(v, 0) for k, v in BS.items()}
    res = {'source_dprime_median': float(np.median(dsrc)), 'source_dprime_ci95': src_ci, 'source_reliable': reliable}
    for (kind, m), v in M.items():
        res[f'{kind}_{m}_retention_median'] = float(np.nanmedian(v)) if reliable else None
        res[f'{kind}_{m}_retention_ci95'] = nci95(e2bs[key][(kind, m)]) if reliable else None
    res['per_layer'] = {str(l): {f'{kind}_{m}': float(M[(kind, m)][j]) for kind, m in M} | {'source_dprime': float(dsrc[j])}
                        for j, l in enumerate(tk_layers)}
    e2[key] = res
    fmt = lambda x: 'undef' if x is None else round(x, 3)  # noqa: E731
    print('E2', key, 'dist shift', fmt(res['distance_shift_retention_median']), 'sem shift', fmt(res['semantic_shift_retention_median']),
          '| dist dprime', fmt(res['distance_dprime_retention_median']), 'sem dprime', fmt(res['semantic_dprime_retention_median']),
          "| src d'", round(res['source_dprime_median'], 2), src_ci, flush=True)
words = ['U_hospital-U_school', 'U_yesterday-U_tomorrow', 'U_ming-U_hong', 'U_pleasant-U_painful']
srcs = ['A-U', 'A_p0-U_p0']
verd = {}
for m in ('shift', 'dprime'):
    for kind in ('distance', 'semantic'):
        verd[f'{m}_self_retention_at_least_0.8_{kind}'] = all(e2[p]['source_reliable'] and e2[p][f'{kind}_{m}_retention_ci95'][0] >= 0.8
                                                             for p in srcs)
        verd[f'{m}_words_below_self_{kind}'] = all(
            (not e2[w]['source_reliable']) or nci95(e2bs[s][(kind, m)] - e2bs[w][(kind, m)])[0] > 0 for s in srcs for w in words)
verd['null_dprime_near_0'] = abs(e2['U_p0-U_p0alt']['source_dprime_median']) < 0.2
verd['primary_measure'] = 'shift'
e2['verdict'] = verd
out['E2'] = e2
print('E2 verdict', verd, flush=True)

# ---- E3 ----
e3 = {}
RAT = {'A/U': ('A', 'U'), 'U/S': ('U', 'S'), 'A_p0/U_p0': ('A_p0', 'U_p0'), 'A_switch/U': ('A_switch', 'U')}
for l in (47, 55, 63):
    X = {c: np.asarray(pooled[:, ci[c], l], np.float32) for c in ('A', 'U', 'S', 'A_switch', 'A_p0', 'U_p0')}
    dist = {c: ((x - x[conf].mean(0)) ** 2).sum(1) for c, x in X.items()}
    row = {}
    for k, (p, q) in RAT.items():
        v = np.sqrt(dist[p][conf].mean() / dist[q][conf].mean())
        bs = [np.sqrt(dist[p][conf_idx[i]].mean() / dist[q][conf_idx[i]].mean()) for i in boots]
        row[k] = {'value': float(v), 'ci95': ci95(bs)}
    tsp = {}
    for c in ('A', 'U', 'S'):
        acc, n = np.zeros(d), 0
        for b in conf_idx:
            h = tk(l, c, b)
            acc += h.sum(0)
            n += len(h)
        cen = acc / n
        tsp[c] = np.sqrt(sum(((tk(l, c, b) - cen) ** 2).sum() for b in conf_idx) / n)
    row['token A/U'] = float(tsp['A'] / tsp['U'])
    row['token U/S'] = float(tsp['U'] / tsp['S'])
    e3[f'L{l}'] = row
    print('E3', l, {k: round(v['value'], 3) for k, v in row.items() if isinstance(v, dict)}, 'token A/U', round(row['token A/U'], 3),
          'token U/S', round(row['token U/S'], 3), flush=True)
e3['verdict'] = {'own_spread_above_1.2': all(e3[f'L{l}']['A/U']['ci95'][0] > 1.2 for l in (47, 55, 63)),
                 'user_system_within_0.9_1.1': all(0.9 <= e3[f'L{l}']['U/S']['ci95'][0] and e3[f'L{l}']['U/S']['ci95'][1] <= 1.1
                                                   for l in (47, 55, 63))}
out['E3'] = e3
print('E3 verdict', e3['verdict'], flush=True)
OUTJ.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('written', OUTJ)
