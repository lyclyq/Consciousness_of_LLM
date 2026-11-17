"""Analysis of the expanded ordinary-word test (11-10, ../../PROTOCOL_zh.md, section 5). Local, numpy only, read-only on inputs.
Input: the extracts of extract_gram_20261110.py, mirrored locally, laid out as
  EXTRACTS/<set>/<order>/<tag>/<source>.{npz,json}   set: b150 (75 held-out bodies, 15 wording families) or p3 (200 phase-3
  held-out bodies, each its own family); order: main or natural; source: pilot (the 11-06 run) and/or one or more words_along runs.
The rows of all sources of one unit (set, order, tag) are merged by name; the sources must cover the same bodies and families
and no row may come twice.
Per row: the common part C of the mean difference over the first and the last third of the body tokens (family U-statistic,
analyze_persistence_20261106.ustats), share kept = sqrt(C_last / C_first), median over the kept layers; initial size =
sqrt(C_first), median over the kept layers; energy = mean over bodies of the mean squared difference in the first third,
median over the kept layers. 95% intervals: 2,000 resamples of families (b150) or bodies (p3), the same resamples for every row
of a unit (paired).
Self = the smaller of the shares of A-U and A-S (A-U alone where the order has no System message).
Advantage: self - median of the word shares, self - largest word share, self - U-S.
Magnitude check (descriptive, written before the data): (a) initial size and energy of every row; (b) Spearman correlation of
initial size and share kept over the word pairs of a unit; (c) word fit share ~ a + b log(initial size) over the word pairs,
the self share minus the fit at the self pairs' initial size (with a flag when that size lies outside the words' range), and
across units the share kept of A-U and A-S against their initial size, separately for the index and the no-index group.
Across models: Spearman correlation of the advantage with the content-index coverage of the paper (main: Table tab:self_kl;
natural: tab:self_natural, tab:self_natural3), permutation p (10,000).
usage: analyze_words_along_20261110.py EXTRACTS OUT_PREFIX --words items/words_v2.json --coverage items/content_index_coverage.json"""
import argparse
import json
from pathlib import Path

import numpy as np

NB = 2000
SEED = {'b150': 20261106, 'p3': 20261107}


def q(G, c):
    """U-statistic of the mean of family means: G (K, F, F), c (R, F) counts -> (R, K)"""
    c = c.astype(np.float64)
    den = c.sum(1) ** 2 - (c ** 2).sum(1)
    full = np.einsum('rf,kfg,rg->rk', c, G, c)
    diag = np.einsum('rf,kff->rk', c ** 2, G)
    return (full - diag) / den[:, None]


def rank(a):
    a = np.asarray(a, float)
    r = np.empty(len(a))
    r[a.argsort(kind='mergesort')] = np.arange(len(a))
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float('nan')
    return float(np.corrcoef(rank(x[ok]), rank(y[ok]))[0, 1])


def perm_p(x, y, n=10000, seed=20261110):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return float('nan')
    r0 = abs(spearman(x, y))
    rng = np.random.default_rng(seed)
    hits = sum(abs(spearman(x, rng.permutation(y))) >= r0 - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [None, None]


def load_unit(d):
    """merge the sources of one unit; returns rows -> dict(G_FF, G_LL, eF), meta"""
    srcs = sorted(p.stem for p in d.glob('*.json'))
    rows, meta0 = {}, None
    for s in srcs:
        meta = json.loads((d / f'{s}.json').read_text(encoding='utf-8'))
        z = np.load(d / f'{s}.npz')
        if meta0 is None:
            meta0 = meta
        else:
            assert meta['ids'] == meta0['ids'] and meta['families'] == meta0['families'] and meta['counts'] == meta0['counts'], \
                ('sources cover different bodies', d, s)
            assert meta['keep_layers'] == meta0['keep_layers'], ('kept layers differ', d, s)
        for r, name in enumerate(meta['rows']):
            if not meta['rows_finite'][r]:
                continue
            assert name not in rows, ('row from two sources', d, name)
            rows[name] = {'G_FF': z['G_FF'][r].astype(np.float64), 'G_LL': z['G_LL'][r].astype(np.float64),
                          'eF': z['eF'][:, r, :], 'source': s, 'unequal_bodies': meta.get('unequal_pairs', {}).get(name, 0)}
    return rows, meta0, srcs


def unit_stats(rows, meta, set_name, word_ids, word_cat):
    F = len(meta['families'])
    rng = np.random.default_rng(SEED[set_name])
    counts = np.concatenate([np.ones((1, F), int), np.stack([np.bincount(rng.integers(0, F, F), minlength=F) for _ in range(NB)])])
    S, INIT, E = {}, {}, {}
    for name, r in rows.items():
        CF, CL = q(r['G_FF'], counts), q(r['G_LL'], counts)
        with np.errstate(invalid='ignore', divide='ignore'):
            size = np.sqrt(CL / CF)
            init = np.sqrt(np.where(CF > 0, CF, np.nan))
        S[name] = np.nanmedian(size, -1)            # (NB+1,): point, then resamples
        INIT[name] = np.nanmedian(init, -1)
        E[name] = float(np.median(r['eF'].mean(0)))
    words = [w for w in word_ids if w in S]
    out = {'n_bodies': len(meta['ids']), 'n_families': F, 'rows': {}, 'n_words': len(words)}
    for name in S:
        out['rows'][name] = {'share': float(S[name][0]), 'share_ci': ci(S[name][1:]), 'init_size': float(INIT[name][0]),
                             'init_size_ci': ci(INIT[name][1:]), 'energy_first': E[name], 'source': rows[name]['source'],
                             'category': word_cat.get(name), 'token_count_differs_in_bodies': rows[name]['unequal_bodies']}
    selfp = [p for p in ('A-U', 'A-S') if p in S]
    if not selfp or not words:
        out['self_pairs'] = selfp
        return out, S, INIT
    Wm = np.stack([S[w] for w in words])           # (W, NB+1)
    Wi = np.stack([INIT[w] for w in words])
    self_s = np.min(np.stack([S[p] for p in selfp]), 0)
    adv_med = self_s - np.median(Wm, 0)
    adv_max = self_s - np.max(Wm, 0)
    out['self_pairs'] = selfp
    out['self'] = {'share': float(self_s[0]), 'share_ci': ci(self_s[1:])}
    out['advantage_vs_word_median'] = {'value': float(adv_med[0]), 'ci': ci(adv_med[1:])}
    out['advantage_vs_word_max'] = {'value': float(adv_max[0]), 'ci': ci(adv_max[1:])}
    out['words_below_self'] = f'{int((Wm[:, 0] < self_s[0]).sum())}/{len(words)}'
    # like-for-like across units: the first pair of each category (the 8 pairs of GLM-5.3-Flash's stage 1)
    first8 = [w for w in words if w.endswith('1')]
    if first8:
        a8 = self_s - np.median(np.stack([S[w] for w in first8]), 0)
        out['advantage_vs_first8_median'] = {'value': float(a8[0]), 'ci': ci(a8[1:]), 'n_words': len(first8)}
    out['word_share_quantiles'] = [float(x) for x in np.quantile(Wm[:, 0], [0, 0.25, 0.5, 0.75, 1])]
    if 'U-S' in S:
        a = self_s - S['U-S']
        out['advantage_vs_US'] = {'value': float(a[0]), 'ci': ci(a[1:])}
    cats = {}
    for w in words:
        cats.setdefault(word_cat[w], []).append(float(S[w][0]))
    out['categories'] = {c: {'median': float(np.median(v)), 'min': float(min(v)), 'max': float(max(v)), 'n': len(v)} for c, v in cats.items()}
    # (b) initial size vs share kept over the word pairs
    rho = np.array([spearman(Wi[:, j], Wm[:, j]) for j in range(Wm.shape[1])])
    out['magnitude_words_spearman'] = {'value': float(rho[0]), 'ci': ci(rho[1:])}
    # (c, within unit) word fit share ~ a + b log(init); self minus the fit at the self pairs' initial size
    resid = {}
    for p in selfp:
        res = np.full(Wm.shape[1], np.nan)
        for j in range(Wm.shape[1]):
            x, y = np.log(Wi[:, j]), Wm[:, j]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 3 and np.isfinite(INIT[p][j]) and INIT[p][j] > 0:
                b, a = np.polyfit(x[ok], y[ok], 1)
                res[j] = S[p][j] - (a + b * np.log(INIT[p][j]))
        resid[p] = {'value': float(res[0]), 'ci': ci(res[1:]),
                    'init_outside_word_range': bool(not (Wi[:, 0].min() <= INIT[p][0] <= Wi[:, 0].max()))}
    out['self_minus_word_fit'] = resid
    return out, S, INIT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('extracts')
    ap.add_argument('out')
    ap.add_argument('--words', required=True)
    ap.add_argument('--coverage', required=True)
    a = ap.parse_args()
    Wd = json.loads(Path(a.words).read_text(encoding='utf-8'))
    word_ids = [p['id'] for p in Wd['pairs']]
    word_cat = {p['id']: p['category'] for p in Wd['pairs']}
    cov = json.loads(Path(a.coverage).read_text(encoding='utf-8'))
    root = Path(a.extracts)
    res = {}
    for d in sorted(root.glob('*/*/*')):
        if not d.is_dir() or not list(d.glob('*.json')):
            continue
        set_name, order, tag = d.parts[-3], d.parts[-2], d.parts[-1]
        rows, meta, srcs = load_unit(d)
        st, _, _ = unit_stats(rows, meta, set_name, word_ids, word_cat)
        st['sources'] = srcs
        grp = cov['main'].get(tag, {}).get('group')
        st['group'] = grp
        c = cov['main' if order == 'main' else 'natural'].get(tag)
        st['coverage'] = (c['layers'] / c['of']) if c else None
        res.setdefault(set_name, {}).setdefault(order, {})[tag] = st
    # across units: relation to coverage, and (c) share vs initial size of the self pairs by group
    across = {}
    for set_name, by in res.items():
        for order, units in by.items():
            tags = sorted(units)
            adv = [units[t].get('advantage_vs_word_median', {}).get('value', np.nan) for t in tags]
            covv = [units[t]['coverage'] if units[t]['coverage'] is not None else np.nan for t in tags]
            adv8 = [units[t].get('advantage_vs_first8_median', {}).get('value', np.nan) for t in tags]
            A = {'n_units': len(tags), 'n_words_per_unit': {t: units[t]['n_words'] for t in tags},
                 'spearman_advantage_coverage': spearman(covv, adv), 'perm_p': perm_p(covv, adv),
                 'spearman_advantage_first8_coverage': spearman(covv, adv8), 'perm_p_first8': perm_p(covv, adv8)}
            for g in ('index', 'no_index'):
                for p in ('A-U', 'A-S'):
                    xs = [units[t]['rows'][p]['init_size'] for t in tags if units[t]['group'] == g and p in units[t]['rows']]
                    ys = [units[t]['rows'][p]['share'] for t in tags if units[t]['group'] == g and p in units[t]['rows']]
                    A[f'{g}_{p}_share_vs_init'] = {'n': len(xs), 'spearman': spearman(xs, ys),
                                                   'points': [[t, units[t]['rows'][p]['init_size'], units[t]['rows'][p]['share']]
                                                              for t in tags if units[t]['group'] == g and p in units[t]['rows']]}
            across[f'{set_name}/{order}'] = A
    full = {'units': res, 'across': across, 'resamples': NB, 'seeds': SEED}
    Path(a.out + '.json').write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding='utf-8')
    L = ['| set | order | setting | group | cover | self | U-S | words (min/med/max) | self - word median | self - word max | words < self | rho(init, share) words | self - word fit (A-U) |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    f2 = lambda x: '-' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.2f}'  # noqa: E731
    for set_name, by in sorted(res.items()):
        for order, units in sorted(by.items()):
            for t, u in sorted(units.items(), key=lambda kv: (kv[1]['group'] or '', kv[0])):
                if 'self' not in u:
                    continue
                wq = u['word_share_quantiles']
                am, ax = u['advantage_vs_word_median'], u['advantage_vs_word_max']
                us = u['rows'].get('U-S', {}).get('share')
                fit = u['self_minus_word_fit'].get('A-U', {})
                L.append(f"| {set_name} | {order} | {t} | {u['group']} | {f2(u['coverage'])} | {f2(u['self']['share'])} | {f2(us)} | "
                         f"{f2(wq[0])}/{f2(wq[2])}/{f2(wq[4])} | {f2(am['value'])} [{f2(am['ci'][0])}, {f2(am['ci'][1])}] | "
                         f"{f2(ax['value'])} [{f2(ax['ci'][0])}, {f2(ax['ci'][1])}] | {u['words_below_self']} | "
                         f"{f2(u['magnitude_words_spearman']['value'])} | {f2(fit.get('value'))}{' (outside)' if fit.get('init_outside_word_range') else ''} |")
    L.append('')
    for k, A in across.items():
        L.append(f"- {k}: Spearman(advantage, coverage) = {f2(A['spearman_advantage_coverage'])} (perm p = {f2(A['perm_p'])}, {A['n_units']} units), "
                 f"first pair of each category only: {f2(A['spearman_advantage_first8_coverage'])} (perm p = {f2(A['perm_p_first8'])}); "
                 + '; '.join(f"{g} {p}: rho(share, init) = {f2(A[f'{g}_{p}_share_vs_init']['spearman'])} (n {A[f'{g}_{p}_share_vs_init']['n']})"
                             for g in ('index', 'no_index') for p in ('A-U', 'A-S')))
    Path(a.out + '.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
