"""95% intervals of the pooled (both orders, each weighted 1/2) values of block C of the evidence figure (11-14, user: the pooled
block needs its statistical confidence too). The two orders use the same 75 held-out bodies and the same 15 wording families, so
one family resample is drawn per replicate and applied to BOTH orders (joint family bootstrap, 2,000 replicates); the pooled value
of a replicate is the mean of the two orders (KL ratios: mean of the logs). Each order is computed exactly as in its own summary:
  next-token KL (kl_words_20261111.summarize): per row the sum over bodies; ratio of sums; words = median over the 32 pairs of the
    12 direction; self = the pair (A-U or A-S) with the smaller point ratio to U-S, or A-U where the order has no System message;
    b = self / U-S, c = self / median word (no System: A-U / median word);
  retention (analyze_words_along_20261110.unit_stats): family U-statistic of the first- and last-third Gram matrices, share kept
    = sqrt(C_last / C_first), median over kept layers; self = min(A-U, A-S); d = self - U-S, e = self - median word.
An order without a System message contributes nothing to b and d (the other order alone is used).
usage: pooled_ci_20261114.py OUT_JSON"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

EV = Path(__file__).resolve().parents[3]
WA = EV / 'p2_words_along_20261110'
EXT = WA / 'results/stage2_20261113/extracts/b150'
FT = json.loads((WA / 'results/final_20261113/final_tables_20261113.json').read_text(encoding='utf-8'))
WORDS = [p['id'] for p in json.loads((WA / 'items/words_v2.json').read_text(encoding='utf-8'))['pairs']]
spec = importlib.util.spec_from_file_location('awa', str(Path(__file__).resolve().parent / 'analyze_words_along_20261110.py'))
awa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(awa)
NB = 2000


def kl_order(src, counts):
    """(b, c) per replicate (row 0 = point estimate); b is None without a System message"""
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((EV / src / 'bodies').glob('*.json'))]
    fams = sorted({r['category'] for r in rows})
    fidx = np.array([fams.index(r['category']) for r in rows])
    fs = lambda v: np.array([v[fidx == f].sum() for f in range(len(fams))])  # noqa: E731
    tot = lambda v: counts @ fs(v)                                             # (NB+1,)  # noqa: E731
    R = {pn: np.array([r['roles'][pn]['kl_content'] for r in rows]) for pn in rows[0]['roles']}
    W = {w: np.array([r['words'][w]['kl_content_12'] for r in rows]) for w in WORDS if w in rows[0]['words']}
    if 'U-S' in R:
        us = tot(R['U-S'])
        pair = 'A-U' if R['A-U'].sum() <= R['A-S'].sum() else 'A-S'
        self_us = tot(R[pair]) / us
        wmed = np.median(np.stack([tot(v) / us for v in W.values()]), 0)
        return self_us, self_us / wmed, fams
    au = tot(R['A-U'])
    wmed = np.median(np.stack([tot(v) / au for v in W.values()]), 0)
    return None, 1 / wmed, fams


def ret_order(order, tag, counts):
    rows, meta, _ = awa.load_unit(EXT / order / tag)
    S = {}
    for name, r in rows.items():
        CF, CL = awa.q(r['G_FF'], counts), awa.q(r['G_LL'], counts)
        with np.errstate(invalid='ignore', divide='ignore'):
            S[name] = np.nanmedian(np.sqrt(CL / CF), -1)
    selfp = [p for p in ('A-U', 'A-S') if p in S]
    self_s = np.min(np.stack([S[p] for p in selfp]), 0)
    wmed = np.median(np.stack([S[w] for w in WORDS if w in S]), 0)
    d = self_s - S['U-S'] if 'U-S' in S else None
    return d, self_s - wmed, meta['families']


def main():
    out = sys.argv[1]
    rows = {o: {r['tag']: r for r in FT['rows'][o]} for o in ('main', 'natural')}
    both = [t for t in rows['main'] if t in rows['natural']]
    rng = np.random.default_rng(20261114)
    F = 15
    counts = np.concatenate([np.ones((1, F)), np.stack([np.bincount(rng.integers(0, F, F), minlength=F) for _ in range(NB)])])
    res = {}
    for t in both:
        per = {'b': [], 'c': [], 'd': [], 'e': []}
        fam_sets = []
        for o in ('main', 'natural'):
            b, c, fk = kl_order(rows[o][t]['kl']['src'], counts)
            d, e, fr = ret_order(o, t, counts)
            fam_sets += [tuple(fk), tuple(fr)]
            if b is not None:
                per['b'].append(np.log10(b))
            per['c'].append(np.log10(c))
            if d is not None:
                per['d'].append(d)
            per['e'].append(e)
        assert len(set(fam_sets)) == 1 and len(fam_sets[0]) == F, (t, 'orders do not share the 15 families')
        r = {}
        for k, v in per.items():
            m = np.mean(np.stack(v), 0)
            if k in 'bc':
                m = 10 ** m
            r[k] = {'value': float(m[0]), 'ci95': [float(np.quantile(m[1:], 0.025)), float(np.quantile(m[1:], 0.975))], 'n_orders': len(v)}
        res[t] = r
        print(f"{t:28s} " + '  '.join(f"{k} {r[k]['value']:.3g} [{r[k]['ci95'][0]:.3g}, {r[k]['ci95'][1]:.3g}]({r[k]['n_orders']})" for k in 'bcde'))
    # per order (blocks A and B): 95% interval of column c (self / median word pair), same family resamples, each order alone
    per_order_c = {}
    for o in ('main', 'natural'):
        for t, r in rows[o].items():
            if r['group'] in ('-',) or not r.get('kl'):
                continue
            _, c, _ = kl_order(r['kl']['src'], counts)
            per_order_c.setdefault(o, {})[t] = {'value': float(c[0]), 'ci95': [float(np.quantile(c[1:], 0.025)), float(np.quantile(c[1:], 0.975))]}
    Path(out).write_text(json.dumps({'method': __doc__.split('usage:')[0].strip(), 'replicates': NB, 'seed': 20261114, 'models': res,
                                     'per_order_c': per_order_c}, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
