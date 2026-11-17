"""Group analysis of the two ordinary-word readouts (11-13, user: "重点在怎么分析"). One point per model, groups as in the paper
(neither 0, self-index only 1, self-reference 2; natural order: groups of the paper appendix, see build_final_tables_20261113.py).
Decomposition into a self-specific and a role-general part:
  retention (shares kept, differences):  self - words = (self - U-S) + (U-S - words)
  next-token KL (content KL, log10 ratios):  log(self / words) = log(self / U||S) + log(U||S / words)
  with self = min(A||U, A||S) (A||U alone where the order has no System message), words = median over the 32 pairs.
Tests, one-sided, permutation of the group labels (20,000): Spearman with the group, Jonckheere-Terpstra, self-reference above the rest.
Models of one developer are not independent draws; the tests describe the ordering, they are not a claim of independence.
usage: group_decomposition_20261113.py FINAL_TABLES_JSON OUT_PREFIX"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

EV = Path(__file__).resolve().parents[3]
ORDER = {'neither': 0, 'self-index only': 1, 'self-reference': 2}
NP = 20000


def rank(a):
    a = np.asarray(a, float)
    r = np.empty(len(a))
    r[a.argsort(kind='mergesort')] = np.arange(len(a))
    for v in np.unique(a):
        m = a == v
        r[m] = r[m].mean()
    return r


def sp(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def jt(v, g):
    s = 0.0
    for i, j in itertools.combinations(range(3), 2):
        a, b = v[g == i], v[g == j]
        s += (b[None, :] > a[:, None]).sum() + 0.5 * (b[None, :] == a[:, None]).sum()
    return float(s)


def tests(v, g, rng):
    v, g = np.asarray(v, float), np.asarray(g)
    ok = ~np.isnan(v)
    v, g = v[ok], g[ok]
    rho, J = sp(g, v), jt(v, g)
    rk = rank(v)
    top = rk[g == 2].mean()
    n_rho = n_j = n_top = 0
    for _ in range(NP):
        pg = rng.permutation(g)
        n_rho += sp(pg, v) >= rho - 1e-12
        n_j += jt(v, pg) >= J - 1e-12
        n_top += rk[pg == 2].mean() >= top - 1e-12
    return {'n': int(ok.sum()), 'medians': {k: (float(np.median(v[g == i])) if (g == i).any() else None) for k, i in ORDER.items()},
            'counts': {k: int((g == i).sum()) for k, i in ORDER.items()},
            'spearman_group': rho, 'p_spearman': n_rho / NP, 'p_jonckheere': n_j / NP, 'p_selfref_above_rest': n_top / NP}


def kl_parts(src):
    s = json.loads((EV / src / 'summary.json').read_text(encoding='utf-8'))['kinds']['kl_content']
    w_au = s['word_median_to_AU_12']['ratio']
    tu = s.get('to_US') or {}
    if 'A-U' not in tu:
        return {'log_self_words': float(np.log10(1 / w_au)), 'log_self_US': np.nan, 'log_US_words': np.nan, 'self_pair': 'A-U'}
    au, as_ = tu['A-U']['ratio'], tu['A-S']['ratio']
    w_us = s['word_median_to_US_12']['ratio']
    self_us = min(au, as_)
    return {'log_self_words': float(np.log10(self_us / w_us)), 'log_self_US': float(np.log10(self_us)),
            'log_US_words': float(np.log10(1 / w_us)), 'self_pair': 'A-U' if au <= as_ else 'A-S'}


def analyse(D):
    """The tests on one final-tables dict (11-15: a function, so that threshold_grid_20261115.py can run them on other groups)."""
    rng = np.random.default_rng(20261113)
    out, per_model = {}, {}
    for order in ('main', 'natural'):
        rows = [r for r in D['rows'][order] if r['group'] in ORDER]
        g = [ORDER[r['group']] for r in rows]
        ret = {'self - words': [r['retention']['adv_words']['value'] for r in rows],
               'self - U-S (self-specific)': [r['retention']['adv_US']['value'] if r['retention']['adv_US'] else np.nan for r in rows],
               'U-S - words (role-general)': [(r['retention']['US'] - r['retention']['word_med']) if r['retention']['US'] is not None
                                              else np.nan for r in rows]}
        kp = [kl_parts(r['kl']['src']) for r in rows]
        kl = {'log10 self / words': [k['log_self_words'] for k in kp],
              'log10 self / U||S (self-specific)': [k['log_self_US'] for k in kp],
              'log10 U||S / words (role-general)': [k['log_US_words'] for k in kp]}
        out[order] = {'retention': {k: tests(v, g, rng) for k, v in ret.items()}, 'kl': {k: tests(v, g, rng) for k, v in kl.items()}}
        per_model[order] = [{'name': r['name'], 'group': r['group'], **{k: v[i] for k, v in ret.items()}, **{k: v[i] for k, v in kl.items()},
                             'kl_self_pair': kp[i]['self_pair']} for i, r in enumerate(rows)]
    return out, per_model


def main():
    D = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    out, per_model = analyse(D)
    Path(sys.argv[2] + '.json').write_text(json.dumps({'tests': out, 'per_model': per_model}, ensure_ascii=False, indent=1,
                                                      default=lambda x: None if isinstance(x, float) and np.isnan(x) else x), encoding='utf-8')
    L = ['# Group decomposition (11-13)', '']
    for order in ('main', 'natural'):
        for kind in ('retention', 'kl'):
            L += [f'## {order}, {kind}', '', '| quantity | n | neither | self-index only | self-reference | Spearman with group (p) | '
                  'Jonckheere p | self-reference above the rest, p |', '|---|---|---|---|---|---|---|---|']
            for k, t in out[order][kind].items():
                m = t['medians']
                L.append(f"| {k} | {t['n']} | " + ' | '.join('-' if m[x] is None else f'{m[x]:+.2f}' for x in ORDER) +
                         f" | {t['spearman_group']:+.2f} ({t['p_spearman']:.4f}) | {t['p_jonckheere']:.4f} | {t['p_selfref_above_rest']:.4f} |")
            L.append('')
    L += ['## Per model, next-token KL (log10)', '', '| order | model | group | self / words | self / U||S | U||S / words | self pair |',
          '|---|---|---|---|---|---|---|']
    for order in ('main', 'natural'):
        for r in per_model[order]:
            f = lambda x: '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:+.2f}'  # noqa: E731
            L.append(f"| {order} | {r['name']} | {r['group']} | {f(r['log10 self / words'])} | {f(r['log10 self / U||S (self-specific)'])} | "
                     f"{f(r['log10 U||S / words (role-general)'])} | {r['kl_self_pair']} |")
    Path(sys.argv[2] + '.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
