"""Rank test of panel c of the evidence figure (11-14): do the pooled values rise with the level of the model in both orders?
The 14 models measured in both orders are ordered by level exactly as in the figure (pooled_rows of
fig_evidence_combined_20261114.py): 0 neither level in either order, 1 a self-index in one order, 2 a self-index in both orders,
3 self-reference in one order with the index in both, 4 self-reference in both orders. Statistic: Spearman rho between the level
and the pooled value of each column (b KL self / external, c KL self / median word pair, both in log10, d share kept self -
external, e share kept self - median word pair), ties at mid-rank, one-sided permutation test with 20,000 permutations of the
levels (seed 20261114). Written so that the values quoted in the README of 11-14 and in the paper can be reproduced.
usage (from projects/consciousness): uv run --offline --no-project --with numpy --with matplotlib python -I pooled_rank_test_20261114.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = [sys.argv[0]]
import fig_evidence_combined_20261114 as F  # noqa: E402

LEVEL = {'neither': 0, 'index-one': 1, 'self-index only': 2, 'selfref-one': 3, 'self-reference': 4}


def rank(a):
    a = np.asarray(a, float)
    o = a.argsort()
    r = np.empty(len(a))
    r[o] = np.arange(len(a))
    for v in np.unique(a):
        m = a == v
        r[m] = r[m].mean()
    return r


def rho(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def run(rows):
    """The rank test on pooled rows (11-15: a function, so that threshold_grid_20261115.py can run it on other groups)."""
    lv = np.array([LEVEL[r['group']] for r in rows], float)
    rng = np.random.default_rng(20261114)
    res = {'n_models': len(rows), 'levels_present': sorted({r['group'] for r in rows}, key=LEVEL.get),
           'models': [{'tag': r['tag'], 'level': r['group']} for r in rows], 'columns': {}}
    for k in 'bcde':
        v = np.array([np.log10(r[k][0]) if k in 'bc' else r[k][0] for r in rows], float)
        r0 = rho(lv, v)
        perm = np.array([rho(rng.permutation(lv), v) for _ in range(20000)])
        med = {g: float(np.median([(r[k][0]) for r in rows if r['group'] == g])) for g in res['levels_present']}
        res['columns'][k] = {'spearman': r0, 'p_one_sided': float((perm >= r0).mean()), 'median_by_level': med}
        print(k, f"rho {r0:+.3f}  p {res['columns'][k]['p_one_sided']:.4f}  medians", {g: round(m, 3) for g, m in med.items()})
    return res


def main():
    res = run(F.pooled_rows())
    out = Path('evidence/p2_words_along_20261110/results/final_20261113/pooled_rank_test_20261114.json')
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('levels', res['levels_present'], 'n', res['n_models'])


if __name__ == '__main__':
    main()
