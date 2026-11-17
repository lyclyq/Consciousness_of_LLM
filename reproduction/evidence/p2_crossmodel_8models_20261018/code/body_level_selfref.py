"""Body-level self-reference from crossmodel_suite G records: per body, P(you) averaged over the three option orders."""
import glob, json, sys
from math import comb
import numpy as np


def sign_p(k, n):
    t = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * t)


for arg in sys.argv[1:]:
    name, d = arg.split('=')
    recs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(d + '/*.json'))]
    P = {r: np.array([x['roles'][r]['mean']['self'] for x in recs]) for r in ('assistant', 'user', 'system')}
    diff = P['assistant'] - np.maximum(P['user'], P['system'])
    k, n = int((diff > 0).sum()), len(diff)
    top = {r: np.mean([max(x['roles'][r]['mean'], key=x['roles'][r]['mean'].get) == t for x in recs]) for r, t in (('assistant', 'self'), ('user', 'user'), ('system', 'system'))}
    print(f'{name}: n={n} P(you|A)={P["assistant"].mean():.3f} P(you|U)={P["user"].mean():.3f} P(you|S)={P["system"].mean():.3f} '
          f'A-max(U,S) {diff.mean():+.3f}, positive {k}/{n}, sign p={sign_p(k, n):.2g} | top answer correct A {top["assistant"]:.2f} U {top["user"]:.2f} S {top["system"]:.2f}')
