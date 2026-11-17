"""Is there a statistical boundary in the next-token (KL) separation? (user 11-10: "整体确实有一个统计学上的界限"). Reads the per-model
values of kl_ratio_distribution_20261110.py (re-derived here from the summaries), no threshold set in advance:
  1. Spearman rho between the content-index share and the smaller content ratio, with a permutation p (100,000 shuffles, seed 1);
  2. one- vs two-component Gaussian mixture on log10(smaller ratio) (EM, 50 random starts, seed 1), BIC of each, and the boundary of
     the two-component fit (the point between the means where the posterior is 0.5);
  3. which models fall on each side, against their content-index share.
Main probe (25 models) and natural order (13 models). Writes results/kl_boundary_stats_20261110.md.
usage: kl_boundary_stats_20261110.py [ARCHIVE]"""
import importlib.util
import math
import random
import sys
from pathlib import Path

import numpy as np

A = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('d', Path(__file__).with_name('kl_ratio_distribution_20261110.py'))
D = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0], str(A)]
spec.loader.exec_module(D)  # builds D.rows (main) and D.nat (natural)


def rank(v):
    o = np.argsort(v, kind='mergesort')
    r = np.empty(len(v))
    r[o] = np.arange(1, len(v) + 1)
    # average ties
    for x in set(v):
        idx = [i for i, y in enumerate(v) if y == x]
        if len(idx) > 1:
            r[idx] = np.mean(r[idx])
    return r


def spearman(x, y):
    a, b = rank(list(x)), rank(list(y))
    return float(np.corrcoef(a, b)[0, 1])


def perm_p(x, y, n=100000, seed=1):
    rng = np.random.default_rng(seed)
    r0 = spearman(x, y)
    a, b = rank(list(x)), rank(list(y))
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    cnt = 0
    for _ in range(n):
        if np.mean(a * rng.permutation(b)) >= r0 - 1e-12:
            cnt += 1
    return r0, (cnt + 1) / (n + 1)


def gmm(z, k, starts=50, seed=1):
    rng = np.random.default_rng(seed)
    best = None
    n = len(z)
    for _ in range(starts):
        mu = rng.choice(z, k, replace=False).astype(float)
        sd = np.full(k, z.std() + 1e-3)
        w = np.full(k, 1 / k)
        for _ in range(500):
            dens = np.array([w[j] * np.exp(-0.5 * ((z - mu[j]) / sd[j]) ** 2) / (sd[j] * math.sqrt(2 * math.pi)) for j in range(k)])
            tot = dens.sum(0) + 1e-300
            g = dens / tot
            nk = g.sum(1)
            if np.any(nk < 1e-6):
                break
            w = nk / n
            mu = (g * z).sum(1) / nk
            sd = np.sqrt((g * (z - mu[:, None]) ** 2).sum(1) / nk) + 1e-3
        ll = float(np.log(tot).sum())
        if best is None or ll > best[0]:
            best = (ll, mu.copy(), sd.copy(), w.copy())
    ll, mu, sd, w = best
    bic = -2 * ll + (3 * k - 1) * math.log(n)
    return ll, bic, mu, sd, w


def boundary(mu, sd, w):
    lo, hi = sorted(range(2), key=lambda j: mu[j])
    xs = np.linspace(mu[lo], mu[hi], 20001)
    p = lambda j: w[j] * np.exp(-0.5 * ((xs - mu[j]) / sd[j]) ** 2) / sd[j]  # noqa: E731
    d = p(hi) / (p(lo) + p(hi))
    return float(xs[np.argmin(abs(d - 0.5))])


L = ['# KL 比值的统计边界（11-10）', '', '没有事先设阈值。较小比值 = min(A–U/U–S, A–S/U–S)，内容；z = log10(较小比值)。', '']
for label, items in (('主探针，25 个模型', [(x['name'] + (' (四流)' if x['hc'] else ''), x['cov'], x['mn']) for x in D.rows]),
                     ('自然顺序，13 个模型', [(n, int(c.split('/')[0]) / int(c.split('/')[1]), mn) for mn, n, c, _ in D.nat])):
    names = [a for a, _, _ in items]
    cov = np.array([b for _, b, _ in items])
    z = np.log10(np.array([c for _, _, c in items]))
    r, p = perm_p(cov, z)
    l1, b1, *_ = gmm(z, 1)
    l2, b2, mu, sd, w = gmm(z, 2)
    bd = boundary(mu, sd, w)
    hi = [(n, c, 10 ** v) for n, c, v in zip(names, cov, z) if v > bd]
    lo = [(n, c, 10 ** v) for n, c, v in zip(names, cov, z) if v <= bd]
    L += [f'## {label}', '',
          f'- Spearman(正文位置索引覆盖, 较小比值) = {r:+.2f}，置换 p = {p:.1e}（单侧，100,000 次）。',
          f'- 单峰 BIC {b1:.1f}，双峰 BIC {b2:.1f}（ΔBIC = {b1 - b2:+.1f}，正数表示双峰更好）；双峰均值 {10 ** min(mu):.2f} 与 {10 ** max(mu):.2f}，'
          f'边界（后验 0.5）在比值 {10 ** bd:.2f}。',
          f'- 边界以上 {len(hi)} 个：' + '；'.join(f'{n} {c:.0%}→{v:.2f}' for n, c, v in sorted(hi, key=lambda x: x[2])),
          f'- 边界以下 {len(lo)} 个：' + '；'.join(f'{n} {c:.0%}→{v:.2f}' for n, c, v in sorted(lo, key=lambda x: x[2])),
          f'- 覆盖：边界以上 {min(c for _, c, _ in hi):.0%}–{max(c for _, c, _ in hi):.0%}，边界以下 {min(c for _, c, _ in lo):.0%}–{max(c for _, c, _ in lo):.0%}。', '']
out = A / 'results' / 'kl_boundary_stats_20261110.md'
out.write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')
print('written', out)
