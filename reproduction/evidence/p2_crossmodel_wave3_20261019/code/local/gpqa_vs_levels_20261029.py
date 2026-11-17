"""User question 10-29: do the two levels go with published capability (GPQA) at all, weakly? Rank correlations over the 14 models
whose developers publish a GPQA score (stats/v5_benchmarks_20261029.csv; EXAONE-4.0.1-32B: reasoning mode, as in the paper), against
  group   : ordinal group of Section 4.4 (neither 0 < self-index only 1 < self-reference after thinking 2 < both levels 3)
  selfref : recognizes its own messages in some mode (0/1)
  index   : self-index share of layers (Figure self_crossmodel; non-thinking rendering where the model has one, else thinking)
  claim   : self-reference in its strongest mode (Figure self_crossmodel points: own minus the larger external, either mode)
Spearman rho with a two-sided permutation p (100,000 permutations, seed 0) and a 95% bootstrap interval over models (10,000, seed 0);
for selfref also the Mann-Whitney comparison of GPQA between models with and without self-reference.
usage: gpqa_vs_levels_20261029.py STATS_DIR"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

S = Path(sys.argv[1])
KEY = {'Qwen3.8-27B': 'qwen38_27b', 'Qwen3.8-Flash-Next': 'qwen38_flashnext', 'gpt-oss-120b': 'gpt_oss_120b', 'gpt-oss-20b': 'gpt_oss_20b',
       'EXAONE-4.0.1-32B': 'exaone4_32b', 'GLM-4.7-Flash': 'glm47_flash', 'Granite-4.2-30B': 'granite42_30b',
       'DeepSeek-R1-Distill-Llama-70B': 'r1_llama70b', 'Granite-4.1-30B': 'granite41_30b', 'Qwen3.6-27B': 'qwen36_27b',
       'Qwen3.6-35B-A3B': 'qwen36_35b', 'Qwen3.5-27B': 'qwen35_27b', 'Gemma-4-31B': 'gemma4_31b', 'Seed-OSS-36B': 'seed_oss_36b'}
GROUP = {'neither level': 0, 'self-index without self-reference': 1, 'self-reference after reasoning': 2, 'both levels': 3}
P = json.load(open(S / 'self_crossmodel_points_v6.json'))
rows = {}
for r in csv.DictReader(open(S / 'v5_benchmarks_20261029.csv', encoding='utf-8')):
    if r['model'] in KEY and r['value'] and r['model'] not in rows:  # first row per model (EXAONE: reasoning mode)
        rows[r['model']] = r
names = sorted(rows)
g = np.array([float(rows[m]['value']) for m in names])
k = [KEY[m] for m in names]
grp = np.array([GROUP[rows[m]['group']] for m in names], float)
sr = (grp >= 2).astype(float)
idx = np.array([(P.get(f'{x}/non_thinking') or P[f'{x}/thinking'])[0] for x in k])
clm = np.array([max(P[f'{x}/{md}'][1] for md in ('non_thinking', 'thinking') if f'{x}/{md}' in P) for x in k])
rng = np.random.default_rng(0)


def rho(a, b):
    return stats.spearmanr(a, b)[0]


print(f'n = {len(names)} models with a published GPQA score')
OUT = {'n': len(names), 'models': {m: {'gpqa': float(x), 'group': int(y), 'index': float(z), 'claim': float(w)}
                                   for m, x, y, z, w in zip(names, g, grp, idx, clm)}, 'rank': {}}
for nm, v in (('group', grp), ('selfref', sr), ('index', idx), ('claim', clm)):
    r0 = rho(g, v)
    perm = np.array([rho(g, rng.permutation(v)) for _ in range(100_000)])
    p = float((np.abs(perm) >= abs(r0) - 1e-12).mean())
    bs = []
    for _ in range(10_000):
        i = rng.integers(0, len(g), len(g))
        if len(set(v[i])) > 1 and len(set(g[i])) > 1:
            bs.append(rho(g[i], v[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f'{nm:8s} Spearman rho = {r0:+.2f}  permutation p = {p:.2f}  95% bootstrap [{lo:+.2f}, {hi:+.2f}]')
    OUT['rank'][nm] = {'rho': float(r0), 'p': p, 'ci95': [float(lo), float(hi)]}
u = stats.mannwhitneyu(g[sr == 1], g[sr == 0], alternative='two-sided')
OUT['mannwhitney'] = {'median_with': float(np.median(g[sr == 1])), 'n_with': int(sr.sum()), 'median_without': float(np.median(g[sr == 0])),
                      'n_without': int((1 - sr).sum()), 'p': float(u.pvalue)}
json.dump(OUT, open(S / 'gpqa_vs_levels_20261029.json', 'w'), indent=1)
print(f'GPQA with self-reference: median {np.median(g[sr == 1]):.1f} (n={int(sr.sum())}); without: median {np.median(g[sr == 0]):.1f} '
      f'(n={int((1 - sr).sum())}); Mann-Whitney p = {u.pvalue:.2f}')
for m, x, y, z, w in sorted(zip(names, g, grp, idx, clm), key=lambda t: -t[1]):
    print(f'  {m:32s} GPQA {x:5.1f}  group {int(y)}  index {z:.2f}  claim {w:+.2f}')
