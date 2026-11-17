"""User proposal 10-29: a GPQA correlation over both conventions, each model weighted equally and, where a model was measured both under
the main probe and in natural order, each measurement with half its weight. Realized by averaging the two values of such a model (the same
weight split, with the ordinary Spearman rho), the value of the one convention otherwise.
  main probe : stats/gpqa_vs_levels_20261029.json (14 models; as tab:self_benchmarks)
  natural    : self-index share, three-role models (stats/self_natural_points_v1.json) and Ministral-3-14B-Reasoning (0 of 40 layers at
               content and onset; natural order only, GPQA 71.2 from its model card, self-reference not measured);
               self-reference strength, three-role models (own minus the larger external, three option orders) and the two-role models of
               tab:self_natural2 (own minus User, in the mode of tab:self_crossmodel)
  group      : unchanged except Gemma-4-31B (neither level under the main probe, both levels in natural order: mean 1.5), as Appendix D states
Spearman rho, two-sided permutation p (100,000, seed 0), 95% bootstrap interval over models (10,000, seed 0).
usage: gpqa_vs_levels_pooled_20261029.py STATS_DIR"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

S = Path(sys.argv[1])
M = json.loads((S / 'gpqa_vs_levels_20261029.json').read_text(encoding='utf-8'))['models']
NP = json.loads((S / 'self_natural_points_v1.json').read_text(encoding='utf-8'))
nat_index = {'Seed-OSS-36B': NP['seed_oss_36b/natural'][0], 'Gemma-4-31B': NP['gemma4_31b/natural'][0],
             'EXAONE-4.0.1-32B': NP['exaone4_32b/natural'][0], 'Granite-4.1-30B': NP['granite41_30b/natural'][0]}
nat_claim = {'Seed-OSS-36B': NP['seed_oss_36b/natural'][1], 'Gemma-4-31B': NP['gemma4_31b/natural'][1],
             'EXAONE-4.0.1-32B': NP['exaone4_32b/natural'][1], 'Granite-4.1-30B': NP['granite41_30b/natural'][1],
             'Qwen3.8-27B': 0.76 - 0.02, 'gpt-oss-20b': 0.93 - 0.03, 'DeepSeek-R1-Distill-Llama-70B': 0.34 - 0.16}
EXTRA = {'Ministral-3-14B-Reasoning': {'gpqa': 71.2, 'index': 0.0}}  # natural order only
rng = np.random.default_rng(0)


def series(key, nat):
    rows = []
    for m, d in M.items():
        if key == 'selfref':  # presence of self-reference (group >= 2); added after the first three, which keep their random draws
            v = 0.5 if m == 'Gemma-4-31B' else float(d['group'] >= 2)
        else:
            v = d[key] if m not in nat else (d[key] + nat[m]) / 2
        if key == 'group' and m == 'Gemma-4-31B':
            v = (0 + 3) / 2
        rows.append((m, d['gpqa'], v))
    if key == 'index':
        rows += [(m, e['gpqa'], e['index']) for m, e in EXTRA.items()]
    return rows


OUT = {}
for key, nat in (('group', {}), ('index', nat_index), ('claim', nat_claim), ('selfref', {})):
    rows = series(key, nat)
    g = np.array([r[1] for r in rows])
    v = np.array([r[2] for r in rows], float)
    r0 = stats.spearmanr(g, v)[0]
    perm = np.array([stats.spearmanr(g, rng.permutation(v))[0] for _ in range(100_000)])
    p = float((np.abs(perm) >= abs(r0) - 1e-12).mean())
    bs = []
    for _ in range(10_000):
        i = rng.integers(0, len(g), len(g))
        if len(set(v[i])) > 1 and len(set(g[i])) > 1:
            bs.append(stats.spearmanr(g[i], v[i])[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    OUT[key] = {'n': len(rows), 'rho': float(r0), 'p': p, 'ci95': [float(lo), float(hi)]}
    print(f'{key:6s} n={len(rows)} rho={r0:+.2f} p={p:.2f} 95% [{lo:+.2f}, {hi:+.2f}]', flush=True)
json.dump(OUT, open(S / 'gpqa_vs_levels_pooled_20261029.json', 'w'), indent=1)
