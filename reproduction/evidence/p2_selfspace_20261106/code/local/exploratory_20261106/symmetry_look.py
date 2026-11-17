"""Exploratory (not for the paper): is the own-message region a congruent copy of the external region, and is the own offset
the same in every content category? Pooled message-content features, one vector per body, role and layer.
(1) congruence: correlation of the body-by-body distance matrices (upper triangle) between roles, and the spread of each role's
    cloud (RMS distance of a body from the role's centroid) relative to User.
(2) uniformity over categories: cosine of each category's mean own offset with the overall mean offset, and the share of the
    offset variance that lies between categories (15 categories x 10 bodies; category = the item family in the body id).
usage: python -I symmetry_look.py NAME=G_DIR [...]"""
import sys
from pathlib import Path
import numpy as np

for spec in sys.argv[1:]:
    name, g = spec.split('=', 1)
    fs = sorted(Path(g).glob('*.npz'))
    cat = np.array(['_'.join(f.stem.split('_')[1:-1]) for f in fs])
    X = {r: np.stack([np.load(f)[f'content_{r}'].astype(np.float32) for f in fs]) for r in ('assistant', 'user', 'system')}
    B, L, _ = X['user'].shape
    iu = np.triu_indices(B, 1)
    ls = sorted({L // 4, L // 2, 3 * L // 4, L - 1})
    print(f'{name}  ({len(set(cat))} categories)  layers {ls}')
    rows = {k: [] for k in ('rsa A-U', 'rsa A-S', 'rsa U-S', 'spread A/U', 'spread S/U', 'cat cos min', 'cat cos med', 'between-cat share')}
    for l in ls:
        D = {}
        sp = {}
        for r in X:
            Z = X[r][:, l]
            D[r] = np.linalg.norm(Z[:, None] - Z[None], axis=-1)[iu]
            sp[r] = np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean())
        c = lambda a, b: np.corrcoef(D[a], D[b])[0, 1]
        rows['rsa A-U'].append(c('assistant', 'user')); rows['rsa A-S'].append(c('assistant', 'system')); rows['rsa U-S'].append(c('user', 'system'))
        rows['spread A/U'].append(sp['assistant'] / sp['user']); rows['spread S/U'].append(sp['system'] / sp['user'])
        o = X['assistant'][:, l] - (X['user'][:, l] + X['system'][:, l]) / 2
        m = o.mean(0)
        cm = np.stack([o[cat == k].mean(0) for k in sorted(set(cat))])
        cs = (cm @ m) / (np.linalg.norm(cm, axis=1) * np.linalg.norm(m))
        rows['cat cos min'].append(cs.min()); rows['cat cos med'].append(np.median(cs))
        oc = o - m
        between = sum((cat == k).sum() * ((cm[i] - m) ** 2).sum() for i, k in enumerate(sorted(set(cat))))
        rows['between-cat share'].append(between / (oc ** 2).sum())
    for k, v in rows.items():
        print(f'   {k:18s} ' + ' '.join(f'{x:5.2f}' for x in v))
