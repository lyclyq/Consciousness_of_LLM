"""Exploratory look (not for the paper): how much of the own-vs-external offset is one shared translation, how big it is
relative to the spread of the texts, and where it lies relative to the directions along which the texts differ.
Pooled features of Qwen3.8-27B (one vector per body, role and layer). Compared with the User-vs-System offset.
usage: python -I translation_look.py G_DIR"""
import json, sys
from pathlib import Path
import numpy as np

g = Path(sys.argv[1])
recs = [{'id': f.stem} for f in sorted(g.glob('*.npz'))]
for kind in ('content', 'onset'):
    X = {}
    for role in ('assistant', 'user', 'system'):
        X[role] = np.stack([np.load(g / f"{r['id']}.npz")[f'{kind}_{role}'].astype(np.float32) for r in recs])  # (B, L, d)
    B, L, d = X['user'].shape
    offs = {'own - external': X['assistant'] - (X['user'] + X['system']) / 2, 'User - System': X['user'] - X['system']}
    print(f'\n== {kind}: {B} bodies, {L} layers, d={d}')
    print('layer | offset | shared share | cos(text, mean) | |mean offset| / text spread | share in top-10 / top-50 text PCs')
    for l in (0, L // 8, L // 4, L // 2, 3 * L // 4, L - 1):
        U = X['user'][:, l]
        Uc = U - U.mean(0)
        spread = np.sqrt((Uc ** 2).sum(1).mean())  # RMS distance of a text from the mean text (User role)
        _, _, Vt = np.linalg.svd(Uc, full_matrices=False)
        for name, O in offs.items():
            o = O[:, l]
            m = o.mean(0)
            shared = (m ** 2).sum() / (o ** 2).sum(1).mean()  # share of the offset energy that is the common translation
            # leave-one-out cosine of each text's offset with the mean of the others
            cs = [np.dot(o[i], (m * B - o[i]) / (B - 1)) / (np.linalg.norm(o[i]) * np.linalg.norm((m * B - o[i]) / (B - 1)) + 1e-9) for i in range(B)]
            mn = m / (np.linalg.norm(m) + 1e-9)
            top = [float(((Vt[:k] @ mn) ** 2).sum()) for k in (10, 50)]
            print(f'L{l:>2} | {name:14s} | {shared:.2f} | {np.mean(cs):.2f} | {np.linalg.norm(m) / spread:.2f} | {top[0]:.2f} / {top[1]:.2f}')
    print(f'(random direction: top-10 {10 / d:.4f}, top-50 {50 / d:.4f})')
