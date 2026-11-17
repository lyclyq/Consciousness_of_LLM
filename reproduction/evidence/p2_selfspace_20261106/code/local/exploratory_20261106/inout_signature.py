"""Exploratory (not for the paper): inside/outside signature of the role geometry on pooled message-content features.
Per body and layer: cos(A-U, A-S) (is the own offset the same vector against both external roles?),
|U-S| / mean(|A-U|, |A-S|) (how close the two external roles are relative to their distance from the own message), and,
along the own axis fitted on the other bodies (leave-one-out), the U-S gap as a share of the own gap.
usage: python -I inout_signature.py NAME=G_DIR [NAME=G_DIR ...]"""
import sys
from pathlib import Path
import numpy as np

for spec in sys.argv[1:]:
    name, g = spec.split('=', 1)
    fs = sorted(Path(g).glob('*.npz'))
    X = {r: np.stack([np.load(f)[f'content_{r}'].astype(np.float32) for f in fs]) for r in ('assistant', 'user', 'system')}
    A, U, S = X['assistant'], X['user'], X['system']
    B, L, _ = A.shape
    au, as_, us = A - U, A - S, U - S
    cos = (au * as_).sum(-1) / (np.linalg.norm(au, axis=-1) * np.linalg.norm(as_, axis=-1) + 1e-9)
    rel = np.linalg.norm(us, axis=-1) / ((np.linalg.norm(au, axis=-1) + np.linalg.norm(as_, axis=-1)) / 2)
    own = A - (U + S) / 2
    tot = own.sum(0)
    gap = np.empty((B, L))
    for i in range(B):  # leave-one-out own axis
        ax = (tot - own[i]) / (B - 1)
        ax /= np.linalg.norm(ax, axis=-1, keepdims=True) + 1e-9
        gap[i] = np.abs((us[i] * ax).sum(-1)) / np.abs((own[i] * ax).sum(-1)).clip(1e-9)
    ls = sorted({0, L // 8, L // 4, L // 2, 3 * L // 4, L - 1})
    row = lambda v: ' '.join(f'{np.median(v[:, l]):5.2f}' for l in ls)
    print(f'{name:24s} layers {ls}')
    print(f'{"":24s} cos(A-U, A-S)       {row(cos)}')
    print(f'{"":24s} |U-S| / |A-ext|     {row(rel)}')
    print(f'{"":24s} U-S gap on own axis {row(gap)}')
