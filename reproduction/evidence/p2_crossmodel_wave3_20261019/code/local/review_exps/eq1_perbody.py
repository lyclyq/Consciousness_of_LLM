"""Review H10/H22 follow-up (10-19): the per-body distribution behind eq1_restricted.py. Residual stream only. For every
held-out body and layer: the share of its Assistant offset o_b = x_A - (x_U + x_S)/2 explained by the discovery-fitted shared
offset u, 1 - ||o_b - u||^2 / ||o_b||^2 (x = mean over the 20 body positions). Reports quantiles over the 75 held-out bodies,
the five lowest bodies, and the mean per content category, at selected layers.
usage: eq1_perbody.py DEFGEO_DIR OUT_JSON"""
import glob
import json
import os
import sys
from collections import defaultdict
import numpy as np

ROLES = ('assistant', 'user', 'system')
SHOW = (1, 3, 7, 15, 23, 31, 39, 47, 55, 63)


def bf16(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


def main(root, out_json):
    ids = sorted({os.path.basename(p).split('__')[0] for p in glob.glob(f'{root}/features/*__assistant.npz')})
    split = {i: 'discovery' if i.rsplit('_', 1)[1][0] == 'd' else 'heldout' for i in ids}
    cat = {i: i.rsplit('_', 1)[0].replace('role150_', '') for i in ids}
    off = {}
    for i in ids:
        x = {}
        for r in ROLES:
            z = np.load(f'{root}/features/{i}__{r}.npz')
            x[r] = np.stack([bf16(z[f'residual__L{l:02d}'][2:22]).mean(0) for l in range(64)])
        off[i] = x['assistant'] - (x['user'] + x['system']) / 2  # (64, 5120)
    D = [i for i in ids if split[i] == 'discovery']
    H = [i for i in ids if split[i] == 'heldout']
    u = np.mean([off[i] for i in D], axis=0)
    share = {i: 1 - ((off[i] - u) ** 2).sum(1) / (off[i] ** 2).sum(1) for i in H}  # (64,) per body
    res = {'layers': {}}
    for l in SHOW:
        v = np.array([share[i][l] for i in H])
        low = sorted(H, key=lambda i: share[i][l])[:5]
        by_cat = defaultdict(list)
        for i in H:
            by_cat[cat[i]].append(share[i][l])
        res['layers'][l] = {'quantiles': {q: float(np.percentile(v, q)) for q in (0, 5, 25, 50, 75, 95, 100)},
                            'lowest': [[i, float(share[i][l])] for i in low],
                            'by_category': {c: float(np.mean(s)) for c, s in sorted(by_cat.items())}}
        q = res['layers'][l]['quantiles']
        cats = sorted(res['layers'][l]['by_category'].items(), key=lambda kv: kv[1])
        print(f'L{l:02d}: min {q[0]:.2f}  p5 {q[5]:.2f}  p25 {q[25]:.2f}  median {q[50]:.2f}  p75 {q[75]:.2f}  max {q[100]:.2f} | '
              f'lowest categories {cats[0][0]} {cats[0][1]:.2f}, {cats[1][0]} {cats[1][1]:.2f}; highest {cats[-1][0]} {cats[-1][1]:.2f}',
              flush=True)
    json.dump(res, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
