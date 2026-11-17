"""Review H10/H22 (10-19): a falsifiable form of Equation 1, h^(r) = c + 1[r=A] u + eps, on the stored geometry of check L
(defgeo150: 150 bodies x 3 roles, 64 blocks x 5 feature families, 20 body positions).

Restriction: one offset u per family and layer, shared by all bodies, fitted ONLY on the discovery bodies as the mean of
o_b = x_A - (x_U + x_S)/2 (x = mean over the 20 body positions). Evaluated on the held-out bodies, which share no wording
with the discovery bodies:
  r2_roles      1 - sum ||d - d_hat||^2 / sum ||d||^2 over the three roles, where d = x_r minus the body's mean over roles
                and d_hat = (2/3)u for Assistant, -(1/3)u for User and System: the share of all role-dependent variation
                explained by the shared offset.
  r2_offset     1 - sum ||o_b - u||^2 / sum ||o_b||^2: the share of the Assistant offset itself explained by u.
  proj_share    sum (o_b . u_hat)^2 / sum ||o_b||^2: the share of the offset's energy along the discovery direction.
  us_share      sum ||x_U - x_S||^2 / sum over roles ||d||^2 (the User-System part, which the model assigns to eps).
  r2_roles_pos  r2_roles per position (residual stream only, every layer): the same u applied at each of the 20
                positions, i.e. an offset shared across bodies AND positions.
1 - r2_offset is the role-content interaction: the part of the Assistant offset that varies with the body.
usage: eq1_restricted.py DEFGEO_DIR OUT_JSON"""
import glob
import json
import os
import sys
import numpy as np

FAMS = ('residual', 'FFN', 'Q', 'K', 'V')
ROLES = ('assistant', 'user', 'system')


def bf16(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


def load(path):
    z = np.load(path)
    # a list per family: Q/K/V widths differ between linear- and full-attention blocks
    mean = {f: [bf16(z[f'{f}__L{l:02d}'][2:22]).mean(0) for l in range(64)] for f in FAMS}
    pos = np.stack([bf16(z[f'residual__L{l:02d}'][2:22]) for l in range(64)])  # (64, 20, 5120)
    return mean, pos


def main(root, out_json):
    ids = sorted({os.path.basename(p).split('__')[0] for p in glob.glob(f'{root}/features/*__assistant.npz')})
    split = {i: 'discovery' if i.rsplit('_', 1)[1][0] == 'd' else 'heldout' for i in ids}
    X, P = {}, {}
    for k, i in enumerate(ids):
        X[i], P[i] = {}, {}
        for r in ROLES:
            X[i][r], P[i][r] = load(f'{root}/features/{i}__{r}.npz')
        if k % 25 == 0:
            print('loaded', k, len(ids), flush=True)
    D = [i for i in ids if split[i] == 'discovery']
    H = [i for i in ids if split[i] == 'heldout']
    print('bodies', len(ids), 'discovery', len(D), 'held-out', len(H), flush=True)
    res = {'n_discovery': len(D), 'n_heldout': len(H), 'families': {}}
    for f in FAMS:
        rows = []
        for l in range(64):
            xs = {i: np.stack([X[i][r][f][l] for r in ROLES]) for i in ids}  # (3, dim)
            off = {i: xs[i][0] - (xs[i][1] + xs[i][2]) / 2 for i in ids}
            u = np.mean([off[i] for i in D], axis=0)
            uh = u / np.linalg.norm(u)
            dhat = np.stack([2 * u / 3, -u / 3, -u / 3])
            num = den = us = o_res = o_tot = proj = 0.0
            for i in H:
                d = xs[i] - xs[i].mean(0)
                num += ((d - dhat) ** 2).sum()
                den += (d ** 2).sum()
                us += ((xs[i][1] - xs[i][2]) ** 2).sum()
                o_res += ((off[i] - u) ** 2).sum()
                o_tot += (off[i] ** 2).sum()
                proj += float(off[i] @ uh) ** 2
            row = {'layer': l, 'r2_roles': 1 - num / den, 'r2_offset': 1 - o_res / o_tot, 'proj_share': proj / o_tot,
                   'us_share': us / den}
            if f == 'residual':
                pn = pd = 0.0
                for i in H:
                    ps = np.stack([P[i][r][l] for r in ROLES])  # (3, 20, dim)
                    d = ps - ps.mean(0, keepdims=True)
                    pn += ((d - dhat[:, None, :]) ** 2).sum()
                    pd += (d ** 2).sum()
                row['r2_roles_pos'] = 1 - pn / pd
            rows.append({k: (float(v) if k != 'layer' else v) for k, v in row.items()})
        res['families'][f] = rows
        r2 = np.array([r['r2_roles'] for r in rows]); ro = np.array([r['r2_offset'] for r in rows])
        print(f'{f:8s} r2_roles min {r2.min():.3f} median {np.median(r2):.3f} max {r2.max():.3f} | '
              f'r2_offset min {ro.min():.3f} median {np.median(ro):.3f}', flush=True)
    for l in (0, 1, 3, 7, 15, 23, 31, 39, 47, 55, 63):
        r = res['families']['residual'][l]
        print(f'residual L{l:02d}: r2_roles {r["r2_roles"]:.3f}  r2_offset {r["r2_offset"]:.3f}  proj {r["proj_share"]:.3f}  '
              f'U-S share {r["us_share"]:.3f}  per-position r2 {r["r2_roles_pos"]:.3f}')
    json.dump(res, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
