"""Review H10/H22, criterion proposed by the user (10-19): after removing the shared offset, is the Assistant message still
singled out? The discovery-fitted shared offset u (mean over discovery bodies of x_A - (x_U + x_S)/2, x = mean over the 20
body positions) is subtracted from every Assistant body position of the held-out bodies. Distances are then computed as in
the paper (Euclidean distance per position, averaged over the 20 positions), and R_A = D_US / ((D_AU + D_AS)/2).
If u carries all that is specific to the model's own message, the leftover Assistant difference is as large as the
User-System difference: R_A after removal is about 1 and Assistant is no longer singled out.
Reports per residual layer, over the 75 held-out bodies: mean R_A before and after removal, and the share of bodies in which
User-System is still the closest pair.
usage: eq1_residual_ra.py DEFGEO_DIR OUT_JSON"""
import glob
import json
import os
import sys
import numpy as np

ROLES = ('assistant', 'user', 'system')


def bf16(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


def dist(a, b):
    return np.linalg.norm(a - b, axis=-1).mean(-1)  # per-position distance, averaged over positions -> (64,)


def main(root, out_json):
    ids = sorted({os.path.basename(p).split('__')[0] for p in glob.glob(f'{root}/features/*__assistant.npz')})
    split = {i: 'discovery' if i.rsplit('_', 1)[1][0] == 'd' else 'heldout' for i in ids}
    P = {}
    for i in ids:
        P[i] = {}
        for r in ROLES:
            z = np.load(f'{root}/features/{i}__{r}.npz')
            P[i][r] = np.stack([bf16(z[f'residual__L{l:02d}'][2:22]) for l in range(64)])  # (64, 20, 5120)
    D = [i for i in ids if split[i] == 'discovery']
    H = [i for i in ids if split[i] == 'heldout']
    u = np.mean([P[i]['assistant'].mean(1) - (P[i]['user'].mean(1) + P[i]['system'].mean(1)) / 2 for i in D], axis=0)  # (64, 5120)
    before, after, us_before, us_after = [], [], [], []
    m_before, m_after, m_cb, m_ca = [], [], [], []  # message level: distances between body-mean vectors
    for i in H:
        a, us, sy = P[i]['assistant'], P[i]['user'], P[i]['system']
        a2 = a - u[:, None, :]
        dus = dist(us, sy)
        for A, R, C in ((a, before, us_before), (a2, after, us_after)):
            dau, das = dist(A, us), dist(A, sy)
            R.append(dus / ((dau + das) / 2))
            C.append((dus < dau) & (dus < das))
        am, um, sm = a.mean(1), us.mean(1), sy.mean(1)
        mus = np.linalg.norm(um - sm, axis=-1)
        for A, R, C in ((am, m_before, m_cb), (am - u, m_after, m_ca)):
            dau, das = np.linalg.norm(A - um, axis=-1), np.linalg.norm(A - sm, axis=-1)
            R.append(mus / ((dau + das) / 2))
            C.append((mus < dau) & (mus < das))
    m_before, m_after, m_cb, m_ca = (np.array(x) for x in (m_before, m_after, m_cb, m_ca))
    print('message level (body-mean vectors):')
    for l in (0, 1, 3, 7, 15, 23, 31, 39, 47, 55, 63):
        print(f'  L{l:02d}: R_A before {m_before[:, l].mean():.2f} -> after {m_after[:, l].mean():.2f} '
              f'[p5 {np.percentile(m_after[:, l], 5):.2f}, p95 {np.percentile(m_after[:, l], 95):.2f}] | Assistant singled out in '
              f'{m_cb[:, l].mean():.2f} -> {m_ca[:, l].mean():.2f}', flush=True)
    ma = m_after.mean(0); mc = m_ca.mean(0)
    print(f'  all 64 layers: R_A after min {ma.min():.2f} median {np.median(ma):.2f} max {ma.max():.2f}; '
          f'still singled out min {mc.min():.2f} median {np.median(mc):.2f} max {mc.max():.2f}', flush=True)
    print('token level (per-position distances, as in the paper):')
    before, after = np.array(before), np.array(after)
    us_before, us_after = np.array(us_before), np.array(us_after)
    rows = [{'layer': l, 'ra_before': float(before[:, l].mean()), 'ra_after': float(after[:, l].mean()),
             'ra_after_p5': float(np.percentile(after[:, l], 5)), 'ra_after_p95': float(np.percentile(after[:, l], 95)),
             'us_closest_before': float(us_before[:, l].mean()), 'us_closest_after': float(us_after[:, l].mean())}
            for l in range(64)]
    for r in rows:
        if r['layer'] in (0, 1, 3, 7, 15, 23, 31, 39, 47, 55, 63):
            print(f"L{r['layer']:02d}: R_A before {r['ra_before']:.2f} -> after {r['ra_after']:.2f} "
                  f"[p5 {r['ra_after_p5']:.2f}, p95 {r['ra_after_p95']:.2f}] | Assistant singled out in "
                  f"{r['us_closest_before']:.2f} -> {r['us_closest_after']:.2f} of held-out bodies", flush=True)
    ra = np.array([r['ra_after'] for r in rows]); sh = np.array([r['us_closest_after'] for r in rows])
    print(f'all 64 layers: R_A after removal min {ra.min():.2f} median {np.median(ra):.2f} max {ra.max():.2f}; '
          f'share of bodies still singled out min {sh.min():.2f} median {np.median(sh):.2f} max {sh.max():.2f}')
    json.dump({'n_discovery': len(D), 'n_heldout': len(H), 'rows': rows}, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
