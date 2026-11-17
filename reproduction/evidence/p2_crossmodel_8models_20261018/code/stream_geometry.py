"""Per-stream geometry for hyper-connection models (Qwen3.8-Flash-Next: 4 residual streams of 2560 dims, stored stream-major in the
10240-dim layer output). For each stream separately, and for the mean of the four per-stream ratios: distance ratios R_X =
D_YZ / mean(D_XY, D_XZ) on held-out bodies, the layers where the Assistant message is singled out (R_A smallest) in every held-out
body and in >= 80% of them, the median R_A, and the cosine between each held-out body's Assistant offset and the discovery direction.
usage: stream_geometry.py G_DIR N_STREAMS OUT_JSON"""
import glob
import json
import sys
import numpy as np

gdir, hc, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
recs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(gdir + '/*.json'))]
V = {r['id']: np.load(f"{gdir}/{r['id']}.npz") for r in recs}
held = [r['id'] for r in recs if r['split'] == 'confirmation']
disc = [r['id'] for r in recs if r['split'] == 'discovery']
ROLES = ('assistant', 'user', 'system')
out = {}
for kind in ('content', 'onset'):
    X = {i: {q: V[i][f'{kind}_{q}'].astype(np.float32) for q in ROLES} for i in V}
    L, D = X[held[0]]['assistant'].shape
    H = D // hc
    res = {}
    allR = []
    for s in range(hc + 1):          # s == hc: the full concatenated state, for reference
        sl = slice(None) if s == hc else slice(s * H, (s + 1) * H)
        Rs, cos = [], []
        dirv = np.mean([X[i]['assistant'][:, sl] - (X[i]['user'][:, sl] + X[i]['system'][:, sl]) / 2 for i in disc], 0)
        for i in held:
            a, u, y = (X[i][q][:, sl] for q in ROLES)
            au, ay, uy = (np.linalg.norm(p - q, axis=-1) for p, q in ((a, u), (a, y), (u, y)))
            Rs.append([uy / ((au + ay) / 2), ay / ((au + uy) / 2), au / ((ay + uy) / 2)])
            off = a - (u + y) / 2
            cos.append((off * dirv).sum(-1) / (np.linalg.norm(off, axis=-1) * np.linalg.norm(dirv, axis=-1) + 1e-9))
        Rs = np.array(Rs)                # [held, 3, L]
        if s < hc:
            allR.append(Rs)
        low = Rs.argmin(1)
        name = 'concat' if s == hc else f'stream{s}'
        res[name] = {'A_lowest_all_bodies': int(((low == 0).mean(0) == 1).sum()), 'A_lowest_80': int(((low == 0).mean(0) >= 0.8).sum()),
                     'R_A_median': float(np.median(Rs[:, 0].mean(0))), 'R_A_by_layer': Rs[:, 0].mean(0).round(3).tolist(),
                     'dir_cos_median': float(np.median(np.mean(cos, 0)))}
    Rm = np.mean(allR, 0)
    low = Rm.argmin(1)
    res['mean_of_streams'] = {'A_lowest_all_bodies': int(((low == 0).mean(0) == 1).sum()), 'A_lowest_80': int(((low == 0).mean(0) >= 0.8).sum()),
                              'R_A_median': float(np.median(Rm[:, 0].mean(0))), 'R_A_by_layer': Rm[:, 0].mean(0).round(3).tolist()}
    out[kind] = res
    for k, v in res.items():
        print(f"{kind:7s} {k:15s} A singled out in all held-out bodies {v['A_lowest_all_bodies']:2d}/{L}, >=80% {v['A_lowest_80']:2d}/{L}, median R_A {v['R_A_median']:.2f}" + (f", dir cos {v['dir_cos_median']:.2f}" if 'dir_cos_median' in v else ''))
json.dump(out, open(outp, 'w'), indent=1)
