"""How much of the self-index is shared: a test of Equation 1 (Section sec:abl_eq1) and the stability of the offset's direction.

gpt-oss-20b (native MXFP4 weights, thinking rendering), from the stored geometry of the main probe (feature package:
suite_think/gpt_oss_20b_mxfp4_en/G/<body>.npz, the per-body means of each role's state on message content and at the response
onset, layers x hidden); the discovery / held-out split of every body from role_bodies.jsonl of the eight-model archive:
  direction (runs/analysis/direction_stability_g_mxfp4.json): for every body, the own-message offset o = h_A - (h_U + h_S)/2 of
    the residual stream at each layer, on message content and at the response onset; the discovery direction u = mean of o over
    the 75 discovery bodies; for each of the 75 held-out bodies, cos(o, u) per layer. Reported: median cosine by layer, its
    minimum and maximum, and the share of held-out bodies with a positive cosine.
  eq1 (runs/analysis/gptoss20b_mxfp4_eq1_{content,onset}.json): u = the mean Assistant offset over the discovery bodies at each
    layer; on each held-out body the share of its offset that u explains, 1 - |o - u|^2 / |o|^2; and whether the own message is
    still apart (R_A the smallest of the three distance ratios) after u is subtracted from h_A, against before.
Qwen3.8-27B (check L: 150 bodies x 3 roles, 64 blocks x 5 feature families, 20 body positions), from the per-layer features of
that run (defgeo150), which are not in the packages; these run only with --defgeo DIR (DIR/features/<body>__<role>.npz):
  restricted (runs/analysis/eq1/eq1_restricted.json): a falsifiable form of Equation 1, h^(r) = c + 1[r=A] u + eps.
    One offset u per family and layer, shared by all bodies, fitted only on the discovery bodies as the mean of
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
  perbody (runs/analysis/eq1/eq1_perbody.json): the per-body distribution behind restricted, residual stream only: for
    every held-out body and layer the share of its Assistant offset explained by u, 1 - ||o_b - u||^2 / ||o_b||^2; quantiles
    over the 75 held-out bodies, the five lowest bodies, and the mean per content category, at selected layers.
  residual_ra (runs/analysis/eq1/eq1_residual_ra.json): after removing the shared offset, is the Assistant message
    still singled out? u is subtracted from every Assistant body position of the held-out bodies; distances per position
    averaged over the 20 positions, R_A = D_US / ((D_AU + D_AS)/2); per residual layer the mean R_A before and after removal
    and the share of bodies in which User-System is still the closest pair (token level), and the same on body-mean vectors
    (message level, printed).

usage: eq1_tests.py --results RES --features FEAT --out OUT [--defgeo DIR]
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import M8, W3, Roots, arguments, need, run_path  # noqa: E402

ROLES = ('assistant', 'user', 'system')


# ---------------------------------------------------------------------------------------------------------------------------
# gpt-oss-20b: direction stability and the test of Equation 1 from the G states

def direction_stability(out, dirs, spl):
    """dirs: {name: G folder}; spl: body id -> split"""
    res = {}
    for d, gdir in dirs.items():
        fs = sorted(glob.glob(os.path.join(gdir, '*.npz')))
        res[d] = {}
        for pos in ('content', 'onset'):
            O = {}
            for f in fs:
                z = np.load(f)
                O[os.path.basename(f)[:-4]] = z[pos + '_assistant'].astype(np.float64) - (z[pos + '_user'].astype(np.float64) + z[pos + '_system'].astype(np.float64)) / 2
            disc = [O[b] for b in O if spl[b] == 'discovery']
            held = [O[b] for b in O if spl[b] == 'confirmation']
            u = np.mean(disc, 0)
            cos = np.array([[float(np.dot(o[l], u[l]) / (np.linalg.norm(o[l]) * np.linalg.norm(u[l]) + 1e-12)) for l in range(u.shape[0])] for o in held])
            med = np.median(cos, 0)
            res[d][pos] = {'n_discovery': len(disc), 'n_heldout': len(held), 'median_cos_by_layer': [round(float(x), 4) for x in med],
                           'min_median': [round(float(med.min()), 4), int(med.argmin())], 'max_median': [round(float(med.max()), 4), int(med.argmax())],
                           'positive_share_by_layer': [round(float(x), 4) for x in (cos > 0).mean(0)],
                           'min_cos_over_bodies_by_layer': [round(float(x), 4) for x in cos.min(0)]}
            print(d, pos, res[d][pos]['min_median'], res[d][pos]['max_median'], min(res[d][pos]['positive_share_by_layer']), flush=True)
    json.dump(res, open(out, 'w', encoding='utf-8'), indent=1)


def eq1_offline(gdir, outp, kind, spl):
    """gdir: G folder; the body records are the bodies of its .npz files with their split (spl)"""
    recs = [{'id': os.path.basename(f)[:-4], 'split': spl[os.path.basename(f)[:-4]]} for f in sorted(glob.glob(os.path.join(gdir, '*.npz')))]
    V = {}
    for r in recs:
        z = np.load(os.path.join(gdir, r['id'] + '.npz'))
        V[r['id']] = {k: z[k].reshape(z[k].shape[0], -1).astype(np.float64) for k in z.files}
    disc = [r['id'] for r in recs if r['split'] == 'discovery']
    held = [r['id'] for r in recs if r['split'] == 'confirmation']

    def off(i):
        v = V[i]
        return v[f'{kind}_assistant'] - (v[f'{kind}_user'] + v[f'{kind}_system']) / 2

    def apart(a, u, s):
        au, as_, us = (np.linalg.norm(x - y, axis=-1) for x, y in ((a, u), (a, s), (u, s)))
        R = np.stack([us / ((au + as_) / 2), as_ / ((au + us) / 2), au / ((as_ + us) / 2)])
        return R.argmin(0) == 0  # per layer: the own message is the one set apart

    u = np.mean([off(i) for i in disc], 0)  # [layers, hidden]
    share = np.array([1 - np.sum((off(i) - u) ** 2, -1) / np.sum(off(i) ** 2, -1) for i in held])  # [bodies, layers]
    before = np.array([apart(V[i][f'{kind}_assistant'], V[i][f'{kind}_user'], V[i][f'{kind}_system']) for i in held])
    after = np.array([apart(V[i][f'{kind}_assistant'] - u, V[i][f'{kind}_user'], V[i][f'{kind}_system']) for i in held])
    m = share.mean(0)
    S = {'kind': kind, 'n_layers': int(u.shape[0]), 'n_fit': len(disc), 'n_heldout': len(held),
         'share_explained_mean_by_layer': m.round(4).tolist(), 'share_range': [float(m.min()), float(m.max())],
         'share_median_over_layers': float(np.median(m)), 'share_min_over_bodies_by_layer': share.min(0).round(4).tolist(),
         'apart_before_by_layer': before.mean(0).round(4).tolist(), 'apart_after_by_layer': after.mean(0).round(4).tolist(),
         'apart_after_median_over_layers': float(np.median(after.mean(0))), 'apart_before_median_over_layers': float(np.median(before.mean(0)))}
    json.dump(S, open(outp, 'w'), indent=1)
    print(json.dumps({k: v for k, v in S.items() if not isinstance(v, list) or k == 'share_range'}))
    print('share by layer', [round(x, 2) for x in m])
    print('apart after by layer', [round(x, 2) for x in after.mean(0)])


# ---------------------------------------------------------------------------------------------------------------------------
# Qwen3.8-27B (check L features)

FAMS = ('residual', 'FFN', 'Q', 'K', 'V')
SHOW = (1, 3, 7, 15, 23, 31, 39, 47, 55, 63)


def bf16(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


def restricted(root, out_json):
    def load(path):
        z = np.load(path)
        # a list per family: Q/K/V widths differ between linear- and full-attention blocks
        mean = {f: [bf16(z[f'{f}__L{l:02d}'][2:22]).mean(0) for l in range(64)] for f in FAMS}
        pos = np.stack([bf16(z[f'residual__L{l:02d}'][2:22]) for l in range(64)])  # (64, 20, 5120)
        return mean, pos

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


def perbody(root, out_json):
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


def residual_ra(root, out_json):
    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1).mean(-1)  # per-position distance, averaged over positions -> (64,)

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


def main(argv=None):
    ap = arguments(__doc__, features=True)
    ap.add_argument('--defgeo', help='folder of the check-L features (features/<body>__<role>.npz), not in the packages')
    args = ap.parse_args(argv)
    r = Roots(args)
    OUTA = r.out(W3 + '/runs/analysis')
    rel = 'suite_think/gpt_oss_20b_mxfp4_en'
    G = run_path(r.feat(W3), rel) / 'G' if r.FEAT else None
    splits = r.res(M8 + '/items/role_bodies.jsonl')
    msg = need([G, splits], 'eq1_tests (gpt-oss-20b)')
    if msg:
        print(msg)
    else:
        spl = {json.loads(l)['id']: json.loads(l)['split'] for l in open(splits, encoding='utf-8') if l.strip()}
        direction_stability(str(OUTA / 'direction_stability_g_mxfp4.json'), {rel: str(G)}, spl)
        for kind in ('content', 'onset'):
            eq1_offline(str(G), str(OUTA / f'gptoss20b_mxfp4_eq1_{kind}.json'), kind, spl)
    if args.defgeo:
        E = r.out(W3 + '/runs/analysis/eq1')
        restricted(args.defgeo, str(E / 'eq1_restricted.json'))
        perbody(args.defgeo, str(E / 'eq1_perbody.json'))
        residual_ra(args.defgeo, str(E / 'eq1_residual_ra.json'))
    else:
        print('eq1_tests (Qwen3.8-27B): needs inputs not in the package (the per-layer features of check L, --defgeo); skipped')


if __name__ == '__main__':
    main()
