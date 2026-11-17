"""Geometry at the response onset for the main geometry figure (check J, template default, non-thinking, turn 'reply':
the role message is followed by a native User message "please respond based on the above" and the generation prompt).
Reads the local mirror of the J run (results/compliant_150_20261015/raw/attrgeo150), position 'decision' (the last prompt
token), all 64 layers of the residual stream.
(1) points: for layers 0/15/31/47/63, each held-out body's three role states centred on their mean (removes the content)
    and projected on x = self-index direction  w_own = mean over discovery bodies of A - (U + S)/2   (unit length)
                       y = User-System direction w_us  = mean over discovery bodies of U - S, orthogonalised to w_own
(2) distance ratios per layer over the held-out bodies (raw L2 and cosine distance): mean R_A, R_U, R_S as in
    analyze_native_geometry.py, the share of bodies whose closest pair is User-System, and the share in which R_A is the
    smallest of the three ratios (the self-index criterion of Definition 1);
(3) direction consistency per layer: cosine between a held-out body's offset of role X from the other two and the mean
    discovery offset of X (as off_X body_cos of the message-content analysis), mean over held-out bodies and over the 15
    categories (mean of category means, as there).
Self-check: the raw-L2 mean ratios over all 150 bodies must reproduce J_native_geometry_default.json (default__reply,
decision) to 1e-6.
usage: onset_geometry_points.py J_ROOT J_SUMMARY OUT_JSON"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROLES = ('assistant', 'user', 'system')
LAYERS = [0, 15, 31, 47, 63]


def dist(a, b, metric):
    if metric == 'raw':
        return np.linalg.norm(a - b, axis=-1)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return 1.0 - (a * b).sum(-1) / (na * nb + 1e-12)


def ratios(H, metric):
    """H: dict role -> [N, 64, d]; per body and layer R_A, R_U, R_S."""
    au, as_, us = dist(H['assistant'], H['user'], metric), dist(H['assistant'], H['system'], metric), dist(H['user'], H['system'], metric)
    return us / ((au + as_) / 2), as_ / ((au + us) / 2), au / ((as_ + us) / 2), np.stack([au, as_, us], -1)


def main(j_root, j_summary, out_json):
    j_root = Path(j_root)
    names = json.load(open(j_root / 'identity.json', encoding='utf-8'))['position_names']
    k = names.index('decision')
    meta = {}
    for f in sorted(glob.glob(str(j_root / 'bodies' / '*.json'))):
        rec = json.load(open(f, encoding='utf-8'))
        meta[rec['id']] = rec['split']
    ids = sorted(meta)
    H = {r: [] for r in ROLES}
    for bid in ids:
        with np.load(j_root / 'features' / f'{bid}.npz') as z:
            for r in ROLES:
                H[r].append(z[f'default__reply__{r}'][:, k].astype(np.float32))
    H = {r: np.stack(v) for r, v in H.items()}          # [150, 64, d]
    split = np.array([meta[b] for b in ids])
    cat = np.array([b.rsplit('_', 1)[0] for b in ids])
    disc, conf = split == 'discovery', split == 'confirmation'
    out = {'source': str(j_root), 'turn': 'default__reply', 'position': 'decision', 'n_heldout': int(conf.sum()),
           'n_discovery': int(disc.sum()), 'layers': LAYERS, 'points': {}, 'heldout': {}, 'direction_consistency': {}}
    # self-check against the archived J summary (all bodies, raw L2, mean over bodies)
    ra, ru, rs, _ = ratios(H, 'raw')
    ref = json.load(open(j_summary, encoding='utf-8'))['curves']['default__reply']['decision']
    err = max(abs(float(ra[:, l].mean()) - ref[l]['R_A']) + abs(float(ru[:, l].mean()) - ref[l]['R_U'])
              + abs(float(rs[:, l].mean()) - ref[l]['R_S']) for l in range(64))
    assert err < 1e-5, err
    out['selfcheck_max_abs_diff_vs_J_summary'] = err
    # (1) points
    for l in LAYERS:
        w_own = np.mean([H['assistant'][i, l] - (H['user'][i, l] + H['system'][i, l]) / 2 for i in np.where(disc)[0]], 0)
        w_own /= np.linalg.norm(w_own)
        w_us = np.mean([H['user'][i, l] - H['system'][i, l] for i in np.where(disc)[0]], 0)
        w_us -= (w_us @ w_own) * w_own
        w_us /= np.linalg.norm(w_us)
        pts = {r: [] for r in ROLES}
        for i in np.where(conf)[0]:
            m = np.mean([H[r][i, l] for r in ROLES], 0)
            for r in ROLES:
                v = H[r][i, l] - m
                pts[r].append([float(v @ w_own), float(v @ w_us)])
        out['points'][str(l)] = pts
    # (2) held-out distance ratios
    Hc = {r: H[r][conf] for r in ROLES}
    for metric in ('raw', 'cos'):
        ra, ru, rs, d = ratios(Hc, metric)
        smallest = (ra < np.minimum(ru, rs))
        out['heldout'][metric] = [{'R_A': float(ra[:, l].mean()), 'R_U': float(ru[:, l].mean()), 'R_S': float(rs[:, l].mean()),
                                   'R_A_median': float(np.median(ra[:, l])),
                                   'US_closest': float(np.mean(d[:, l].argmin(-1) == 2)),
                                   'A_smallest': float(smallest[:, l].mean())} for l in range(64)]
    # (3) direction consistency
    sym = {'assistant': ('user', 'system'), 'user': ('assistant', 'system'), 'system': ('assistant', 'user')}
    for r, (y, z) in sym.items():
        off = H[r] - (H[y] + H[z]) / 2                       # [150, 64, d]
        axis = off[disc].mean(0)                             # [64, d]
        num = (off[conf] * axis[None]).sum(-1)
        cos = num / (np.linalg.norm(off[conf], axis=-1) * np.linalg.norm(axis, axis=-1)[None] + 1e-12)   # [75, 64]
        by_cat = defaultdict(list)
        for c, row in zip(cat[conf], cos):
            by_cat[c].append(row)
        cat_means = np.stack([np.mean(v, 0) for v in by_cat.values()])
        out['direction_consistency'][r] = {'mean_of_category_means': cat_means.mean(0).tolist(), 'mean': cos.mean(0).tolist(),
                                          'min_body': cos.min(0).tolist(), 'positive_share': (cos > 0).mean(0).tolist(),
                                          'categories': len(by_cat), 'bodies_per_category': sorted({len(v) for v in by_cat.values()})}
    json.dump(out, open(out_json, 'w'))
    hr = out['heldout']['raw']
    print('self-check vs J summary: max |diff|', err)
    print('held-out R_A range %.3f-%.3f; A smallest in %.2f-%.2f of bodies; US closest in %.2f-%.2f'
          % (min(x['R_A'] for x in hr), max(x['R_A'] for x in hr), min(x['A_smallest'] for x in hr), max(x['A_smallest'] for x in hr),
             min(x['US_closest'] for x in hr), max(x['US_closest'] for x in hr)))
    dc = out['direction_consistency']['assistant']['mean_of_category_means']
    print('direction consistency (Assistant): L0 %.3f, L3 %.3f, min %.3f, max %.3f' % (dc[0], dc[3], min(dc), max(dc)))


if __name__ == '__main__':
    main(*sys.argv[1:])
