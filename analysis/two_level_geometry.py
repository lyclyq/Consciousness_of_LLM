"""Qwen3.8-27B, native renderings (check J: template default and preserve_thinking=False, 150 bodies x 3 roles): the geometry of
the self-index at the response onset and the self-reference baselines, from the per-layer feature arrays (feature package,
attrgeo150/features/*.npz) and the per-body records (compliant_150_per_body.tgz of the results package: attrgeo150/identity.json
and attrgeo150/bodies/*.json, extracted into OUT/_work).

native (summaries/native_geometry150_summary.json of the compliant_150 archive; its default-mode part is
  summaries/J_native_geometry_default.json of the two-level archive): role geometry by position x layer, decision JS,
  option-letter baselines. Distances: raw L2 and cosine distance between role states at the same position.
  R_X = D(other two) / mean(D(X, other)); R_X < 2/3 means X is farther from both others than they are from each other. Table rows
  print median R; the stored 64-layer curves use the mean over bodies. 'closest' = share of bodies in which each pair is the
  closest pair. Also: the agreement of the final-layer lens with the native logits, the option-letter mass of the attribution
  question, and the attribution baselines (both option orders pooled).
numbers (summaries/j150_numbers.txt of the compliant_150 archive): the check-J numbers quoted in the paper, printed from the
  native summary (odd role and closest pairs per position and layer, decision JS, attribution baselines).
onset (summaries/J_onset_geometry.json of the two-level archive): geometry at the response onset for the main geometry figure
  (template default, non-thinking, turn 'reply': the role message is followed by a native User message "please respond based on
  the above" and the generation prompt), position 'decision' (the last prompt token), all 64 layers of the residual stream.
  (1) points: for layers 0/15/31/47/63, each held-out body's three role states centred on their mean (removes the content)
      and projected on x = self-index direction  w_own = mean over discovery bodies of A - (U + S)/2   (unit length)
                         y = User-System direction w_us  = mean over discovery bodies of U - S, orthogonalised to w_own
  (2) distance ratios per layer over the held-out bodies (raw L2 and cosine distance): mean R_A, R_U, R_S as in native, the share
      of bodies whose closest pair is User-System, and the share in which R_A is the smallest of the three ratios (the self-index
      criterion of Definition def:self);
  (3) direction consistency per layer: cosine between a held-out body's offset of role X from the other two and the mean
      discovery offset of X (as off_X body_cos of the message-content analysis), mean over held-out bodies and over the 15
      categories (mean of category means, as there).
  Self-check: the raw-L2 mean ratios over all 150 bodies must reproduce the native summary (default__reply, decision) to 1e-6.

usage: two_level_geometry.py --results RES --features FEAT --out OUT
"""
import glob
import json
import math
import os
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C150, TL, Roots, arguments, need, shown  # noqa: E402

ROLES = ('assistant', 'user', 'system')
PAIRS = {'AU': (0, 1), 'AS': (0, 2), 'US': (1, 2)}
SHOW_POS = ['header_newline', 'body_mean', 'body_p5', 'body_im_end', 'turn_newline', 'decision']
SHOW_LAYERS = [0, 1, 2, 3, 5, 7, 11, 15, 23, 31, 39, 47, 55, 59, 63]
ORD = ('self', 'user', 'system')


def extract(tgz, dest):
    """identity.json and bodies/*.json of the J run (attrgeo150) from the per-body archive"""
    pat = re.compile(r'^[^/]+/attrgeo150/(identity\.json|bodies/[^/]+\.json)$')
    with tarfile.open(tgz) as T:
        for mem in T.getmembers():
            m = pat.match(mem.name)
            if m and mem.isfile():
                p = Path(dest) / m.group(1)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(T.extractfile(mem).read())
    return Path(dest)


def dist(a, b, metric):
    if metric == 'raw':
        return np.linalg.norm(a - b, axis=-1)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return 1.0 - (a * b).sum(-1) / (na * nb + 1e-12)


def native(root, feat, out_json):
    """root: identity.json and bodies/ of the run; feat: its features/ folder"""
    ident = json.load(open(f'{root}/identity.json', encoding='utf-8'))
    names = ident['position_names']
    files = sorted(glob.glob(f'{feat}/*.npz'))
    res = {'n_bodies': len(files), 'cells': []}
    print(f'check J: {len(files)} bodies; R_X = D(other two)/mean(D(X,other)); closest = share of bodies per closest pair')
    for mode in ('default', 'preserve_false'):
        for turn in ('reply', 'attr'):
            D = {m: {p: [] for p in PAIRS} for m in ('raw', 'cos')}
            for f in files:
                with np.load(f) as z:
                    X = [z[f'{mode}__{turn}__{r}'].astype(np.float32) for r in ROLES]
                for m in D:
                    for p, (i, j) in PAIRS.items():
                        D[m][p].append(dist(X[i], X[j], m))
            for m in D:
                for p in PAIRS:
                    D[m][p] = np.stack(D[m][p])  # [N, 64, P]
            curves = res.setdefault('curves', {}).setdefault(f'{mode}__{turn}', {})
            for pos in names:
                k = names.index(pos)
                per_layer = []
                for l in range(D['raw']['AU'].shape[1]):
                    d = np.stack([D['raw'][p][:, l, k] for p in PAIRS], -1)
                    closest = np.bincount(d.argmin(-1), minlength=3) / len(d)
                    per_layer.append({'R_A': float(np.mean(d[:, 2] / ((d[:, 0] + d[:, 1]) / 2))),
                                      'R_U': float(np.mean(d[:, 1] / ((d[:, 0] + d[:, 2]) / 2))),
                                      'R_S': float(np.mean(d[:, 0] / ((d[:, 1] + d[:, 2]) / 2))),
                                      'closest': {'AU': float(closest[0]), 'AS': float(closest[1]), 'US': float(closest[2])}})
                curves[pos] = per_layer
            print(f'\n==== mode {mode}, turn {turn}')
            for pos in SHOW_POS:
                k = names.index(pos)
                print(f'  -- {pos}:  layer | raw closest AU/AS/US | raw median R_A R_U R_S | cos closest AU/AS/US')
                for l in SHOW_LAYERS:
                    row = {}
                    for m in ('raw', 'cos'):
                        d = np.stack([D[m][p][:, l, k] for p in PAIRS], -1)  # [N, 3] AU AS US
                        closest = np.bincount(d.argmin(-1), minlength=3) / len(d)
                        RA = d[:, 2] / ((d[:, 0] + d[:, 1]) / 2)
                        RU = d[:, 1] / ((d[:, 0] + d[:, 2]) / 2)
                        RS = d[:, 0] / ((d[:, 1] + d[:, 2]) / 2)
                        row[m] = {'closest': closest.tolist(), 'R': [float(np.median(RA)), float(np.median(RU)), float(np.median(RS))]}
                    res['cells'].append({'mode': mode, 'turn': turn, 'pos': pos, 'layer': l, **row})
                    rc, rr, cc = row['raw']['closest'], row['raw']['R'], row['cos']['closest']
                    print(f'     L{l:<3d} | {rc[0]:.2f} {rc[1]:.2f} {rc[2]:.2f} | {rr[0]:.2f} {rr[1]:.2f} {rr[2]:.2f} | {cc[0]:.2f} {cc[1]:.2f} {cc[2]:.2f}')
    # decision-point readouts
    js, base, agree, masses = {}, {}, [], []
    for f in sorted(glob.glob(f'{root}/bodies/*.json')):
        d = json.load(open(f, encoding='utf-8'))
        for c in d['conditions']:
            key = (c['mode'], c['turn'])
            for p, v in c['decision_js_by_layer'].items():
                js.setdefault(key, {}).setdefault(p, []).append(v)
            for r, info in c['roles'].items():
                agree.append(info['lens_l63_vs_native_maxabs'])
                if c['turn'] in ('attr', 'attr2'):
                    pr = {L: math.exp(v) for L, v in info['native_letter_logprob'].items()}
                    order = ('self', 'user', 'system') if c['turn'] == 'attr' else ('system', 'self', 'user')
                    val = dict(zip(order, (pr['A'], pr['B'], pr['C'])))
                    base.setdefault((c['mode'], r), []).append([val['self'], val['user'], val['system']])
                    masses.append(pr['A'] + pr['B'] + pr['C'])
    print(f'\nfinal-layer lens vs native logits: max |diff| = {max(agree):.2e} over {len(agree)} forwards')
    print(f'attribution question: option-letter mass min {min(masses):.4f}, mean {sum(masses)/len(masses):.4f} over {len(masses)} forwards')
    res['lens_l63_maxabs'] = max(agree)
    res['attr_letter_mass'] = {'min': min(masses), 'mean': sum(masses) / len(masses), 'n': len(masses)}
    print('decision-point JS (nats), mean over bodies, layers 0/23/39/55/63:')
    res['decision_js'] = {}
    for key, v in js.items():
        line = []
        for p, arr in v.items():
            a = np.mean(np.array(arr), 0)
            res['decision_js'][f'{key[0]}__{key[1]}__{p}'] = a.tolist()
            line.append(f'{p} ' + '/'.join(f'{a[l]:.3f}' for l in (0, 23, 39, 55, 63)))
        print(f'  {key[0]:14s} {key[1]:5s} ' + '   '.join(line))
    print('attribution question (both option orders pooled when present), native P per role:')
    res['attr_baseline'] = {}
    for (mode, r), v in sorted(base.items()):
        m = np.mean(np.array(v), 0)
        res['attr_baseline'][f'{mode}__{r}'] = m.tolist()
        print(f'  {mode:14s} {r:9s} P(self) {m[0]:.3f}  P(user) {m[1]:.3f}  P(system) {m[2]:.3f}  n={len(v)}')
    json.dump(res, open(out_json, 'w'), indent=1)
    return res


def default_part(res, out_json):
    """the default-mode part of the native summary (curves, cells, decision JS and baselines of mode 'default'; the lens and
    letter-mass entries cover both modes)"""
    J = {'n_bodies': res['n_bodies'], 'cells': [c for c in res['cells'] if c['mode'] == 'default'],
         'curves': {k: v for k, v in res['curves'].items() if k.startswith('default__')},
         'lens_l63_maxabs': res['lens_l63_maxabs'], 'attr_letter_mass': res['attr_letter_mass'],
         'decision_js': {k: v for k, v in res['decision_js'].items() if k.startswith('default__')},
         'attr_baseline': {k: v for k, v in res['attr_baseline'].items() if k.startswith('default__')}}
    json.dump(J, open(out_json, 'w'), indent=1)


def numbers(j_json):
    """Print the check-J numbers quoted in the paper (from the native summary JSON)."""
    J = json.load(open(j_json, encoding='utf-8'))
    print('bodies', J['n_bodies'])
    C = J['curves']

    def row(key, pos, layers):
        out = []
        for l in layers:
            c = C[key][pos][l]
            r = min('AUS', key=lambda k: c[f'R_{k}'])
            out.append(f"L{l}: odd {r} R {c[f'R_{r}']:.2f} AU {c['closest']['AU']:.2f} AS {c['closest']['AS']:.2f} US {c['closest']['US']:.2f}")
        return out

    LAY = [0, 1, 2, 3, 5, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
    for key in ('preserve_false__reply', 'default__reply', 'preserve_false__attr', 'default__attr'):
        for pos in ('decision', 'turn_newline', 'body_mean', 'body_im_end', 'header_newline'):
            print(f'\n## {key} / {pos}')
            for line in row(key, pos, LAY):
                print('  ', line)
    print('\n## decision JS (mean over bodies) at L47/L55/L63')
    for k, v in J['decision_js'].items():
        print(f'  {k:32s}', ' '.join(f'{v[l]:.3f}' for l in (47, 55, 63)), '| max over L0-L47 %.3f' % max(v[:48]))
    print('\n## attribution baselines (both option orders)')
    for k, v in J['attr_baseline'].items():
        print(f'  {k:28s} self {v[0]:.3f} user {v[1]:.3f} system {v[2]:.3f}')


LAYERS = [0, 15, 31, 47, 63]


def ratios(H, metric):
    """H: dict role -> [N, 64, d]; per body and layer R_A, R_U, R_S."""
    au, as_, us = dist(H['assistant'], H['user'], metric), dist(H['assistant'], H['system'], metric), dist(H['user'], H['system'], metric)
    return us / ((au + as_) / 2), as_ / ((au + us) / 2), au / ((as_ + us) / 2), np.stack([au, as_, us], -1)


def onset(j_root, feat, j_summary, out_json, source):
    """j_root: identity.json and bodies/ of the run; feat: its features/ folder; source: recorded as the input folder"""
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
        with np.load(Path(feat) / f'{bid}.npz') as z:
            for r in ROLES:
                H[r].append(z[f'default__reply__{r}'][:, k].astype(np.float32))
    H = {r: np.stack(v) for r, v in H.items()}          # [150, 64, d]
    split = np.array([meta[b] for b in ids])
    cat = np.array([b.rsplit('_', 1)[0] for b in ids])
    disc, conf = split == 'discovery', split == 'confirmation'
    out = {'source': source, 'turn': 'default__reply', 'position': 'decision', 'n_heldout': int(conf.sum()),
           'n_discovery': int(disc.sum()), 'layers': LAYERS, 'points': {}, 'heldout': {}, 'direction_consistency': {}}
    # self-check against the J summary (all bodies, raw L2, mean over bodies)
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


def main(argv=None):
    import contextlib
    import io
    args = arguments(__doc__, features=True).parse_args(argv)
    r = Roots(args)
    tgz = r.res(C150 + '/raw/compliant_150_per_body.tgz')
    feat = r.feat(C150 + '/raw/attrgeo150/features')
    msg = need([tgz, feat], 'two_level_geometry')
    if msg:
        print(msg)
        return
    meta = extract(tgz, r.out('_work/attrgeo150'))
    S150, STL = r.out(C150 + '/summaries'), r.out(TL + '/summaries')
    res = native(meta, feat, str(S150 / 'native_geometry150_summary.json'))
    default_part(res, str(STL / 'J_native_geometry_default.json'))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        numbers(str(S150 / 'native_geometry150_summary.json'))
    print(buf.getvalue(), end='')
    (S150 / 'j150_numbers.txt').write_text(buf.getvalue(), encoding='utf-8')
    onset(meta, feat, str(S150 / 'native_geometry150_summary.json'), str(STL / 'J_onset_geometry.json'), shown(feat.parent))


if __name__ == '__main__':
    main()
