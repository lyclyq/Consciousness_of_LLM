"""Summary for check J (native renderings, both template modes): role geometry by position x layer, decision JS,
option-letter baselines. Distances: raw L2 and cosine distance between role states at the same position.
R_X = D(other two) / mean(D(X, other)); R_X < 2/3 means X is farther from both others than they are from each other. Table rows print median R; the stored
64-layer curves use the mean over bodies (as in the check-D summary).
'closest' = share of bodies in which each pair is the closest pair."""
import glob
import json
import math
import sys
import numpy as np

ROLES = ('assistant', 'user', 'system')
PAIRS = {'AU': (0, 1), 'AS': (0, 2), 'US': (1, 2)}
SHOW_POS = ['header_newline', 'body_mean', 'body_p5', 'body_im_end', 'turn_newline', 'decision']
SHOW_LAYERS = [0, 1, 2, 3, 5, 7, 11, 15, 23, 31, 39, 47, 55, 59, 63]
ORD = ('self', 'user', 'system')


def dist(a, b, metric):
    if metric == 'raw':
        return np.linalg.norm(a - b, axis=-1)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return 1.0 - (a * b).sum(-1) / (na * nb + 1e-12)


def main(root='attrgeo', out_json='analysis_pilot/native_geometry_summary.json'):
    ident = json.load(open(f'{root}/identity.json'))
    names = ident['position_names']
    files = sorted(glob.glob(f'{root}/features/*.npz'))
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
        d = json.load(open(f))
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


if __name__ == '__main__':
    main(*sys.argv[1:])
