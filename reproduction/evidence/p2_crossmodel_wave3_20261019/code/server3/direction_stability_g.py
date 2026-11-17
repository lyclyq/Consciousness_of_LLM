"""Direction stability of the self-index offset from the stored G states (CPU only, user 10-23: align the gpt-oss-20b analyses
with those of Qwen3.8-27B). For every body, the own-message offset o = h_A - (h_U + h_S)/2 of the residual stream at each layer,
on message content and at the response onset (the per-body means saved by crossmodel_suite.py phase G); the discovery direction
u = mean of o over the 75 discovery bodies; for each of the 75 held-out bodies, cos(o, u) per layer. Reported: median cosine by
layer, its minimum and maximum, and the share of held-out bodies with a positive cosine. The paper's Qwen3.8-27B figure (0.98 at
L3, above 0.77 at every layer) is reproduced from the same files as a check (0.99 at L3, at least 0.80).
usage: direction_stability_g.py OUT.json G_DIR [G_DIR ...]"""
import glob
import json
import sys

import numpy as np

SPLITS = '/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl'


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    spl = {json.loads(l)['id']: json.loads(l)['split'] for l in open(SPLITS, encoding='utf-8') if l.strip()}
    res = {}
    for d in dirs:
        fs = sorted(glob.glob(d.rstrip('/') + '/G/*.npz'))
        res[d] = {}
        for pos in ('content', 'onset'):
            O = {}
            for f in fs:
                z = np.load(f)
                O[f.split('/')[-1][:-4]] = z[pos + '_assistant'].astype(np.float64) - (z[pos + '_user'].astype(np.float64) + z[pos + '_system'].astype(np.float64)) / 2
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


if __name__ == '__main__':
    main()
