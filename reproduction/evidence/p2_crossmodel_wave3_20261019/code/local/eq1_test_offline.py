"""Test of Equation 1 (paper section 'How Much of the Self-Index Is Shared') from the stored geometry of one model (G phase
of crossmodel_suite.py: per body and layer the content-mean state of each role). As in the paper: u = the mean Assistant
offset o = h_A - (h_U + h_S)/2 over the discovery bodies at each layer; on each held-out body the share of its offset that u
explains, 1 - |o - u|^2 / |o|^2; and whether the own message is still apart (R_A the smallest of the three distance ratios)
after u is subtracted from h_A, against before. usage: eq1_test_offline.py G_DIR OUT.json [content|onset]"""
import glob
import json
import os
import sys

import numpy as np

gdir, outp = sys.argv[1], sys.argv[2]
kind = sys.argv[3] if len(sys.argv) > 3 else 'content'
recs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(os.path.join(gdir, '*.json')))]
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
