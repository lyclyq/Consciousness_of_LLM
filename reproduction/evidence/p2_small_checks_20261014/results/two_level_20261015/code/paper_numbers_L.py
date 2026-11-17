"""Numbers quoted in the paper from check L (template default, non-thinking mode, 150 bodies).
usage: paper_numbers_L.py ROOT ANALYSIS_DIR OUT_JSON"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np


def decode(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


root, ana, out_json = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
S = json.load(open(ana / 'summary.json'))
R = S['results']['native']
out = {}
# Table: held-out (confirmation) bodies
rows = [json.loads(l) for l in open(ana / 'per_body_rows.jsonl', encoding='utf-8')]
by = defaultdict(list)
for r in rows:
    by[(r['split'], r['key'])].append(r)
table = {}
for key in ('FFN__L00', 'Q__L01', 'residual__L00'):
    c = R['confirmation'][key]
    rs = by[('confirmation', key)]
    table[key] = {'raw_R_A': c['raw']['R_A']['mean'], 'ci': c['raw']['R_A'].get('ci95'), 'cos_R_A': c['cos']['R_A']['mean'],
                  'z_R_A': c['z']['R_A']['mean'], 'drop_R_A': c['drop1pct']['R_A']['mean'],
                  'n_R_A_lt_2_3': int(sum(r['dist']['raw']['R_A'] < 2 / 3 for r in rs)), 'n_US_closest': int(sum(r['dist']['raw']['closest'] == 'US' for r in rs)),
                  'n_bodies': len(rs), 'disc_n_R_A_lt_2_3': int(sum(r['dist']['raw']['R_A'] < 2 / 3 for r in by[('discovery', key)]))}
out['table'] = table
# depth profile, residual, all variants: is Assistant singled out at every layer?
prof = {}
for v in ('raw', 'cos', 'z', 'drop1pct'):
    ra = [R['confirmation'][f'residual__L{l:02d}'][v]['R_A']['mean'] for l in range(64)]
    us = [R['confirmation'][f'residual__L{l:02d}'][v]['closest_fraction']['US'] for l in range(64)]
    prof[v] = {'R_A_min': min(ra), 'R_A_max': max(ra), 'US_closest_min': min(us), 'layers_US_closest_all': sum(u == 1.0 for u in us)}
out['residual_profile'] = prof
# R_A < 2/3 in every held-out body at which residual layers?
lt = []
for l in range(64):
    rs = by[('confirmation', f'residual__L{l:02d}')]
    lt.append(int(sum(r['dist']['raw']['R_A'] < 2 / 3 for r in rs)))
out['residual_R_A_lt_2_3_per_layer'] = lt
# all components: any site where U-S is not the closest pair in all held-out bodies?
comp = {}
for c in ('residual', 'FFN', 'Q', 'K', 'V'):
    us = [R['confirmation'][f'{c}__L{l:02d}']['raw']['closest_fraction']['US'] for l in range(64)]
    comp[c] = {'layers_all_bodies_US_closest': sum(u == 1.0 for u in us), 'min_share': min(us)}
out['components_US_closest'] = comp
# direction consistency (held-out body cosine) and positive alignment count over 320 sites
dc = {}
for l in (0, 3, 23, 39, 63):
    c = R['confirmation'][f'residual__L{l:02d}']
    dc[l] = {x: c[f'off_{x}']['body_cos']['mean'] for x in 'AUS'}
out['direction_consistency_residual'] = dc
pos = sum(R['confirmation'][f'{c}__L{l:02d}']['off_A']['body_cos']['mean'] > 0 for c in ('residual', 'FFN', 'Q', 'K', 'V') for l in range(64))
out['positive_heldout_alignment_sites'] = int(pos)
neg = [f'{c}__L{l:02d}' for c in ('residual', 'FFN', 'Q', 'K', 'V') for l in range(64) if R['confirmation'][f'{c}__L{l:02d}']['off_A']['body_cos']['mean'] <= 0]
out['nonpositive_sites'] = neg
# header newline (identical for all bodies) and embeddings
nl = S['header_geometry']['native']['newline_pos2']
out['newline_R'] = {l: {k: nl[f'residual__L{l:02d}']['raw'][f'R_{k}'] for k in 'AUS'} for l in (0, 1, 3, 7, 11, 23, 39, 63)}
out['embedding'] = S['embedding_geometry']['native']
# content-prediction invariance: native next-token JS at body positions (layer 63) and P(<|im_end|>) at the last body token
js = S['lens_pair_js']['native']
out['body_js_L63'] = {k: js[f'mean_JS_{k}'][63] for k in ('AU', 'AS', 'US')}
p_end = []
for f in glob.glob(str(root / 'lens' / '*.npz')):
    with np.load(f) as z:
        p_end.append(np.exp(z['observed_next_logprob'][:, 19, 63]))
out['p_im_end_last_body_token_min'] = float(np.min(p_end))
# offset size relative to the residual norm at L63 (body positions)
ratios = []
for f in sorted(glob.glob(str(root / 'features' / '*__assistant.npz')))[:150]:
    bid = Path(f).name.split('__')[0]
    h = {r: decode(np.load(root / 'features' / f'{bid}__{r}.npz')['residual__L63'])[2:] for r in ('assistant', 'user', 'system')}
    o = h['assistant'] - (h['user'] + h['system']) / 2
    ratios.append(float(np.median(np.linalg.norm(o, axis=-1) / np.linalg.norm(h['assistant'], axis=-1))))
out['L63_offset_over_norm_median'] = float(np.median(ratios))
json.dump(out, open(out_json, 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:6000])
