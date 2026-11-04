"""Qwen3.8-27B, message content geometry (check L: template default, non-thinking mode, 150 bodies x 3 roles, 64 blocks x 5
feature families, 20 body positions): the summaries L_message_content_geometry.json and L_paper_numbers.json of the two-level
archive. Both need the per-layer features of that run (defgeo150: features/<body>__<word>.npz, lens/<body>.npz,
word_embeddings.npz, identity.json and bodies/), which are not in the results or feature package; the archived summaries are in
the results package and are what the figure scripts read.

analysis: the CPU-only geometry analysis below (its definitions are stored in the summary), run with --source pilot on the
  check-L folder; writes summary.json (= L_message_content_geometry.json) and per_body_rows.jsonl.
paper_numbers: the numbers quoted in the paper from that summary and the features (Table tab:self_early, the depth profile, the
  closest pair over all components, direction consistency, the header newline, the embeddings, the next-token JS, the size of the
  offset at L63) -> L_paper_numbers.json.

usage: message_geometry.py --defgeo DIR --out OUT [--workers N]
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TL  # noqa: E402

DEFINITIONS = """P2 small checks, CPU-only geometry analysis (read-only on inputs).

--source d3    : existing D3 features (150 bodies x assistant/user/system, 20 body positions).
--source pilot : outputs of p2_null_pilot.py (native + null header triplets, same 20 body
                 positions plus body-invariant header-word / newline positions).

Per triplet (members A,U,S; for null triplets A/U/S are frequency-matched stand-ins)
and per component-layer, per body:
  * three distances D_AU, D_AS, D_US (mean over the 20 positions) in four spaces:
    raw L2 (D3 definition), cosine distance, per-dimension z-scored L2 (sigma from
    discovery bodies), raw L2 with the top-1% variance dimensions removed;
  * symmetric ratios R_A=D_US/mean(D_AU,D_AS), R_U=D_AS/mean(D_AU,D_US), R_S=D_AU/mean(D_AS,D_US),
    closest pair, and the double inequality D_US<min(D_AU,D_AS);
  * offsets o_X = X - mean(other two), X in {A,U,S}; discovery-only direction; held-out
    body-mean cosine and position-mean cosine (D3 definitions).
Aggregation: template-group means (D3: category__split groups of 5 bodies), group bootstrap.
Descriptive only; no multiple-comparison-corrected confirmation claim; no causal claim.
"""

BLOCKS = 64
COMPONENTS = ('residual', 'FFN', 'Q', 'K', 'V')
KEYS = [f'{c}__L{l:02d}' for c in COMPONENTS for l in range(BLOCKS)]
VARIANTS = ('raw', 'cos', 'z', 'drop1pct')
PAIRS = (('A', 'U'), ('A', 'S'), ('U', 'S'))
CFG = {}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.replace(path)


def decode(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def body_conditions(body):
    """Return {condition_name: {key: float32 [P, d]}} and optional header arrays."""
    if CFG['source'] == 'd3':
        out = {}
        for role in ('assistant', 'user', 'system'):
            path = None
            for lane in ('demo__D3_role_lane1', 'demo__D3_role_lane2'):
                candidate = Path(CFG['root'])/lane/'features'/f"{body['id']}__{role}__D3_role.npz"
                if candidate.exists():
                    path = candidate
                    break
            if path is None:
                raise FileNotFoundError((body['id'], role))
            with np.load(path, allow_pickle=False) as saved:
                out[role] = {k: decode(saved[k]).astype(np.float64) for k in KEYS}
        return out, None
    out, header = {}, {}
    for word in CFG['words']:
        with np.load(Path(CFG['root'])/'features'/f"{body['id']}__{word}.npz", allow_pickle=False) as saved:
            full = {k: decode(saved[k]).astype(np.float64) for k in KEYS}
        out[word] = {k: v[2:] for k, v in full.items()}
        header[word] = {k: v[:2] for k, v in full.items()}
    return out, header


def cos_rows(x, y):
    num = (x*y).sum(-1)
    den = np.linalg.norm(x, axis=-1)*np.linalg.norm(y, axis=-1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-30)


def pass1(body):
    conds, _ = body_conditions(body)
    stats, offsets = {}, {}
    for key in KEYS:
        stack = np.concatenate([conds[c][key] for c in conds], axis=0)
        stats[key] = (stack.sum(0), (stack**2).sum(0), stack.shape[0])
        for t in CFG['triplets']:
            m = {slot: conds[t['members'][slot]][key] for slot in ('A', 'U', 'S')}
            for x in ('A', 'U', 'S'):
                others = [y for y in ('A', 'U', 'S') if y != x]
                offsets[(t['name'], x, key)] = (m[x]-(m[others[0]]+m[others[1]])/2).mean(0)
    return body['id'], stats, offsets


def distances(m, key, sigma, keep):
    result = {}
    for variant in VARIANTS:
        d = {}
        for x, y in PAIRS:
            a, b = m[x], m[y]
            if variant == 'raw':
                v = np.linalg.norm(a-b, axis=-1).mean()
            elif variant == 'cos':
                v = (1-cos_rows(a, b)).mean()
            elif variant == 'z':
                v = np.linalg.norm((a-b)/sigma[key], axis=-1).mean()
            else:
                v = np.linalg.norm((a-b)[:, keep[key]], axis=-1).mean()
            d[x+y] = float(v)
        au, as_, us = d['AU'], d['AS'], d['US']
        result[variant] = {'D_AU': au, 'D_AS': as_, 'D_US': us,
                           'R_A': us/((au+as_)/2) if au+as_ > 0 else None,
                           'R_U': as_/((au+us)/2) if au+us > 0 else None,
                           'R_S': au/((as_+us)/2) if as_+us > 0 else None,
                           'closest': min(d, key=d.get), 'double_US': bool(us < min(au, as_))}
    return result


def pass2(body):
    conds, header = body_conditions(body)
    sigma, keep, direction = CFG['sigma'], CFG['keep'], CFG['direction']
    rows = []
    for t in CFG['triplets']:
        for key in KEYS:
            m = {slot: conds[t['members'][slot]][key] for slot in ('A', 'U', 'S')}
            rec = {'body': body['id'], 'split': body['split'], 'group': body['template_group'],
                   'category': body['category'], 'triplet': t['name'], 'key': key,
                   'dist': distances(m, key, sigma, keep)}
            for x in ('A', 'U', 'S'):
                others = [y for y in ('A', 'U', 'S') if y != x]
                per_token = m[x]-(m[others[0]]+m[others[1]])/2
                axis = direction[(t['name'], x, key)]
                rec['off_'+x] = {'body_cos': float(cos_rows(per_token.mean(0)[None], axis[None])[0]),
                                 'pos_cos': float(cos_rows(per_token, np.broadcast_to(axis, per_token.shape)).mean()),
                                 'first_cos': float(cos_rows(per_token[:1], axis[None])[0]),
                                 'last_cos': float(cos_rows(per_token[-1:], axis[None])[0]),
                                 'offset_l2': float(np.linalg.norm(per_token, axis=-1).mean())}
            rows.append(rec)
    return rows


def init(cfg):
    CFG.update(cfg)


def grouped(values, seed=0, draws=1000):
    groups = defaultdict(list)
    for g, v in values:
        if v is not None:
            groups[g].append(v)
    means = np.asarray([np.mean(v) for v in groups.values()], dtype=np.float64)
    if len(means) == 0:
        return None
    out = {'mean': float(means.mean()), 'groups': int(len(means))}
    if len(means) >= 3:
        rng = np.random.default_rng(seed)
        boot = means[rng.integers(0, len(means), size=(draws, len(means)))].mean(-1)
        out['ci95'] = [float(q) for q in np.quantile(boot, [.025, .975])]
    return out


def aggregate(rows, triplets):
    summary = {}
    for t in triplets:
        tname = t['name']
        sub = [r for r in rows if r['triplet'] == tname]
        for split in ('confirmation', 'discovery', 'all'):
            part = sub if split == 'all' else [r for r in sub if r['split'] == split]
            by_key = defaultdict(list)
            for r in part:
                by_key[r['key']].append(r)
            for key, rs in by_key.items():
                cell = summary.setdefault(tname, {}).setdefault(split, {}).setdefault(key, {})
                for variant in VARIANTS:
                    vc = cell.setdefault(variant, {})
                    for name in ('R_A', 'R_U', 'R_S', 'D_AU', 'D_AS', 'D_US'):
                        vc[name] = grouped([(r['group'], r['dist'][variant][name]) for r in rs])
                    vc['closest_fraction'] = {p: float(np.mean([r['dist'][variant]['closest'] == p for r in rs]))
                                              for p in ('AU', 'AS', 'US')}
                    vc['double_US_fraction'] = float(np.mean([r['dist'][variant]['double_US'] for r in rs]))
                if split != 'discovery':
                    for x in ('A', 'U', 'S'):
                        cell['off_'+x] = {m: grouped([(r['group'], r['off_'+x][m]) for r in rs])
                                         for m in ('body_cos', 'pos_cos', 'first_cos', 'last_cos', 'offset_l2')}
    return summary


def header_geometry(cfg, triplets):
    """Body-invariant header-word (input pos 1) and newline (pos 2) states, pilot only."""
    body = cfg['bodies'][0]
    _, header = body_conditions(body)
    out = {}
    for t in triplets:
        for pos_i, pos_name in ((0, 'header_word_pos1'), (1, 'newline_pos2')):
            for key in KEYS:
                m = {s: header[t['members'][s]][key][pos_i] for s in ('A', 'U', 'S')}
                d = {x+y: float(np.linalg.norm(m[x]-m[y])) for x, y in PAIRS}
                c = {x+y: float(1-cos_rows(m[x][None], m[y][None])[0]) for x, y in PAIRS}
                rec = {'raw': d, 'cos': c}
                for v in ('raw', 'cos'):
                    au, as_, us = rec[v]['AU'], rec[v]['AS'], rec[v]['US']
                    rec[v]['R_A'] = us/((au+as_)/2) if au+as_ > 0 else None
                    rec[v]['R_U'] = as_/((au+us)/2) if au+us > 0 else None
                    rec[v]['R_S'] = au/((as_+us)/2) if as_+us > 0 else None
                out.setdefault(t['name'], {}).setdefault(pos_name, {})[key] = rec
    return out


def analysis(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=('d3', 'pilot'), required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--bodies', default='')
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.source == 'd3':
        bodies = [json.loads(l) for l in Path(args.bodies).read_text(encoding='utf-8').splitlines() if l.strip()]
        triplets = [{'name': 'native', 'members': {'A': 'assistant', 'U': 'user', 'S': 'system'}}]
        words = ['assistant', 'user', 'system']
    else:
        identity = json.loads((Path(args.root)/'identity.json').read_text(encoding='utf-8'))
        triplets, words = identity['triplets'], identity['words']
        bodies = []
        for bid in identity['bodies']:
            path = Path(args.root)/'bodies'/f'{bid}.json'
            if path.exists():
                meta = json.loads(path.read_text(encoding='utf-8'))
                bodies.append({'id': bid, 'split': meta['split'], 'category': meta['category'],
                               'template_group': meta['template_group']})
    cfg = {'source': args.source, 'root': args.root, 'triplets': triplets, 'words': words, 'bodies': bodies}
    discovery = [b for b in bodies if b['split'] == 'discovery']
    with Pool(args.workers, initializer=init, initargs=(cfg,)) as pool:
        s1, s2, n = {}, {}, {}
        sums = defaultdict(lambda: 0)
        counts = defaultdict(int)
        for _, stats, offsets in pool.imap_unordered(pass1, discovery):
            for key, (a, b, c) in stats.items():
                s1[key] = s1.get(key, 0)+a
                s2[key] = s2.get(key, 0)+b
                n[key] = n.get(key, 0)+c
            for k, v in offsets.items():
                sums[k] = sums[k]+v
                counts[k] += 1
    sigma, keep = {}, {}
    for key in KEYS:
        mean = s1[key]/n[key]
        var = np.maximum(s2[key]/n[key]-mean**2, 0)
        sd = np.sqrt(var)
        sigma[key] = sd+1e-3*np.median(sd)+1e-12
        cut = np.quantile(var, .99)
        keep[key] = var < cut if (var >= cut).sum() < len(var) else np.ones_like(var, dtype=bool)
    direction = {k: sums[k]/counts[k] for k in sums}
    cfg.update(sigma=sigma, keep=keep, direction=direction)
    with Pool(args.workers, initializer=init, initargs=(cfg,)) as pool:
        rows = [r for rs in pool.imap_unordered(pass2, bodies) for r in rs]
    with (out/'per_body_rows.jsonl').open('w', encoding='utf-8') as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False)+'\n')
    summary = {'source': args.source, 'root': args.root, 'bodies': len(bodies),
               'discovery_bodies': len(discovery), 'triplets': triplets,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'definitions': DEFINITIONS, 'results': aggregate(rows, triplets), 'seconds': time.time()-t0}
    if args.source == 'pilot':
        init(cfg)
        summary['header_geometry'] = header_geometry(cfg, triplets)
        emb = np.load(Path(args.root)/'word_embeddings.npz', allow_pickle=False)
        names = [str(w) for w in emb['words']]
        geo = {}
        for space in ('input_embedding', 'output_unembedding'):
            mat = emb[space].astype(np.float64)
            for t in triplets:
                m = {s: mat[names.index(t['members'][s])] for s in ('A', 'U', 'S')}
                d = {x+y: float(np.linalg.norm(m[x]-m[y])) for x, y in PAIRS}
                c = {x+y: float(1-cos_rows(m[x][None], m[y][None])[0]) for x, y in PAIRS}
                geo.setdefault(t['name'], {})[space] = {
                    'raw': d, 'cos': c,
                    'R_A_raw': d['US']/((d['AU']+d['AS'])/2), 'R_A_cos': c['US']/((c['AU']+c['AS'])/2)}
        summary['embedding_geometry'] = geo
        lens = defaultdict(lambda: defaultdict(list))
        for b in bodies:
            with np.load(Path(args.root)/'lens'/f"{b['id']}.npz", allow_pickle=False) as saved:
                pj = saved['triplet_pair_js_nats']  # [T, 3, 20, 64]
            for ti, t in enumerate(triplets):
                means = pj[ti].mean(1)  # [3, 64]
                lens[t['name']]['AU'].append(means[0])
                lens[t['name']]['AS'].append(means[1])
                lens[t['name']]['US'].append(means[2])
        summary['lens_pair_js'] = {}
        for t in triplets:
            au, as_, us = (np.asarray(lens[t['name']][p]) for p in ('AU', 'AS', 'US'))
            summary['lens_pair_js'][t['name']] = {
                'mean_JS_AU': au.mean(0).tolist(), 'mean_JS_AS': as_.mean(0).tolist(),
                'mean_JS_US': us.mean(0).tolist(),
                'R_js_A': (us/((au+as_)/2+1e-30)).mean(0).tolist()}
    atomic(out/'summary.json', summary)
    print('done', len(rows), 'rows', round(time.time()-t0, 1), 's')



def paper_numbers(root, ana, out_json):
    """Numbers quoted in the paper from check L (template default, non-thinking mode, 150 bodies)."""
    S = json.load(open(ana / 'summary.json', encoding='utf-8'))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--defgeo', required=True, help='the check-L folder (features/, lens/, bodies/, identity.json, word_embeddings.npz)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    out = Path(os.path.abspath(a.out)) / TL / 'summaries'
    ana = out / 'message_geometry'
    analysis(['--source', 'pilot', '--root', a.defgeo, '--out', str(ana), '--workers', str(a.workers)])
    (out / 'L_message_content_geometry.json').write_bytes((ana / 'summary.json').read_bytes())
    paper_numbers(Path(a.defgeo), ana, str(out / 'L_paper_numbers.json'))


if __name__ == '__main__':
    main()
