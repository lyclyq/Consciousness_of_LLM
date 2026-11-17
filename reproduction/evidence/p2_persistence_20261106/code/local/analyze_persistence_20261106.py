"""Analysis of the persistence pilot (PROTOCOL_zh.md, sections 4-5, with the clarifications of 11-06 in its revision log).
Held-out (confirmation) bodies only; the resampling unit is the held-out wording family (15), 2000 resamples.
  M1  index at length k: per layer, the share of held-out bodies in which R_A = D_US / ((D_AU + D_AS) / 2) is the smallest of
      R_A, R_U, R_S (the paper's ratios); index layers = layers beyond the first third with share >= 0.8. Evaluated at
      k = 1..5 (every body has >= 5 clauses) and at the full body; other k are reported.
  M2  size kept: sqrt(C_full / C_1), C = squared norm of the mean A-U readout difference across families (U-statistic of the
      family means; in a resample, pairs of copies of the same family are left out), median over the 6 kept layers.
      Reliable at k = 1 if the resampled C_1 (median over layers) has a 2.5% quantile > 0.
  M3  direction kept: cosine between the mean A-U differences at k = 1 and at the full body (same U-statistics), median over layers.
Descriptive: M2/M3 for A-S, U-S and the word pairs (lengths 1 -> full, frame before / after); total readout energy ratio;
on the body tokens of the full body, the shrink / turn decomposition (as explore_decay_decomp).
Verdict (main order; natural order where defined): P1 for the index settings, P2 for the no-index settings.
usage: analyze_persistence_20261106.py RUN_DIR OUT_JSON [--tag TAG]"""
import argparse
import json
from pathlib import Path
import numpy as np

INDEX = {'qwen38_27b', 'qwen38_27b_think', 'gpt_oss_20b_en', 'gpt_oss_120b_en', 'exaone4_32b_en', 'mimo_v26_distill_qwen9b',
         'r1_distill_llama70b', 'glm46v_flash', 'granite41_30b_en', 'phi4_en'}
NOINDEX = {'gemma4_31b_en', 'seed_oss_36b', 'ernie45_21b_a3b_thinking', 'mimo7b_rl_0530', 'moonlight16b_a3b_instruct',
           'apertus8b_en', 'olmo2_7b_en', 'mimo_vl7b_rl_2508'}
NB = 2000
KS = (1, 2, 3, 4, 5)


def ustats(MA, MB, counts):
    """MA, MB: (F, K, d) family means; counts: (R, F) copies per family. Returns C_A, C_B, X (R, K): squared norms of the mean
    of A, of B, and the inner product of the two means, each without the pairs of copies of one family."""
    GA = np.einsum('fkd,gkd->kfg', MA, MA)
    GB = np.einsum('fkd,gkd->kfg', MB, MB)
    GX = np.einsum('fkd,gkd->kfg', MA, MB)
    c = counts.astype(np.float64)
    den = c.sum(1) ** 2 - (c ** 2).sum(1)                          # (R,)
    def q(G):
        full = np.einsum('rf,kfg,rg->rk', c, G, c)
        diag = np.einsum('rf,kff->rk', c ** 2, G)
        return (full - diag) / den[:, None]
    return q(GA), q(GB), q(GX)


def summarize_pair(MA, MB, counts):
    CA, CB, X = ustats(MA, MB, counts)
    with np.errstate(invalid='ignore', divide='ignore'):
        size = np.sqrt(CB / CA)
        cos = X / np.sqrt(CA * CB)
    return CA, CB, size, cos


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [None, None]


def med(x, axis=-1):
    with np.errstate(invalid='ignore'):
        return np.nanmedian(x, axis=axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('out')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    run = Path(args.run)
    tag = args.tag or run.name
    ident = json.loads((run / 'identity.json').read_text(encoding='utf-8'))
    L, keep, order = ident['n_layers'], ident['keep_layers'], ident['order']
    metas = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(run.glob('*.json')) if p.name != 'identity.json']
    held = [m for m in metas if m['split'] == 'confirmation']
    fams = sorted({m['category'] for m in held})
    F = len(fams)
    rng = np.random.default_rng(20261106)
    counts = np.stack([np.bincount(rng.integers(0, F, F), minlength=F) for _ in range(NB)])
    full_counts = np.ones((1, F), int)
    Z = {m['id']: dict(np.load(run / f'{m["id"]}.npz')) for m in held}
    fam_idx = {f: [m['id'] for m in held if m['category'] == f] for f in fams}
    out = {'tag': tag, 'order': order, 'n_layers': L, 'keep_layers': keep, 'n_held_out': len(held), 'families': F,
           'n_bodies_on_disk': len(metas), 'group': 'index' if tag in INDEX else 'no_index' if tag in NOINDEX else None,
           'script_identity': {k: ident.get(k) for k in ('script_sha256', 'model', 'tmpl_kwargs', 'lang', 'fixed_date', 'natural_system_ok')}}
    first = int(np.ceil(L / 3))

    # ---- M1 ----
    def index_at(getk):
        R = []
        for m in held:
            k = getk(m)
            if k is None:
                continue
            d = Z[m['id']]['role_dist'][k].astype(np.float64)      # (L, 3): AU, AS, US
            au, as_, us = d[:, 0], d[:, 1], d[:, 2]
            R.append(np.stack([us / ((au + as_) / 2), as_ / ((au + us) / 2), au / ((as_ + us) / 2)], -1))
        if not R:
            return None
        R = np.array(R)
        if not np.isfinite(R).all():
            return {'n_bodies': len(R), 'defined': False}
        share = (R.argmin(-1) == 0).mean(0)                         # (L,)
        return {'n_bodies': len(R), 'defined': True, 'index_layers_beyond_first_third': int((share[first:] >= 0.8).sum()),
                'layers_beyond_first_third': L - first, 'index_layers_all': int((share >= 0.8).sum()),
                'share_A_lowest_median_beyond_first_third': float(np.median(share[first:]))}
    m1 = {}
    for k in KS:
        m1[str(k)] = index_at(lambda m, k=k: k if m['n_clauses'] >= k else None)
    m1['full'] = index_at(lambda m: m['n_clauses'])
    m1['empty'] = index_at(lambda m: 0)
    for k in range(6, 11):
        r = index_at(lambda m, k=k: k if m['n_clauses'] >= k else None)
        if r:
            m1[str(k)] = r
    out['M1'] = m1

    # ---- M2 / M3 (role pairs) ----
    def fam_means(get):
        return np.stack([np.mean([get(Z[i]).astype(np.float64) for i in fam_idx[f]], 0) for f in fams])
    roles = {}
    for pi, pn in enumerate(('A-U', 'A-S', 'U-S')):
        M1_ = fam_means(lambda z, pi=pi: z['role_diff'][1, pi])
        Mf = fam_means(lambda z, pi=pi: z['role_diff'][len(z['role_diff']) - 1, pi])
        if not (np.isfinite(M1_).all() and np.isfinite(Mf).all()):
            roles[pn] = {'defined': False}
            continue
        C1, Cf, size, cos = summarize_pair(M1_, Mf, full_counts)
        bC1, _, bsize, bcos = summarize_pair(M1_, Mf, counts)
        e1 = np.mean([np.sum(Z[m['id']]['role_diff'][1, pi].astype(np.float64) ** 2, -1) for m in held], 0)
        ef = np.mean([np.sum(Z[m['id']]['role_diff'][-1, pi].astype(np.float64) ** 2, -1) for m in held], 0)
        roles[pn] = {'defined': True, 'M2_size_kept': float(med(size[0])), 'M2_ci95': ci(med(bsize)),
                     'M3_direction_cos': float(med(cos[0])), 'M3_ci95': ci(med(bcos)),
                     'C1_median': float(med(C1[0])), 'C1_ci95': ci(med(bC1)), 'C1_reliable': bool(ci(med(bC1))[0] is not None and ci(med(bC1))[0] > 0),
                     'total_energy_ratio_sqrt': float(med(np.sqrt(ef / e1))),
                     'per_layer': {str(l): {'size': float(size[0, j]), 'cos': float(cos[0, j])} for j, l in enumerate(keep)}}
    out['roles'] = roles

    # ---- words (main order) ----
    if 'word_diff' in next(iter(Z.values())):
        words = {}
        for pi, w in enumerate(ident['words']):
            for si, pos in enumerate(ident['word_positions']):
                Ma = fam_means(lambda z, pi=pi, si=si: z['word_diff'][pi, si, 0])
                Mb = fam_means(lambda z, pi=pi, si=si: z['word_diff'][pi, si, -1])   # full body (index 2, or 1 in lean runs)
                C1, Cf, size, cos = summarize_pair(Ma, Mb, full_counts)
                _, _, bsize, bcos = summarize_pair(Ma, Mb, counts)
                words[f'{w[0]}_{pos}'] = {'pair': [w[1], w[2]], 'M2_size_kept': float(med(size[0])), 'M2_ci95': ci(med(bsize)),
                                          'M3_direction_cos': float(med(cos[0])), 'C1_median': float(med(C1[0]))}
        out['words'] = words
        # self minus the best-kept word at the same position as the model's own marker is decided per template, so report both
        out['A-U_minus_max_word_M2'] = {pos: (roles['A-U']['M2_size_kept'] - max(v['M2_size_kept'] for k, v in words.items() if k.endswith(pos)))
                                        if roles.get('A-U', {}).get('defined') else None for pos in ident['word_positions']}

    # ---- body tokens of the full body (descriptive) ----
    dec = {}
    names = ['A-U', 'A-S', 'U-S'] + ([f'{w[0]}_before' for w in ident['words']] if 'word_diff' in next(iter(Z.values())) else [])
    names = names[:next(iter(Z.values()))['dec_mF'].shape[0]]   # lean runs keep only the role pairs
    for qi, qn in enumerate(names):
        MF = fam_means(lambda z, qi=qi: z['dec_mF'][qi])
        ML = fam_means(lambda z, qi=qi: z['dec_mL'][qi])
        if not (np.isfinite(MF).all() and np.isfinite(ML).all()):
            continue
        CF, CL, size, cos = summarize_pair(MF, ML, full_counts)
        eF = np.mean([Z[m['id']]['dec_eF'][qi] for m in held], 0)
        eL = np.mean([Z[m['id']]['dec_eL'][qi] for m in held], 0)
        dec[qn] = {'total_ratio': float(med(np.sqrt(eL / eF))), 'common_ratio': float(med(size[0])), 'cos_first_last': float(med(cos[0])),
                   'share_first': float(med(CF[0] / eF)), 'share_last': float(med(CL[0] / eL))}
    out['body_tokens_full'] = dec

    # ---- verdict ----
    v = {}
    au = roles.get('A-U', {})
    m1_eval = [m1[str(k)] for k in KS] + [m1['full']]
    m1_defined = all(x and x.get('defined') for x in m1_eval)
    if out['group'] == 'index':
        p1_m1 = all(x['index_layers_beyond_first_third'] >= 1 for x in m1_eval) if m1_defined else None
        p1_m2 = bool(au.get('defined') and au['C1_reliable'] and au['M2_size_kept'] >= 0.8)
        v = {'prediction': 'P1', 'M1_index_at_every_length': p1_m1, 'M2_at_least_0.8_and_reliable': p1_m2,
             'holds': bool(p1_m2 and (p1_m1 is None or p1_m1)), 'M1_evaluable': m1_defined}
    elif out['group'] == 'no_index':
        p2 = all(x['index_layers_beyond_first_third'] == 0 for x in m1_eval) if m1_defined else None
        v = {'prediction': 'P2', 'M1_no_index_at_any_length': p2, 'holds': p2, 'M1_evaluable': m1_defined}
    out['verdict'] = v
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    au_s = (f"A-U M2 {au['M2_size_kept']:.3f} {[round(x, 3) for x in au['M2_ci95']]} M3 {au['M3_direction_cos']:.3f} "
            f"C1 reliable {au['C1_reliable']}") if au.get('defined') else 'A-U undefined'
    print(tag, order, f"held {len(held)}", au_s, '| M1', {k: (x['index_layers_beyond_first_third'] if x and x.get('defined') else None)
                                                      for k, x in m1.items() if k in ('1', '2', '3', '4', '5', 'full', 'empty')}, '|', v)


if __name__ == '__main__':
    main()
