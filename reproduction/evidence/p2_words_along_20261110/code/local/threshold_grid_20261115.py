"""Threshold robustness grid of the NMI version (11-15, Supplementary Table "Classification and statistics at three thresholds").
For each threshold of the self-index criterion (70%, 80% = the paper's, 90%) the groups of both orders are recomputed by rule
(classify_groups_by_rule_20261115.py), the final tables are regrouped (values unchanged, rows in the order build_final_tables
would give them), and the paper's statistics are rerun with the project code itself: group_decomposition_20261113.analyse (main
and natural order, Spearman with the group, one-sided permutation tests) and pooled_rank_test_20261114.run on the pooled rows of
fig_evidence_combined_20261114 (14 models measured in both orders). The 80% row must equal the files of the paper (assert).
Writes results/final_20261113/threshold_grid_20261115.json and .md. Read-only on all inputs.
usage (from projects/consciousness): uv run --offline --no-project --with numpy --with matplotlib python -I threshold_grid_20261115.py"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classify_groups_by_rule_20261115 as CG  # noqa: E402
import group_decomposition_20261113 as GD  # noqa: E402
import pooled_rank_test_20261114 as PRT  # noqa: E402  (imports fig_evidence_combined_20261114 as PRT.F)
from build_final_tables_20261113 import EXTRA  # noqa: E402

R = HERE.parents[1] / 'results/final_20261113'
OUT = R / 'threshold_grid_20261115'
BASE = json.loads((R / 'final_tables_20261113.json').read_text(encoding='utf-8'))
GO = ('neither', 'self-index only', 'self-reference')
LEVELS = ('neither', 'index-one', 'self-index only', 'selfref-one', 'self-reference')
NAN = lambda x: None if isinstance(x, float) and np.isnan(x) else x  # noqa: E731


def regrouped(groups):
    ft = copy.deepcopy(BASE)
    for o, key in (('main', 'groups'), ('natural', 'natural_groups')):
        by = {r['tag']: r for r in ft['rows'][o]}
        order = [t for g in ('self-reference', 'self-index only', 'neither') for t in groups[o][g]] + (EXTRA if o == 'main' else [])
        rows = []
        for t in order:
            if t in by:
                r = by[t]
                if t not in EXTRA:
                    r['group'] = next(g for g in groups[o] if t in groups[o][g])
                rows.append(r)
        assert len(rows) == len(ft['rows'][o])
        ft['rows'][o] = rows
        ft[key] = groups[o]
    return ft


def jsonable(x):
    return json.loads(json.dumps(x, default=NAN))


def main():
    out, changes = {}, []
    for thr in CG.THRESHOLDS:
        t = f'{int(round(thr * 100))}'
        C = CG.classify(thr)
        ft = regrouped(C['groups'])
        tests, _ = GD.analyse(ft)
        tests = jsonable(tests)
        PRT.F.FT = ft
        pooled = PRT.run(PRT.F.pooled_rows())
        if t == '80':   # the paper's threshold: the regrouped tables and every statistic equal the files of the paper
            assert ft['rows'] == BASE['rows'] and ft['groups'] == BASE['groups'] and ft['natural_groups'] == BASE['natural_groups']
            assert tests == json.loads((R / 'group_decomposition_20261113.json').read_text(encoding='utf-8'))['tests']
            assert pooled == json.loads((R / 'pooled_rank_test_20261114.json').read_text(encoding='utf-8'))
        mk, mr, nk, nr = tests['main']['kl'], tests['main']['retention'], tests['natural']['kl'], tests['natural']['retention']
        sk, sr = mk['log10 self / U||S (self-specific)'], mr['self - U-S (self-specific)']
        rk, rr = mk['log10 U||S / words (role-general)'], mr['U-S - words (role-general)']
        nsk, nsr = nk['log10 self / U||S (self-specific)'], nr['self - U-S (self-specific)']
        lev = {m['tag']: m['level'] for m in pooled['models']}
        S = C['summary']
        out[t] = {'threshold': thr,
                  'main_counts': S['main']['counts'], 'main_n_index': len(S['main']['with_index']),
                  'main_nesting_P': S['main']['nesting_P_exact'], 'main_nesting_P_formula': S['main']['nesting_P_formula'],
                  'natural_counts': S['natural']['counts'], 'natural_n_index': len(S['natural']['with_index']),
                  'selfref_without_index': {o: S[o]['selfref_without_index'] for o in ('main', 'natural')},
                  'main_kl_self_ext': {'rho': sk['spearman_group'], 'P': sk['p_spearman'], 'medians_ratio': {g: 10 ** sk['medians'][g] for g in GO}},
                  'main_share_self_ext': {'rho': sr['spearman_group'], 'P': sr['p_spearman'], 'medians': {g: sr['medians'][g] for g in GO}},
                  'main_role_general': {'kl_rho': rk['spearman_group'], 'kl_P': rk['p_spearman'], 'share_rho': rr['spearman_group'], 'share_P': rr['p_spearman']},
                  'natural_kl_self_ext': {'rho': nsk['spearman_group'], 'P': nsk['p_spearman'],
                                          'medians_ratio': {g: (None if nsk['medians'][g] is None else 10 ** nsk['medians'][g]) for g in GO}},
                  'natural_share_self_ext': {'rho': nsr['spearman_group'], 'P': nsr['p_spearman'], 'medians': {g: nsr['medians'][g] for g in GO}},
                  'pooled_kl_b': {'rho': pooled['columns']['b']['spearman'], 'P': pooled['columns']['b']['p_one_sided']},
                  'pooled_share_d': {'rho': pooled['columns']['d']['spearman'], 'P': pooled['columns']['d']['p_one_sided']},
                  'pooled_level_counts': {k: sum(v == k for v in lev.values()) for k in LEVELS}, 'pooled_levels': lev}
        names = {r['tag']: r['name'] for o in ('main', 'natural') for r in BASE['rows'][o]}
        for o, base in (('main', CG.GROUPS), ('natural', CG.NAT_GROUPS)):
            for ch in S[o]['changed_from_paper']:
                tg = ch['tag']
                src = CG.MAIN[tg] if o == 'main' else CG.NAT[tg]
                sh = CG.shares(src, o)
                q = C['models'][o][tg]['qualifying_layers_0based']
                changes.append({'threshold': t, 'order': o, 'model': names[tg], 'at_80': ch['paper'], 'here': ch['here'],
                                'deciding': {p: {i: round(sh[p][i], 3) for i in q[p]} for p in ('content', 'onset')}})
    f2 = lambda x: f'{x:+.2f}'  # noqa: E731
    L = ['# Threshold robustness grid (11-15)', '', 'Rows: threshold of the self-index criterion (80% is the paper\'s). Groups SR self-reference, IO self-index only, '
         'N neither. P one-sided permutation (seeds of the paper\'s scripts); nesting P exact.', '',
         '| threshold | main: with index / SR / IO / N | nesting P (main) | natural: with index / SR / IO / N | main KL self/ext rho (P) | '
         'main share kept own-ext rho (P) | role-general rho, KL / share kept | main medians KL self/ext (N / IO / SR) | '
         'main medians share kept own-ext (N / IO / SR) | natural rho KL (P) | natural rho share kept (P) | pooled rho KL (P) | '
         'pooled rho share kept (P) | pooled levels N / index one order / index both / SR one order / SR both |', '|' + '---|' * 15]
    for t, r in out.items():
        mc, nc, pc = r['main_counts'], r['natural_counts'], r['pooled_level_counts']
        m1, m2 = r['main_kl_self_ext'], r['main_share_self_ext']
        L.append(f"| {t}% | {r['main_n_index']} / {mc['self-reference']} / {mc['self-index only']} / {mc['neither']} | "
                 f"{r['main_nesting_P_formula']} = {r['main_nesting_P']:.4f} | {r['natural_n_index']} / {nc['self-reference']} / {nc['self-index only']} / "
                 f"{nc['neither']} | {m1['rho']:.2f} ({m1['P']:.4f}) | {m2['rho']:.2f} ({m2['P']:.4f}) | "
                 f"{r['main_role_general']['kl_rho']:.2f} / {r['main_role_general']['share_rho']:.2f} | "
                 + ' / '.join(f"{m1['medians_ratio'][g]:.2f}" for g in GO) + ' | ' + ' / '.join(f2(m2['medians'][g]) for g in GO) + ' | '
                 f"{r['natural_kl_self_ext']['rho']:.2f} ({r['natural_kl_self_ext']['P']:.4f}) | {r['natural_share_self_ext']['rho']:.2f} "
                 f"({r['natural_share_self_ext']['P']:.4f}) | {r['pooled_kl_b']['rho']:.2f} ({r['pooled_kl_b']['P']:.4f}) | "
                 f"{r['pooled_share_d']['rho']:.2f} ({r['pooled_share_d']['P']:.4f}) | "
                 + ' / '.join(str(pc[k]) for k in LEVELS) + ' |')
    L += ['', 'No model recognizes its own messages without a self-index at any threshold, in either order.', '',
          '| threshold | order | model | group at 80% | group at this threshold | deciding layers beyond L/3 (0-based index: share) |', '|---|---|---|---|---|---|']
    for c in changes:
        d = '; '.join(f"{p} " + ', '.join(f'{i}: {v:.3f}' for i, v in c['deciding'][p].items()) for p in ('content', 'onset') if c['deciding'][p])
        L.append(f"| {c['threshold']}% | {c['order']} | {c['model']} | {c['at_80']} | {c['here']} | {d} |")
    OUT.with_suffix('.json').write_text(json.dumps({'method': __doc__.split('usage')[0].strip(), 'rows': out, 'changes': changes},
                                                   ensure_ascii=False, indent=1, default=NAN), encoding='utf-8')
    OUT.with_suffix('.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
