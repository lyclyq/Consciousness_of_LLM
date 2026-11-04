"""Cross-model comparison: the per-model tables from which the two levels of every model are read (Table tab:self_crossmodel,
Figures self_crossmodel and self_crossmodel_depth, the classification of Section sec:res_crossmodel). One set of rules for all
archives:
  attribution_non_thinking_body_level.csv  body-level mean P(you) for the model's own, User and System messages, first two option
                                           orders ('2_orders_as_paper') and all three ('3_orders'); own minus the larger external
                                           probability per body, the bodies in which it is positive and a two-sided sign test
  thinking_attribution_counts.csv          per source: finished items, answers, you_share = answered 'you' / finished
  geometry_summary.csv                     layers at which Assistant has the smallest R_A in >= 80% / all held-out bodies, the
                                           median R_A; for the four-stream models (hyper-connection streams) the mean of the
                                           per-stream ratios and the strongest stream
  v4_*.csv                                 the same for the models added later (Qwen3.5-27B, Granite-4.1-30B, Granite-4.2-30B)
                                           and the 900-item thinking runs (Qwen3.8-Flash-Next, Qwen3.6-27B, Seed-OSS-36B), with the
                                           layers of the last third (the last floor(L/3) layers) at which the own message is
                                           singled out in >= 80%
  v5_olmo_*.csv                            the same for OLMo-2-1124-7B-Instruct and OLMo-2-1124-13B-Instruct (English, no
                                           thinking mode)
Outputs: evidence/p2_crossmodel_8models_20261018/stats/{attribution_non_thinking_body_level,thinking_attribution_counts,
geometry_summary}.csv (first eight models) and evidence/p2_crossmodel_wave3_20261019/stats/ (the other models).

--body-level NAME=DIR ... prints, for record folders of the main probe (DIR relative to the results root), the body-level
self-reference over the three option orders (P(you) per role, own minus the larger external, positive bodies, sign test, and
whether the top answer names the true source); it writes no file.

usage: crossmodel_classification.py --results RES --out OUT [--body-level NAME=DIR ...]
"""
import csv
import glob
import json
import os
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import M8, W3, Roots, arguments, run_path  # noqa: E402


def sign_p(k, n):
    t = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * t)


# ---------------------------------------------------------------------------------------------------------------------------
# first eight models (models/<model>/<mode>/<measure>/ of the eight-model archive)

def eight_models(A, S):
    """A: the eight-model archive in the results package; S: its output stats folder."""
    def write_csv(name, rows):
        if not rows:
            return
        with open(S / name, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def attribution_stats():
        out = []
        for d in sorted(glob.glob(str(A / 'models' / '*' / 'non_thinking' / 'geo_attr'))):
            model = Path(d).parts[-3]
            recs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(d + '/role150_*.json'))]
            if not recs or 'mean' not in recs[0]['roles']['assistant']:
                continue
            for label, idx in (('3_orders', [0, 1, 2]), ('2_orders_as_paper', [0, 1])):
                m = lambda r, role, key: float(np.mean([r['roles'][role]['by_order'][i][key] for i in idx]))  # noqa: E731
                pa = np.array([m(r, 'assistant', 'self') for r in recs]); pu = np.array([m(r, 'user', 'self') for r in recs])
                ps = np.array([m(r, 'system', 'self') for r in recs])
                dl = pa - np.maximum(pu, ps); k = int((dl > 0).sum())
                out.append(dict(model=model, orders=label, n_bodies=len(recs), p_you_own=round(pa.mean(), 4), p_you_user=round(pu.mean(), 4),
                                p_you_system=round(ps.mean(), 4), own_minus_max_other=round(dl.mean(), 4), bodies_positive=k,
                                sign_p=float(f'{sign_p(k, len(dl)):.3g}'),
                                p_system_as_user=round(float(np.mean([m(r, 'system', 'user') for r in recs])), 4)))
        write_csv('attribution_non_thinking_body_level.csv', out)
        return out

    def thinking_stats():
        out = []
        true = {'assistant': 'self', 'user': 'user', 'system': 'system'}
        for d in sorted(glob.glob(str(A / 'models' / '*' / 'thinking' / 'think_attr'))):
            model = Path(d).parts[-3]
            recs = [json.load(open(f, encoding='utf-8')) for f in glob.glob(d + '/role150_*.json')]
            for r in ('assistant', 'user', 'system'):
                rs = [x for x in recs if x['role'] == r]
                fin = [x for x in rs if x.get('finished')]
                c = Counter(x['on_answer'] for x in fin)
                out.append(dict(model=model, role=r, n_items=len(rs), n_finished=len(fin), answered_you=c.get('self', 0),
                                answered_user=c.get('user', 0), answered_system=c.get('system', 0), unparsed=c.get(None, 0),
                                correct_share=round(sum(x['on_answer'] == true[r] for x in fin) / max(1, len(fin)), 4),
                                you_share=round(c.get('self', 0) / max(1, len(fin)), 4)))
        write_csv('thinking_attribution_counts.csv', out)
        return out

    def geometry_stats():
        out = []
        for f in sorted(glob.glob(str(A / 'models' / '*' / '*' / 'geo*' / '_run_summary.json'))):
            p = Path(f)
            model, mode, measure = p.parts[-4], p.parts[-3], p.parts[-2]
            s = json.load(open(f, encoding='utf-8'))
            L = s.get('n_layers')
            for kind in ('content', 'onset'):
                if kind not in s:
                    continue
                k = s[kind]
                out.append(dict(model=model, mode=mode, measure=measure, position=kind, n_layers=L,
                                layers_A_apart_all_heldout=k['layers_A_lowest_all_bodies'], layers_A_apart_80pct=k['layers_A_lowest_80'],
                                median_R_A=round(float(np.median(k['mean_R'][0])), 4),
                                median_direction_cos=round(float(np.median(k['direction_cos_mean'])), 4)))
        write_csv('geometry_summary.csv', out)
        return out

    a = attribution_stats()
    t = thinking_stats()
    g = geometry_stats()
    print(f'eight-model archive: attribution rows {len(a)}, thinking rows {len(t)}, geometry rows {len(g)}')


# ---------------------------------------------------------------------------------------------------------------------------
# the five wave-3 models (Seed-OSS-36B, EXAONE-4.0.1-32B, gpt-oss-20b, gpt-oss-120b, GLM-5.3-Flash)

def wave3(A, S):
    NONTHINK = {'seed_oss_36b': run_path(A, 'suite/seed_oss_36b'), 'exaone4_32b': run_path(A, 'suite/exaone4_32b_en')}
    GEO = {('seed_oss_36b', 'non_thinking'): run_path(A, 'suite/seed_oss_36b'), ('seed_oss_36b', 'thinking'): run_path(A, 'suite_think/seed_oss_36b'),
           ('exaone4_32b', 'non_thinking'): run_path(A, 'suite/exaone4_32b_en'), ('exaone4_32b', 'thinking'): run_path(A, 'suite_think/exaone4_32b_en'),
           ('gpt_oss_20b', 'thinking'): run_path(A, 'suite_think/gpt_oss_20b_mxfp4_en'), ('gpt_oss_120b', 'thinking'): run_path(A, 'suite_think/gpt_oss_120b_en')}
    STREAMS = {('glm53_flash', 'thinking'): run_path(A, 'suite_think/glm53_flash_stream_geometry.json')}
    THINK = {'seed_oss_36b': run_path(A, 'thinking/seed_oss_36b'), 'exaone4_32b': run_path(A, 'thinking/exaone4_32b_en'),
             'gpt_oss_20b': run_path(A, 'thinking/gpt_oss_20b_mxfp4_en'), 'gpt_oss_120b': run_path(A, 'thinking/gpt_oss_120b_en'),
             'glm53_flash': A / 'runs' / 'api' / 'glm53_flash'}

    def write_csv(name, rows):
        S.mkdir(exist_ok=True)
        with open(S / name, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(name, len(rows), 'rows')

    out = []
    for model, d in NONTHINK.items():
        recs = [json.loads(Path(f).read_text(encoding='utf-8')) for f in sorted(glob.glob(str(d / 'G' / 'role150_*.json')))]
        for label, idx in (('3_orders', [0, 1, 2]), ('2_orders_as_paper', [0, 1])):
            m = lambda r, role, key: float(np.mean([r['roles'][role]['by_order'][i][key] for i in idx]))  # noqa: E731
            pa = np.array([m(r, 'assistant', 'self') for r in recs]); pu = np.array([m(r, 'user', 'self') for r in recs])
            ps = np.array([m(r, 'system', 'self') for r in recs])
            dl = pa - np.maximum(pu, ps); k = int((dl > 0).sum())
            out.append(dict(model=model, orders=label, n_bodies=len(recs), p_you_own=round(pa.mean(), 4), p_you_user=round(pu.mean(), 4),
                            p_you_system=round(ps.mean(), 4), own_minus_max_other=round(dl.mean(), 4), bodies_positive=k,
                            sign_p=float(f'{sign_p(k, len(dl)):.3g}'),
                            p_system_as_user=round(float(np.mean([m(r, 'system', 'user') for r in recs])), 4)))
    write_csv('attribution_non_thinking_body_level.csv', out)

    out = []
    true = {'assistant': 'self', 'user': 'user', 'system': 'system'}
    for model, d in THINK.items():
        recs = [json.loads(Path(f).read_text(encoding='utf-8')) for f in glob.glob(str(d / 'items' / 'role150_*.json'))]
        for r in ('assistant', 'user', 'system'):
            rs = [x for x in recs if x['role'] == r]
            fin = [x for x in rs if x.get('finished')]
            c = Counter(x['on_answer'] for x in fin)
            out.append(dict(model=model, role=r, n_items=len(rs), n_finished=len(fin), answered_you=c.get('self', 0),
                            answered_user=c.get('user', 0), answered_system=c.get('system', 0), unparsed=c.get(None, 0),
                            correct_share=round(sum(x['on_answer'] == true[r] for x in fin) / max(1, len(fin)), 4),
                            you_share=round(c.get('self', 0) / max(1, len(fin)), 4)))
    write_csv('thinking_attribution_counts.csv', out)

    out = []
    for (model, mode), d in GEO.items():
        s = json.loads((d / 'summary.json').read_text(encoding='utf-8'))
        L = s['n_layers']
        for kind in ('content', 'onset'):
            k = s[kind]
            out.append(dict(model=model, mode=mode, measure='geo', position=kind, n_layers=L, statistic='single_state',
                            layers_A_apart_all_heldout=k['layers_A_lowest_all_bodies'], layers_A_apart_80pct=k['layers_A_lowest_80'],
                            strongest_stream_80pct='', median_R_A=round(float(np.median(k['mean_R'][0])), 4)))
    for (model, mode), f in STREAMS.items():
        s = json.loads(Path(f).read_text(encoding='utf-8'))
        for kind in ('content', 'onset'):
            k = s[kind]
            L = len(k['mean_of_streams']['R_A_by_layer'])
            best = max(v['A_lowest_80'] for n, v in k.items() if n.startswith('stream'))
            out.append(dict(model=model, mode=mode, measure='geo', position=kind, n_layers=L, statistic='mean_of_streams',
                            layers_A_apart_all_heldout=k['mean_of_streams']['A_lowest_all_bodies'], layers_A_apart_80pct=k['mean_of_streams']['A_lowest_80'],
                            strongest_stream_80pct=best, median_R_A=round(k['mean_of_streams']['R_A_median'], 4)))
    write_csv('geometry_summary.csv', out)


# ---------------------------------------------------------------------------------------------------------------------------
# models added or completed later (v4 tables) and the two OLMo-2 models (v5 tables), with the rules above

def later_models(A, S, NONTHINK, THINK, prefix, GEO=None):
    """NONTHINK: model -> suite run (non-thinking geometry and attribution); THINK: model -> thinking run (or None); GEO: (model,
    mode) -> suite run of the geometry (default: the NONTHINK runs, non-thinking)."""
    GEO = GEO if GEO is not None else {(m, 'non_thinking'): d for m, d in NONTHINK.items()}

    def write_csv(name, rows):
        S.mkdir(exist_ok=True)
        with open(S / name, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(name, len(rows), 'rows')
        for r in rows:
            print('  ', r)

    out = []
    for model, d in NONTHINK.items():
        recs = [json.loads(Path(f).read_text(encoding='utf-8')) for f in sorted(glob.glob(str(d / 'G' / 'role150_*.json')))]
        recs = [r for r in recs if 'roles' in r]
        for label, idx in (('3_orders', [0, 1, 2]), ('2_orders_as_paper', [0, 1])):
            m = lambda r, role, key: float(np.mean([r['roles'][role]['by_order'][i][key] for i in idx]))  # noqa: E731
            pa = np.array([m(r, 'assistant', 'self') for r in recs]); pu = np.array([m(r, 'user', 'self') for r in recs])
            ps = np.array([m(r, 'system', 'self') for r in recs])
            dl = pa - np.maximum(pu, ps); k = int((dl > 0).sum())
            out.append(dict(model=model, orders=label, n_bodies=len(recs), p_you_own=round(pa.mean(), 4), p_you_user=round(pu.mean(), 4),
                            p_you_system=round(ps.mean(), 4), own_minus_max_other=round(dl.mean(), 4), bodies_positive=k,
                            sign_p=float(f'{sign_p(k, len(dl)):.3g}'),
                            p_system_as_user=round(float(np.mean([m(r, 'system', 'user') for r in recs])), 4)))
    write_csv(f'{prefix}attribution_non_thinking.csv', out)

    if THINK:
        out = []
        true = {'assistant': 'self', 'user': 'user', 'system': 'system'}
        for model, d in THINK.items():
            recs = [json.loads(Path(f).read_text(encoding='utf-8')) for f in glob.glob(str(d / 'items' / 'role150_*.json'))]
            for r in ('assistant', 'user', 'system'):
                rs = [x for x in recs if x['role'] == r]
                fin = [x for x in rs if x.get('finished')]
                c = Counter(x['on_answer'] for x in fin)
                out.append(dict(model=model, role=r, n_items=len(rs), n_finished=len(fin), answered_you=c.get('self', 0),
                                answered_user=c.get('user', 0), answered_system=c.get('system', 0), unparsed=c.get(None, 0),
                                correct_share=round(sum(x['on_answer'] == true[r] for x in fin) / max(1, len(fin)), 4),
                                you_share=round(c.get('self', 0) / max(1, len(fin)), 4)))
        write_csv(f'{prefix}thinking_counts.csv', out)

    out = []
    for (model, mode), d in GEO.items():
        s = json.loads((d / 'summary.json').read_text(encoding='utf-8'))
        L = s['n_layers']
        last = range(L - L // 3, L)
        for kind in ('content', 'onset'):
            k = s[kind]
            sh = k['share_A_lowest']
            lay80 = [i for i in range(L) if sh[i] >= 0.8]
            out.append(dict(model=model, mode=mode, position=kind, n_layers=L, layers_A_apart_all_heldout=k['layers_A_lowest_all_bodies'],
                            layers_A_apart_80pct=k['layers_A_lowest_80'], last_third_80pct=sum(i in last for i in lay80),
                            last_third_size=len(last), layers_80pct=' '.join(map(str, lay80)),
                            median_R_A=round(float(np.median(k['mean_R'][0])), 4)))
    write_csv(f'{prefix}geometry_summary.csv', out)


def v4_models(A, S):
    NONTHINK = {'qwen35_27b': run_path(A, 'suite/qwen35_27b'), 'granite41_30b': run_path(A, 'suite/granite41_30b_en'),
                'granite42_30b': run_path(A, 'suite/granite42_30b_en')}
    THINK = {'qwen38_flashnext': run_path(A, 'thinking/flashnext'), 'qwen36_27b': run_path(A, 'thinking/qwen36_27b'),
             'seed_oss_36b': run_path(A, 'thinking/seed_oss_36b'), 'qwen35_27b': run_path(A, 'thinking/qwen35_27b'),
             'granite42_30b': run_path(A, 'thinking/granite42_30b_en')}
    later_models(A, S, NONTHINK, THINK, 'v4_')


def olmo_models(A, S):
    NONTHINK = {'olmo2_7b': run_path(A, 'suite/olmo2_7b_en'), 'olmo2_13b': run_path(A, 'suite/olmo2_13b_en')}
    later_models(A, S, NONTHINK, None, 'v5_olmo_')


def newdev_models(A, S):
    """The seven models of newdev.py (v6_newdev_*.csv): non-thinking attribution, thinking counts, geometry in each measured mode."""
    from newdev import NEWDEV
    NONTHINK = {k: run_path(A, v['attr']) for k, v in NEWDEV.items() if v['attr']}
    THINK = {k: run_path(A, v['think']) for k, v in NEWDEV.items() if v['think']}
    GEO = {(k, mode): run_path(A, d) for k, v in NEWDEV.items() for mode, d in v['geo'].items()}
    later_models(A, S, NONTHINK, THINK, 'v6_newdev_', GEO)


# ---------------------------------------------------------------------------------------------------------------------------
# body-level self-reference of record folders (printout)

def body_level(RES, args):
    for arg in args:
        name, d = arg.split('=')
        d = str(RES / d)
        recs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(d + '/*.json'))]
        P = {r: np.array([x['roles'][r]['mean']['self'] for x in recs]) for r in ('assistant', 'user', 'system')}
        diff = P['assistant'] - np.maximum(P['user'], P['system'])
        k, n = int((diff > 0).sum()), len(diff)
        top = {r: np.mean([max(x['roles'][r]['mean'], key=x['roles'][r]['mean'].get) == t for x in recs]) for r, t in (('assistant', 'self'), ('user', 'user'), ('system', 'system'))}
        print(f'{name}: n={n} P(you|A)={P["assistant"].mean():.3f} P(you|U)={P["user"].mean():.3f} P(you|S)={P["system"].mean():.3f} '
              f'A-max(U,S) {diff.mean():+.3f}, positive {k}/{n}, sign p={sign_p(k, n):.2g} | top answer correct A {top["assistant"]:.2f} U {top["user"]:.2f} S {top["system"]:.2f}')


def main(argv=None):
    ap = arguments(__doc__)
    ap.add_argument('--body-level', nargs='*', default=None, metavar='NAME=DIR')
    args = ap.parse_args(argv)
    r = Roots(args)
    if args.body_level:
        body_level(r.RES, args.body_level)
        return
    eight_models(r.res(M8), r.out(M8 + '/stats'))
    A, S = r.res(W3), r.out(W3 + '/stats')
    wave3(A, S)
    v4_models(A, S)
    olmo_models(A, S)
    newdev_models(A, S)


if __name__ == '__main__':
    main()
