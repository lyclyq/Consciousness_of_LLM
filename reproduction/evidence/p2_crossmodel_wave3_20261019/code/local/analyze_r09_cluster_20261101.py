"""R09 of the supervisor's review (11-01): uncertainty unit, completion and robustness of the causal tests, from the archived items of
the paper's main runs (no new runs). The 75 held-out bodies are 15 wording families x 5 numerical variants (item id
role150_<family>_c<variant>), each in two option orders. Per readout and condition, against the unedited message of the same role:
  - n, finished (edited / unedited), and the paired intersection (both finished), on which the paper's change is computed;
  - the change with the paper's item bootstrap (make_v4_mechanism.boot, seed 1, 4000 draws) and with a family-cluster bootstrap
    (15 families drawn with replacement, every item of a drawn family kept with its option orders and paired conditions; seed 1,
    4000 draws);
  - with thinking, bounds over all items that treat an unfinished output as unknown (worst case for the edited minus unedited share);
  - control tasks (content, arithmetic; 75 items per condition): paired change in accuracy, item and family-cluster intervals.
Outputs (ARCHIVE/stats): r09_cluster_20261101.json, r09_cluster_20261101.md.
usage: analyze_r09_cluster_20261101.py ARCHIVE"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_v4_mechanism import boot, items  # noqa: E402

READOUTS = [('qsp', 'Qwen3.8-27B, no thinking', 'server3/singlepass_swap/qwen38_27b_all', False),
            ('qth', 'Qwen3.8-27B, thinking', 'server3/thinking_swap/qwen38_27b_all', True),
            ('g20', 'gpt-oss-20b, thinking', 'pro/thinking_swap/gpt_oss_20b_all', True),
            ('g120', 'gpt-oss-120b, thinking', 'pro/thinking_swap/gpt_oss_120b_all', True)]
ATTR = [('own <- User', 'swU8_A', 'base_A'), ('own <- System', 'swS8_A', 'base_A'), ('own <- User, random', 'rswU8_A', 'base_A'),
        ('own <- own (copy)', 'idA8_A', 'base_A'), ('User <- own', 'swA8_U', 'base_U'), ('User <- own, random', 'rswA8_U', 'base_U'),
        ('System <- own', 'swA8_S', 'base_S')]


def fam(i):
    return i.rsplit('_c', 1)[0]


def cluster_ci(pairs, reps=4000, seed=1):
    """pairs: list of (family, diff); resample families with replacement"""
    fams = sorted({f for f, _ in pairs})
    byf = {f: [d for g, d in pairs if g == f] for f in fams}
    rng = random.Random(seed)
    ms = []
    for _ in range(reps):
        ds = [d for f in (rng.choice(fams) for _ in fams) for d in byf[f]]
        ms.append(sum(ds) / len(ds))
    ms.sort()
    return ms[int(0.025 * reps)], ms[int(0.975 * reps) - 1]


def value(r, thinking):
    if not thinking:
        return r['p']['self']
    return float(r.get('on_answer') == 'self') if r.get('finished') else None


def correct(r):
    for k in ('content', 'arith', 'answer'):
        if isinstance(r.get(k), dict) and 'correct' in r[k]:
            return float(bool(r[k]['correct'])) if r.get('finished', True) else None
    raise KeyError(sorted(r.keys()))


def compare(recs, cond, base, val):
    by = {}
    for r in recs:
        if r['cond'] in (cond, base):
            by.setdefault((r['id'], str(r.get('order'))), {})[r['cond']] = r
    keys = [k for k, v in by.items() if cond in v and base in v]
    e = {k: val(by[k][cond]) for k in keys}
    b = {k: val(by[k][base]) for k in keys}
    both = [k for k in keys if e[k] is not None and b[k] is not None]
    d = [e[k] - b[k] for k in both]
    out = {'n': len(keys), 'finished': [sum(e[k] is not None for k in keys), sum(b[k] is not None for k in keys)], 'pairs': len(both)}
    if d:
        out.update(change=sum(d) / len(d), item_ci=list(boot(d, seed=1)), cluster_ci=list(cluster_ci([(fam(k[0]), x) for k, x in zip(both, d)])))
    lo = sum((0.0 if e[k] is None else e[k]) - (1.0 if b[k] is None else b[k]) for k in keys) / len(keys)
    hi = sum((1.0 if e[k] is None else e[k]) - (0.0 if b[k] is None else b[k]) for k in keys) / len(keys)
    out['bounds'] = [lo, hi]
    return out


def main():
    A = Path(sys.argv[1])
    res = {}
    for key, name, d, thinking in READOUTS:
        recs = items(str(A / 'runs' / d))
        res[key] = {'attribution': {lab: compare(recs, c, b, lambda r: value(r, thinking)) for lab, c, b in ATTR}}
        for task in ('content', 'arith'):
            trecs = items(str(A / 'runs' / f'{d}_{task}'))
            conds = sorted({r['cond'] for r in trecs})
            res[key][task] = {}
            for c in conds:
                if c.startswith('base_'):
                    continue
                base = 'base_U' if c.endswith('_U') else ('base_S' if c.endswith('_S') else 'base_A')
                if base in conds:
                    res[key][task][c] = compare(trecs, c, base, correct)
    f3 = lambda x: f'{x:+.2f}'  # noqa: E731
    L = ['# R09: uncertainty unit, completion and bounds of the causal tests (11-01)', '',
         'Change: item-paired difference on the items finished in both conditions (the paper\'s estimand). Item CI: the paper\'s item '
         'bootstrap; cluster CI: 15 wording families resampled with all their items. Bounds: over all items, unfinished outputs '
         'treated as unknown (thinking). Finished: edited / unedited.', '']
    for key, name, _, thinking in READOUTS:
        L += [f'## {name}', '', '| condition | finished | pairs | change | item CI | cluster CI | bounds (unknown unfinished) |', '|---|---|---|---|---|---|---|']
        for lab, s in res[key]['attribution'].items():
            if 'change' not in s:
                continue
            L.append(f"| {lab} | {s['finished'][0]}/{s['finished'][1]} of {s['n']} | {s['pairs']} | {f3(s['change'])} | "
                     f"[{f3(s['item_ci'][0])}, {f3(s['item_ci'][1])}] | [{f3(s['cluster_ci'][0])}, {f3(s['cluster_ci'][1])}] | "
                     f"[{f3(s['bounds'][0])}, {f3(s['bounds'][1])}] |")
        for task in ('content', 'arith'):
            L += ['', f'{task}: paired change in accuracy', '', '| condition | finished | pairs | change | item CI | cluster CI |', '|---|---|---|---|---|---|']
            for c, s in res[key][task].items():
                if 'change' in s:
                    L.append(f"| {c} | {s['finished'][0]}/{s['finished'][1]} of {s['n']} | {s['pairs']} | {f3(s['change'])} | "
                             f"[{f3(s['item_ci'][0])}, {f3(s['item_ci'][1])}] | [{f3(s['cluster_ci'][0])}, {f3(s['cluster_ci'][1])}] |")
        L.append('')
    (A / 'stats' / 'r09_cluster_20261101.json').write_text(json.dumps({'results': res}, indent=1), encoding='utf-8')
    (A / 'stats' / 'r09_cluster_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
