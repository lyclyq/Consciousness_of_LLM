"""R09 numbers (family-cluster bootstrap, completion, bounds) for MiMo-V2.6-Distill-Qwen-9B in its default call, the fifth readout
of the causal tests (user 11-10: "four models and five readouts"), with the functions of analyze_r09_cluster_20261101.py unchanged
(compare: item bootstrap seed 1, 15 wording families resampled with all their items, seed 1, 4000 draws; bounds with unfinished
outputs treated as unknown). Runs (crossmodel_role_swap_r05.py, KL archive ../p2_modeswitch_20261107/results/pro/thinking_swap):
attribution mimo_v26_r05q_all (the paper's procedure: fit_q, every position), 64 dimensions mimo_v26_all_k64, content
mimo_v26_content_all, arithmetic mimo_v26_arith_all. MiMo has no System conditions.
Outputs (ARCHIVE/stats): r09_cluster_mimo_20261110.json, r09_cluster_mimo_20261110.md.
usage: analyze_r09_cluster_mimo_20261110.py ARCHIVE"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_r09_cluster_20261101 as R9  # noqa: E402

RUNS = Path('..') / 'p2_modeswitch_20261107' / 'results' / 'pro' / 'thinking_swap'
ATTR64 = [('own <- User, 64', 'swU64_A', 'base_A'), ('own <- User, 64, random', 'rswU64_A', 'base_A'),
          ('User <- own, 64', 'swA64_U', 'base_U'), ('User <- own, 64, random', 'rswA64_U', 'base_U')]


def main():
    A = Path(sys.argv[1])
    T = (A / RUNS).resolve()  # without "..": the item paths would pass the Windows path limit of 260 characters
    val = lambda r: R9.value(r, True)  # noqa: E731
    recs = R9.items(str(T / 'mimo_v26_r05q_all'))
    res = {'attribution': {lab: R9.compare(recs, c, b, val) for lab, c, b in R9.ATTR if c in {r['cond'] for r in recs}}}
    recs64 = R9.items(str(T / 'mimo_v26_all_k64'))
    res['attribution_k64'] = {lab: R9.compare(recs64, c, b, val) for lab, c, b in ATTR64}
    for task, run in (('content', 'mimo_v26_content_all'), ('arith', 'mimo_v26_arith_all')):
        trecs = R9.items(str(T / run))
        conds = sorted({r['cond'] for r in trecs})
        res[task] = {c: R9.compare(trecs, c, 'base_U' if c.endswith('_U') else 'base_A', R9.correct) for c in conds if not c.startswith('base_')}
        res[task]['_accuracy'] = {c: sum(R9.correct(r) or 0 for r in trecs if r['cond'] == c) / sum(1 for r in trecs if r['cond'] == c) for c in conds}
    f3 = lambda x: f'{x:+.2f}'  # noqa: E731
    L = ['# R09 numbers for MiMo-V2.6-Distill-Qwen-9B, default call (11-10)', '', 'As r09_cluster_20261101.md.', '',
         '| condition | finished | pairs | change | item CI | cluster CI | bounds (unknown unfinished) |', '|---|---|---|---|---|---|---|']
    for part in ('attribution', 'attribution_k64'):
        for lab, s in res[part].items():
            if 'change' in s:
                L.append(f"| {lab} | {s['finished'][0]}/{s['finished'][1]} of {s['n']} | {s['pairs']} | {f3(s['change'])} | "
                         f"[{f3(s['item_ci'][0])}, {f3(s['item_ci'][1])}] | [{f3(s['cluster_ci'][0])}, {f3(s['cluster_ci'][1])}] | "
                         f"[{f3(s['bounds'][0])}, {f3(s['bounds'][1])}] |")
    for task in ('content', 'arith'):
        L += ['', f"{task}: accuracy {json.dumps({k: round(v, 3) for k, v in res[task]['_accuracy'].items()})}", '',
              '| condition | finished | pairs | change | item CI | cluster CI |', '|---|---|---|---|---|---|']
        for c, s in res[task].items():
            if c != '_accuracy' and 'change' in s:
                L.append(f"| {c} | {s['finished'][0]}/{s['finished'][1]} of {s['n']} | {s['pairs']} | {f3(s['change'])} | "
                         f"[{f3(s['item_ci'][0])}, {f3(s['item_ci'][1])}] | [{f3(s['cluster_ci'][0])}, {f3(s['cluster_ci'][1])}] |")
    (A / 'stats' / 'r09_cluster_mimo_20261110.json').write_text(json.dumps({'results': res}, indent=1), encoding='utf-8')
    (A / 'stats' / 'r09_cluster_mimo_20261110.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
