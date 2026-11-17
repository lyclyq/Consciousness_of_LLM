"""R08 of the supervisor's review, the reasoning traces of the gpt-oss models for System messages (11-01). The gpt-oss template
renders a System message as a developer message; the paper (Section 5.1, installation paragraph) counts the traces that call the
edited System message "the developer message" (147 of 150 for gpt-oss-20b, System <- own). This script counts the same thing,
with the same rule, for the exchange of the User and System coordinates (R08 runs) and, for comparison, in the paper's runs:
  rule: "developer message" in the generated text, case-insensitive, over all 150 items of a condition (finished or not)
  runs: thinking_swap/<model>_r08_us (base_S, swU8_S: the User coordinates written into a System message, rswU8_S: the same in
        the random subspace; also base_U, swS8_U, rswS8_U) and thinking_swap/<model>_all (base_S, swA8_S), runs/pro
Also given per condition: the answers of the finished items (you / user / system).
Outputs (ARCHIVE/stats): r08_traces_20261101.json, r08_traces_20261101.md.
usage: analyze_r08_traces_20261101.py ARCHIVE"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_v4_mechanism import items  # noqa: E402

MODELS = [('g20', 'gpt_oss_20b', 'gpt-oss-20b'), ('g120', 'gpt_oss_120b', 'gpt-oss-120b')]
RUNS = [('r08_us', ['base_S', 'swU8_S', 'rswU8_S', 'base_U', 'swS8_U', 'rswS8_U']), ('all', ['base_S', 'swA8_S'])]
NAMES = {'base_S': 'System, unedited', 'swU8_S': 'System <- User', 'rswU8_S': 'System <- User, random',
         'base_U': 'User, unedited', 'swS8_U': 'User <- System', 'rswS8_U': 'User <- System, random', 'swA8_S': 'System <- own'}


def main():
    A = Path(sys.argv[1])
    res = {}
    for key, d, _ in MODELS:
        res[key] = {}
        for run, conds in RUNS:
            by = {}
            for r in items(str(A / 'runs/pro/thinking_swap' / f'{d}_{run}')):
                by.setdefault(r['cond'], []).append(r)
            res[key][run] = {}
            for c in conds:
                rs = by[c]
                fin = [r for r in rs if r.get('finished')]
                ans = Counter(r.get('on_answer') for r in fin)
                res[key][run][c] = {'n': len(rs), 'developer_message': sum('developer message' in r['gen'].lower() for r in rs),
                                    'finished': len(fin), 'answers': {k: ans.get(k, 0) for k in ('self', 'user', 'system')}}
    L = ['# R08: reasoning traces of the gpt-oss models for System and User messages (11-01)', '',
         'Traces that contain "developer message" (case-insensitive), over all items of a condition; answers of the finished items. '
         'Runs: thinking_swap/<model>_r08_us (exchange of the User and System coordinates, k = 8, fit_q, all positions) and '
         'thinking_swap/<model>_all (the paper\'s runs).', '',
         '| model | run | condition | items | "developer message" | finished | you / user / system |', '|---|---|---|---|---|---|---|']
    for key, _, name in MODELS:
        for run, conds in RUNS:
            for c in conds:
                s = res[key][run][c]
                a = s['answers']
                L.append(f"| {name} | {run} | {NAMES[c]} | {s['n']} | {s['developer_message']} | {s['finished']} | "
                         f"{a['self']} / {a['user']} / {a['system']} |")
    (A / 'stats' / 'r08_traces_20261101.json').write_text(json.dumps({'results': res}, indent=1), encoding='utf-8')
    (A / 'stats' / 'r08_traces_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
