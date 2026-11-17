"""R05 / R06 of the supervisor's review (PRO runs of 10-30/31, code/pro/r05r06_pro.sh), from the per-item results in the wave-3
archive. Statistics as in the paper (make_v4_mechanism.py: items, summarize): the share answering "you" (thinking) or mean
P("you") (no thinking), and the item-paired difference from the unedited condition of the same recipient with a 95% bootstrap
interval. Runs per readout (the edited conditions of the paper, removal swU8_A and installation swA8_U, k = 8):
  r05q_all  the paper's fit (fit_q) and positions (all), with the random controls of base seed 20261022 (rerun of the paper's run)
  r05q_msg  fit_q, edits only in the message (body and end)
  r05m_msg  fit_m (fitted on the message only), edits only in the message: neither the fit nor the edit sees the question
  r05m_all  fit_m, all positions
  r06_s1..4 random subspaces of base seeds 20261023..20261026, all positions (with r05q_all: five consecutive seeds)
Dose: per edited item, the relative size of the edit over all edited layers and positions, sqrt(sum ||delta||^2 / sum ||x||^2).
The ratio random / V is also given without the final layer (the band ends at the model's last block, whose output at the edited
prompt positions reaches no later layer and no read logit: the read position is not edited) and per quarter of the layers.
Runs without summary.json (not finished) are left out and listed. The paper's run (v4_mechanism.json, 'all') is shown alongside.
Outputs (ARCHIVE/stats): r05r06_20261031.json, r05r06_20261031.md.
usage: analyze_r05r06_20261031.py ARCHIVE"""
import json
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_v4_mechanism import items, summarize  # noqa: E402

READOUTS = [('qsp', 'qwen_sp', 'singlepass_swap/qwen38_27b', 'Qwen3.8-27B, no thinking'),
            ('qth', 'qwen_th', 'thinking_swap/qwen38_27b', 'Qwen3.8-27B, thinking'),
            ('g20', 'oss_th', 'thinking_swap/gpt_oss_20b', 'gpt-oss-20b, thinking'),
            ('g120', 'oss120_th', 'thinking_swap/gpt_oss_120b', 'gpt-oss-120b, thinking')]
RUNS = ['r05q_all', 'r05q_msg', 'r05m_msg', 'r05m_all', 'r06_s1', 'r06_s2', 'r06_s3', 'r06_s4']
SEED = {'r05q_all': 20261022, 'r06_s1': 20261023, 'r06_s2': 20261024, 'r06_s3': 20261025, 'r06_s4': 20261026}
EDITED = ('swU8_A', 'swA8_U', 'rswU8_A', 'rswA8_U')


def layer_sets(n):
    last = n - 1
    sets = [('without_final', lambda l: l != last)]
    for j in range(4):
        sets.append((f'quarter{j + 1}', lambda l, j=j: j * n // 4 <= l < (j + 1) * n // 4 and l != last))
    return sets


def dose(recs, n_layers):
    out = {}
    for c in EDITED:
        ds = [r['dose'] for r in recs if r['cond'] == c and r.get('dose')]
        if ds:
            rel = [math.sqrt(sum(d['edit2']) / sum(d['state2'])) for d in ds]
            q = statistics.quantiles(rel, n=4)
            out[c] = {'n': len(rel), 'median': statistics.median(rel), 'q1': q[0], 'q3': q[2],
                      'n_pos_median': statistics.median(d['n_pos'] for d in ds),
                      'final_share': sum(d['edit2'][d['layers'].index(n_layers - 1)] for d in ds) / sum(sum(d['edit2']) for d in ds)}
            for name, keep in layer_sets(n_layers):
                ix = [[i for i, l in enumerate(d['layers']) if keep(l)] for d in ds]
                out[c]['median_' + name] = statistics.median(
                    math.sqrt(sum(d['edit2'][i] for i in k) / sum(d['state2'][i] for i in k)) for d, k in zip(ds, ix))
    return out


def f(s, key='diff'):
    if not s or key not in s:
        return '—'
    lo, hi = s[key + '_ci'] if key == 'diff' else s['ci']
    return f"{s[key]:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    A = Path(sys.argv[1])
    paper = json.load(open(A / 'stats/v4_mechanism.json', encoding='utf-8'))['results']
    res, missing = {}, []
    for rd, key, root, _ in READOUTS:
        res[rd] = {'paper_all': {c: paper[key]['all'].get(c) for c in ('base_A', 'base_U') + EDITED}}
        for run in RUNS:
            d = A / 'runs/pro' / f'{root}_{run}'
            if not (d / 'summary.json').exists():
                missing.append(f'{root}_{run}')
                continue
            recs = items(str(d))
            s = summarize(recs, None)
            n_layers = json.load(open(d / 'identity.json', encoding='utf-8'))['n_layers']
            res[rd][run] = {'stats': s, 'dose': dose(recs, n_layers), 'n_items': len(recs)}
    L = ['# R05 / R06 (PRO, 10-30/31)', '',
         'Item-paired difference from the unedited condition (95% bootstrap interval); thinking: share answering "you", '
         'no thinking: mean P("you"). k = 8.', '']
    L += ['## R05: removal (← User, in the own message) and installation (← own, in the User message)', '',
          '| readout | edit | paper | fit_q, all | fit_q, message | fit_m, message | fit_m, all |', '|---|---|---|---|---|---|---|']
    for rd, _, _, name in READOUTS:
        for c, lab in (('swU8_A', 'removal'), ('swA8_U', 'installation')):
            row = [name, lab, f(res[rd]['paper_all'].get(c))]
            row += [f(res[rd][r]['stats'].get(c)) if r in res[rd] else '—' for r in ('r05q_all', 'r05q_msg', 'r05m_msg', 'r05m_all')]
            L.append('| ' + ' | '.join(row) + ' |')
    L += ['', '## R06: random subspaces, five consecutive base seeds (all positions)', '',
          '| readout | edit | paper (20261022) | ' + ' | '.join(str(v) for v in SEED.values()) + ' | max abs |', '|---' * 9 + '|']
    for rd, _, _, name in READOUTS:
        for c, lab in (('rswU8_A', 'random, own message'), ('rswA8_U', 'random, User message')):
            vals = [res[rd][r]['stats'].get(c) if r in res[rd] else None for r in SEED]
            ds = [v['diff'] for v in vals if v and 'diff' in v]
            row = [name, lab, f(res[rd]['paper_all'].get(c))] + [f(v) for v in vals]
            row.append(f'{max(abs(x) for x in ds):.3f} (n={len(ds)})' if ds else '—')
            L.append('| ' + ' | '.join(row) + ' |')
    L += ['', '## Dose: relative size of the edit, sqrt(sum ||delta||^2 / sum ||x||^2), median [quartiles] over items', '',
          '| readout | run | ← User (V) | ← own (V) | ← random, own message (R) | ← random, User message (R) | positions (median, ← User) |',
          '|---|---|---|---|---|---|---|']
    for rd, _, _, name in READOUTS:
        for r in ('r05q_all', 'r05q_msg', 'r05m_msg', 'r05m_all'):
            if r not in res[rd]:
                continue
            D = res[rd][r]['dose']
            cell = lambda c: f"{D[c]['median']:.4f} [{D[c]['q1']:.4f}, {D[c]['q3']:.4f}]" if c in D else '—'  # noqa: E731
            L.append(f"| {name} | {r} | {cell('swU8_A')} | {cell('swA8_U')} | {cell('rswU8_A')} | {cell('rswA8_U')} | "
                     f"{D['swU8_A']['n_pos_median'] if 'swU8_A' in D else '—'} |")
    keys = ['median', 'median_without_final'] + [f'median_quarter{j}' for j in range(1, 5)]
    L += ['', '## Dose ratio random / V by layers', '',
          'V: r05q_all; random: median over the five base seeds (r05q_all, r06_s1..4). The final layer\'s output at the edited prompt '
          'positions reaches no later layer and no read logit; the quarters split the layers 0..n-1, the fourth without the final one. '
          'Last column: the final layer\'s share of the summed squared V edit.', '',
          '| readout | edit | all layers | without final | quarter 1 | quarter 2 | quarter 3 | quarter 4 | final layer, share of V |',
          '|---|---|---|---|---|---|---|---|---|']
    for rd, _, _, name in READOUTS:
        if 'r05q_all' not in res[rd]:
            continue
        for v, rr, lab in (('swU8_A', 'rswU8_A', 'removal'), ('swA8_U', 'rswA8_U', 'installation')):
            Dv = res[rd]['r05q_all']['dose'].get(v)
            Dr = [res[rd][r]['dose'][rr] for r in SEED if r in res[rd] and rr in res[rd][r]['dose']]
            if Dv and Dr:
                L.append(f'| {name} | {lab} | ' + ' | '.join(f"{statistics.median(x[kk] for x in Dr) / Dv[kk]:.2f}" for kk in keys)
                         + f" | {Dv['final_share']:.2f} |")
    if missing:
        L += ['', 'Not finished (left out): ' + ', '.join(missing)]
    (A / 'stats/r05r06_20261031.json').write_text(json.dumps({'results': res, 'missing': missing}, indent=1), encoding='utf-8')
    (A / 'stats/r05r06_20261031.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
