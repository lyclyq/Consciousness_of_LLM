"""Trust in the model's own records (Appendix app:ownstep, Table tab:self_ownstep): the reasoning task, 120 two-step problems.

ownstep (stats/v4_ownstep.csv): every model and mode. Single pass (suite M of the main probe): mean adoption index of wrong steps
  in Assistant history minus that of the same steps from the user, with a 95% bootstrap interval over problems. Thinking
  (thinking M): share of finished wrong-step items whose answer adopts the error, own minus user, item-paired by problem (both
  finished), 95% bootstrap interval over problems; and the share of finished correct-step items answered correctly, per source.
  The single-pass runs of the eight-model archive (models/<model>/non_thinking/reason) are included under the same rule (rows
  <model>_8models).
reasoning (summaries/M_reasoning_summary.json; Qwen3.8-27B, the implementation of the two-level analyses): baselines (greedy
  answers, labelled generation): accuracy with a correct step; with a wrong step, the share that adopts the error (answer =
  f + delta) and the share that corrects it (answer = f). Teacher-forced readout: adoption index
  a = P(first digit of f + delta) / (P(first digit of f + delta) + P(first digit of f)). Role effect = assistant - user, paired
  by problem, 95% bootstrap interval over problems (10,000 resamples, seed 0). Patching (wrong step): mean change of a in the
  recipient, and the fraction of the role gap closed (mean a_patched - mean a_recipient) / (mean a_donor - mean a_recipient).
  Needs the per-problem records of that run (a folder with problems.json and problems/*.json), which are not in the results
  package; it runs only when --reasoning-dir is given.

usage: ownstep.py --results RES --out OUT [--reasoning-dir DIR]
"""
import csv
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import M8, TL, W3, Roots, arguments, run_glob  # noqa: E402


def ownstep(A, A8, S):
    SUITE = [(d, d + '/M') for d in run_glob(A, 'suite/*_M')]
    SUITE += [(str(Path(d).parent.parent) + '_8models', d) for d in sorted(glob.glob(str(A8 / 'models' / '*' / 'non_thinking' / 'reason')))]
    THINK = sorted(run_glob(A, 'thinking_m/*') + glob.glob(str(A / 'runs' / 'api' / '*_thinkm')))

    def boot(d, reps=4000, seed=0):
        rng = random.Random(seed)
        m = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(reps))
        return m[int(0.025 * reps)], m[int(0.975 * reps) - 1]

    rows = []
    for d, md in SUITE:
        probs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(md + '/m*.json'))]
        if not probs:
            continue
        g = lambda p, s, r: next(x['adoption_index'] for x in p['results'] if x['step'] == s and x['role'] == r)  # noqa: E731
        diff = [g(p, 'wrong', 'assistant') - g(p, 'wrong', 'user') for p in probs]
        lo, hi = boot(diff)
        rows.append(dict(model=Path(d).name, mode='single_pass', n=len(probs), own=round(sum(g(p, 'wrong', 'assistant') for p in probs) / len(probs), 4),
                         user=round(sum(g(p, 'wrong', 'user') for p in probs) / len(probs), 4), own_minus_user=round(sum(diff) / len(diff), 4),
                         ci_lo=round(lo, 4), ci_hi=round(hi, 4), correct_own='', correct_user='', unfinished=''))
    for d in THINK:
        if 'smoke' in Path(d).name:
            continue
        its = [json.load(open(f, encoding='utf-8')) for f in glob.glob(d + '/items/*.json')]
        if not its:
            continue
        by = {(x['id'], x['step'], x['role']): x for x in its}
        ids = sorted({x['id'] for x in its})
        adopt = lambda x: float(x.get('class') == 'adopted')  # noqa: E731
        pairs = [(by[(i, 'wrong', 'assistant')], by[(i, 'wrong', 'user')]) for i in ids
                 if (i, 'wrong', 'assistant') in by and (i, 'wrong', 'user') in by
                 and by[(i, 'wrong', 'assistant')].get('finished') and by[(i, 'wrong', 'user')].get('finished')]
        diff = [adopt(a) - adopt(u) for a, u in pairs]
        lo, hi = boot(diff) if diff else (float('nan'),) * 2
        fin = lambda s, r: [x for x in its if x['step'] == s and x['role'] == r and x.get('finished')]  # noqa: E731
        share = lambda xs, c: round(sum(x.get('class') == c for x in xs) / max(1, len(xs)), 4)  # noqa: E731
        rows.append(dict(model=Path(d).name, mode='thinking', n=len(pairs), own=share(fin('wrong', 'assistant'), 'adopted'),
                         user=share(fin('wrong', 'user'), 'adopted'), own_minus_user=round(sum(diff) / max(1, len(diff)), 4),
                         ci_lo=round(lo, 4), ci_hi=round(hi, 4), correct_own=share(fin('correct', 'assistant'), 'correct'),
                         correct_user=share(fin('correct', 'user'), 'correct'),
                         unfinished=sum(not x.get('finished') for x in its)))
    with open(S / 'v4_ownstep.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']:28s} {r['mode']:11s} n={r['n']:3d} own={r['own']:.3f} user={r['user']:.3f} diff={r['own_minus_user']:+.3f} "
              f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] corr={r['correct_own']}/{r['correct_user']} unf={r['unfinished']}")


def reasoning(out_json, *patterns):
    def first_int(s):
        m = re.search(r'-?\d+', s)
        return int(m.group()) if m else None

    def boot(x, seed=0, n=10000):
        x = np.asarray(x, float)
        rng = np.random.default_rng(seed)
        m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    files = sorted({f for p in patterns for f in glob.glob(p)})
    probs = json.load(open(files[0].rsplit('problems', 1)[0] + 'problems.json', encoding='utf-8'))
    meta = {p['id']: p for p in probs}
    res = {'n_problems': len(files)}
    for split in ('all', 'confirmation'):
        base = defaultdict(dict)
        patch = defaultdict(list)
        for f in files:
            d = json.load(open(f, encoding='utf-8'))
            if split != 'all' and d['split'] != split:
                continue
            p = meta[d['id']]
            for r in d['results']:
                a = r['p_adopt_digit'] / (r['p_adopt_digit'] + r['p_correct_digit'] + 1e-12)
                if r['cond'] == 'baseline':
                    g = first_int(r['greedy'])
                    base[d['id']][(r['step'], r['role'])] = {'a': a, 'correct': g == p['answer'], 'adopt': g == p['answer_if_adopted'],
                                                             'mass': r['digit_mass']}
                else:
                    patch[(r['recipient'], r['region'], str(r['layer']))].append((d['id'], a))
        ids = sorted(base)
        out = {'n': len(ids)}
        print(f'\n== split {split}: {len(ids)} problems')
        for step in ('correct', 'wrong'):
            for role in ('assistant', 'user'):
                rows = [base[i][(step, role)] for i in ids]
                acc = np.mean([r['correct'] for r in rows]); adopt = np.mean([r['adopt'] for r in rows])
                am = np.mean([r['a'] for r in rows]); ms = np.min([r['mass'] for r in rows])
                out[f'{step}_{role}'] = {'accuracy': float(acc), 'adopt_rate': float(adopt), 'adoption_index': float(am), 'min_digit_mass': float(ms)}
                print(f'   step {step:7s} {role:9s}: greedy correct {acc:.3f}  adopts error {adopt:.3f}  adoption index {am:.3f}  (min digit mass {ms:.3f})')
            da = [base[i][(step, 'assistant')]['a'] - base[i][(step, 'user')]['a'] for i in ids]
            dc = [float(base[i][(step, 'assistant')]['correct']) - float(base[i][(step, 'user')]['correct']) for i in ids]
            out[f'{step}_role_effect'] = {'d_adoption_index': float(np.mean(da)), 'ci': boot(da), 'd_accuracy': float(np.mean(dc)), 'ci_acc': boot(dc)}
            print(f'   step {step}: assistant - user  adoption index {np.mean(da):+.3f} {boot(da)}  accuracy {np.mean(dc):+.3f} {boot(dc)}')
        a_rec = {'assistant': np.mean([base[i][('wrong', 'assistant')]['a'] for i in ids]),
                 'user': np.mean([base[i][('wrong', 'user')]['a'] for i in ids])}
        print('   patching (wrong step): mean adoption index after patch [gap closed]')
        out['patch'] = []
        for rec, don in (('assistant', 'user'), ('user', 'assistant')):
            gap = a_rec[don] - a_rec[rec]
            for region in ('role_header', 'body', 'post', 'body_post'):
                cells = []
                for l in ('0', '3', '7', '15', '23', '31', '39', '47', '55', '63', 'all'):
                    v = [a for i, a in patch[(rec, region, l)] if i in base]
                    if not v:
                        continue
                    m = float(np.mean(v)); g = (m - a_rec[rec]) / gap if abs(gap) > 1e-6 else float('nan')
                    out['patch'].append({'recipient': rec, 'donor': don, 'region': region, 'layer': l, 'a': m, 'gap_closed': g})
                    cells.append(f'L{l} {m:.3f}[{g:+.2f}]')
                print(f'     {rec}<-{don} {region:11s} ' + ' '.join(cells))
        res[split] = out
    json.dump(res, open(out_json, 'w'), indent=1)


def main(argv=None):
    ap = arguments(__doc__)
    ap.add_argument('--reasoning-dir', help='folder with problems.json and problems/*.json of the two-level reasoning run')
    args = ap.parse_args(argv)
    r = Roots(args)
    ownstep(r.res(W3), r.res(M8), r.out(W3 + '/stats'))
    if args.reasoning_dir:
        d = os.path.abspath(args.reasoning_dir)
        reasoning(str(r.out(TL + '/summaries') / 'M_reasoning_summary.json'), os.path.join(d, 'problems', '*.json'))
    else:
        print('reasoning: needs inputs not in the package (the per-problem records of the two-level reasoning run); skipped')


if __name__ == '__main__':
    main()
