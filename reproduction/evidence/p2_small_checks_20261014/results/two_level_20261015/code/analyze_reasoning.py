"""Summary for check M (reasoning with an intermediate step written by the model itself vs by the user).

Baselines (greedy answers, labelled generation): accuracy with a correct step; with a wrong step, the share that
adopts the error (answer = f + delta) and the share that corrects it (answer = f). Teacher-forced readout:
adoption index a = P(first digit of f + delta) / (P(first digit of f + delta) + P(first digit of f)).
Role effect = assistant - user, paired by problem, 95% bootstrap interval over problems.
Patching (wrong step): mean change of a in the recipient, and the fraction of the role gap closed
(mean a_patched - mean a_recipient) / (mean a_donor - mean a_recipient).
usage: analyze_reasoning.py OUT_JSON PATTERN [PATTERN ...]"""
import glob
import json
import re
import sys
from collections import defaultdict
import numpy as np


def first_int(s):
    m = re.search(r'-?\d+', s)
    return int(m.group()) if m else None


def boot(x, seed=0, n=10000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main(out_json, *patterns):
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


if __name__ == '__main__':
    main(*sys.argv[1:])
