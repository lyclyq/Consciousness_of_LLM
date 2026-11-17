"""R03 of the supervisor's review (PRO runs of 10-31, code/pro/r03_pro.sh and r03_qth_retry1_pro.sh): does the readout of the source
go through the eight coordinates or through the rest of the state (epsilon of Equation 1)?
Part A (<dir>_r03c_all, crossmodel_role_swap_r03.py): at every layer of the band and aligned position, all positions, fit_q of
r05r06_fits, k = 8,
  V  the recipient takes the donor's 8 coordinates (the paper's edit; swU8_A / swA8_U of <dir>_r05q_all)
  C  the recipient takes the donor's state outside the 8 directions and keeps its own coordinates inside them (cswU8_A / cswA8_U)
  F  the recipient takes the donor's whole state (fswU8_A / fswA8_U)
  R  V in the random subspace of base seed 20261022 (rswU8_A / rswA8_U of <dir>_r05q_all)
Removal: the own message given the User state; installation: the User message given the own state. The unedited items of
r05q_all are copied into r03c_all: one baseline with R05, R06 and R08. Statistics as in the paper (make_v4_mechanism.py): the share
of finished items answered "you" (thinking) or the mean probability of the "you" option (no thinking); the item-paired change from
the unedited message of the same role, 95% bootstrap interval (seed 1). A thinking cell with unfinished items also gets the bounds
of the change with the unfinished items counted as "you" and as not "you" (user 10-31: unfinished items are reported as such).
Edit size: per item sqrt(sum of squared changes / sum of squared states) over all layers and edited positions, median over items.
Part B (<dir>_r03fd, crossmodel_readout_fd_r03.py): central differences D_v f of f = logit("you" letter) - logit("user" letter),
eps = 0.25, along v = d (total), P_V d (V), d - P_V d (perp) and P_R d (R), d = h^A - h^U, added at one layer at a time; per item the
sum over the layers of --layer-step (the first-order change of f when the role difference is applied at every such layer), mean
over the items complete at both endpoints, with the share of D_V and D_perp in D_total (95% bootstrap interval of the ratio of
means, seed 1). The logits are bfloat16, so every D is a multiple of 0.25; the share of zero entries is given.
Outputs (ARCHIVE/stats): r03_20261101.json, r03_20261101.md.
usage: analyze_r03_20261101.py ARCHIVE"""
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_r05r06_20261031 import READOUTS  # noqa: E402
from make_v4_mechanism import BASE, boot, items  # noqa: E402

EDITS = [('removal', 'base_A', [('V', 'swU8_A'), ('C', 'cswU8_A'), ('F', 'fswU8_A'), ('R', 'rswU8_A')]),
         ('installation', 'base_U', [('V', 'swA8_U'), ('C', 'cswA8_U'), ('F', 'fswA8_U'), ('R', 'rswA8_U')])]
FROM_R05 = ('swU8_A', 'swA8_U', 'rswU8_A', 'rswA8_U')
DIRS = ('total', 'V', 'perp', 'R')


def you(r, unfinished=None):
    if 'p' in r:
        return r['p']['self']
    if not r.get('finished', True):
        return unfinished
    return float(r.get('on_answer') == 'self')


def level(rs):
    v = [x for x in (you(r) for r in rs) if x is not None]
    out = {'n': len(rs), 'you': sum(v) / len(v) if v else None}
    if rs and 'p' not in rs[0]:
        out['finished'] = sum(r.get('finished', True) for r in rs)
    return out


def size(r):
    d = r.get('dose')
    if not d or not d.get('state2'):
        return None
    return math.sqrt(sum(d['edit2']) / sum(d['state2']))


def change(rs, bs):
    b = {(r['id'], tuple(r['order'])): r for r in bs}
    pairs = [(r, b[k]) for r in rs for k in [(r['id'], tuple(r['order']))] if k in b]
    d = [you(r) - you(q) for r, q in pairs if you(r) is not None and you(q) is not None]
    out = {'diff': sum(d) / len(d), 'ci': list(boot(d, seed=1)), 'n_pairs': len(d)}
    if any(not r.get('finished', True) for r, _ in pairs):
        bounds = []
        for u in (0.0, 1.0):
            e = [you(r, u) - you(q) for r, q in pairs if you(q) is not None]
            bounds.append(sum(e) / len(e))
        out['bounds'] = bounds
    s = [x for x in (size(r) for r in rs) if x is not None]
    out['size'] = statistics.median(s) if s else None
    return out


def part_b(d):
    recs = items(str(d))
    ok = [r for r in recs if all('D' in r['endpoints'].get(e, {}) for e in ('A', 'U'))]
    out = {'n_items': len(recs), 'n_complete': len(ok)}
    for e in ('A', 'U'):
        S = {k: [sum(r['endpoints'][e]['D'][k]) for r in ok] for k in DIRS}
        mean = {k: sum(v) / len(v) for k, v in S.items()}
        rng = random.Random(1)
        share = {}
        for k in ('V', 'perp'):
            bs = []
            for _ in range(4000):
                ix = [rng.randrange(len(ok)) for _ in ok]
                t = sum(S['total'][i] for i in ix)
                if t:
                    bs.append(sum(S[k][i] for i in ix) / t)
            bs.sort()
            share[k] = {'ratio': mean[k] / mean['total'] if mean['total'] else None,
                        'ci': [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]] if bs else None}
        zero = sum(x == 0 for r in ok for x in r['endpoints'][e]['D']['total']) / sum(len(r['endpoints'][e]['D']['total']) for r in ok)
        out[e] = {'mean_sum': mean, 'share': share, 'zero_total': zero,
                  'zero_dir_nonzero': sum(x != 0 for r in ok for x in r['endpoints'][e]['D']['zero'])}
    out['norm_share'] = {k: statistics.median(sum(r['norms'][k]) / sum(r['norms']['total']) for r in ok) for k in ('V', 'perp', 'R')}
    out['n_layers'] = len(ok[0]['layers']) if ok else 0
    return out


def main():
    A = Path(sys.argv[1])
    res, missing = {}, []
    for rd, _, root, _ in READOUTS:
        da, db = A / 'runs/pro' / f'{root}_r03c_all', A / 'runs/pro' / f'{root}_r03fd'
        if not (da / 'summary.json').exists():
            missing.append(f'{root}_r03c_all')
            continue
        by = {}
        for r in items(str(da)):
            by.setdefault(r['cond'], []).append(r)
        for r in items(str(A / 'runs/pro' / f'{root}_r05q_all')):
            if r['cond'] in FROM_R05:
                by.setdefault(r['cond'], []).append(r)
        res[rd] = {'levels': {c: level(rs) for c, rs in sorted(by.items())}}
        for _, base, eds in EDITS:
            for _, c in eds:
                if c in by:
                    assert BASE[by[c][0]['recipient']] == base
                    res[rd][c] = change(by[c], by[base])
        if (db / 'summary.json').exists():
            res[rd]['B'] = part_b(db)
        else:
            missing.append(f'{root}_r03fd')
    f3 = lambda x: '—' if x is None else f'{x:.3f}'  # noqa: E731
    ch = lambda s: f"{s['diff']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]" if s else '—'  # noqa: E731
    L = ['# R03: is the source read through the eight coordinates or through the rest of the state? (PRO, 10-31)', '',
         '## Part A: replacing the coordinates (V), everything else (C), the whole state (F)', '',
         'Thinking: share of finished items answered "you"; no thinking: mean probability of the "you" option. Change: item-paired '
         'difference from the unedited message of the same role (95% bootstrap interval). V = the paper\'s edit (8 coordinates of the '
         'donor); C = the donor\'s state outside the 8 directions, the recipient\'s coordinates inside them; F = the donor\'s whole '
         'state; R = V in a random 8-dim subspace (base seed 20261022). fit_q, all positions, every layer of the band. V and R are the '
         'items of r05q_all. Size: median over items of the edit\'s norm relative to the state\'s (all layers).', '',
         '| readout | test | unedited "you" | basis | edited "you" | change in "you" | size | finished |', '|---|---|---|---|---|---|---|---|']
    notes = []
    for rd, _, _, name in READOUTS:
        if rd not in res:
            continue
        Lv = res[rd]['levels']
        for test, base, eds in EDITS:
            for w, c in eds:
                if c not in Lv:
                    continue
                s = res[rd].get(c)
                fin = f"{Lv[c]['finished']}/{Lv[c]['n']}" if 'finished' in Lv[c] else '—'
                L.append(f"| {name} | {test} | {f3(Lv[base]['you'])} | {w} | {f3(Lv[c]['you'])} | {ch(s)} | "
                         f"{f3(s.get('size') if s else None)} | {fin} |")
                if s and 'bounds' in s:
                    notes.append(f"{name}, {test}, {w}: {Lv[c]['finished']} of {Lv[c]['n']} finished; change {s['bounds'][0]:+.3f} with the "
                                 f"unfinished items counted as not \"you\", {s['bounds'][1]:+.3f} counted as \"you\".")
    if notes:
        L += ['', 'Unfinished items:'] + [f'- {x}' for x in notes]
    L += ['', '## Part B: first-order sensitivity of the readout to the role difference, inside and outside the 8 coordinates', '',
          'f = logit("you" letter) - logit("user" letter) where the answer letter is read (onset without thinking; after the model\'s own '
          'reasoning, teacher-forced, with thinking). D_v f by central differences (eps = 0.25) along d = h^A - h^U (total), its part '
          'in the 8 fitted directions (V), the rest (perp) and its part in a random 8-dim subspace (R), at one layer at a time; per item '
          'summed over the layers, mean over items. Share: mean D_V / mean D_total and mean D_perp / mean D_total (95% bootstrap '
          'interval over items). bf16 logits: every D is a multiple of 0.25.', '',
          '| readout | endpoint | items | D_total | D_V | D_perp | D_R | V share | perp share | D_total = 0 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    sh = lambda s: f"{s['ratio']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]" if s and s['ratio'] is not None and s['ci'] else '—'  # noqa: E731
    for rd, _, _, name in READOUTS:
        B = res.get(rd, {}).get('B')
        if not B:
            continue
        for e, lab in (('A', 'own message'), ('U', 'User message')):
            m = B[e]['mean_sum']
            L.append(f"| {name} | {lab} | {B['n_complete']}/{B['n_items']} | {m['total']:+.2f} | {m['V']:+.2f} | {m['perp']:+.2f} | "
                     f"{m['R']:+.2f} | {sh(B[e]['share']['V'])} | {sh(B[e]['share']['perp'])} | {B[e]['zero_total']:.2f} |")
    L += ['', 'Norm of the parts relative to |d| (median over items, summed over layers): ' + '; '.join(
        f"{name} V {res[rd]['B']['norm_share']['V']:.2f}, perp {res[rd]['B']['norm_share']['perp']:.2f}, R {res[rd]['B']['norm_share']['R']:.3f} "
        f"({res[rd]['B']['n_layers']} layers)" for rd, _, _, name in READOUTS if 'B' in res.get(rd, {}))]
    if missing:
        L += ['', 'Not finished (left out): ' + ', '.join(missing)]
    (A / 'stats/r03_20261101.json').write_text(json.dumps({'results': res, 'missing': missing}, indent=1), encoding='utf-8')
    (A / 'stats/r03_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    sys.stdout.buffer.write(('\n'.join(L) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
