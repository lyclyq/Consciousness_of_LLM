"""R08 of the supervisor's review (PRO runs of 10-31, code/pro/r08_pro.sh): do the self-index coordinates also tell one external
source from the other? A User and a System message exchange their eight coordinates (User <- System: swS8_U; System <- User:
swU8_S), with the same exchange in the random subspace of base seed 20261022 (rswS8_U, rswU8_S); fit_q of r05r06_fits, all
positions, k = 8. The unedited own and User items are those of r05q_all, copied into every run: R05, R06 and R08 share one
baseline. For comparison, the installation of the own coordinates into the User message (swA8_U) is read from r05q_all.
Statistics as in the paper (make_v4_mechanism.py: items, boot): per answer (you / user / system), the share of finished items so
answered (thinking) or the mean probability of that option (no thinking), and the item-paired difference from the unedited message
of the same role with a 95% bootstrap interval.
Outputs (ARCHIVE/stats): r08_20261031.json, r08_20261031.md.
usage: analyze_r08_20261031.py ARCHIVE"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_r05r06_20261031 import READOUTS  # noqa: E402
from make_v4_mechanism import BASE, boot, items  # noqa: E402

ANSWERS = ('self', 'user', 'system')
EDITS = [('swS8_U', 'User ← System'), ('rswS8_U', 'User ← System, random'), ('swU8_S', 'System ← User'),
         ('rswU8_S', 'System ← User, random'), ('swA8_U', 'User ← own (R05)')]


def value(r, a):
    if 'p' in r:
        return r['p'][a]
    return float(r.get('on_answer') == a) if r.get('finished', True) else None


def level(rs):
    out = {'n': len(rs)}
    if rs and 'p' not in rs[0]:
        out['finished'] = sum(r.get('finished', True) for r in rs)
    for a in ANSWERS:
        v = [x for x in (value(r, a) for r in rs) if x is not None]
        out[a] = sum(v) / len(v) if v else None
    return out


def paired(rs, bs):
    b = {(r['id'], tuple(r['order'])): r for r in bs}
    out = {}
    for a in ANSWERS:
        d = [value(r, a) - value(b[k], a) for r in rs for k in [(r['id'], tuple(r['order']))]
             if k in b and value(r, a) is not None and value(b[k], a) is not None]
        if d:
            out[a] = {'diff': sum(d) / len(d), 'ci': list(boot(d, seed=1)), 'n_pairs': len(d)}
    return out


def main():
    A = Path(sys.argv[1])
    res, missing = {}, []
    for rd, _, root, _ in READOUTS:
        d = A / 'runs/pro' / f'{root}_r08_us'
        if not (d / 'summary.json').exists():
            missing.append(f'{root}_r08_us')
            continue
        recs = items(str(d))
        by = {}
        for r in recs:
            by.setdefault(r['cond'], []).append(r)
        by['swA8_U'] = [r for r in items(str(A / 'runs/pro' / f'{root}_r05q_all')) if r['cond'] == 'swA8_U']
        res[rd] = {'levels': {c: level(rs) for c, rs in sorted(by.items())}, 'n_items': len(recs)}
        for c, _ in EDITS:
            if c in by:
                res[rd][c] = paired(by[c], by[BASE[by[c][0]['recipient']]])
    fmt = lambda x: '—' if x is None else f'{x:.3f}'  # noqa: E731
    lv = lambda s: ' / '.join(fmt(s.get(a)) for a in ANSWERS)  # noqa: E731
    dd = lambda s, a: f"{s[a]['diff']:+.3f} [{s[a]['ci'][0]:+.3f}, {s[a]['ci'][1]:+.3f}]" if s and a in s else '—'  # noqa: E731
    L = ['# R08: exchange of the User and System coordinates (PRO, 10-31)', '',
         'Thinking: share of finished items answered you / user / system; no thinking: mean probability of the three options. '
         'Changes: item-paired difference from the unedited message of the same role (95% bootstrap interval). k = 8, fit_q, '
         'all positions; random = the same exchange in the random subspace of base seed 20261022.', '',
         '| readout | message | edit | unedited you / user / system | edited you / user / system | change in "you" | change in "user" | '
         'change in "system" |', '|---|---|---|---|---|---|---|---|']
    for rd, _, _, name in READOUTS:
        if rd not in res:
            continue
        Lv = res[rd]['levels']
        for c, lab in EDITS:
            if c not in Lv:
                continue
            msg, base = ('User', 'base_U') if c.endswith('_U') else ('System', 'base_S')
            L.append(f'| {name} | {msg} | {lab} | {lv(Lv[base])} | {lv(Lv[c])} | {dd(res[rd].get(c), "self")} | '
                     f'{dd(res[rd].get(c), "user")} | {dd(res[rd].get(c), "system")} |')
    L += ['', 'Items: ' + ', '.join(f"{name} {res[rd]['n_items']}" for rd, _, _, name in READOUTS if rd in res)]
    if missing:
        L += ['', 'Not finished (left out): ' + ', '.join(missing)]
    (A / 'stats/r08_20261031.json').write_text(json.dumps({'results': res, 'missing': missing}, indent=1), encoding='utf-8')
    (A / 'stats/r08_20261031.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    sys.stdout.buffer.write(('\n'.join(L) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
