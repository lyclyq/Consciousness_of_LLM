"""Full runs of the R05/R06 signal pilots of the supervisor's review (PRO, from 10-31; code/pro/p56_full_pro.sh). The 8-item pilots
(pilot_check_p56.py, rules fixed before any pilot ran) gave: T1 attr_alt signal, T2 attr_num no signal, T3 game signal, T4 polarity
no signal, T5 content direction not valid; only T1 and T3 run in full (user 10-31: "如果有信号我们才做，不然不做也不写"), in all four
readouts, with fit_m of r05r06_fits (fitted on message positions only) and the paper's k = 8.
  T1 <dir>_p56full_alt: the attribution question in a held-out wording (attr_alt), the message edited only, 75 held-out bodies x 2
     option orders. Reference: <dir>_r05m_msg, the same fit and region with the paper's wording (R05).
  T3 games_swap/<model>_p56full_game_{main,control}: the secret game v6, the coordinates edited at X only (the setter's or the
     user's announcement), all games. Reference: the paper's game runs (*_s6_main / *_s6_control; the paper's fit, all positions).
Statistics as in the paper. T1: share of finished items answered "you" (thinking) or mean probability of the "you" option (no
thinking), item-paired change from the unedited message of the same role, 95% bootstrap interval (seed 1). T3: share of finished
games in which the model acts as the setter, item-paired change on games finished in both conditions (make_v4_mech_ext.game_summary);
control: share answered correctly. Runs not yet in the archive are left out.
Outputs (ARCHIVE/stats): p56full_20261101.json, p56full_20261101.md.
usage: analyze_p56full_20261101.py ARCHIVE"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_r03_20261101 import change, level  # noqa: E402
from analyze_r05r06_20261031 import READOUTS  # noqa: E402
from make_v4_mech_ext import GAMES, game_summary  # noqa: E402
from make_v4_mechanism import items  # noqa: E402

GN = {'qsp': 'qwen38_27b_sp', 'qth': 'qwen38_27b_th', 'g20': 'gpt_oss_20b', 'g120': 'gpt_oss_120b'}
REF_GAME = {'qsp': 'qwen_sp', 'qth': 'qwen_th', 'g20': 'oss_th', 'g120': 'oss120_th'}
T1 = [('removal', 'swU8_A', 'base_A'), ('installation', 'swA8_U', 'base_U'), ('random removal', 'rswU8_A', 'base_A')]


def attr(d):
    by = {}
    for r in items(str(d)):
        by.setdefault(r['cond'], []).append(r)
    out = {'levels': {c: level(rs) for c, rs in sorted(by.items())}}
    for _, c, base in T1:
        if c in by and base in by:
            out[c] = change(by[c], by[base])
    return out


def main():
    A = Path(sys.argv[1])
    res, missing = {'T1': {}, 'T3': {}}, []
    for rd, _, root, _ in READOUTS:
        d = A / 'runs/pro' / f'{root}_p56full_alt'
        if (d / 'summary.json').exists():
            res['T1'][rd] = {'full': attr(d), 'ref': attr(A / 'runs/pro' / f'{root}_r05m_msg')}
        else:
            missing.append(f'{root}_p56full_alt')
        g = {q: A / 'runs/pro/games_swap' / f'{GN[rd]}_p56full_game_{q}' for q in ('main', 'control')}
        if all((x / 'summary.json').exists() for x in g.values()):
            res['T3'][rd] = {'full': {q: game_summary(str(x)) for q, x in g.items()},
                             'ref': {q: game_summary(str(A / GAMES[REF_GAME[rd]].format(q))) for q in ('main', 'control')}}
        else:
            missing += [f'games_swap/{GN[rd]}_p56full_game_{q}' for q, x in g.items() if not (x / 'summary.json').exists()]
    f3 = lambda x: '—' if x is None else f'{x:.3f}'  # noqa: E731
    ch = lambda s: f"{s['diff']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]" if s else '—'  # noqa: E731
    L = ['# Full runs of the signal pilots (R05-1: held-out wording, X-only game; PRO, from 10-31)', '',
         'Pilot verdicts (pilot_check_p56.py): T1 attr_alt signal, T2 attr_num no signal, T3 game signal, T4 polarity no signal, '
         'T5 content direction not valid. fit_m of r05r06_fits, k = 8.', '',
         '## T1: attribution question in a held-out wording, the message edited only', '',
         'Thinking: share of finished items answered "you"; no thinking: mean probability of the "you" option. Change: item-paired '
         'difference from the unedited message of the same role (95% bootstrap interval). Reference: r05m_msg, the same fit and region '
         'with the paper\'s wording.', '',
         '| readout | wording | unedited own / User | removal | installation | random removal | finished (fewest) |',
         '|---|---|---|---|---|---|---|']
    for rd, _, _, name in READOUTS:
        if rd not in res['T1']:
            continue
        for lab, k in (('held-out', 'full'), ('paper (r05m_msg)', 'ref')):
            s = res['T1'][rd][k]
            Lv = s['levels']
            fins = [(v['finished'], v['n']) for v in Lv.values() if 'finished' in v]
            fin = '{}/{}'.format(*min(fins)) if fins else '—'
            L.append(f"| {name} | {lab} | {f3(Lv['base_A']['you'])} / {f3(Lv['base_U']['you'])} | {ch(s.get('swU8_A'))} | "
                     f"{ch(s.get('swA8_U'))} | {ch(s.get('rswU8_A'))} | {fin} |")
    L += ['', '## T3: the secret game, the coordinates edited at X only', '',
          'Share of finished games in which the model acts as the setter; change: item-paired difference from the unedited '
          'announcement of the same role (95% bootstrap interval); control: share of the control question answered correctly. '
          'Reference: the paper\'s game runs (s6; the paper\'s fit, all positions).', '',
          '| readout | run | unedited own / User | removal | installation | random removal | control correct (own / User) |',
          '|---|---|---|---|---|---|---|']
    gd = lambda s: f"{s['diff']:+.3f} [{s['diff_ci'][0]:+.3f}, {s['diff_ci'][1]:+.3f}]" if s and 'diff' in s else '—'  # noqa: E731
    for rd, _, _, name in READOUTS:
        if rd not in res['T3']:
            continue
        for lab, k in (('X only (fit_m)', 'full'), ('paper (s6)', 'ref')):
            M, Ct = res['T3'][rd][k]['main'], res['T3'][rd][k]['control']
            L.append(f"| {name} | {lab} | {f3(M['base_A']['stat'])} / {f3(M['base_U']['stat'])} | {gd(M.get('swU8_A'))} | "
                     f"{gd(M.get('swA8_U'))} | {gd(M.get('rswU8_A'))} | {f3(Ct['base_A']['accuracy'])} / {f3(Ct['base_U']['accuracy'])} |")
    if missing:
        L += ['', 'Not finished or not yet archived (left out): ' + ', '.join(missing)]
    (A / 'stats/p56full_20261101.json').write_text(json.dumps({'results': res, 'missing': missing}, indent=1), encoding='utf-8')
    (A / 'stats/p56full_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    sys.stdout.buffer.write(('\n'.join(L) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
