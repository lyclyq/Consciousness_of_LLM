"""Signal rules of the R05/R06 pilots (user 10-31: "这三类先做一个 8 题冒烟看看信号，如果有信号我们才做，不然不做也不写"), fixed before
any pilot ran. 8 items per condition (4 held-out bodies x 2 orders; 8 bodies in the third option order for attr_num; 8 bodies for the
content question; the first 8 secret games). Measure: with thinking, the share of finished items answered "you" / "yes" / as the
setter; without thinking, the mean probability of the "you" option / of "yes" (games: share of finished replies as the setter).
Changes: item-paired differences from the unedited message of the same role.
  T1 attr_alt, T2 attr_num (R05, held-out wording / answer mapping; fit_m, message only)
     a readout is testable if unedited own >= 0.5 and unedited User <= 0.25 (thinking: at least 6 of 8 finished in every condition);
     it shows a signal if removal <= -0.25 and the random removal is within +-0.125. Test: signal in at least 3 of the 4 readouts.
  T3 game (R05, the secret game with fit_m, X only)
     testable as the rule of the paper: own X -> setter >= 0.5, User X -> setter <= 0.25, at least 6 of 8 finished, control correct
     in at least 7 of 8 under both unedited conditions; signal: removal <= -0.25 or installation >= +0.25, random within +-0.125.
     Test: signal in at least 2 of the 4 readouts (the main subspace moved the game in 3 of 4).
  T4 polarity (R06; yn_self and yn_user, fit_m, message only)
     testable if, unedited, yn_self is answered yes for own >= 0.75 and for User <= 0.25, and yn_user yes for own <= 0.25 and for
     User >= 0.75; signal: removal lowers yes on yn_self by >= 0.25 AND raises yes on yn_user by >= 0.25, random removal on yn_self
     within +-0.125. Test: signal in at least 3 of the 4 readouts.
  T5 relevant direction (R06; basis P/M at the size of the V edit): the test is valid (the direction is relevant) if, in the content
     question, the content-direction edit of either sign lowers the share answered correctly by >= 0.25, in at least 3 of the 4
     readouts; the attribution changes under P/M are printed but are not part of the rule (once valid, the full run is reported as
     it comes out).
usage: pilot_check_p56.py CROSSMODEL_DIR   (prints one line per test and readout, then the verdicts)"""
import json
import sys
from pathlib import Path

C = Path(sys.argv[1])
RD = [('qsp', 'singlepass_swap/qwen38_27b', 'games_swap/qwen38_27b_sp'), ('qth', 'thinking_swap/qwen38_27b', 'games_swap/qwen38_27b_th'),
      ('g20', 'thinking_swap/gpt_oss_20b', 'games_swap/gpt_oss_20b'), ('g120', 'thinking_swap/gpt_oss_120b', 'games_swap/gpt_oss_120b')]


def load(d):
    recs = {}
    for f in sorted((C / d / 'items').glob('*.json')) if (C / d / 'items').is_dir() else []:
        r = json.loads(f.read_text(encoding='utf-8'))
        recs.setdefault(r['cond'], []).append(r)
    return recs


def value(r, kind):
    if kind == 'content':
        return float(r['content']['correct']) if r.get('finished', True) else None
    if kind == 'game':
        return float(r['answer'].get('reads_as') == 'assistant') if r['finished'] else None
    if kind == 'control':
        return float(bool(r['answer'].get('correct'))) if r['finished'] else None
    if 'p' in r:
        return r['p']['yes'] if kind == 'yn' else r['p']['self']
    if not r['finished']:
        return None
    return float(r.get('yn') == 'yes') if kind == 'yn' else float(r.get('on_answer') == 'self')


def key(r):
    return r['id'], tuple(r.get('order') or ())


def level(recs, c, kind):
    xs = [value(r, kind) for r in recs.get(c, [])]
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs) if xs else None), len(xs), len(recs.get(c, []))


def change(recs, c, base, kind):
    b = {key(r): value(r, kind) for r in recs.get(base, [])}
    d = [value(r, kind) - b[key(r)] for r in recs.get(c, []) if key(r) in b and value(r, kind) is not None and b[key(r)] is not None]
    return sum(d) / len(d) if d else None


def fmt(x):
    return '--' if x is None else f'{x:+.2f}'


def main():
    verdict = {}
    for test, kind, suffix in (('T1 attr_alt', 'attr', '_p56_alt'), ('T2 attr_num', 'attr', '_p56_num')):
        hits = 0
        for rd, d, _ in RD:
            R = load(d + suffix)
            a, na, ta = level(R, 'base_A', kind)
            u, nu, tu = level(R, 'base_U', kind)
            fin_ok = all(level(R, c, kind)[1] >= 6 for c in R) if 'singlepass' not in d else True
            ok = a is not None and u is not None and a >= 0.5 and u <= 0.25 and fin_ok and len(R) >= 5
            rem, ins, rnd = change(R, 'swU8_A', 'base_A', kind), change(R, 'swA8_U', 'base_U', kind), change(R, 'rswU8_A', 'base_A', kind)
            sig = ok and rem is not None and rem <= -0.25 and rnd is not None and abs(rnd) <= 0.125
            hits += sig
            print(f'{test} {rd}: unedited own {fmt(a)} User {fmt(u)} testable {ok} | removal {fmt(rem)} installation {fmt(ins)} random {fmt(rnd)} | signal {sig}')
        verdict[test] = hits >= 3
    hits = 0
    for rd, _, g in RD:
        M, Ct = load(g + '_p56_game_main'), load(g + '_p56_game_control')
        a, _, _ = level(M, 'base_A', 'game')
        u, _, _ = level(M, 'base_U', 'game')
        fin_ok = all(level(M, c, 'game')[1] >= 6 for c in M)
        ctrl = [level(Ct, c, 'control')[0] for c in ('base_A', 'base_U')]
        ok = a is not None and u is not None and a >= 0.5 and u <= 0.25 and fin_ok and all(x is not None and x >= 7 / 8 for x in ctrl)
        rem, ins, rnd = change(M, 'swU8_A', 'base_A', 'game'), change(M, 'swA8_U', 'base_U', 'game'), change(M, 'rswU8_A', 'base_A', 'game')
        sig = ok and rnd is not None and abs(rnd) <= 0.125 and ((rem is not None and rem <= -0.25) or (ins is not None and ins >= 0.25))
        hits += sig
        print(f'T3 game {rd}: unedited own {fmt(a)} User {fmt(u)} control {[fmt(x) for x in ctrl]} testable {ok} | removal {fmt(rem)} '
              f'installation {fmt(ins)} random {fmt(rnd)} | signal {sig}')
    verdict['T3 game'] = hits >= 2
    hits = 0
    for rd, d, _ in RD:
        S, U = load(d + '_p56_yn_self'), load(d + '_p56_yn_user')
        sa, su = level(S, 'base_A', 'yn')[0], level(S, 'base_U', 'yn')[0]
        ua, uu = level(U, 'base_A', 'yn')[0], level(U, 'base_U', 'yn')[0]
        ok = None not in (sa, su, ua, uu) and sa >= 0.75 and su <= 0.25 and ua <= 0.25 and uu >= 0.75
        rs, ru = change(S, 'swU8_A', 'base_A', 'yn'), change(U, 'swU8_A', 'base_A', 'yn')
        is_, iu = change(S, 'swA8_U', 'base_U', 'yn'), change(U, 'swA8_U', 'base_U', 'yn')
        rnd = change(S, 'rswU8_A', 'base_A', 'yn')
        sig = ok and None not in (rs, ru, rnd) and rs <= -0.25 and ru >= 0.25 and abs(rnd) <= 0.125
        hits += sig
        print(f'T4 polarity {rd}: unedited yes (you-question own/User {fmt(sa)}/{fmt(su)}; user-question own/User {fmt(ua)}/{fmt(uu)}) '
              f'testable {ok} | removal: you-q {fmt(rs)} user-q {fmt(ru)}; installation: you-q {fmt(is_)} user-q {fmt(iu)}; random {fmt(rnd)} | signal {sig}')
    verdict['T4 polarity'] = hits >= 3
    hits = 0
    for rd, d, _ in RD:
        K, A = load(d + '_p56_cdir_content'), load(d + '_p56_cdir')
        acc = level(K, 'base_A', 'content')[0]
        ch = [change(K, c, 'base_A', 'content') for c in ('pcsU8_A', 'mcsU8_A')]
        vch = change(K, 'swU8_A', 'base_A', 'content')
        valid = any(x is not None and x <= -0.25 for x in ch)
        hits += valid
        att = [change(A, c, b, 'attr') for c, b in (('pcsU8_A', 'base_A'), ('mcsU8_A', 'base_A'), ('pcsA8_U', 'base_U'), ('mcsA8_U', 'base_U'))]
        print(f'T5 content direction {rd}: content correct unedited {fmt(acc)}, change under +/- {[fmt(x) for x in ch]}, under the V removal '
              f'{fmt(vch)} | valid {valid} | attribution (information): own +/- {fmt(att[0])} {fmt(att[1])}, User +/- {fmt(att[2])} {fmt(att[3])}')
    verdict['T5 content direction'] = hits >= 3
    print('VERDICT ' + '; '.join(f'{k}: {"signal" if v else "no signal"}' if not k.startswith('T5') else f'{k}: {"valid" if v else "not valid"}'
                                 for k, v in verdict.items()))


if __name__ == '__main__':
    main()
