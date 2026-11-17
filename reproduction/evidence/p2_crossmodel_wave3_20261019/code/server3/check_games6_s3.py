"""Release check of the secret game v6 smokes (make_games_secret6.py) on server3 (user 10-27), the rule fixed in advance (as for the earlier games): unedited,
the main question separates the sources (setter's role >= 0.5 with the model's own X, <= 0.25 with the user's X, on finished games),
at least 75% of the games finished, and the control answered correctly in at least 7 of 8 games under both unedited conditions.
PASS writes <full out>.GO for the main and control runs of that readout, FAIL writes .SKIP (the runner then exits); every verdict
goes to games6_check_s3.log. usage: check_games6_s3.py"""
import json
import time
from pathlib import Path

TAGS = ('qwen38_27b_sp', 'gemma4_31b_nat_sp', 'qwen38_27b')
for tag in TAGS:
    try:
        m = json.load(open(f'games_swap/{tag}_s6_smoke_main/summary.json'))['secret']
        c = json.load(open(f'games_swap/{tag}_s6_smoke_control/summary.json'))['secret']
        a, u = m['base_A'], m['base_U']
        ok = (a['finished'] >= 0.75 * a['n'] and u['finished'] >= 0.75 * u['n'] and a['own_share'] >= 0.5 and u['own_share'] <= 0.25
              and all(round(c[k]['accuracy'] * c[k]['finished']) >= 7 for k in ('base_A', 'base_U')))
        msg = (f"{'PASS' if ok else 'FAIL'} own {a['own_share']} {a['reads_as']} | user {u['own_share']} {u['reads_as']} | control "
               f"{c['base_A']['accuracy']} {c['base_U']['accuracy']} | finished {a['finished']} {u['finished']}")
    except Exception as e:  # a missing or broken smoke counts as FAIL
        ok, msg = False, f'FAIL {type(e).__name__}: {e}'
    for q in ('main', 'control'):
        Path(f'games_swap/{tag}_s6_{q}.' + ('GO' if ok else 'SKIP')).touch()
    with open('games6_check_s3.log', 'a', encoding='utf-8') as fh:
        fh.write(f"{time.strftime('%F %T')} {tag} {msg}\n")
    print(tag, msg, flush=True)
