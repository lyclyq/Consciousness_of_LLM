"""Numbers for the parts of the causal tests decided for paper v4 after make_v4_mechanism.py (user 10-24 to 10-27), computed from
the per-item results in the wave-3 archive:
  - Gemma-4-31B without thinking, in natural order (its claim to its own message appears only when the message answers a user
    request; appendix only, user 10-27): every layer, groups of positions, thirds, content and arithmetic questions
    (runs/server3/singlepass_swap/gemma4_31b_nat_*), and its layer windows (8 layers, step 4, every position; 14 windows);
  - the single layers L39 (Qwen) and L30 (Gemma);
  - 64 instead of 8 dimensions at every layer, attribution question, every readout (*_all_k64; their unedited conditions are
    copies of the 8-dimension runs, identical prompts);
  - the secret game v6 (runs/*/games_swap/*_s6_main and *_s6_control; make_games_secret6.py, v5 plus one System sentence; v5 is
    sealed, user 10-27 "都用 v6 处理覆盖"): which role the model takes after
    "I'll be the setter, number ready, the guesser has m tries" was said by itself or by the user, and the content control (the
    number of tries), 8 dimensions only (user 10-26).
The layer windows of Qwen3.8-27B and gpt-oss-20b are in make_v4_mechanism.py (10-27). The 10-27 version of this script (v5 games) is
make_v4_mech_ext_1027.py; the 10-26 version (v1 games,
64-dimension games, Qwen/Gemma window figure) is make_v4_mech_ext_1026.py.
Attribution statistics as in make_v4_mechanism.py (same functions): single pass, mean P("you") with a bootstrap interval; thinking,
share of finished items answered "you" (Wilson); item-paired change from the unedited message of the same role. Game: share of
finished items in which the model acts as the setter (its reply reads the setter's message as its own), item-paired change on items
finished in both conditions; control: share answered correctly.
Outputs: ARCHIVE/stats/v4_mech_ext.json, v4_mech_ext_tables.md.
usage: make_v4_mech_ext.py ARCHIVE"""
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_v4_mechanism import items, summarize, wilson, boot  # noqa: E402

READ = {'qwen_sp': ('Qwen3.8-27B, no thinking', 'runs/server3/singlepass_swap/qwen38_27b_{}', 64),
        'qwen_th': ('Qwen3.8-27B, thinking', 'runs/server3/thinking_swap/qwen38_27b_{}', 64),
        'oss_th': ('gpt-oss-20b, thinking', 'runs/pro/thinking_swap/gpt_oss_20b_{}', 24),
        'oss120_th': ('gpt-oss-120b, thinking', 'runs/pro/thinking_swap/gpt_oss_120b_{}', 36),
        'gemma_sp': ('Gemma-4-31B, no thinking, natural order', 'runs/server3/singlepass_swap/gemma4_31b_nat_{}', 60)}
GAMES = {'qwen_sp': 'runs/server3/games_swap/qwen38_27b_sp_s6_{}', 'qwen_th': 'runs/server3/games_swap/qwen38_27b_s6_{}',
         'oss_th': 'runs/pro/games_swap/gpt_oss_20b_s6_{}', 'oss120_th': 'runs/pro/games_swap/gpt_oss_120b_s6_{}',
         'gemma_sp': 'runs/server3/games_swap/gemma4_31b_nat_sp_s6_{}'}
SINGLE = {'qwen_sp': ('L39', 39), 'gemma_sp': ('L30', 30)}


def game_summary(d):
    recs = [r for r in items(d) if r.get('game') == 'secret']
    by = {}
    for r in recs:
        by.setdefault(r['id'], {})[r['cond']] = r
    out = {}
    for c in sorted({r['cond'] for r in recs}):
        rs = [r for r in recs if r['cond'] == c]
        fin = [r for r in rs if r['finished']]
        s = {'n': len(rs), 'finished': len(fin)}
        if 'correct' in rs[0]['answer']:
            s['accuracy'] = sum(r['answer']['correct'] for r in fin) / max(1, len(fin))
            out[c] = s
            continue
        k = sum(r['answer']['reads_as'] == 'assistant' for r in fin)
        s.update(stat=k / max(1, len(fin)), ci=list(wilson(k, len(fin))),
                 roles={a: sum(r['answer']['reads_as'] == a for r in fin) for a in ('assistant', 'user', None)})
        base = {'assistant': 'base_A', 'user': 'base_U'}[rs[0]['recipient']]
        if c != base:
            v = lambda r: float(r['answer']['reads_as'] == 'assistant')  # noqa: E731
            dd = [v(x[c]) - v(x[base]) for x in by.values() if c in x and base in x and x[c]['finished'] and x[base]['finished']]
            if dd:
                s.update(diff=sum(dd) / len(dd), diff_ci=list(boot(dd, seed=1)), n_pairs=len(dd))
        out[c] = s
    return out


def get(A, pattern, suffix, fn=summarize):
    d = A / pattern.format(suffix)
    if not d.exists() or os.path.exists(d / 'NOT_USED.txt'):
        return None
    recs = items(d) if fn is summarize else None
    return fn(recs, suffix) if fn is summarize else fn(d)


def fmt(s, key='diff'):
    if s is None:
        return ''
    if 'accuracy' in s:
        return f"acc {s['accuracy']:.2f} ({s['finished']}/{s['n']})"
    t = f"{s['stat']:.2f}"
    if 'finished' in s and s['finished'] != s['n']:
        t += f" ({s['finished']}/{s['n']} fin.)"
    if key in s:
        t += f"; {s[key]:+.2f} [{s[key + '_ci'][0]:+.2f}, {s[key + '_ci'][1]:+.2f}]"
    return t


def main():
    A = Path(sys.argv[1])
    R = {'gemma': {}, 'windows': {}, 'single': {}, 'k64': {}, 'games': {}}
    for k in ('all', 'all_content', 'all_arith', 'region_body', 'region_end', 'region_question', 'third_early', 'third_middle',
              'third_late', 'third_late_content', 'third_late_arith'):
        s = get(A, READ['gemma_sp'][1], k)
        if s:
            R['gemma'][k] = s
    for ro in ('gemma_sp',):
        R['windows'][ro] = {}
        for d in sorted(glob.glob(str(A / READ[ro][1].format('window_L*')))):
            if os.path.exists(os.path.join(d, 'NOT_USED.txt')):
                continue
            a, b = map(int, re.search(r'window_L(\d+)_(\d+)$', d).groups())
            R['windows'][ro][f'L{a}-L{b}'] = {'layers': [a, b], **summarize(items(d), 'window')}
    for ro, (lab, layer) in SINGLE.items():
        s = get(A, READ[ro][1], lab)
        if s:
            R['single'][ro] = {'layer': layer, **s}
    for ro in READ:
        s = get(A, READ[ro][1], 'all_k64')
        if s:
            R['k64'][ro] = s
        R['games'][ro] = {}
        for q in ('main', 'control'):
            s = get(A, GAMES[ro], q, fn=game_summary)
            if s:
                R['games'][ro][q] = s
    (A / 'stats').mkdir(exist_ok=True)
    json.dump(R, open(A / 'stats' / 'v4_mech_ext.json', 'w', encoding='utf-8'), indent=1)

    L = ['# v4 mechanism, extensions (generated by make_v4_mech_ext.py)', '']
    L += ['## Gemma-4-31B, natural order, single pass', '| run | condition | value; change [95%] |', '|---|---|---|']
    for k, S in R['gemma'].items():
        for c, s in S.items():
            L.append(f'| {k} | {c} | {fmt(s)} |')
    L += ['', '## layer windows (8 layers, every position)', '| readout | window | own <- User | User <- own | own <- User, random |',
          '|---|---|---|---|---|']
    for ro, W in R['windows'].items():
        for w, S in W.items():
            L.append(f"| {ro} | {w} | {fmt(S.get('swU8_A'))} | {fmt(S.get('swA8_U'))} | {fmt(S.get('rswU8_A'))} |")
    L += ['', '## single layer', '| readout | condition | value; change |', '|---|---|---|']
    for ro, S in R['single'].items():
        for c, s in S.items():
            if isinstance(s, dict):
                L.append(f"| {ro} L{S['layer']} | {c} | {fmt(s)} |")
    L += ['', '## 64 dimensions, every layer, attribution', '| readout | condition | value; change |', '|---|---|---|']
    for ro, S in R['k64'].items():
        for c, s in S.items():
            L.append(f'| {ro} | {c} | {fmt(s)} |')
    L += ['', '## secret game v6 (share acting as the setter; control = accuracy)', '| readout | run | condition | value; change |',
          '|---|---|---|---|']
    for ro, G in R['games'].items():
        for q, S in G.items():
            for c, s in S.items():
                L.append(f'| {ro} | {q} | {c} | {fmt(s)} |')
    (A / 'stats' / 'v4_mech_ext_tables.md').write_text('\n'.join(L) + '\n', encoding='utf-8')

    print(open(A / 'stats' / 'v4_mech_ext_tables.md', encoding='utf-8').read())


if __name__ == '__main__':
    main()
