"""Answer check of the Qwen3.8-27B English thinking runs of q38er (content, arithmetic, secret game v6 main and control) and q38ek
(64-dim installation), pulled into runs/pro/qwen38_en_20261102 on 11-02: the as-run reading of every finished item against a
strict reading of its final text (the text after the last </think>, end tokens removed); disagreements and ambiguous items are
listed with their final text.
  attribution letter  as-run: first capital A/B/C of the tail (crossmodel_role_swap.py f2362716);
                      strict: 'answer ... X', else the only standalone A/B/C;
  content, arithmetic as-run: first number of the tail, commas removed; strict: the only number, else the number after 'answer';
  game main           as-run: 'go ahead' -> setter, else a number -> guesser (crossmodel_games.py 5b29640d);
                      strict: exactly one of the two present;
  game control        as-run: first number; strict: the only number.
usage: python check_q38er_ek_20261102.py WAVE3_DIR"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

W = Path(sys.argv[1])
EN = W / 'runs/pro/qwen38_en_20261102'
STANDALONE = re.compile(r'(?<![A-Za-z])([ABC])(?![A-Za-z])')
AFTER_ANSWER = re.compile(r'answer\W{0,20}?([ABC])(?![A-Za-z])', re.I)
GO_AHEAD = re.compile(r'请猜|go ahead', re.I)


def final(x):
    t = (x.get('gen') or x['tail']).split('</think>')[-1]
    return re.sub(r'<\|[^|]*\|>', ' ', t).replace('**', '')


def strict_letter(t):
    m = AFTER_ANSWER.search(t)
    if m:
        return m.group(1)
    found = set(STANDALONE.findall(t))
    return found.pop() if len(found) == 1 else None


def strict_number(t):
    nums = set(re.findall(r'\d+', t.replace(',', '')))
    if len(nums) == 1:
        return nums.pop()
    m = re.search(r'answer\D{0,20}?(\d+)', t.replace(',', ''), re.I)
    return m.group(1) if m else None


def strict_move(t):
    go, num = bool(GO_AHEAD.search(t)), bool(re.search(r'\d+', t))
    return 'setter' if go and not num else ('guesser' if num and not go else None)


assert strict_letter('\n\nA ') == 'A' and strict_letter('Answer: C') == 'C' and strict_letter('A or B') is None
assert strict_letter('Because it was the assistant') is None
assert strict_number('\n\n2,414 ') == '2414' and strict_number('12 or 13') is None and strict_number('The answer is 7 (not 8)') == '7'
assert strict_move('Go ahead') == 'setter' and strict_move('50') == 'guesser' and strict_move('Go ahead, 50') is None


def long_path(p):
    """Item paths below EN exceed the Windows path limit; the extended-length prefix lifts it."""
    return Path('\\\\?\\' + str(p.resolve())) if sys.platform == 'win32' else p


def items(run):
    return [json.loads(f.read_text(encoding='utf-8')) for f in sorted(long_path(EN / run / 'items').glob('*.json'))]


def report(run, asrun, strict):
    its = items(run)
    fin = [x for x in its if x['finished']]
    dis, amb = [], []
    for x in fin:
        a, s = asrun(x), strict(final(x))
        if s is None:
            amb.append((x, a))
        elif s != a:
            dis.append((x, a, s))
    print(f'== {run}: {len(its)} items, {len(fin)} finished, {len(dis)} disagree, {len(amb)} ambiguous (strict reading)')
    for x, a, s in dis[:12]:
        print(f'   DISAGREE {x["cond"]} {x["id"]} as-run {a} strict {s} | {final(x)[-140:]!r}')
    for x, a in amb[:12]:
        print(f'   AMBIGUOUS {x["cond"]} {x["id"]} as-run {a} | {final(x)[-140:]!r}')
    unf = Counter(x['cond'] for x in its if not x['finished'])
    if unf:
        print(f'   unfinished by condition: {dict(unf)}')


report('thinking_swap/qwen38_27b_en_all_k64', lambda x: x['on_letter'], strict_letter)
report('thinking_swap/qwen38_27b_en_all_content', lambda x: x['content']['given'], strict_number)
report('thinking_swap/qwen38_27b_en_all_arith', lambda x: x['content']['given'], strict_number)
report('games_swap/qwen38_27b_en_s6_main', lambda x: x['answer']['type'], strict_move)
report('games_swap/qwen38_27b_en_s6_control', lambda x: x['answer']['given'], strict_number)
