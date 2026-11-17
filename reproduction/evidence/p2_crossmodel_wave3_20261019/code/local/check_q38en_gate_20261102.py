"""Gate check before GO_Q38EN_MAIN (Qwen3.8-27B English causal core, PRO, 11-02): read the two fit runs pulled into
runs/pro/qwen38_en_20261102/ and compare with the Chinese runs and the English main probe.
  1. single-pass fit run (singlepass_ablate/qwen38_27b_en/items.json): P(you) of base_A / base_U on the 75 held-out bodies x 2 orders,
     next to the English main probe on the same bodies (8-model archive, geo_attr_en, first two orders) and the Chinese run;
  2. thinking fit run (thinking_ablate4/qwen38_27b_en/items/base_*): finished share, answer shares, and the as-run letter
     (on_letter, first capital A/B/C after the reasoning) against a strict reading of the text after </think>:
     a standalone A/B/C (not inside a word), preferring one after "answer"; disagreements and ambiguous items are listed, and
     N_SHOW random items are printed with their final text for a hand check.
usage: python check_q38en_gate_20261102.py WAVE3_DIR M8_DIR"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

W, M8 = Path(sys.argv[1]), Path(sys.argv[2])
EN = W / 'runs/pro/qwen38_en_20261102'
N_SHOW = 25
STANDALONE = re.compile(r'(?<![A-Za-z])([ABC])(?![A-Za-z])')
AFTER_ANSWER = re.compile(r'answer\W{0,20}?([ABC])(?![A-Za-z])', re.I)


def strict_letter(text):
    """Letter of the final answer: text after the last </think>, end tokens removed; 'answer ... X' first, else the only standalone
    letter; None when absent or ambiguous."""
    t = text.split('</think>')[-1]
    t = re.sub(r'<\|[^|]*\|>', ' ', t)
    m = AFTER_ANSWER.search(t)
    if m:
        return m.group(1)
    found = set(STANDALONE.findall(t))
    return found.pop() if len(found) == 1 else None


assert strict_letter('\n\nA<|im_end|>') == 'A'
assert strict_letter('Answer: C') == 'C'
assert strict_letter('Based on the above, B') == 'B'
assert strict_letter('**B**') == 'B'
assert strict_letter('A or B') is None
assert strict_letter('Because it was the assistant') is None

# 1. single pass
sp = json.loads((EN / 'singlepass_ablate/qwen38_27b_en/items.json').read_text(encoding='utf-8'))
zh = json.loads((W / 'runs/server3/singlepass_ablate/qwen38_27b/items.json').read_text(encoding='utf-8'))


def pyou(items, cond):
    v = [x['p']['self'] for x in items if x['cond'] == cond]
    return len(v), sum(v) / len(v), min(x['mass'] for x in items if x['cond'] == cond)


held = sorted({x['id'] for x in sp})
probe = []
for i in held:
    d = json.loads((M8 / 'models/qwen38_27b/non_thinking/geo_attr_en' / f'{i}.json').read_text(encoding='utf-8'))
    r = d['roles']
    probe.append({k: sum(o['self'] for o in r[k]['by_order'][:2]) / 2 for k in ('assistant', 'user')})
print('1. single pass, 75 held-out bodies x 2 orders')
for c in sorted({x['cond'] for x in sp}, key=lambda c: (c[-1], c)):
    n, m, mass = pyou(sp, c)
    nz, mz, _ = pyou(zh, c) if any(x['cond'] == c for x in zh) else (0, float('nan'), 0)
    print(f'   {c:8s} n {n:3d}  P(you) en {m:.3f}  (zh {mz:.3f})  letter mass min {mass:.3f}')
print(f'   main probe en, same bodies: own {sum(p["assistant"] for p in probe) / len(probe):.3f}, User {sum(p["user"] for p in probe) / len(probe):.3f}')

# 2. thinking
items = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((EN / 'thinking_ablate4/qwen38_27b_en/items').glob('base_*.json'))]
print(f'\n2. thinking fit run: {len(items)} items')
for c in ('base_A', 'base_U'):
    it = [x for x in items if x['cond'] == c]
    fin = [x for x in it if x['finished']]
    ans = Counter(x['on_answer'] for x in fin)
    print(f'   {c}: n {len(it)}, finished {len(fin)}, answers {dict(ans)}, P(you) among finished '
          f'{ans.get("self", 0) / max(1, len(fin)):.3f}, mean new tokens {sum(x["n_new"] for x in it) / max(1, len(it)):.0f}')
dis, amb = [], []
for x in items:
    if not x['finished']:
        continue
    s = strict_letter(x.get('gen') or x['tail'])
    if s is None:
        amb.append(x)
    elif s != x['on_letter']:
        dis.append((x, s))
print(f'   strict reading vs as-run letter (finished items): {len(dis)} disagree, {len(amb)} ambiguous')
for x, s in dis[:10]:
    print('     DISAGREE', x['cond'], x['id'], x['order'], 'as-run', x['on_letter'], 'strict', s, '|', repr((x.get('gen') or x['tail']).split('</think>')[-1][-160:]))
for x in amb[:10]:
    print('     AMBIGUOUS', x['cond'], x['id'], 'as-run', x['on_letter'], '|', repr((x.get('gen') or x['tail']).split('</think>')[-1][-160:]))
print(f'\n   {N_SHOW} random finished items (as-run letter | final text after </think>):')
rng = random.Random(0)
for x in rng.sample([x for x in items if x['finished']], min(N_SHOW, sum(x['finished'] for x in items))):
    print(f'     {x["cond"]} {x["on_letter"]} -> {x["on_answer"]:6s} | {(x.get("gen") or x["tail"]).split("</think>")[-1].strip()[:80]!r}')
ex = next(x for x in items if x['cond'] == 'base_A' and x.get('gen'))
print('\n   one base_A generation (first 600 characters):\n', ex['gen'][:600])
