"""Reasoning-trace counts of Section 4.5 ("They are sufficient where nothing else names the source"), reproduced from the per-item
results (user 10-26: every number of the paper has a script). Rules as used for the text on 10-24:
  - Qwen3.8-27B thinking, runs/server3/thinking_swap/qwen38_27b_all: a trace "describes an assistant message" if it mentions the
    assistant ("assistant", case-insensitive, not followed by ")", or 助手) after the answer option is removed from the trace
    ("AI assistant" / "AI Assistant" / "AI 助手" / "AI助手", the option text of the attribution question, which every trace quotes);
    paper: 102 of 150 for the edited User message (swA8_U), none unedited (base_U), 3 after the random-subspace edit (rswA8_U).
  - gpt-oss-20b, runs/pro/thinking_swap/gpt_oss_20b_all: traces for the edited System message (swA8_S) containing the exact phrase
    "the developer message" (case-insensitive); paper: 147 of 150 (149 contain "developer" at all, which the text does not claim).
usage: trace_counts_20261026.py   (run from evidence/p2_crossmodel_wave3_20261019)"""
import glob
import json
import re

QWEN, GPTOSS = 'runs/server3/thinking_swap/qwen38_27b_all', 'runs/pro/thinking_swap/gpt_oss_20b_all'


def items(d, cond):
    xs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(d + '/items/*.json'))]
    return [x for x in xs if x['cond'] == cond]


def describes_assistant(s):
    s_en = s.replace('AI assistant', '').replace('AI Assistant', '')
    s_zh = s.replace('AI 助手', '').replace('AI助手', '')
    return bool(re.search(r'assistant(?! *\))', s_en, re.I)) or '助手' in s_zh


rows = []
for cond, paper in (('swA8_U', 102), ('base_U', 0), ('rswA8_U', 3)):
    xs = items(QWEN, cond)
    got = sum(describes_assistant(x.get('gen', '')) for x in xs)
    rows.append((f'Qwen3.8-27B thinking {cond}: describes an assistant message', paper, got, len(xs)))
    plain = sum(bool(re.search('assistant', x.get('gen', ''), re.I)) or '助手' in x.get('gen', '') for x in xs)
    rows.append((f'  (without removing the option text: {plain}/{len(xs)}, not the paper rule)', None, None, None))
xs = items(GPTOSS, 'swA8_S')
g = [x.get('gen', '').lower() for x in xs]
rows.append(('gpt-oss-20b swA8_S: "the developer message"', 147, sum('the developer message' in s for s in g), len(xs)))
rows.append((f'  (any "developer": {sum("developer" in s for s in g)}/{len(xs)}, not claimed)', None, None, None))

ok = True
for name, paper, got, n in rows:
    if paper is None:
        print(name)
        continue
    flag = 'OK' if paper == got else 'MISMATCH'
    ok &= paper == got
    print(f'{name:62s} paper {paper:>3}/150  recomputed {got:>3}/{n}  {flag}')
print('all match' if ok else 'MISMATCH FOUND')
