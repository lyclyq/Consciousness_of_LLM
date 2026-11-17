"""Reasoning-trace counts for the English thinking runs of Qwen3.8-27B (PRO, 11-02), with the rule of trace_counts_20261026.py that
gives the paper's 102 of 150 in Chinese: a trace "describes an assistant message" if it mentions the assistant ("assistant",
case-insensitive, not followed by ")") after the option text "AI assistant" is removed. Runs: thinking_swap/qwen38_27b_en_all
(8 dims: swA8_U, base_U, rswA8_U) and thinking_swap/qwen38_27b_en_all_k64 (64 dims: swA64_U, rswA64_U); the Chinese runs of the
paper (runs/server3/thinking_swap/qwen38_27b_all and _all_k64) are printed alongside. For each condition also the answers of the
items whose trace describes an assistant message.
usage: trace_counts_q38en_20261102.py   (run from evidence/p2_crossmodel_wave3_20261019)"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()


def long_path(p):
    """Item paths exceed the Windows path limit; the extended-length prefix lifts it."""
    return Path('\\\\?\\' + str(p.resolve())) if sys.platform == 'win32' else p


def items(d, cond):
    xs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted(long_path(ROOT / d / 'items').glob('*.json'))]
    return [x for x in xs if x['cond'] == cond]


def describes_assistant(s):
    s_en = s.replace('AI assistant', '').replace('AI Assistant', '')
    s_zh = s.replace('AI 助手', '').replace('AI助手', '')
    return bool(re.search(r'assistant(?! *\))', s_en, re.I)) or '助手' in s_zh


assert describes_assistant('there is an assistant message before') and not describes_assistant('A. the AI assistant (you)')
assert not describes_assistant('the assistant) said') and describes_assistant('这是助手的回复')

EN, ZH = 'runs/pro/qwen38_en_20261102/thinking_swap/qwen38_27b_en_all', 'runs/server3/thinking_swap/qwen38_27b_all'
for lang, d8, d64 in (('en', EN, EN + '_k64'), ('zh', ZH, ZH + '_k64')):
    for d, conds in ((d8, ('base_U', 'swA8_U', 'rswA8_U')), (d64, ('swA64_U', 'rswA64_U'))):
        for c in conds:
            xs = items(d, c)
            hit = [x for x in xs if describes_assistant(x.get('gen', ''))]
            print(f'{lang} {c:8s} describes an assistant message {len(hit):>3}/{len(xs)}; answers of those {dict(Counter(x.get("on_answer") for x in hit))}; '
                  f'all answers {dict(Counter(x.get("on_answer") for x in xs))}')
