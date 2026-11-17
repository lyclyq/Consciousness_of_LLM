"""Strict re-read of the thinking-attribution answers of the models added on 11-03 (memory rule: answer parsing must be tested
and spot-checked against the raw text). For every item of each run: the text after the last reasoning-close marker (the record's
'tail', or the full 'gen' where saved) is stripped of end-of-turn tokens, '**' and trailing punctuation; the strict reading is
the letter when that text is exactly A/B/C, or starts with "A." / "A)" / "(A)" / "答案：A" / "Answer: A"; otherwise it is read
by hand from the printed text. The strict reading must equal the recorded on_letter for every finished item, and 'finished'
must mean a close marker within the 3072 new tokens.
usage: python -I check_newdev_answers_20261104.py ARCHIVE_W   (prints a table and writes stats/check_newdev_answers_20261104.txt)"""
import glob
import json
import os
import re
import sys
from collections import Counter

W = sys.argv[1]
RUNS = {'MiMo-7B-RL-0530': 'runs/server3/thinking/mimo7b_rl_0530', 'ERNIE-4.5-21B-A3B-Thinking': 'runs/server3/thinking/ernie45_21b_a3b_thinking',
        'MiMo-VL-7B-RL-2508': 'runs/pro/thinking/mimo_vl7b_rl_2508', 'MiMo-V2.6-Distill-Qwen-9B': 'runs/pro/thinking/mimo_v26_distill_qwen9b'}
EOT = ('<|im_end|>', '<|endoftext|>', '</s>', '<|end_of_sentence|>', '<eos>', '<response>', '</response>')  # ERNIE wraps its answer in <response>
STRICT = re.compile(r'^(?:答案[:：]\s*|Answer:\s*|\()?([ABC])(?:[\.\)）:：]|$)')
lines = []
for name, rel in RUNS.items():
    fs = sorted(glob.glob(os.path.join(W, rel, 'items', '*.json')))
    c, bad, hand = Counter(), [], []
    for f in fs:
        r = json.load(open(f, encoding='utf-8'))
        if not r['finished']:
            c['unfinished'] += 1
            if r['n_new'] < 3072 and r['on_letter'] is not None:
                bad.append((os.path.basename(f), 'unfinished but a letter was read', r['tail'][-80:]))
            continue
        if r['n_new'] > 3072:
            bad.append((os.path.basename(f), 'finished beyond 3072 new tokens', r['n_new']))
        t = r['tail']
        for e in EOT:
            t = t.replace(e, '')
        t = t.replace('**', '').strip().rstrip('。.').strip()
        m = STRICT.match(t)
        strict = m.group(1) if m else None
        if strict is None:
            # tail is the last 300 characters: when the answer text is longer, list the distinct answer letters it shows
            shown = sorted(set(re.findall(r'(?:答案[:：]\s*|选\s*|^|\n)([ABC])(?=[\s\.。:：）)]|$)', t)))
            hand.append((os.path.basename(f), r['on_letter'], 'letters shown ' + ''.join(shown), len(r['tail']), t[-90:]))
            c['by hand'] += 1
        elif strict != r['on_letter']:
            bad.append((os.path.basename(f), f'strict {strict} vs recorded {r["on_letter"]}', t[:120]))
        else:
            c['strict = recorded'] += 1
    lines.append(f'{name}: {len(fs)} items, ' + ', '.join(f'{k} {v}' for k, v in sorted(c.items())) + f', disagreements {len(bad)}')
    lines += [f'   DISAGREE {b}' for b in bad[:20]]
    lines += [f'   by hand: {h}' for h in hand[:40]]
out = '\n'.join(lines)
print(out)
open(os.path.join(W, 'stats', 'check_newdev_answers_20261104.txt'), 'w', encoding='utf-8').write(out + '\n')
