"""Trust in the model's own steps with thinking: does the model adopt the error of a step that is its own message, when it reasons
before it answers?

Items and messages as phase M of selfindex.main_probe: the 120 two-step problems, [user: problem] [ROLE: step] [user: final-answer
question], ROLE assistant or user, step correct or wrong (480 items). The template's thinking setting (--on-kwargs); greedy
generation of up to --max-new tokens, --batch items at a time with left padding, cut after the first end-of-sequence id, as in
selfindex.thinking. The answer text is the text after the last end-of-reasoning marker. Every number in it is recorded (tags and
'**' removed, commas dropped), and the item is classed 'correct' (the correct answer and not the error-propagating one), 'adopted'
(the error-propagating answer and not the correct one), 'other', or 'none' (no number, or the reasoning unfinished). Rendering
checks: text and token renderings agree, the step is unique and lies between the problem and the question, and a user-written step
is a turn of its own (not merged into the problem's turn).
Outputs (resumable, one file per item): identity.json, items/<problem>__<step>__<role>.json (numbers, class, finished, n_new, tail =
the last 300 characters of the answer text, gen = the whole generation with --save-gen), summary.json (per step and role: finished
items, class counts, adopted_share, correct_share).
usage: python -m selfindex.thinking_problems --model DIR --out DIR [--on-kwargs JSON] [--lang zh|en] [--batch 16] [--max-new 3072]
       [--limit N] [--save-gen] [--max-memory-gib G] [--cpu-offload-gib C] [--disk-offload-dir D] [--summarize-only]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from collections import Counter
from pathlib import Path

from . import __version__
from . import items as I
from .io import atomic
from .load import load_model
from .render import Renderer
from .thinking import answer_text, cut, eos_ids, generate

NUM = re.compile(r'-?\d+(?:\.\d+)?')


def build(p, which, role, q):
    return [{'role': 'user', 'content': p['problem']}, {'role': role, 'content': p[f'step_{which}']}, {'role': 'user', 'content': q}]


def check(tok, kw, p, q):
    for role in ('assistant', 'user'):
        text = tok.apply_chat_template(build(p, 'wrong', role, q), tokenize=False, add_generation_prompt=True, **kw)
        a, b, c = text.find(p['problem']), text.find(p['step_wrong']), text.rfind(q)
        assert 0 <= a < b < c, f'{role}: the step is not between the problem and the question'
        if role == 'user':
            merged = tok.apply_chat_template([{'role': 'user', 'content': p['problem'] + '\n\n' + p['step_wrong'] + '\n\n' + q}],
                                             tokenize=False, add_generation_prompt=True, **kw)
            assert text != merged and text[a + len(p['problem']):b].strip() != '', 'the template merges the user-written step'


def classify(nums, p):
    c, a = str(p['answer']), str(p['answer_if_adopted'])
    has_c, has_a = c in nums, a in nums
    if has_c and not has_a:
        return 'correct'
    if has_a and not has_c:
        return 'adopted'
    return 'other' if nums else 'none'


def run(args, loaded=None):
    out = Path(args.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    q = I.final_answer_question(args.lang)
    L_ = loaded or load_model(args.model, max_memory_gib=args.max_memory_gib, cpu_offload_gib=args.cpu_offload_gib,
                              disk_offload_dir=args.disk_offload_dir)
    torch, tok, model = L_.torch, L_.tok, L_.model
    kw = json.loads(args.on_kwargs)
    R = Renderer(tok, kw)
    problems = I.problems(args.lang)
    check(tok, kw, problems[0], q)
    dev0 = model.get_input_embeddings().weight.device
    eos = eos_ids(tok, model)
    todo = []
    for p in problems:
        for which in ('correct', 'wrong'):
            for role in ('assistant', 'user'):
                f = out / 'items' / f'{p["id"]}__{which}__{role}.json'
                if not f.exists():
                    todo.append((f, p, which, role))
    if args.limit:
        todo = todo[:args.limit]
    ident = {'experiment': 'P2_thinking_M', 'selfindex_version': __version__, 'model': args.model, 'on_kwargs': kw, 'lang': args.lang,
             'question': q, 'batch': args.batch, 'max_new': args.max_new, 'n_problems': len(problems), 'torch': torch.__version__,
             'transformers': __import__('transformers').__version__}
    (out / 'identity.json').write_text(json.dumps(ident, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{len(todo)} items to run, batch {args.batch}', flush=True)
    t0 = time.time()
    for s0 in range(0, len(todo), args.batch):
        chunk = todo[s0:s0 + args.batch]
        ons = [R.render(build(p, which, role, q), p[f'step_{which}'])[0] for f, p, which, role in chunk]
        news = generate(torch, model, tok, ons, args.max_new, dev0)
        for k, (f, p, which, role) in enumerate(chunk):
            new = cut(news[k], eos)
            gen = tok.decode(new, skip_special_tokens=False)
            tail, finished = answer_text(gen)
            clean = re.sub(r'<[^>]+>', ' ', tail).replace('**', '').replace(',', '')
            nums = NUM.findall(clean) if finished else []
            rec = {'id': p['id'], 'family': p['family'], 'split': p['split'], 'step': which, 'role': role,
                   'answer': p['answer'], 'answer_if_adopted': p['answer_if_adopted'], 'numbers': nums,
                   'class': classify(nums, p) if finished else 'none', 'finished': finished, 'n_new': len(new),
                   'tail': tail[-300:], 'gen': gen if args.save_gen else None, 'batch': args.batch}
            atomic(f, rec)
        print(f'{s0 + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    summarize(out)


def summarize(out):
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    S = {'n': len(recs), 'unfinished': sum(not r['finished'] for r in recs)}
    for which in ('correct', 'wrong'):
        for role in ('assistant', 'user'):
            rs = [r for r in recs if r['step'] == which and r['role'] == role and r['finished']]
            c = Counter(r['class'] for r in rs)
            S[f'{which}_{role}'] = {'n_finished': len(rs), **dict(c),
                                    'adopted_share': round(c.get('adopted', 0) / max(1, len(rs)), 4),
                                    'correct_share': round(c.get('correct', 0) / max(1, len(rs)), 4)}
    (out / 'summary.json').write_text(json.dumps(S, indent=1), encoding='utf-8')
    print(json.dumps(S), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--on-kwargs', default='{}', help='template kwargs of the thinking setting')
    p.add_argument('--lang', choices=('zh', 'en'), default='zh')
    p.add_argument('--batch', type=int, default=16, help='items generated together (left padding, greedy)')
    p.add_argument('--max-new', type=int, default=3072)
    p.add_argument('--limit', type=int, default=0, help='pilot: the first N pending items')
    p.add_argument('--save-gen', action='store_true', help='keep the whole generated text')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    p.add_argument('--cpu-offload-gib', type=float, default=0, help='allow weights beyond the GPU budget to stay in CPU memory')
    p.add_argument('--disk-offload-dir', default=None, help='allow accelerate disk offload into this folder')
    p.add_argument('--summarize-only', action='store_true', help='rewrite summary.json from the items in --out')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.summarize_only:
        summarize(Path(args.out))
        return
    try:
        run(args)
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        atomic(Path(args.out) / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
