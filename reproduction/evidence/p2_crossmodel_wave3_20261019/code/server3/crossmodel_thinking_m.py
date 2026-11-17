"""Trust in the model's own steps WITH thinking (user 10-21: the own-step test is run in each mode in which the model recognizes
its own messages, so models with self-reference under thinking get it under thinking, models with both get both).

Same items and messages as phase M of crossmodel_suite.py: the 120 two-step problems (the paper's set), [user: problem]
[ROLE: step] [user: final-answer question], ROLE assistant or user, step correct or wrong -> 480 items. The template's thinking
setting (--on-kwargs); greedy generation up to --max-new tokens, as crossmodel_thinking_b.py; the answer is read from the text
after the end of the reasoning (the last reasoning-close marker). Each item records every number in that text and is classed
'correct' (the correct answer and not the error-propagating one), 'adopted' (the error-propagating answer and not the correct
one), 'other', or 'none' (no number / reasoning unfinished). Rendering checks: text and token renderings agree, the step is
unique and lies between the problem and the question, and a user-written step is a turn of its own (not merged).
usage: crossmodel_thinking_m.py --model DIR --on-kwargs JSON --problems F --out DIR [--lang zh|en] [--batch 16] [--max-new 3072]
                                [--max-memory-gib 44] [--cpu-offload-gib 0] [--limit N]"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402
from crossmodel_suite import Renderer, load  # noqa: E402

CLOSE = ('</think>', '<channel|>', '</seed:think>', '<|channel|>final<|message|>')  # as crossmodel_thinking_b.py
NUM = re.compile(r'-?\d+(?:\.\d+)?')


def messages(p, step, q):
    return [{'role': 'user', 'content': p['problem']}, {'role': None, 'content': step}, {'role': 'user', 'content': q}]


def build(p, which, role, q):
    m = messages(p, p[f'step_{which}'], q)
    m[1]['role'] = role
    return m


def check(tok, kw, p, q):
    for role in ('assistant', 'user'):
        msgs = build(p, 'wrong', role, q)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
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


def run(args):
    out = Path(args.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    q = cs.Q_FINAL_EN if args.lang == 'en' else cs.Q_FINAL
    torch, tok, model, cfg, placement, cls = load(args)
    kw = json.loads(args.on_kwargs)
    R = Renderer(tok, kw)
    problems = json.loads(Path(args.problems).read_text(encoding='utf-8'))
    check(tok, kw, problems[0], q)
    dev0 = model.get_input_embeddings().weight.device
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = {tok.eos_token_id} | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    eos.discard(None)
    todo = []
    for p in problems:
        for which in ('correct', 'wrong'):
            for role in ('assistant', 'user'):
                f = out / 'items' / f'{p["id"]}__{which}__{role}.json'
                if not f.exists():
                    todo.append((f, p, which, role))
    if args.limit:
        todo = todo[:args.limit]
    ident = {'experiment': 'P2_thinking_M', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'suite_sha256': hashlib.sha256(Path(cs.__file__).read_bytes()).hexdigest(), 'model': args.model, 'on_kwargs': kw, 'lang': args.lang, 'question': q,
             'batch': args.batch, 'max_new': args.max_new, 'n_problems': len(problems), 'torch': torch.__version__,
             'transformers': __import__('transformers').__version__}
    (out / 'identity.json').write_text(json.dumps(ident, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{len(todo)} items to run, batch {args.batch}', flush=True)
    t0 = time.time()
    for s0 in range(0, len(todo), args.batch):
        chunk = todo[s0:s0 + args.batch]
        ons = [R.render(build(p, which, role, q), p[f'step_{which}'])[0] for f, p, which, role in chunk]
        L = max(len(x) for x in ons)
        x = torch.tensor([[pad] * (L - len(s)) + s for s in ons], dtype=torch.long, device=dev0)
        am = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in ons], dtype=torch.long, device=dev0)
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=am, max_new_tokens=args.max_new, do_sample=False, pad_token_id=pad)
        for k, (f, p, which, role) in enumerate(chunk):
            new = g[k, L:].tolist()
            n_new = len(new)
            for j, t in enumerate(new):
                if t in eos:
                    n_new = j + 1
                    break
            gen = tok.decode(new[:n_new], skip_special_tokens=False)
            finished = any(c in gen for c in CLOSE)
            tail = gen
            for c in CLOSE:
                if c in gen:
                    tail = gen.split(c)[-1]
            clean = re.sub(r'<[^>]+>', ' ', tail).replace('**', '').replace(',', '')
            nums = NUM.findall(clean) if finished else []
            rec = {'id': p['id'], 'family': p['family'], 'split': p['split'], 'step': which, 'role': role,
                   'answer': p['answer'], 'answer_if_adopted': p['answer_if_adopted'], 'numbers': nums,
                   'class': classify(nums, p) if finished else 'none', 'finished': finished, 'n_new': int(n_new),
                   'tail': tail[-300:], 'gen': gen if args.save_gen else None, 'batch': args.batch}
            f.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--on-kwargs', default='{}')
    p.add_argument('--problems', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--lang', choices=('zh', 'en'), default='zh')
    p.add_argument('--max-new', type=int, default=3072)
    p.add_argument('--max-memory-gib', type=float, default=44)
    p.add_argument('--cpu-offload-gib', type=float, default=0)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--limit', type=int, default=0, help='pilot: first N pending items')
    p.add_argument('--save-gen', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    a = p.parse_args()
    if a.summarize_only:
        summarize(Path(a.out))
    else:
        run(a)


if __name__ == '__main__':
    main()
