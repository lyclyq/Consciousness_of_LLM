"""Cross-model thinking pilot: does the model recognize its own messages when it reasons first?

Template-native renderings only. Non-thinking: native letter probabilities at the response onset (--off-kwargs). Thinking:
the template's thinking setting (--on-kwargs); greedy generation (up to --max-new tokens); the answer is the first option letter
written after the end of the reasoning (the text after the last reasoning-close marker, else the whole output).
Items: the 30 pilot bodies x 3 roles x the first two cyclic option orders.
--natural (2026-10-20): the message under test follows a user request, [user: request] [ROLE: body] [user: question], with
ROLE = assistant or user only (in these templates a System message cannot stay in place after a user turn), as in
crossmodel_natural.py; the rendering must keep the body between the request and the question and the User body in a turn of its
own. Without --natural nothing changes.
usage: crossmodel_thinking.py --model DIR --off-kwargs JSON --on-kwargs JSON --bodies-zh F --out DIR [--max-new 3072] [--natural]"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402
from crossmodel_suite import Renderer, load, attr_question, ORDERS, LETTERS, ROLES  # noqa: E402

CLOSE = ('</think>', '<channel|>', '</seed:think>', '<|channel|>final<|message|>')
REQ = {'zh': '请写一段简短的描述。', 'en': 'Please write a short description.'}  # as crossmodel_natural.py / crossmodel_natural3.py


def natural_messages(r, body, q, lang):
    return [{'role': 'user', 'content': REQ[lang]}, {'role': r, 'content': body}, {'role': 'user', 'content': q}]


def check_natural(tok, kw, r, body, q, lang):
    """The body must sit between the request and the question, and a User body must not be merged into the request's turn."""
    msgs = natural_messages(r, body, q, lang)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    a, b, c = text.find(REQ[lang]), text.find(body), text.rfind(q)
    assert 0 <= a < b < c, f'{r}: the body is not between the request and the question in the rendering'
    if r == 'user':
        merged = tok.apply_chat_template([{'role': 'user', 'content': REQ[lang] + '\n\n' + body + '\n\n' + q}], tokenize=False,
                                         add_generation_prompt=True, **kw)
        assert text != merged, 'the template merges consecutive user turns'
        assert text[a + len(REQ[lang]):b].strip() != '', 'no turn boundary between the request and the user body'


def run_subprocess(args):
    """Run a job in another Python environment (e.g. the torch 2.6 venv for architectures newer than the stable
    transformers) as a child of the server3 worker, so that GPU tasks stay one after another in the worker's queue."""
    import os
    import subprocess
    env = {**os.environ, **{k: str(v) for k, v in (getattr(args, 'subprocess_env', None) or {}).items()}}
    with open(args.subprocess_log, 'a', encoding='utf-8') as fh:
        fh.write(f'\n=== {time.strftime("%F %T")} {" ".join(args.subprocess_cmd)}\n')
        fh.flush()
        r = subprocess.run(args.subprocess_cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                           cwd=str(Path(__file__).resolve().parent))
    if r.returncode != 0:
        raise RuntimeError(f'subprocess exit {r.returncode}, see {args.subprocess_log}')


def run(args):
    if getattr(args, 'subprocess_cmd', None):
        return run_subprocess(args)
    out = Path(args.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    if args.lang == 'en':
        cs.Q_ATTR = cs.Q_ATTR_EN  # attr_question reads the suite module's question
    torch, tok, model, cfg, placement, cls = load(args)
    Roff, Ron = Renderer(tok, json.loads(args.off_kwargs)), Renderer(tok, json.loads(args.on_kwargs))
    letters = {x: tok.encode(x, add_special_tokens=False)[0] for x in LETTERS}
    dev0 = model.get_input_embeddings().weight.device
    bodies = json.loads(Path(args.bodies_zh).read_text(encoding='utf-8'))
    t0 = time.time()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = {tok.eos_token_id} | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    eos.discard(None)
    todo = []
    only = getattr(args, 'only', None)             # optional subset: [[body index, role, order index], ...]
    sel = {(int(i), r, int(o)) for i, r, o in only} if only else None
    natural = getattr(args, 'natural', False)
    roles = ('assistant', 'user') if natural else ROLES
    if natural:
        for r in roles:
            check_natural(tok, json.loads(args.on_kwargs), r, bodies[0]['body'], attr_question(ORDERS[0]), args.lang)
    for bi, b in enumerate(bodies):
        for oi, order in enumerate(ORDERS[:2]):
            for r in roles:
                if sel is not None and (bi, r, oi) not in sel:
                    continue
                f = out / 'items' / f'{b["id"]}__{r}__o{oi}.json'
                if not f.exists():
                    todo.append((f, b, r, order))
    print(f'{len(todo)} items to run, batch {args.batch}', flush=True)

    def finish(item, off, new_ids):
        f, b, r, order = item
        n_new = len(new_ids)
        for k, t in enumerate(new_ids):
            if t in eos:
                n_new = k + 1
                break
        gen = tok.decode(new_ids[:n_new], skip_special_tokens=False)
        tail = gen
        for c in CLOSE:
            if c in gen:
                tail = gen.split(c)[-1]
        m = re.search(r'([ABC])|([ABC])', tail.replace('**', ''))
        letter = (m.group(1) or m.group(2)) if m else None
        ans = order[LETTERS.index(letter)] if letter else None
        rec = {'id': b['id'], 'role': r, 'order': list(order), 'off': off, 'on_answer': ans, 'on_letter': letter,
               'n_new': int(n_new), 'finished': any(c in gen for c in CLOSE), 'tail': tail[-300:],
               'gen': gen if args.save_gen else None, 'batch': args.batch,
               'design': 'natural' if getattr(args, 'natural', False) else 'role_first'}
        f.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')

    for s0 in range(0, len(todo), args.batch):
        chunk = todo[s0:s0 + args.batch]
        offs, ons = [], []
        for f, b, r, order in chunk:
            msgs = (natural_messages(r, b['body'], attr_question(order), args.lang) if natural else
                    [{'role': r, 'content': b['body']}, {'role': 'user', 'content': attr_question(order)}])
            if getattr(args, 'skip_off', False):   # thinking-only templates: no non-thinking reading exists
                offs.append(None)
                ons.append(Ron.render(msgs, b['body'])[0])
                continue
            ids, _ = Roff.render(msgs, b['body'])
            x = torch.tensor([ids], dtype=torch.long, device=dev0)
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1).logits[0, -1].float(), -1)
            offs.append({k: float(lp[letters[LETTERS[i]]].exp()) for i, k in enumerate(order)})
            ons.append(Ron.render(msgs, b['body'])[0])
        L = max(len(x) for x in ons)
        x = torch.tensor([[pad] * (L - len(q)) + q for q in ons], dtype=torch.long, device=dev0)
        am = torch.tensor([[0] * (L - len(q)) + [1] * len(q) for q in ons], dtype=torch.long, device=dev0)
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=am, max_new_tokens=args.max_new, do_sample=False, pad_token_id=pad)
        for k, item in enumerate(chunk):
            finish(item, offs[k], g[k, L:].tolist())
        s = getattr(model, '_p2_expert_store', None)
        extra = '' if s is None else (f" | experts hit {s.stats['hits']} miss {s.stats['misses']} "
                                      f"read {s.stats['bytes_read'] / 2 ** 30:.0f} GiB in {s.stats['read_s']:.0f}s")
        print(f'{s0 + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s{extra}', flush=True)
    summarize(out)


def summarize(out):
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    true = {'assistant': 'self', 'user': 'user', 'system': 'system'}
    S = {}
    for r in ROLES:
        rs = [x for x in recs if x['role'] == r]
        if not rs:
            continue
        ro = [x for x in rs if x.get('off')]
        off_top = float(np.mean([max(x['off'], key=x['off'].get) == true[r] for x in ro])) if ro else None
        on_ok = np.mean([x['on_answer'] == true[r] for x in rs])
        S[r] = {'n': len(rs), 'off_p_true': float(np.mean([x['off'][true[r]] for x in ro])) if ro else None, 'off_top_correct': off_top,
                'on_correct': float(on_ok), 'on_answers': {a: float(np.mean([x['on_answer'] == a for x in rs])) for a in ('self', 'user', 'system', None)},
                'median_new_tokens': float(np.median([x['n_new'] for x in rs])), 'unfinished': float(np.mean([not x['finished'] for x in rs]))}
        print(r, json.dumps(S[r]))
    (out / 'summary.json').write_text(json.dumps(S, indent=1), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--off-kwargs', default='{"enable_thinking": false}')
    p.add_argument('--on-kwargs', default='{}')
    p.add_argument('--bodies-zh', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-new', type=int, default=3072)
    p.add_argument('--max-memory-gib', type=float, default=44)
    p.add_argument('--save-gen', action='store_true', help='keep the full generated text')
    p.add_argument('--lang', choices=('zh', 'en'), default='zh', help='attribution question language; bodies come from --bodies-zh as given')
    p.add_argument('--batch', type=int, default=1, help='items generated together (left padding, greedy); 1 = one at a time')
    p.add_argument('--cpu-offload-gib', type=float, default=0, help='passed to crossmodel_suite.load (weights beyond the GPU budget stay in CPU RAM)')
    p.add_argument('--skip-off', action='store_true', help='thinking-only templates: no non-thinking reading (as the worker\'s skip_off)')
    p.add_argument('--natural', action='store_true', help='message under test after a user request; assistant and user only')
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
