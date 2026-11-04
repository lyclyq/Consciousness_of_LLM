"""Self-reference after reasoning: is the model's own message attributed to the model when the model reasons before it answers?

Every input is the model's own chat-template rendering, checked against apply_chat_template(tokenize=True). Items: the bodies
(--bodies all: the 150 bodies; pilot: the 30 pilot bodies) x 3 roles x the first two cyclic option orders, [ROLE: body] [user:
attribution question]. Two readings of each item:
  off  the template's non-thinking setting (--off-kwargs): the probabilities of the three option letters at the response onset,
       one teacher-forced forward, use_cache=False. --skip-off for templates without a non-thinking setting.
  on   the template's thinking setting (--on-kwargs): greedy generation of up to --max-new tokens, --batch items at a time with
       left padding. The generation is cut after the first end-of-sequence id (the tokenizer's or the generation config's). The
       answer text is the text after the last end-of-reasoning marker (render.CLOSE; when several kinds occur, the one listed last
       decides; the whole generation if there is none), and the answer is the first of the characters A, B, C in it ('**'
       removed), mapped to the author that letter names in the item's option order.
--natural: the message under test follows a user request, [user: request] [ROLE: body] [user: question], with ROLE assistant or
  user only (these templates cannot keep a System message in place after a user turn); the rendering must keep the body between
  the request and the question, and a User body in a turn of its own.
--only: a subset of the items, a JSON list of [body index, role, order index] (or the path of a file holding it).
Outputs (resumable, one file per item): items/<body>__<role>__o<order>.json with off (the letter probabilities by author, or null),
on_answer (self, user, system or null), on_letter, n_new (generated tokens), finished (an end-of-reasoning marker was written),
tail (the last 300 characters of the answer text), gen (the whole generation, kept with --save-gen), batch, design (role_first or
natural); summary.json per role: n, off_p_true (mean off probability of the true author), off_top_correct (share of items whose
most probable letter names the true author), on_correct, on_answers, median_new_tokens, unfinished.
--layout reproduces the records of earlier runs. v1: the first implementation, which generated one item at a time: no batch and
design fields, every generated token counted (no cut), the end-of-reasoning markers without gpt-oss's. b3: the batched records
without the design field, read with the markers without gpt-oss's. b: the batched records without the design field.
usage: python -m selfindex.thinking --model DIR --out DIR [--off-kwargs JSON] [--on-kwargs JSON] [--bodies all|pilot]
       [--lang zh|en] [--batch N] [--max-new N] [--skip-off] [--natural] [--only JSON|FILE] [--save-gen] [--layout v1|b3|b|final]
       [--max-memory-gib G] [--cpu-offload-gib C] [--disk-offload-dir D] [--summarize-only]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from pathlib import Path

import numpy as np

from . import items as I
from .io import atomic
from .load import load_model
from .render import CLOSE, Renderer

ROLES = I.ROLES
LETTERS = ('A', 'B', 'C')
TRUE = {'assistant': 'self', 'user': 'user', 'system': 'system'}
CLOSE_V1 = CLOSE[:3]  # the markers before gpt-oss's final channel was added (Qwen, Gemma, Seed-OSS)


def body_set(which, lang):
    """The bodies as dicts with 'id' and 'body': all 150 (in the order of items/bodies), or the 30 pilot bodies in their own order."""
    if which == 'all':
        return list(I.bodies(lang))
    zh = I.pilot_bodies('zh')
    if lang == 'zh':
        return zh
    en = I.pilot_bodies('en')  # id -> English text
    return [{**b, 'body': en[b['id']]} for b in zh]


def natural_messages(role, body, q, lang):
    return [{'role': 'user', 'content': I.natural_order_request(lang)}, {'role': role, 'content': body}, {'role': 'user', 'content': q}]


def check_natural(tok, kw, role, body, q, lang):
    """The body must sit between the request and the question, and a User body must not be merged into the request's turn."""
    req = I.natural_order_request(lang)
    text = tok.apply_chat_template(natural_messages(role, body, q, lang), tokenize=False, add_generation_prompt=True, **kw)
    a, b, c = text.find(req), text.find(body), text.rfind(q)
    assert 0 <= a < b < c, f'{role}: the body is not between the request and the question in the rendering'
    if role == 'user':
        merged = tok.apply_chat_template([{'role': 'user', 'content': req + '\n\n' + body + '\n\n' + q}], tokenize=False,
                                         add_generation_prompt=True, **kw)
        assert text != merged, 'the template merges consecutive user turns'
        assert text[a + len(req):b].strip() != '', 'no turn boundary between the request and the user body'


def eos_ids(tok, model):
    """End-of-sequence ids: the tokenizer's and the generation config's."""
    gen_eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = {tok.eos_token_id} | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    eos.discard(None)
    return eos


def generate(torch, model, tok, seqs, max_new, dev0):
    """Greedy generation for several token sequences at once (left padding); the new tokens of each, padding included."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    T = max(len(s) for s in seqs)
    x = torch.tensor([[pad] * (T - len(s)) + s for s in seqs], dtype=torch.long, device=dev0)
    am = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs], dtype=torch.long, device=dev0)
    with torch.no_grad():
        g = model.generate(input_ids=x, attention_mask=am, max_new_tokens=max_new, do_sample=False, pad_token_id=pad)
    return [g[k, T:].tolist() for k in range(len(seqs))]


def cut(new, eos):
    """The generated tokens up to and including the first end-of-sequence id (the rest of a batch row is padding)."""
    for k, t in enumerate(new):
        if t in eos:
            return new[:k + 1]
    return new


def answer_text(gen, close=CLOSE):
    """(the text after the last end-of-reasoning marker, whether any marker occurs); markers are taken in the order of `close`,
    so when several kinds occur the one listed last decides."""
    tail = gen
    for c in close:
        if c in gen:
            tail = gen.split(c)[-1]
    return tail, any(c in gen for c in close)


def first_letter(tail):
    """The first of the characters A, B, C ('**' removed). Every implementation found this character, also where its pattern
    was r'\\b([ABC])\\b|([ABC])': the second alternative matches any A, B or C, so the search stops at the first one."""
    m = re.search('[ABC]', tail.replace('**', ''))
    return m.group(0) if m else None


def parse_only(value):
    if not value:
        return None
    only = json.loads(Path(value).read_text(encoding='utf-8') if os.path.isfile(value) else value)
    return {(int(i), r, int(o)) for i, r, o in only} if only else None


def run(args, loaded=None):
    out = Path(args.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    v1 = args.layout == 'v1'
    if v1 and args.batch != 1:
        raise SystemExit('--layout v1 generates one item at a time (--batch 1)')
    if args.natural and args.layout != 'final':
        raise SystemExit('--natural writes the design field (--layout final)')
    L_ = loaded or load_model(args.model, max_memory_gib=args.max_memory_gib, cpu_offload_gib=args.cpu_offload_gib,
                              disk_offload_dir=args.disk_offload_dir)
    torch, tok, model = L_.torch, L_.tok, L_.model
    off_kw, on_kw = json.loads(args.off_kwargs), json.loads(args.on_kwargs)
    Roff, Ron = Renderer(tok, off_kw), Renderer(tok, on_kw)
    letters = {x: tok.encode(x, add_special_tokens=False)[0] for x in LETTERS}
    dev0 = model.get_input_embeddings().weight.device
    lang = args.lang
    bodies = body_set(args.bodies, lang)
    orders = I.orders()[:2]
    close = CLOSE_V1 if args.layout in ('v1', 'b3') else CLOSE
    eos = eos_ids(tok, model)
    t0 = time.time()
    sel = parse_only(args.only)
    roles = ('assistant', 'user') if args.natural else ROLES
    if args.natural:
        for r in roles:
            check_natural(tok, on_kw, r, bodies[0]['body'], I.attr_question(orders[0], lang), lang)
    todo = []
    for bi, b in enumerate(bodies):
        for oi, order in enumerate(orders):
            for r in roles:
                if sel is not None and (bi, r, oi) not in sel:
                    continue
                f = out / 'items' / f'{b["id"]}__{r}__o{oi}.json'
                if not f.exists():
                    todo.append((f, b, r, order))
    print(f'{len(todo)} items to run, batch {args.batch}', flush=True)
    for s0 in range(0, len(todo), args.batch):
        chunk = todo[s0:s0 + args.batch]
        offs, ons = [], []
        for f, b, r, order in chunk:
            q = I.attr_question(order, lang)
            msgs = (natural_messages(r, b['body'], q, lang) if args.natural else
                    [{'role': r, 'content': b['body']}, {'role': 'user', 'content': q}])
            if args.skip_off:
                offs.append(None)
            else:
                ids, _ = Roff.render(msgs, b['body'])
                x = torch.tensor([ids], dtype=torch.long, device=dev0)
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                                                 logits_to_keep=1).logits[0, -1].float(), -1)
                offs.append({k: float(lp[letters[LETTERS[i]]].exp()) for i, k in enumerate(order)})
            ons.append(Ron.render(msgs, b['body'])[0])
        news = generate(torch, model, tok, ons, args.max_new, dev0)
        for k, (f, b, r, order) in enumerate(chunk):
            new = news[k] if v1 else cut(news[k], eos)
            gen = tok.decode(new, skip_special_tokens=False)
            tail, finished = answer_text(gen, close)
            letter = first_letter(tail)
            rec = {'id': b['id'], 'role': r, 'order': list(order), 'off': offs[k], 'on_answer': order[LETTERS.index(letter)] if letter else None,
                   'on_letter': letter, 'n_new': len(new), 'finished': finished, 'tail': tail[-300:], 'gen': gen if args.save_gen else None}
            if not v1:
                rec['batch'] = args.batch
            if args.layout == 'final':
                rec['design'] = 'natural' if args.natural else 'role_first'
            atomic(f, rec)
        print(f'{s0 + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    summarize(out)


def summarize(out):
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    S = {}
    for r in ROLES:
        rs = [x for x in recs if x['role'] == r]
        if not rs:
            continue
        ro = [x for x in rs if x.get('off')]
        off_top = float(np.mean([max(x['off'], key=x['off'].get) == TRUE[r] for x in ro])) if ro else None
        S[r] = {'n': len(rs), 'off_p_true': float(np.mean([x['off'][TRUE[r]] for x in ro])) if ro else None, 'off_top_correct': off_top,
                'on_correct': float(np.mean([x['on_answer'] == TRUE[r] for x in rs])),
                'on_answers': {a: float(np.mean([x['on_answer'] == a for x in rs])) for a in ('self', 'user', 'system', None)},
                'median_new_tokens': float(np.median([x['n_new'] for x in rs])), 'unfinished': float(np.mean([not x['finished'] for x in rs]))}
        print(r, json.dumps(S[r]))
    (out / 'summary.json').write_text(json.dumps(S, indent=1), encoding='utf-8')


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--off-kwargs', default='{"enable_thinking": false}', help='template kwargs of the non-thinking reading')
    p.add_argument('--on-kwargs', default='{}', help='template kwargs of the thinking reading')
    p.add_argument('--bodies', choices=('all', 'pilot'), default='all', help='the 150 bodies or the 30 pilot bodies')
    p.add_argument('--lang', choices=('zh', 'en'), default='zh', help='language of the bodies, the question and the request')
    p.add_argument('--batch', type=int, default=1, help='items generated together (left padding, greedy)')
    p.add_argument('--max-new', type=int, default=3072)
    p.add_argument('--skip-off', action='store_true', help='no non-thinking reading (templates that always think)')
    p.add_argument('--natural', action='store_true', help='message under test after a user request; assistant and user only')
    p.add_argument('--only', default=None, help='subset: JSON list of [body index, role, order index], or a file holding it')
    p.add_argument('--save-gen', action='store_true', help='keep the whole generated text')
    p.add_argument('--layout', choices=('v1', 'b3', 'b', 'final'), default='final', help='records of earlier runs (see above)')
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
