"""P2 self-space experiment (11-06, see PREREG_zh.md): token states of the same message body under 18 context conditions.

Every condition renders [ROLE: PREFIX + BODY][user: neutral request] with the model's own chat template (template-native input
only; the same layout as phase G of crossmodel_suite.py) and changes exactly one thing before the body: the role of the message
(Assistant / User / System), the template's own option that drops the empty think block of an earlier Assistant message
(preserve_thinking=False), or one word of a short prefix inside a User message. The body tokens are identical in all 18
conditions (asserted); with causal attention their states depend only on the tokens before them. Each rendering is run as one
sequence per forward with an attention mask of ones, as in phase G (batches change the late-layer states, see forward_one).

Per body it stores (atomic, resumable per body):
  pooled   (C, L, d) fp16  mean over the body tokens common to all 18 conditions, every decoder layer
  pooled3  (3, L, d) fp16  A / U / S without prefix, mean over the body tokens common to these three (= phase G 'content')
  tokens   (C, K, T, d) fp16  states of the T common body tokens at the K stored layers
The json next to it holds the conditions, spans, token ids of the body and of each rendering, split and category.
usage: selfspace_tokens_20261106.py --model DIR --bodies role_bodies.jsonl --out DIR [--limit N] [--dry-run]
       [--max-memory-gib 44] [--min-free-gib 32] [--tokenizer DIR]"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import socket
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossmodel_suite import Q_REPLY, Renderer, atomic, checked_tokenizer, common_body, find_layers, ids_of, load  # noqa: E402

BASE = {'enable_thinking': False}
SWITCH = {'enable_thinking': False, 'preserve_thinking': False}
P0, P0ALT = '以下是一段记录：', '下面是一段记录：'
# each pair differs in one token at the same position (Qwen3.8 tokenizer, checked by the dry run): 医院的/学校的, 昨天的/明天的,
# 小+明的/小+红的, 愉快的/痛苦的, 你/我
WORDS = (('hospital', '医院'), ('school', '学校'), ('yesterday', '昨天'), ('tomorrow', '明天'), ('ming', '小明'), ('hong', '小红'),
         ('pleasant', '愉快'), ('painful', '痛苦'))
COND = ([('A', 'assistant', '', BASE), ('U', 'user', '', BASE), ('S', 'system', '', BASE), ('A_switch', 'assistant', '', SWITCH),
         ('A_p0', 'assistant', P0, BASE), ('U_p0', 'user', P0, BASE), ('S_p0', 'system', P0, BASE), ('U_p0alt', 'user', P0ALT, BASE),
         ('U_you', 'user', '以下是你写的：', BASE), ('U_me', 'user', '以下是我写的：', BASE)]
        + [(f'U_{k}', 'user', f'以下是{w}的记录：', BASE) for k, w in WORDS])
# pairs whose two prefixes must have the same number of tokens, so that the body sits at the same positions
SAME_LEN = [('U_p0', 'U_p0alt'), ('U_you', 'U_me')] + [(f'U_{WORDS[i][0]}', f'U_{WORDS[i + 1][0]}') for i in range(0, len(WORDS), 2)]
KEEP_LAYERS = (0, 8, 16, 24, 32, 39, 47, 55, 63)


def render_body(tok, b):
    """All 18 renderings of one body; returns ids per condition, the 18-condition common body span and the A/U/S span."""
    rend = {}
    for name, role, prefix, kw in COND:
        msgs = [{'role': role, 'content': prefix + b['body']}, {'role': 'user', 'content': Q_REPLY}]
        rend[name] = Renderer(tok, kw).render(msgs, b['body'])
    # the default rendering gives an earlier Assistant message an empty think block before the body; the switch removes it
    head = {k: tok.decode(rend[k][0][:rend[k][1][0]]) for k in ('A', 'A_switch', 'U')}
    assert '<think>' in head['A'] and '<think>' not in head['A_switch'] and '<think>' not in head['U'], head
    span = common_body(rend)
    span3 = common_body({k: rend[k] for k in ('A', 'U', 'S')})
    body = {tuple(rend[k][0][i] for i in span[k]) for k in rend}
    assert len(body) == 1, 'body tokens differ between conditions'
    for a, c in SAME_LEN:
        assert span[a] == span[c], f'{a} and {c} put the body at different positions'
    return {k: v[0] for k, v in rend.items()}, span, span3


def prefix_lengths(tok):
    return {name: len(ids_of(tok(prefix, add_special_tokens=False))) if prefix else 0 for name, _, prefix, _ in COND}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tokenizer', default=None, help='tokenizer directory for --dry-run (default: --model)')
    ap.add_argument('--bodies', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true', help='render and check every condition, no model, no GPU')
    ap.add_argument('--max-memory-gib', type=int, default=44)
    ap.add_argument('--min-free-gib', type=float, default=32.0, help='stop cleanly before a body if the disk has less free')
    args = ap.parse_args()
    bodies = [json.loads(l) for l in open(args.bodies, encoding='utf-8') if l.strip()]
    if args.limit:
        bodies = bodies[:args.limit]
    out = Path(args.out)

    if args.dry_run:
        from transformers import AutoTokenizer
        tdir = args.tokenizer or args.model
        tok = checked_tokenizer(AutoTokenizer.from_pretrained(tdir, trust_remote_code=True), tdir)
        lens = prefix_lengths(tok)
        n_tok, trimmed = [], []
        for b in bodies:
            ids, span, span3 = render_body(tok, b)
            n_tok.append(len(span['A']))
            trimmed.append(len(span3['A']) - len(span['A']))
        print(json.dumps({'bodies': len(bodies), 'conditions': len(COND), 'prefix_tokens': lens,
                          'body_tokens_min_max': [min(n_tok), max(n_tok)], 'trimmed_vs_AUS_max': max(trimmed)},
                         ensure_ascii=False))
        print('DRY RUN OK')
        return

    out.mkdir(parents=True, exist_ok=True)
    largs = argparse.Namespace(model=args.model, max_memory_gib=args.max_memory_gib)
    torch, tok, model, cfg, placement, cls = load(largs)
    lname, layers = find_layers(model)
    L = len(layers)
    assert max(KEEP_LAYERS) < L, (KEEP_LAYERS, L)
    ident = {'experiment': 'P2_selfspace_tokens_20261106', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'suite_sha256': hashlib.sha256((Path(__file__).resolve().parent / 'crossmodel_suite.py').read_bytes()).hexdigest(),
             'model': args.model, 'model_class': cls, 'layers_module': lname, 'n_layers': L, 'keep_layers': KEEP_LAYERS,
             'conditions': [{'name': n, 'role': r, 'prefix': p, 'tmpl_kwargs': kw} for n, r, p, kw in COND],
             'same_length_pairs': SAME_LEN, 'prefix_tokens': prefix_lengths(tok), 'device_map': placement,
             'torch': torch.__version__, 'transformers': __import__('transformers').__version__, 'host': socket.gethostname(),
             'n_bodies': len(bodies), 'bodies_file': args.bodies,
             'bodies_sha256': hashlib.sha256(Path(args.bodies).read_bytes()).hexdigest()}
    atomic(out / 'identity.json', ident)
    dev0 = model.get_input_embeddings().weight.device
    cap = {}

    def hook(i):
        def h(mod, inp, outp):
            cap[i] = (outp[0] if isinstance(outp, tuple) else outp).detach()
        return h
    handles = [layers[i].register_forward_hook(hook(i)) for i in range(L)]
    names = [c[0] for c in COND]

    def forward_one(seq):
        # one sequence per forward with an attention mask of ones, exactly as phase G: batching changes the late-layer states
        # (diag_batching_20261106.py: up to 8.4% with right padding, up to 1.8% for equal-length batches, <= 0.05% single)
        x = torch.tensor([seq], dtype=torch.long, device=dev0)
        cap.clear()
        with torch.no_grad():
            model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=1)
        return {i: cap[i][0].float() for i in range(L)}
    t0 = time.time()
    done = 0
    for b in bodies:
        if (out / f'{b["id"]}.json').exists():
            continue
        free = shutil.disk_usage(out).free / 2 ** 30
        if free < args.min_free_gib:
            print(f'[stop] {free:.1f} GiB free < {args.min_free_gib}: stopping before {b["id"]} (rerun resumes)', flush=True)
            break
        ids, span, span3 = render_body(tok, b)
        nb = len(span['A'])
        for ci, n in enumerate(names):
            h = forward_one(ids[n])
            if ci == 0:
                d = h[0].shape[-1]
                pooled = np.zeros((len(names), L, d), np.float16)
                pooled3 = np.zeros((3, L, d), np.float16)
                tokens = np.zeros((len(names), len(KEEP_LAYERS), nb, d), np.float16)
            for li in range(L):
                hb = h[li].index_select(0, torch.tensor(span[n], device=h[li].device))
                pooled[ci, li] = hb.mean(0).cpu().numpy()
                if li in KEEP_LAYERS:
                    tokens[ci, KEEP_LAYERS.index(li)] = hb.cpu().numpy()
                if n in ('A', 'U', 'S'):
                    pooled3[('A', 'U', 'S').index(n), li] = h[li].index_select(0, torch.tensor(span3[n], device=h[li].device)).mean(0).cpu().numpy()
        assert np.isfinite(pooled).all() and np.isfinite(tokens).all(), f'non-finite states in {b["id"]}'
        part = out / f'{b["id"]}.npz.part'
        with open(part, 'wb') as f:
            np.savez(f, pooled=pooled, pooled3=pooled3, tokens=tokens)
        part.replace(out / f'{b["id"]}.npz')
        body_ids = [ids['A'][i] for i in span['A']]
        atomic(out / f'{b["id"]}.json', {'id': b['id'], 'split': b.get('split'), 'category': b.get('category'), 'conditions': names,
                                          'n_body_tokens': nb, 'body_ids': body_ids, 'spans': span, 'spans_AUS': span3,
                                          'render_lengths': {n: len(ids[n]) for n in names},
                                          'render_sha256': {n: hashlib.sha256(json.dumps(ids[n]).encode()).hexdigest()[:16] for n in names}})
        done += 1
        if done % 10 == 0 or done == 1:
            print(f'{done} bodies, {time.time() - t0:.0f}s, last {b["id"]} ({nb} body tokens), {free:.0f} GiB free', flush=True)
    for hd in handles:
        hd.remove()
    n_all = sum(1 for b in bodies if (out / f'{b["id"]}.json').exists())
    print(f'finished: {n_all}/{len(bodies)} bodies on disk', flush=True)
    if n_all == len(bodies):
        (out / 'DONE').write_text(time.strftime('%F %T') + '\n')


if __name__ == '__main__':
    main()
