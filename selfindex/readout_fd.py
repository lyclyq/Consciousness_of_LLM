"""First-order sensitivity of the attribution readout by layer: projected central differences along the own-User difference
(template-native renderings only; forward passes only, so the weights and kernels are those of the main runs, e.g. MXFP4 expert
kernels without a backward pass).

Per held-out item (body x the first two option orders, attribution question) the model's own message (A) and the same text as a
User message (U) are rendered and aligned as in the causal tests (render.align: the body, the end of the message and the question
up to the generation prompt); d_l = h^A_l - h^U_l at the aligned positions of layer l. The readout f is the logit of the option
letter of "you" minus that of the letter of "user" where the answer letter is read: at the response onset in single-pass mode;
in thinking mode after the model's own greedy reasoning, teacher-forced on the generated prefix up to the answer letter (the first
option-letter token, plain or with a leading space, after the last end-of-reasoning marker of the generated tokens; an item
without one is recorded as skipped). At every layer l of --band (every --layer-step-th, from its first) five directions are added
to the states of layer l at the aligned positions only, every other layer as computed:
  total  d_l
  V      P_V d_l, its part in the fitted subspace (the 8 leading directions of V of --fit-from)
  perp   d_l - P_V d_l, the rest (the residual outside the subspace)
  R      P_R d_l, its part in the random 8-dim subspace R of --fit-from
  zero   no change (two unperturbed rows: the numerical noise of the batch)
and D_v f = (f(h + eps v) - f(h - eps v)) / (2 eps), eps = 0.25, at the own message (endpoint A: its states and reasoning) and at
the User message (endpoint U). If the readout of the role difference went through the residual outside the subspace, D along perp
would carry it; if through the subspace, D along V. The ten rows of one layer run as one batch of identical sequences.
Outputs in --out: items/<id>__o<order>.json (per layer and endpoint: D for the five directions, the norms of the vectors and of
the states, f and the letter probabilities unperturbed; a run resumes from the items on disk), traces/ (thinking: the generated
ids of each rendering), identity.json, summary.json (per layer, the medians over the complete items of D for each direction and of
D_V / D_total, D_perp / D_total, D_R / D_total and D_zero / D_total at both endpoints). A teacher-forced sequence longer than
--max-len tokens is recorded as skipped.
usage: python -m selfindex.readout_fd --mode single|thinking --model DIR --on-kwargs JSON --lang zh|en --fit-from FIT --band A-B
         --out DIR [--layer-step N] [--limit-bodies N] [--batch N] [--max-new N] [--max-len N]
       python -m selfindex.readout_fd --summarize-only --out DIR
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

from . import __version__
from . import items as I
from .io import atomic
from .load import find_layers, load_model
from .render import CLOSE, Renderer, align
from .role_swap import selected_bodies

EPS = 0.25
DIRS = ('total', 'V', 'perp', 'R', 'zero')
LETTERS = ('A', 'B', 'C')
TURN_STARTS = ('<|start|>', '<|im_start|>')


def run(a, loaded=None):
    out = Path(a.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    (out / 'traces').mkdir(exist_ok=True)
    L_ = loaded or load_model(a.model, max_memory_gib=a.max_memory_gib, cpu_offload_gib=a.cpu_offload_gib)
    torch, tok, model = L_.torch, L_.tok, L_.model
    _, layers = find_layers(model)
    nL = len(layers)
    b0, b1 = (int(x) for x in a.band.split('-'))
    band = tuple(range(b0, b1 + 1))
    fd_layers = band[::a.layer_step]  # the layers at which the directions are applied
    assert b1 < nL
    R = Renderer(tok, json.loads(a.on_kwargs))
    ORDERS = I.orders()
    held = [b for b in selected_bodies(a) if b['split'] == 'confirmation']
    if a.limit_bodies:
        held = held[:a.limit_bodies]
    dev0 = model.get_input_embeddings().weight.device
    cands = [tok.convert_tokens_to_ids(t) for t in TURN_STARTS]
    F = torch.load(a.fit_from, weights_only=True)
    Bv = {l: F['V'][l][:, :8].float().contiguous() for l in band}
    Br = {l: F['R'][l][:, :8].float().contiguous() for l in band}
    # the option letters as single tokens, plain and (where single tokens) with a leading space
    sets = [{x: tok.encode(x, add_special_tokens=False) for x in LETTERS}, {x: tok.encode(' ' + x, add_special_tokens=False) for x in LETTERS}]
    sets = [{x: v[0] for x, v in s.items()} for s in sets if all(len(v) == 1 for v in s.values())]
    assert sets and len(sets[0]) == 3, 'single-token option letters needed'
    closes = [tok.encode(c, add_special_tokens=False) for c in CLOSE]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = {tok.eos_token_id} | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    eos.discard(None)

    def render(role, b, oi):
        msgs = [{'role': role, 'content': b['body']}, {'role': 'user', 'content': I.attr_question(ORDERS[oi], a.lang)}]
        ids, span = R.render(msgs, b['body'])
        g0 = max(i for i in range(len(ids)) if ids[i] in cands)
        assert g0 > span[-1], 'no generation prompt after the body'
        return ids, span[0], span[-1] + 1, g0

    st = {'mode': None, 'pos': None, 'layer': None, 'vecs': None, 'caps': None}

    def make_hook(l):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if st['mode'] is None or h.shape[1] == 1:
                return output
            sl = torch.tensor(st['pos'], dtype=torch.long, device=h.device)
            if st['mode'] == 'capture':
                st['caps'][l] = h[0, sl].float().clone()
            elif l == st['layer']:
                h[:, sl] = (h[:, sl].float() + st['vecs'].to(h.device)).to(h.dtype)
            return output
        return hook

    def last_logits(seq, nrows):
        x = torch.tensor([seq] * nrows, dtype=torch.long, device=dev0)
        with torch.no_grad():
            return model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1).logits[:, -1].float()

    def generate(seqs):
        res = []
        for s0 in range(0, len(seqs), a.batch):
            chunk = seqs[s0:s0 + a.batch]
            Lmax = max(len(q) for q in chunk)
            x = torch.tensor([[pad] * (Lmax - len(q)) + q for q in chunk], dtype=torch.long, device=dev0)
            am = torch.tensor([[0] * (Lmax - len(q)) + [1] * len(q) for q in chunk], dtype=torch.long, device=dev0)
            with torch.no_grad():
                g = model.generate(input_ids=x, attention_mask=am, max_new_tokens=a.max_new, do_sample=False, pad_token_id=pad)
            for j in range(len(chunk)):
                new = g[j, Lmax:].tolist()
                n = next((i + 1 for i, t in enumerate(new) if t in eos), len(new))
                res.append(new[:n])
            print(f'generated {min(s0 + a.batch, len(seqs))}/{len(seqs)}', flush=True)
        return res

    def answer_at(gen):
        """The index in gen of the answer letter after the last end-of-reasoning marker (token sequences), and its letter set."""
        start = None
        for c in closes:
            for i in range(len(gen) - len(c), -1, -1):
                if gen[i:i + len(c)] == c:
                    start = i + len(c) if start is None else max(start, i + len(c))
                    break
        if start is None:
            return None, None
        for j in range(start, len(gen)):
            for k, s in enumerate(sets):
                if gen[j] in s.values():
                    return j, k
        return None, None

    items = [(b, oi) for b in held for oi in (0, 1)]
    ident = {'experiment': 'P2_readout_fd', 'selfindex_version': __version__, 'model': a.model, 'mode': a.mode,
             'on_kwargs': a.on_kwargs, 'n_layers': nL, 'band': [b0, b1], 'layer_step': a.layer_step, 'fit_from': a.fit_from, 'eps': EPS,
             'directions': DIRS, 'lang': a.lang, 'n_eval': len(held), 'batch': a.batch, 'max_new': a.max_new, 'placement': L_.placement,
             'torch': str(torch.__version__), 'transformers': str(__import__('transformers').__version__), 'max_len': a.max_len,
             'body_ids': a.body_ids}
    (out / 'identity.json').write_text(json.dumps(ident, indent=1), encoding='utf-8')
    print(json.dumps(ident), flush=True)
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in band]
    try:
        todo = [(b, oi) for b, oi in items if not (out / 'items' / f'{b["id"]}__o{oi}.json').exists()]
        rend = {(b['id'], oi, role): render(role, b, oi) for b, oi in todo for role in ('assistant', 'user')}
        traces = {}
        if a.mode == 'thinking':
            keys = [(b['id'], oi, role) for b, oi in todo for role in ('assistant', 'user')]
            need = [k for k in keys if not (out / 'traces' / f'{k[0]}__o{k[1]}__{k[2]}.json').exists()]
            for k, g in zip(need, generate([rend[k][0] for k in need])):
                (out / 'traces' / f'{k[0]}__o{k[1]}__{k[2]}.json').write_text(json.dumps({'gen': g, 'text': tok.decode(g)}, ensure_ascii=False),
                                                                              encoding='utf-8')
            for k in keys:
                traces[k] = json.loads((out / 'traces' / f'{k[0]}__o{k[1]}__{k[2]}.json').read_text(encoding='utf-8'))['gen']
        t0 = time.time()
        for n_done, (b, oi) in enumerate(todo):
            rA, rU = rend[(b['id'], oi, 'assistant')], rend[(b['id'], oi, 'user')]
            ri, di = align(rA, rU)
            rec = {'id': b['id'], 'order': list(ORDERS[oi]), 'n_pos': len(ri), 'layers': list(fd_layers), 'eps': EPS, 'endpoints': {}}
            ends = {}
            for e, r_, pos, role in (('A', rA, ri, 'assistant'), ('U', rU, di, 'user')):
                seq, ls = r_[0], 0
                if a.mode == 'thinking':
                    gen = traces[(b['id'], oi, role)]
                    j, ls = answer_at(gen)
                    if j is None:
                        rec['endpoints'][e] = {'skipped': 'no answer letter after the reasoning', 'n_new': len(gen)}
                        continue
                    seq = r_[0] + gen[:j]
                if len(seq) > a.max_len:
                    rec['endpoints'][e] = {'skipped': f'sequence of {len(seq)} tokens', 'n_tokens': len(seq)}
                    continue
                ends[e] = (seq, pos, ls)
            if len(ends) < 2:
                atomic(out / 'items' / f'{b["id"]}__o{oi}.json', rec)
                continue
            caps = {}
            for e, (seq, pos, ls) in ends.items():
                st.update(mode='capture', pos=pos, caps={})
                lg = last_logits(seq, 1)[0]
                st.update(mode=None)
                caps[e] = st['caps']
                lt = sets[ls]
                ly, lu = lt[LETTERS[ORDERS[oi].index('self')]], lt[LETTERS[ORDERS[oi].index('user')]]
                p3 = torch.softmax(lg[[lt[x] for x in LETTERS]], -1).tolist()
                rec['endpoints'][e] = {'f0': float(lg[ly] - lg[lu]), 'p_letters': dict(zip(LETTERS, p3)), 'letter_set': ls,
                                       'n_tokens': len(seq), 'D': {k: [] for k in DIRS}}
                ends[e] = (seq, pos, ly, lu)
            norms = {k: [] for k in DIRS[:4]}
            rec['state_norm'] = {'A': [], 'U': []}
            for l in fd_layers:
                d = caps['A'][l] - caps['U'][l]
                vV = (d @ Bv[l].to(d.device)) @ Bv[l].to(d.device).T
                vR = (d @ Br[l].to(d.device)) @ Br[l].to(d.device).T
                vec = {'total': d, 'V': vV, 'perp': d - vV, 'R': vR, 'zero': torch.zeros_like(d)}
                for k in DIRS[:4]:
                    norms[k].append(float(vec[k].norm()))
                for e in ('A', 'U'):
                    rec['state_norm'][e].append(float(caps[e][l].norm()))
                rows = torch.stack([s * EPS * vec[k] for k in DIRS for s in (1, -1)])
                for e, (seq, pos, ly, lu) in ends.items():
                    st.update(mode='perturb', pos=pos, layer=l, vecs=rows)
                    lg = last_logits(seq, len(rows))
                    st.update(mode=None, vecs=None)
                    f = (lg[:, ly] - lg[:, lu]).tolist()
                    for i, k in enumerate(DIRS):
                        rec['endpoints'][e]['D'][k].append((f[2 * i] - f[2 * i + 1]) / (2 * EPS))
            rec['norms'] = norms
            atomic(out / 'items' / f'{b["id"]}__o{oi}.json', rec)
            print(f'item {n_done + 1}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    finally:
        for hd in handles:
            hd.remove()
    summarize(out)


def summarize(out):
    out = Path(out)
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    ok = [r for r in recs if all('D' in r['endpoints'].get(e, {}) for e in ('A', 'U'))]
    S = {'n_items': len(recs), 'n_complete': len(ok), 'skipped': [[r['id'], r['order'], {e: v.get('skipped') for e, v in r['endpoints'].items()
                                                                                         if 'skipped' in v}] for r in recs if r not in ok]}
    if ok:
        L = ok[0]['layers']
        for e in ('A', 'U'):
            per = []
            for i, l in enumerate(L):
                tot = [r['endpoints'][e]['D']['total'][i] for r in ok]
                row = {'layer': l, 'median_total': statistics.median(tot)}
                for k in ('V', 'perp', 'R', 'zero'):
                    xs = [r['endpoints'][e]['D'][k][i] for r in ok]
                    row[f'median_{k}'] = statistics.median(xs)
                    row[f'median_{k}_over_total'] = statistics.median(x / t for x, t in zip(xs, tot) if t != 0) if any(tot) else None
                per.append(row)
            S[e] = per
    (out / 'summary.json').write_text(json.dumps(S, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in S.items() if k in ('n_items', 'n_complete')}), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--mode', choices=('thinking', 'single'))
    p.add_argument('--model')
    p.add_argument('--on-kwargs', default='{}', help='chat-template keyword arguments of the rendering (JSON)')
    p.add_argument('--lang', choices=('zh', 'en'))
    p.add_argument('--fit-from', default='', help='fit file with the subspaces V and R (fit_q.pt of selfindex.role_swap --make-fits)')
    p.add_argument('--band', default='', help='first-last layer, e.g. 0-63')
    p.add_argument('--out', default='')
    p.add_argument('--layer-step', type=int, default=1, help='apply the directions at every n-th layer of the band, from its first')
    p.add_argument('--limit-bodies', type=int, default=0, help='the first N held-out bodies only')
    p.add_argument('--body-ids', default='', help='JSON list of body ids (or objects with an id): only these bodies, in this order')
    p.add_argument('--batch', type=int, default=16, help='generation batch (thinking)')
    p.add_argument('--max-new', type=int, default=3072, help='new tokens at most (thinking)')
    p.add_argument('--max-len', type=int, default=4096, help='longest teacher-forced sequence; longer items are recorded as skipped')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    p.add_argument('--cpu-offload-gib', type=float, default=0, help='allow weights beyond the GPU budget to stay in CPU memory')
    p.add_argument('--summarize-only', action='store_true', help='rewrite summary.json from the items in --out')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.summarize_only:
        if not args.out:
            p.error('--summarize-only needs --out')
        summarize(Path(args.out))
        return
    missing = [k for k in ('mode', 'model', 'lang', 'fit_from', 'band', 'out') if not getattr(args, k)]
    if missing:
        p.error('missing: ' + ', '.join('--' + k.replace('_', '-') for k in missing))
    try:
        run(args)
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        atomic(Path(args.out) / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
