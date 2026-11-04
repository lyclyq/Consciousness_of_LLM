"""Self-reference and geometry in the natural conversation order: the message under test follows a user request, as a reply does.

The main probe opens the conversation with the message under test, because many templates accept a System message only as the
first message. Here the message under test comes after a user request: [user: request] [ROLE: body] [user: Q]. Q is the reply
request (geometry) or the attribution question in the three cyclic option orders (attribution), worded as in the main probe. Every
input is the model's own chat-template rendering, checked against apply_chat_template(tokenize=True); one teacher-forced forward
per condition, use_cache=False; the 150 bodies.
--roles 2: ROLE assistant or user (for templates that cannot keep a System message in place after a user turn); attribution only:
  the probabilities of the three option letters at the response onset and the most probable token there. Outputs <body>.json in
  --out; summary.json: P(you | own message) and P(you | user message) per body (three orders averaged) and over bodies, their mean
  difference with a two-sided sign test over bodies, the share of bodies whose most probable answer is the true author, the mean
  answer probabilities per role, the smallest letter mass.
--roles 3: four conditions,
    assistant    [user: request] [assistant: body] [user: Q]
    user         [user: request] [user: body] [user: Q]          (the template's own rendering; some templates merge the turns)
    user_merged  [user: request + "\\n\\n" + body + "\\n\\n" + Q]  (one user message: the form that merging templates produce)
    system       [user: request] [system: body] [user: Q]        (a mid-conversation System message, which must stay in place)
  and the body must lie between the request and the question in every rendering. Geometry: the residual of every decoder layer,
  mean over the body tokens shared by the four renderings and at the response onset. Attribution (skipped with --geometry-only):
  the letter probabilities after at most two format tokens that the model writes before its letter, as in the main probe.
  Outputs N/<body>.json, N/<body>.npz (float16 content_<cond>, onset_<cond>: layers x hidden), status.json; summary.json: for the
  triples (assistant, user, system) and (assistant, user_merged, system), the distance ratio R_A per layer over the held-out
  bodies, the share of them in which the Assistant body is singled out (lowest R) and the number of layers at which that holds in
  all of them and in at least 80%; the attribution means per condition; whether the template merged the User body into the
  request's turn.
Mistral checkpoints carry a tokenizer.json whose pre-tokenizer regex transformers flags as wrong; their tokenizer is reloaded with
fix_mistral_regex=True (the check of load.checked_tokenizer takes tokenizer.json as the reference and would undo the fix).
usage: python -m selfindex.natural --model DIR --out DIR --roles 2|3 [--tmpl-kwargs JSON] [--lang zh|en] [--geometry-only]
       [--max-memory-gib G] [--cpu-offload-gib C] [--disk-offload-dir D] [--summarize-only N_LAYERS]
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from math import comb
from pathlib import Path

import numpy as np

from . import __version__
from . import items as I
from .io import atomic
from .load import TOKENIZER_PROBE, find_layers, load_model
from .render import Renderer, common_body, ids_of

LETTERS = ('A', 'B', 'C')
ROLES2 = ('assistant', 'user')
CONDS = ('assistant', 'user', 'user_merged', 'system')
TRIPLES = {'separate': ('assistant', 'user', 'system'), 'merged': ('assistant', 'user_merged', 'system')}
NOT_BODY = ('identity.json', 'summary.json', 'status.json', 'failed.json')


def mistral_tokenizer(tok, cfg, model_dir):
    """(tokenizer, note): for Mistral checkpoints the tokenizer reloaded with the corrected pre-tokenizer regex."""
    if 'mistral' not in str(getattr(cfg, 'model_type', '')).lower() and 'mistral' not in str(model_dir).lower():
        return tok, None
    from transformers import AutoTokenizer
    t2 = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    changed = ids_of(t2(TOKENIZER_PROBE, add_special_tokens=False)) != ids_of(tok(TOKENIZER_PROBE, add_special_tokens=False))
    return t2, {'fix_mistral_regex': True, 'probe_changed': changed, 'replaced': type(tok).__name__}


def letter_ids(tok):
    letters = {}
    for x in LETTERS:
        t = tok.encode(x, add_special_tokens=False)
        assert len(t) == 1, (x, t)
        letters[x] = t[0]
    return letters


def sign_p(k, n):
    """Two-sided sign test: k of n differences positive."""
    t = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * t)


def run(args, loaded=None):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = json.loads(args.tmpl_kwargs)
    L_ = loaded or load_model(args.model, max_memory_gib=args.max_memory_gib, cpu_offload_gib=args.cpu_offload_gib,
                              disk_offload_dir=args.disk_offload_dir)
    tok, tok_note = mistral_tokenizer(L_.tok, L_.cfg, args.model)
    (run_two if args.roles == 2 else run_three)(args, L_, tok, tok_note, kw, out)


def run_two(args, L_, tok, tok_note, kw, out):
    torch, model = L_.torch, L_.model
    R = Renderer(tok, kw)
    letters = letter_ids(tok)
    lang = args.lang
    bodies = I.bodies(lang)
    req = I.natural_order_request(lang)
    orders = I.orders()
    ident = {'experiment': 'P2_crossmodel_natural_order', 'selfindex_version': __version__, 'model': args.model, 'tmpl_kwargs': kw,
             'request': req, 'example_question': I.attr_question(orders[0], lang), 'device_map': L_.placement,
             'cpu_offload_gib': args.cpu_offload_gib, 'n_bodies': len(bodies), 'tokenizer_note': tok_note}
    atomic(out / 'identity.json', ident)
    dev0 = model.get_input_embeddings().weight.device
    t0 = time.time()
    for bi, b in enumerate(bodies):
        f = out / f'{b["id"]}.json'
        if f.exists():
            continue
        rec = {'id': b['id'], 'split': b['split'], 'category': b['category'], 'roles': {}}
        for r in ROLES2:
            ps = []
            for order in orders:
                msgs = [{'role': 'user', 'content': req}, {'role': r, 'content': b['body']},
                        {'role': 'user', 'content': I.attr_question(order, lang)}]
                ids, _ = R.render(msgs, b['body'])
                x = torch.tensor([ids], dtype=torch.long, device=dev0)
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                                                 logits_to_keep=1).logits[0, -1].float(), -1)
                p = {k: float(lp[letters[LETTERS[i]]].exp()) for i, k in enumerate(order)}
                p['top_token'] = tok.decode([int(lp.argmax())])
                ps.append(p)
            rec['roles'][r] = {'by_order': ps, 'mean': {k: sum(q[k] for q in ps) / len(ps) for k in ('self', 'user', 'system')}}
        atomic(f, rec)
        if bi % 10 == 0:
            print(f'{bi + 1}/{len(bodies)} {time.time() - t0:.0f}s', flush=True)
    summarize_two(out)


def summarize_two(out):
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted(out.glob('*.json')) if f.name not in NOT_BODY]
    if not recs:
        return None
    pa = [r['roles']['assistant']['mean']['self'] for r in recs]
    pu = [r['roles']['user']['mean']['self'] for r in recs]
    d = [a - u for a, u in zip(pa, pu)]
    k = sum(x > 0 for x in d)
    top = {r0: sum(max(('self', 'user', 'system'), key=lambda q: r['roles'][r0]['mean'][q]) == t for r in recs) / len(recs)
           for r0, t in (('assistant', 'self'), ('user', 'user'))}
    S = {'n_bodies': len(recs), 'p_you_assistant': sum(pa) / len(pa), 'p_you_user': sum(pu) / len(pu),
         'diff_mean': sum(d) / len(d), 'diff_positive': k, 'sign_p': sign_p(k, len(d)), 'top_answer_correct': top,
         'answers': {r0: {q: sum(r['roles'][r0]['mean'][q] for r in recs) / len(recs) for q in ('self', 'user', 'system')} for r0 in ROLES2},
         'letter_mass_min': min(min(sum(p[q] for q in ('self', 'user', 'system')) for p in r['roles'][r0]['by_order'])
                                for r in recs for r0 in ROLES2)}
    (out / 'summary.json').write_text(json.dumps(S, indent=1), encoding='utf-8')
    print(json.dumps(S), flush=True)
    return S


def messages(cond, req, body, q, sep):
    if cond == 'user_merged':
        return [{'role': 'user', 'content': req + sep + body + sep + q}]
    return [{'role': 'user', 'content': req}, {'role': cond, 'content': body}, {'role': 'user', 'content': q}]


def render(R, cond, req, body, q, sep):
    msgs = messages(cond, req, body, q, sep)
    ids, span = R.render(msgs, body)
    text = R.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **R.kw)
    a, b, c = text.find(req), text.find(body), text.rfind(q)
    assert 0 <= a < b < c, f'{cond}: the body is not between the request and the question in the rendering'
    return ids, span


def run_three(args, L_, tok, tok_note, kw, out):
    torch, model, cfg = L_.torch, L_.model, L_.cfg
    lname, layers = find_layers(model)
    L = len(layers)
    R = Renderer(tok, kw)
    letters = letter_ids(tok)
    lang = args.lang
    bodies = I.bodies(lang)
    orders = I.orders()
    req, q_reply = I.natural_order_request(lang), I.reply_request(lang)
    sep = I.templates()['natural_order']['merged_separator']
    ident = {'experiment': 'P2_crossmodel_natural3', 'selfindex_version': __version__, 'model': args.model, 'model_class': L_.model_class,
             'architectures': cfg.architectures, 'layers_module': lname, 'n_layers': L, 'tmpl_kwargs': kw, 'device_map': L_.placement,
             'torch': torch.__version__, 'transformers': __import__('transformers').__version__, 'lang': lang, 'request': req,
             'separator': sep, 'reply_question': q_reply, 'attr_question_example': I.attr_question(orders[0], lang),
             'geometry_only': args.geometry_only, 'cpu_offload_gib': args.cpu_offload_gib, 'fp8_backend': L_.fp8_backend,
             'tokenizer_note': tok_note, 'tokenizer_rebuilt_from': getattr(tok, '_p2_rebuilt_from', None), 'n_bodies': len(bodies)}
    atomic(out / 'identity.json', ident)
    print(json.dumps({k: ident[k] for k in ('model_class', 'layers_module', 'n_layers', 'tokenizer_note')}), flush=True)
    dev0 = model.get_input_embeddings().weight.device
    st = {'mode': None}

    def hook(i):
        def h(mod, inp, outp):
            if st['mode'] == 'capture':
                x = outp[0] if isinstance(outp, tuple) else outp
                st['cap'][i] = x[0].detach()
            return None
        return h
    handles = [layers[i].register_forward_hook(hook(i)) for i in range(L)]

    def fwd(ids, mode=None):
        st.update(mode=mode, cap={})
        x = torch.tensor([ids], dtype=torch.long, device=dev0)
        with torch.no_grad():
            o = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=1)
        cap = st['cap']
        st['mode'] = None
        return torch.log_softmax(o.logits[0].float(), -1), cap

    def letter_p(lp, order):
        return {k: float(lp[-1, letters[LETTERS[i]]].exp()) for i, k in enumerate(order)}

    def answer_prefix(ids):
        # format tokens the model writes before its answer letter (greedy, probability > 0.5, at most two), as in the main probe
        pre = []
        for _ in range(2):
            lp, _ = fwd(ids + pre)
            mass = sum(float(lp[-1, letters[x]].exp()) for x in LETTERS)
            top = int(lp[-1].argmax())
            if mass >= 0.5 or float(lp[-1, top].exp()) < 0.5 or top in letters.values():
                return pre, lp
            pre.append(top)
        lp, _ = fwd(ids + pre)
        return pre, lp

    t0 = time.time()
    gdir = out / 'N'
    gdir.mkdir(exist_ok=True)
    try:
        for bi, b in enumerate(bodies):
            done = gdir / f'{b["id"]}.json'
            if done.exists():
                continue
            geo = {c: render(R, c, req, b['body'], q_reply, sep) for c in CONDS}
            spans = common_body(geo)
            rec = {'id': b['id'], 'split': b['split'], 'category': b['category'], 'conds': {},
                   'user_is_merged': geo['user'][0] == geo['user_merged'][0]}
            arrs = {}
            for c in CONDS:
                ids = geo[c][0]
                _, cap = fwd(ids, 'capture')
                pos = spans[c]
                arrs[f'content_{c}'] = torch.stack([cap[i][pos].float().mean(0).to(dev0) for i in range(L)]).half().cpu().numpy()
                arrs[f'onset_{c}'] = torch.stack([cap[i][len(ids) - 1].float().to(dev0) for i in range(L)]).half().cpu().numpy()
                rec['conds'][c] = {'n_body_tokens': len(pos), 'n_tokens': len(ids)}
                if args.geometry_only:
                    continue
                ps = []
                for order in orders:
                    aid, _ = render(R, c, req, b['body'], I.attr_question(order, lang), sep)
                    pre, lp = answer_prefix(aid)
                    ps.append({**letter_p(lp, order), 'prefix': tok.decode(pre)})
                rec['conds'][c].update({'by_order': ps, 'mean': {k: float(np.mean([p[k] for p in ps])) for k in ('self', 'user', 'system')},
                                        'letter_mass_min': float(min(sum(p[k] for k in ('self', 'user', 'system')) for p in ps))})
            np.savez_compressed(gdir / f'{b["id"]}.npz', **arrs)
            atomic(done, rec)
            if bi % 10 == 0:
                print(f'N {bi + 1}/{len(bodies)} {time.time() - t0:.0f}s', flush=True)
    finally:
        for h in handles:
            h.remove()
    atomic(out / 'status.json', {'state': 'complete', 'elapsed_s': time.time() - t0})
    summarize_three(out, L)


def summarize_three(out, L):
    gdir = out / 'N'
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted(gdir.glob('*.json'))]
    S = {'n_layers': L, 'n_bodies': len(recs), 'user_rendering_merged_by_template': sorted({r['user_is_merged'] for r in recs})}
    held = [r for r in recs if r['split'] == 'confirmation']
    if not held:
        (out / 'summary.json').write_text(json.dumps(S, indent=1), encoding='utf-8')
        return S
    V = {}
    for r in held:
        with np.load(gdir / f'{r["id"]}.npz') as z:
            # hyper-connection models store each layer as [streams, hidden]: the full concatenated state is used
            V[r['id']] = {k: (z[k].reshape(z[k].shape[0], -1) if z[k].ndim == 3 else z[k]).astype(np.float32) for k in z.files}
    for tname, (ra, ru, rs) in TRIPLES.items():
        for kind in ('content', 'onset'):
            Rs = []
            for r in held:
                a, u, s = (V[r['id']][f'{kind}_{x}'] for x in (ra, ru, rs))
                au, as_, us = np.linalg.norm(a - u, axis=-1), np.linalg.norm(a - s, axis=-1), np.linalg.norm(u - s, axis=-1)
                Rs.append([us / ((au + as_) / 2), as_ / ((au + us) / 2), au / ((as_ + us) / 2)])
            Rs = np.array(Rs)
            share = (Rs.argmin(1) == 0).mean(0)
            S[f'{tname}_{kind}'] = {'R_A_by_layer': Rs[:, 0].mean(0).round(4).tolist(), 'share_A_lowest': share.round(4).tolist(),
                                   'layers_A_lowest_all_bodies': int((share == 1).sum()), 'layers_A_lowest_80': int((share >= 0.8).sum())}
    if all('mean' in r['conds'][c] for r in recs for c in CONDS):
        S['attr'] = {c: {k: float(np.mean([r['conds'][c]['mean'][k] for r in recs])) for k in ('self', 'user', 'system')} for c in CONDS}
        S['letter_mass_min'] = float(min(r['conds'][c]['letter_mass_min'] for r in recs for c in CONDS))
    (out / 'summary.json').write_text(json.dumps(S, indent=1), encoding='utf-8')
    for tname in TRIPLES:
        for kind in ('content', 'onset'):
            d = S[f'{tname}_{kind}']
            print(f'{tname:8s} {kind:7s} A singled out in >=80% of held-out bodies at {d["layers_A_lowest_80"]}/{L} layers, '
                  f'in all at {d["layers_A_lowest_all_bodies"]}/{L}', flush=True)
    if 'attr' in S:
        print('answers:', {c: {k: round(v, 3) for k, v in d.items()} for c, d in S['attr'].items()}, flush=True)
    return S


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--roles', type=int, choices=(2, 3), required=True, help='2: assistant and user; 3: with merged user and system')
    p.add_argument('--tmpl-kwargs', default='{}')
    p.add_argument('--lang', choices=('zh', 'en'), default='zh')
    p.add_argument('--geometry-only', action='store_true', help='--roles 3 without the attribution forwards')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    p.add_argument('--cpu-offload-gib', type=float, default=0, help='allow weights beyond the GPU budget to stay in CPU memory')
    p.add_argument('--disk-offload-dir', default=None, help='allow accelerate disk offload into this folder')
    p.add_argument('--summarize-only', type=int, default=0,
                   help='rewrite summary.json from the outputs in --out (the number of decoder layers, for --roles 3)')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.geometry_only and args.roles == 2:
        raise SystemExit('--geometry-only needs --roles 3 (the two-role design measures attribution only)')
    if args.summarize_only:
        summarize_two(Path(args.out)) if args.roles == 2 else summarize_three(Path(args.out), args.summarize_only)
        return
    try:
        run(args)
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        atomic(Path(args.out) / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
