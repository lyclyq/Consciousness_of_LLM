"""Natural-order calibration with three roles (user-approved 2026-10-20; results for templates that merge user turns go to an
appendix, with the reason stated).

Mistral and DeepSeek-V4.1 cannot render the main probe: their templates refuse or degrade a conversation that opens with an
Assistant message, and they merge consecutive user messages into one turn, so a User body is never a message of its own. This
check puts the message under test after a user request, as a reply normally is, and measures what the merge does in models
whose templates keep every message separate:
    assistant    [user: REQ] [assistant: body] [user: Q]
    user         [user: REQ] [user: body] [user: Q]          (the template's own rendering; merged by Mistral / DeepSeek)
    user_merged  [user: REQ + "\\n\\n" + body + "\\n\\n" + Q]  (one user message: the form those templates produce)
    system       [user: REQ] [system: body] [user: Q]        (mid-conversation System message, which must stay in place)
Q is the neutral reply request (geometry) or the attribution question in the three cyclic option orders (attribution), worded
as in crossmodel_suite phase G. Every rendering is checked against apply_chat_template(tokenize=True), and the body must lie
between the request and the question in the rendered text (a template that moves a System message to the top fails here).
Geometry: residual of every decoder layer, mean over the body tokens (ids common to all four conditions) and at the response
onset. The summary gives the distance ratios and the share of held-out bodies in which the Assistant body is singled out, for
the triple (assistant, user, system) and for (assistant, user_merged, system), plus the attribution means per condition.
usage: crossmodel_natural3.py --model DIR --tmpl-kwargs JSON --bodies role_bodies.jsonl --out DIR [--lang zh|en --bodies-en F]
       [--geometry-only] [--max-memory-gib G] [--cpu-offload-gib C]"""
from __future__ import annotations
import argparse
import hashlib
import json
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402

REQ = {'zh': '请写一段简短的描述。', 'en': 'Please write a short description.'}
CONDS = ('assistant', 'user', 'user_merged', 'system')
TRIPLES = {'separate': ('assistant', 'user', 'system'), 'merged': ('assistant', 'user_merged', 'system')}
SEP = '\n\n'


def messages(cond, req, body, q):
    if cond == 'user_merged':
        return [{'role': 'user', 'content': req + SEP + body + SEP + q}]
    return [{'role': 'user', 'content': req}, {'role': cond, 'content': body}, {'role': 'user', 'content': q}]


def render(R, cond, req, body, q):
    msgs = messages(cond, req, body, q)
    ids, span = R.render(msgs, body)
    text = R.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **R.kw)
    a, b, c = text.find(req), text.find(body), text.rfind(q)
    assert 0 <= a < b < c, f'{cond}: the body is not between the request and the question in the rendering'
    return ids, span


def summarize(out, L):
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
            V[r['id']] = {k: (z[k].reshape(z[k].shape[0], -1) if z[k].ndim == 3 else z[k]).astype(np.float32) for k in z.files}
    for tname, (ra, ru, rs) in TRIPLES.items():
        for kind in ('content', 'onset'):
            Rs = []
            for r in held:
                a, u, s = (V[r['id']][f'{kind}_{x}'] for x in (ra, ru, rs))
                au, as_, us = np.linalg.norm(a - u, axis=-1), np.linalg.norm(a - s, axis=-1), np.linalg.norm(u - s, axis=-1)
                Rs.append([us / ((au + as_) / 2), as_ / ((au + us) / 2), au / ((as_ + us) / 2)])
            Rs = np.array(Rs)
            low = Rs.argmin(1)
            share = (low == 0).mean(0)
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


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = json.loads(args.tmpl_kwargs)
    torch, tok, model, cfg, placement, cls = cs.load(args)
    tok_note = None
    if 'mistral' in str(getattr(cfg, 'model_type', '')).lower() or 'mistral' in args.model.lower():
        # the tokenizer.json of these checkpoints carries a pre-tokenizer regex that transformers flags as wrong; load the
        # corrected one (cs.checked_tokenizer would take tokenizer.json itself as the reference and undo the fix)
        from transformers import AutoTokenizer
        t2 = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=True)
        probe = cs.TOKENIZER_PROBE
        tok_note = {'fix_mistral_regex': True, 'probe_changed': cs.ids_of(t2(probe, add_special_tokens=False)) != cs.ids_of(tok(probe, add_special_tokens=False)),
                    'replaced': type(tok).__name__}
        tok = t2
    lname, layers = cs.find_layers(model)
    L = len(layers)
    R = cs.Renderer(tok, kw)
    letters = {}
    for x in cs.LETTERS:
        t = tok.encode(x, add_special_tokens=False)
        assert len(t) == 1, (x, t)
        letters[x] = t[0]
    bodies = [json.loads(l) for l in open(args.bodies, encoding='utf-8') if l.strip()]
    if args.lang == 'en':
        cs.Q_ATTR, cs.Q_REPLY = cs.Q_ATTR_EN, cs.Q_REPLY_EN
        en = json.loads(Path(args.bodies_en).read_text(encoding='utf-8'))
        bodies = [{**b, 'body': en[b['id']]} for b in bodies]
    req = REQ[args.lang]
    ident = {'experiment': 'P2_crossmodel_natural3', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'suite_sha256': hashlib.sha256(Path(cs.__file__).read_bytes()).hexdigest(), 'model': args.model, 'model_class': cls,
             'architectures': cfg.architectures, 'layers_module': lname, 'n_layers': L, 'tmpl_kwargs': kw, 'device_map': placement,
             'torch': torch.__version__, 'transformers': __import__('transformers').__version__, 'host': socket.gethostname(),
             'lang': args.lang, 'request': req, 'separator': SEP, 'reply_question': cs.Q_REPLY, 'attr_question_example': cs.attr_question(cs.ORDERS[0]),
             'geometry_only': args.geometry_only, 'cpu_offload_gib': args.cpu_offload_gib, 'fp8_backend': getattr(model, '_p2_fp8_backend', None),
             'tokenizer_note': tok_note, 'tokenizer_rebuilt_from': getattr(tok, '_p2_rebuilt_from', None), 'n_bodies': len(bodies)}
    cs.atomic(out / 'identity.json', ident)
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
        return {k: float(lp[-1, letters[cs.LETTERS[i]]].exp()) for i, k in enumerate(order)}

    def answer_prefix(ids):
        # format tokens the model writes before its answer letter (greedy, probability > 0.5, at most two), as in phase G
        pre = []
        for _ in range(2):
            lp, _ = fwd(ids + pre)
            mass = sum(float(lp[-1, letters[x]].exp()) for x in cs.LETTERS)
            top = int(lp[-1].argmax())
            if mass >= 0.5 or float(lp[-1, top].exp()) < 0.5 or top in letters.values():
                return pre, lp
            pre.append(top)
        lp, _ = fwd(ids + pre)
        return pre, lp

    t0 = time.time()
    gdir = out / 'N'
    gdir.mkdir(exist_ok=True)
    for bi, b in enumerate(bodies):
        done = gdir / f'{b["id"]}.json'
        if done.exists():
            continue
        geo = {c: render(R, c, req, b['body'], cs.Q_REPLY) for c in CONDS}
        spans = cs.common_body(geo)
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
            for order in cs.ORDERS:
                aid, _ = render(R, c, req, b['body'], cs.attr_question(order))
                pre, lp = answer_prefix(aid)
                ps.append({**letter_p(lp, order), 'prefix': tok.decode(pre)})
            rec['conds'][c].update({'by_order': ps, 'mean': {k: float(np.mean([p[k] for p in ps])) for k in ('self', 'user', 'system')},
                                    'letter_mass_min': float(min(sum(p[k] for k in ('self', 'user', 'system')) for p in ps))})
        np.savez_compressed(gdir / f'{b["id"]}.npz', **arrs)
        cs.atomic(done, rec)
        if bi % 10 == 0:
            print(f'N {bi + 1}/{len(bodies)} {time.time() - t0:.0f}s', flush=True)
    for h in handles:
        h.remove()
    cs.atomic(out / 'status.json', {'state': 'complete', 'elapsed_s': time.time() - t0})
    summarize(out, L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--tmpl-kwargs', default='{}')
    p.add_argument('--bodies', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--lang', choices=('zh', 'en'), default='zh')
    p.add_argument('--bodies-en', default=None)
    p.add_argument('--geometry-only', action='store_true')
    p.add_argument('--max-memory-gib', type=float, default=44)
    p.add_argument('--cpu-offload-gib', type=float, default=0)
    p.add_argument('--summarize-only', type=int, default=0)
    args = p.parse_args()
    if args.summarize_only:
        summarize(Path(args.out), args.summarize_only)
        return
    try:
        run(args)
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        cs.atomic(Path(args.out) / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
