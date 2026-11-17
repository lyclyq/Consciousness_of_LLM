"""P2 persistence pilot (11-06, ../../PROTOCOL_zh.md): does the own-message offset at the reply position keep its size as more of
the message is visible? Each body is cut at clause boundaries (k = 0..n clauses; k = n is the full body). The message under test
is rendered with the model's own chat template in the main order [ROLE: text][user: Q] or the natural order
[user: REQUEST][ROLE: text][user: Q]. One teacher-forced forward per rendering, one sequence per forward, attention mask of
ones, no generation. The readout is the last prompt token (the response onset of phase G in crossmodel_suite.py).
Stored per body (resumable, atomic):
  role_dist  (n+1, L, 3) f32      readout distances |A-U|, |A-S|, |U-S| at every layer, per length k (NaN if not rendered)
  role_diff  (n+1, 3, K, d) f16   readout differences A-U, A-S, U-S at the K kept layers
  role_norm  (n+1, 3, K) f32      readout norms of A, U, S
  full_state (3, K, d) f16        readout states of A, U, S for the full body
  word_diff  (P, 2, 3, K, d) f16  main order only: readout difference between the two words of a pair; frame before / after the
                                  text; lengths k = 1, ceil(n/2), n; User message
  word_norm  (P, 2, 3, 2, K) f32
  dec_mF, dec_mL (Q, K, d) f16, dec_eF, dec_eL (Q, K) f32: on the body tokens of the full body, the mean difference vector and
                                  the mean squared difference over the first / last third of the tokens. Q = role pairs, plus the
                                  word pairs with the frame before the text (main order).
The json next to it holds lengths, token counts, flags and the gate against phase G (full body, main order, every layer).
usage: persistence_readout_20261106.py --model DIR --tmpl-kwargs JSON --bodies role_bodies.jsonl --out DIR
       [--lang en --bodies-en FILE] [--order natural --request TEXT] [--gate-g DIR] [--limit N] [--dry-run]"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import socket
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402
from crossmodel_suite import Renderer, atomic, checked_tokenizer, common_body, find_layers, ids_of, load  # noqa: E402

ROLES = (('A', 'assistant'), ('U', 'user'), ('S', 'system'))
RPAIRS = ((0, 1), (0, 2), (1, 2))                       # A-U, A-S, U-S
KEEP_FRACS = (0.375, 0.5, 0.625, 0.75, 0.875, 1.0)      # L = 64 -> 24 32 39 47 55 63 (the self-space token layers)
PUNCT = {'zh': '，。；：！？、', 'en': ',.;:!?'}
WORDS = {
    'zh': [('place', '医院', '学校', '以下是{w}的记录：', '（以上是{w}的记录）'),
           ('time', '昨天', '明天', '以下是{w}的记录：', '（以上是{w}的记录）'),
           ('name', '小明', '小红', '以下是{w}的记录：', '（以上是{w}的记录）'),
           ('feeling', '愉快', '痛苦', '以下是{w}的记录：', '（以上是{w}的记录）')],
    'en': [('place', 'hospital', 'school', "The following is the {w}'s record: ", " (That was the {w}'s record.)"),
           ('time', 'yesterday', 'tomorrow', "The following is {w}'s record: ", " (That was {w}'s record.)"),
           ('name', 'Ming', 'Hong', "The following is {w}'s record: ", " (That was {w}'s record.)"),
           ('feeling', 'pleasant', 'painful', 'The following is a {w} record: ', ' (That was a {w} record.)')]}
POS = ('before', 'after')
VDT = np.float16   # storage dtype of vectors (--store-f32: float32)


def clause_texts(body, lang):
    """texts[k] for k = 1..n: the body up to and including its k-th clause; texts[n] is the whole body."""
    cuts = []
    for i, ch in enumerate(body):
        if ch in PUNCT[lang] and (lang == 'zh' or i + 1 == len(body) or body[i + 1].isspace()):
            cuts.append(i + 1)
    end = len(body.rstrip())
    if not cuts or cuts[-1] < end:
        cuts.append(end)
    keep = []
    for c in cuts:
        if body[(keep[-1] if keep else 0):c].strip():
            keep.append(c)
    texts = [body[:c].rstrip() for c in keep]
    assert texts[-1] == body.rstrip(), 'clause split does not reproduce the body'
    return texts


def msgs_for(order, role, content, q, request):
    m = [{'role': role, 'content': content}, {'role': 'user', 'content': q}]
    return m if order == 'main' else [{'role': 'user', 'content': request}] + m


def render(tok, kw, msgs, target):
    """ids and the target's token span; target '' (empty message): ids only, checked against tokenize=True."""
    if target:
        return Renderer(tok, kw).render(msgs, target)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    ids = ids_of(tok(text, add_special_tokens=False))
    ref = ids_of(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw))
    assert ids == ref, 'text and token renderings disagree'
    return ids, None


def natural_system_ok(tok, kw, q, request, text):
    """the template keeps a System message after a user turn, in place (else: error, dropped, or moved)"""
    try:
        t = tok.apply_chat_template(msgs_for('natural', 'system', text, q, request), tokenize=False, add_generation_prompt=True, **kw)
    except Exception:
        return False
    return t.count(text) == 1 and t.index(text) > t.index(request)


def plan_body(tok, kw, body, lang, order, q, request, sys_ok, lean=False):
    """every rendering of one body: roles at k = 0..n, words (main order) at k = 1, ceil(n/2), n (lean: k = 1, n)"""
    texts = clause_texts(body, lang)
    n = len(texts)
    roles = {}
    for k in range(n + 1):
        text = texts[k - 1] if k else ''
        for ri, (rn, role) in enumerate(ROLES):
            if order == 'natural' and role == 'system' and not sys_ok:
                continue
            try:
                roles[(k, ri)] = render(tok, kw, msgs_for(order, role, text, q, request), text)
            except Exception as e:
                if k:
                    raise
                roles[(k, ri)] = ('unsupported', repr(e)[:160])   # empty message refused by the template
    words = {}
    wl = [1, n] if lean else [1, math.ceil(n / 2), n]
    if order == 'main':
        for pi, (_, w1, w2, pre, post) in enumerate(WORDS[lang]):
            for si, sd in enumerate(POS):
                for j, k in enumerate(wl):
                    text = texts[k - 1]
                    for wi, w in enumerate((w1, w2)):
                        content = pre.format(w=w) + text if sd == 'before' else text + post.format(w=w)
                        words[(pi, si, j, wi)] = render(tok, kw, msgs_for(order, 'user', content, q, request), text)
    full = {ROLES[ri][0]: roles[(n, ri)] for ri in range(3) if (n, ri) in roles}
    spans = {'roles': common_body(full)}
    if order == 'main' and not lean:            # lean: no body-token decomposition for the words
        jf = len(wl) - 1
        for pi in range(len(WORDS[lang])):
            spans[f'word{pi}'] = common_body({'w1': words[(pi, 0, jf, 0)], 'w2': words[(pi, 0, jf, 1)]})
    return texts, wl, roles, words, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tmpl-kwargs', default='{}')
    ap.add_argument('--bodies', required=True)
    ap.add_argument('--lang', choices=('zh', 'en'), default='zh')
    ap.add_argument('--bodies-en', default=None, help='JSON id -> English body (with --lang en)')
    ap.add_argument('--order', choices=('main', 'natural'), default='main')
    ap.add_argument('--request', default=None, help='natural order: the user request before the message under test')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gate-g', default=None, help="the model's phase-G directory (npz with onset_<role>), main order only")
    ap.add_argument('--gate-strict', action='store_true', help='exit with an error if the gate fails on a body')
    ap.add_argument('--max-memory-gib', type=float, default=44)
    ap.add_argument('--cpu-offload-gib', type=float, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true', help='render and check every condition; no model, no GPU')
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('--min-free-gib', type=float, default=34.0)
    ap.add_argument('--stop-file', default=None, help='stop cleanly before the next body when this file exists')
    ap.add_argument('--store-f32', action='store_true', help='store vectors as float32 (models whose readout differences overflow float16)')
    ap.add_argument('--fixed-date', default=None, help="YYYY-MM-DD: the template's strftime_now() returns this date (templates that write"
                    " the date of the call into a default system message, e.g. Apertus; reproduces the rendering of the paper's run)")
    ap.add_argument('--lean', action='store_true', help='store the role differences only at k = 0, 1, n (the distances stay at every k)'
                    ' and run the words only at k = 1, n, without their body-token decomposition (phase 3, to save disk)')
    args = ap.parse_args()
    global VDT
    VDT = np.float32 if args.store_f32 else np.float16
    kw = json.loads(args.tmpl_kwargs)
    if args.fixed_date:
        import datetime as _dt
        _day = _dt.date.fromisoformat(args.fixed_date)
        kw['strftime_now'] = lambda fmt: _dt.datetime(_day.year, _day.month, _day.day).strftime(fmt)  # shadows the jinja global
    q = cs.Q_REPLY_EN if args.lang == 'en' else cs.Q_REPLY
    if args.order == 'natural':
        assert args.request, '--request is required in natural order'
    bodies = [json.loads(l) for l in open(args.bodies, encoding='utf-8') if l.strip()]
    if args.lang == 'en':
        en = json.loads(Path(args.bodies_en).read_text(encoding='utf-8'))
        bodies = [{**b, 'body': en[b['id']]} for b in bodies]
    if args.limit:
        bodies = bodies[:args.limit]
    out = Path(args.out)

    if args.dry_run:
        from transformers import AutoTokenizer
        tdir = args.tokenizer or args.model
        tok = checked_tokenizer(AutoTokenizer.from_pretrained(tdir, trust_remote_code=True), tdir)
        sys_ok = natural_system_ok(tok, kw, q, args.request, bodies[0]['body']) if args.order == 'natural' else True
        nfw, ncl, k0, unequal, ntok = 0, [], {}, {}, []
        for b in bodies:
            texts, wl, roles, words, spans = plan_body(tok, kw, b['body'], args.lang, args.order, q, args.request, sys_ok, args.lean)
            ncl.append(len(texts))
            nfw += sum(1 for v in roles.values() if v[0] != 'unsupported') + len(words)
            for (k, ri), v in roles.items():
                if k == 0:
                    k0[ROLES[ri][0]] = k0.get(ROLES[ri][0], True) and v[0] != 'unsupported'
            for (pi, si, j, wi), v in words.items():
                if wi == 0 and len(v[0]) != len(words[(pi, si, j, 1)][0]):
                    unequal[f'{WORDS[args.lang][pi][0]}_{POS[si]}'] = unequal.get(f'{WORDS[args.lang][pi][0]}_{POS[si]}', 0) + 1
            ntok.append(len(spans['roles']['A']))
        print(json.dumps({'bodies': len(bodies), 'order': args.order, 'lang': args.lang, 'natural_system_ok': sys_ok,
                          'clauses_min_med_max': [min(ncl), int(np.median(ncl)), max(ncl)], 'forwards': nfw,
                          'empty_message_rendered': k0, 'word_pairs_unequal_length_renderings': unequal,
                          'full_body_tokens_min_max': [min(ntok), max(ntok)]}, ensure_ascii=False))
        print('DRY RUN OK')
        return

    out.mkdir(parents=True, exist_ok=True)
    largs = argparse.Namespace(model=args.model, max_memory_gib=args.max_memory_gib, cpu_offload_gib=args.cpu_offload_gib)
    torch, tok, model, cfg, placement, cls = load(largs)
    lname, layers = find_layers(model)
    L = len(layers)
    keep = sorted({round(f * (L - 1)) for f in KEEP_FRACS})
    K = len(keep)
    kpos = {l: i for i, l in enumerate(keep)}
    sys_ok = natural_system_ok(tok, kw, q, args.request, bodies[0]['body']) if args.order == 'natural' else True
    me = Path(__file__).resolve()
    ident = {'experiment': 'P2_persistence_readout_20261106', 'script_sha256': hashlib.sha256(me.read_bytes()).hexdigest(),
             'suite_sha256': hashlib.sha256((me.parent / 'crossmodel_suite.py').read_bytes()).hexdigest(),
             'model': args.model, 'model_class': cls, 'layers_module': lname, 'n_layers': L, 'keep_layers': keep,
             'tmpl_kwargs': json.loads(args.tmpl_kwargs), 'fixed_date': args.fixed_date, 'order': args.order, 'request': args.request, 'reply_question': q, 'lang': args.lang,
             'natural_system_ok': sys_ok, 'lean': args.lean, 'role_diff_slots': 'k = 0, 1, n' if args.lean else 'k = 0..n', 'roles': ROLES, 'role_pairs': ['A-U', 'A-S', 'U-S'], 'words': WORDS[args.lang],
             'word_positions': POS, 'device_map': placement, 'torch': torch.__version__,
             'transformers': __import__('transformers').__version__, 'host': socket.gethostname(),
             'tokenizer_rebuilt_from': getattr(tok, '_p2_rebuilt_from', None), 'cpu_offload_gib': args.cpu_offload_gib,
             'n_bodies': len(bodies), 'bodies_file': args.bodies, 'bodies_sha256': hashlib.sha256(Path(args.bodies).read_bytes()).hexdigest(),
             'bodies_en_sha256': hashlib.sha256(Path(args.bodies_en).read_bytes()).hexdigest() if args.bodies_en else None,
             'gate_g': args.gate_g}
    atomic(out / 'identity.json', ident)
    print(json.dumps({k: ident[k] for k in ('model_class', 'n_layers', 'keep_layers', 'order', 'lang', 'natural_system_ok')}), flush=True)
    dev0 = model.get_input_embeddings().weight.device
    TDT = torch.float32 if VDT == np.float32 else torch.float16
    st = {'span': None}
    cap_last, cap_span = {}, {}

    def hook(i):
        def h(mod, inp, outp):
            x = outp[0] if isinstance(outp, tuple) else outp
            assert x.dim() == 3, f'layer output of rank {x.dim()} (hyper-connection streams are not handled here)'
            cap_last[i] = x[0, -1].detach()
            if st['span'] is not None and i in kpos:
                a, e = st['span']
                cap_span[i] = x[0, a:e].detach()
        return h
    handles = [layers[i].register_forward_hook(hook(i)) for i in range(L)]

    def fwd(ids, span=None):
        cap_last.clear()
        cap_span.clear()
        st['span'] = None if span is None else (span[0], span[-1] + 1)
        x = torch.tensor([ids], dtype=torch.long, device=dev0)
        with torch.no_grad():
            model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=1)
        st['span'] = None
        last = torch.stack([cap_last[i].float().to(dev0) for i in range(L)])
        sp = torch.stack([cap_span[l].float().to(dev0) for l in keep]) if span is not None else None
        return last, sp

    def decomp(x1, x2):
        D = x1 - x2                                   # (K, T, d)
        T = D.shape[1]
        p = torch.arange(T, device=D.device, dtype=torch.float32) / T
        f, la = p < 1 / 3, p >= 2 / 3
        return (D[:, f].mean(1), D[:, la].mean(1), (D[:, f] ** 2).sum(-1).mean(1), (D[:, la] ** 2).sum(-1).mean(1))

    gate_dir = Path(args.gate_g) if args.gate_g else None
    t0 = time.time()
    done = 0
    for b in bodies:
        if (out / f'{b["id"]}.json').exists():
            continue
        if args.stop_file and Path(args.stop_file).exists():
            print(f'[stop] {args.stop_file} exists: stopping before {b["id"]} (rerun resumes)', flush=True)
            break
        free = shutil.disk_usage(out).free / 2 ** 30
        if free < args.min_free_gib:
            print(f'[stop] {free:.1f} GiB free < {args.min_free_gib}: stopping before {b["id"]} (rerun resumes)', flush=True)
            break
        tb = time.time()
        texts, wl, roles, words, spans = plan_body(tok, kw, b['body'], args.lang, args.order, q, args.request, sys_ok, args.lean)
        n = len(texts)
        nw = len(WORDS[args.lang]) if args.order == 'main' else 0
        Q = 3 + (0 if args.lean else nw)
        slots = {0: 0, 1: 1, n: 2} if args.lean else {k: k for k in range(n + 1)}   # where role_diff / role_norm of length k go
        NS = 3 if args.lean else n + 1
        WJ = len(wl)
        role_dist = np.full((n + 1, L, 3), np.nan, np.float32)
        role_diff = role_norm = full_state = word_diff = word_norm = None
        dec = None
        gate = {}
        tokcount = {}
        for k in range(n + 1):
            H = {}
            for ri in range(3):
                v = roles.get((k, ri))
                if v is None or v[0] == 'unsupported':
                    continue
                ids, span = v
                rspan = spans['roles'][ROLES[ri][0]] if k == n else None
                last, sp = fwd(ids, rspan)
                H[ri] = (last, sp)
                tokcount[f'{ROLES[ri][0]}{k}'] = len(ids)
            if role_diff is None and H:
                d = next(iter(H.values()))[0].shape[-1]
                role_diff = np.full((NS, 3, K, d), np.nan, VDT)
                role_norm = np.full((NS, 3, K), np.nan, np.float32)
                full_state = np.full((3, K, d), np.nan, VDT)
                dec = {'mF': np.full((Q, K, d), np.nan, VDT), 'mL': np.full((Q, K, d), np.nan, VDT),
                       'eF': np.full((Q, K), np.nan, np.float32), 'eL': np.full((Q, K), np.nan, np.float32)}
            sk = slots.get(k)
            for ri, (last, _) in H.items():
                if sk is not None:
                    role_norm[sk, ri] = last[keep].norm(dim=-1).cpu().numpy()
            for pi, (i1, i2) in enumerate(RPAIRS):
                if i1 in H and i2 in H:
                    role_dist[k, :, pi] = (H[i1][0] - H[i2][0]).norm(dim=-1).cpu().numpy()
                    if sk is not None:
                        role_diff[sk, pi] = (H[i1][0][keep] - H[i2][0][keep]).to(TDT).cpu().numpy()
            if k == n:
                for ri, (last, sp) in H.items():
                    full_state[ri] = last[keep].to(TDT).cpu().numpy()
                for pi, (i1, i2) in enumerate(RPAIRS):
                    if i1 in H and i2 in H:
                        mF, mL, eF, eL = decomp(H[i1][1], H[i2][1])
                        dec['mF'][pi], dec['mL'][pi] = mF.to(TDT).cpu().numpy(), mL.to(TDT).cpu().numpy()
                        dec['eF'][pi], dec['eL'][pi] = eF.cpu().numpy(), eL.cpu().numpy()
                g = gate_dir / 'G' / f'{b["id"]}.npz' if gate_dir else None
                if g is not None and g.exists() and args.order == 'main':
                    with np.load(g) as z:
                        for ri, (rn, role) in enumerate(ROLES):
                            if ri in H and f'onset_{role}' in z.files:
                                ref = torch.tensor(z[f'onset_{role}'].astype(np.float32), device=dev0).reshape(L, -1)
                                ours = H[ri][0]
                                cosv = torch.nn.functional.cosine_similarity(ours, ref, dim=-1)
                                rel = (ours - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)
                                gate[rn] = {'min_cos': float(cosv.min()), 'max_rel': float(rel.max())}
                    gate['ok'] = bool(gate) and all(v['min_cos'] >= 0.999 and v['max_rel'] <= 0.03 for kk, v in gate.items() if kk != 'ok')
                    if args.gate_strict and not gate['ok']:
                        sys.exit(f'GATE FAILED on {b["id"]}: {gate}')
            del H
        if nw:
            word_diff = np.full((nw, 2, WJ, K, d), np.nan, VDT)
            word_norm = np.full((nw, 2, WJ, 2, K), np.nan, np.float32)
            for pi in range(nw):
                for si in range(2):
                    for j in range(WJ):
                        Hw = []
                        dec_here = si == 0 and j == WJ - 1 and not args.lean
                        for wi in range(2):
                            ids, span = words[(pi, si, j, wi)]
                            wspan = spans[f'word{pi}'][('w1', 'w2')[wi]] if dec_here else None
                            Hw.append(fwd(ids, wspan))
                            tokcount[f'w{pi}{POS[si][0]}{j}{wi}'] = len(ids)
                            word_norm[pi, si, j, wi] = Hw[-1][0][keep].norm(dim=-1).cpu().numpy()
                        word_diff[pi, si, j] = (Hw[0][0][keep] - Hw[1][0][keep]).to(TDT).cpu().numpy()
                        if dec_here:
                            mF, mL, eF, eL = decomp(Hw[0][1], Hw[1][1])
                            dec['mF'][3 + pi], dec['mL'][3 + pi] = mF.to(TDT).cpu().numpy(), mL.to(TDT).cpu().numpy()
                            dec['eF'][3 + pi], dec['eL'][3 + pi] = eF.cpu().numpy(), eL.cpu().numpy()
                        del Hw
        arrs = {'role_dist': role_dist, 'role_diff': role_diff, 'role_norm': role_norm, 'full_state': full_state,
                'dec_mF': dec['mF'], 'dec_mL': dec['mL'], 'dec_eF': dec['eF'], 'dec_eL': dec['eL']}
        if nw:
            arrs.update(word_diff=word_diff, word_norm=word_norm)
        for kk, v in arrs.items():
            assert not np.isinf(v).any(), f'{kk} has inf in {b["id"]} (float16 overflow? rerun with --store-f32)'
        part = out / f'{b["id"]}.npz.part'
        with open(part, 'wb') as fh:
            np.savez(fh, **arrs)
        part.replace(out / f'{b["id"]}.npz')
        unsup = [f'{ROLES[ri][0]}{k}' for (k, ri), v in roles.items() if v[0] == 'unsupported']
        atomic(out / f'{b["id"]}.json', {'id': b['id'], 'split': b.get('split'), 'category': b.get('category'), 'n_clauses': n,
                                          'text_chars': [len(t) for t in texts], 'word_lengths': wl if nw else None,
                                          'n_full_body_tokens': len(spans['roles']['A']), 'token_counts': tokcount,
                                          'unsupported': unsup, 'gate': gate or None, 'seconds': round(time.time() - tb, 1)})
        done += 1
        if done % 10 == 0 or done == 1:
            print(f'{done} bodies, {time.time() - t0:.0f}s, last {b["id"]} ({n} clauses, {time.time() - tb:.1f}s), {free:.0f} GiB free'
                  + (f', gate {gate}' if gate else ''), flush=True)
    for hd in handles:
        hd.remove()
    n_all = sum(1 for b in bodies if (out / f'{b["id"]}.json').exists())
    print(f'finished: {n_all}/{len(bodies)} bodies on disk', flush=True)
    if n_all == len(bodies):
        (out / 'DONE').write_text(time.strftime('%F %T') + '\n')


if __name__ == '__main__':
    main()
