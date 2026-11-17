"""P2 ordinary words along the message, expanded (11-10, ../../PROTOCOL_zh.md). Exploratory, for the Discussion conjecture
sec:disc_along. Same measurement as the body-token decomposition of the 11-06 pilot (persistence_readout_20261106.py, fab9c7b2):
  full body only; for a pair of renderings (x1, x2) of the same body, D = x1 - x2 on the body tokens common to the two renderings
  (crossmodel_suite.common_body), at the KEEP_FRACS layers; mean difference vector and mean squared difference over the first
  third (p < 1/3) and the last third (p >= 2/3) of these tokens, p = position / number of common tokens.
Rows:
  roles (--roles): A-U, A-S, U-S, the body as an Assistant, User, System message, [ROLE: body][user: Q] (main) or
                   [user: REQ][ROLE: body][user: Q] (natural); S is left out (NaN row) where the template does not keep a System
                   message after a user turn (persistence_readout_20261106.natural_system_ok).
  words: for each pair (w1, w2) of ../../items/words_v2.json, the body as a User message preceded by the frame with w1, and with
         w2 (zh: 以下是{w}的记录：, en: the pair's frame_en), same order and question as the roles. --pairs new runs only the
         pairs that the pilot did not measure (main order of the pilot settings), --pairs all every pair.
Readout states (last prompt token, every layer) are used by the gates only; the per-body npz keeps the decomposition, the
three role readout states at the kept layers (with --roles) and token counts.
Gates (before any measurement, on the first --gate-n bodies of the selected split; the run refuses to start on a failure):
  --gate-pilot DIR  the pilot run of the same setting (same order, template arguments, language, date): A/U/S readout states at
                    the kept layers against its full_state, the role-pair decomposition against its dec rows 0-2, and in the
                    main order of a non-lean pilot run the four pilot word pairs against its dec rows 3-6.
  --gate-g DIR      the paper's feature run (DIR/G or DIR/N npz with onset_<role>, every layer): A/U/S readout states, every
                    layer, rendered with --gate-tmpl-kwargs (default: the measurement's) in the measurement order.
  criterion: cosine >= 0.999 and relative error <= 0.03 everywhere (the pilot's gate).
Hyper-connection models: a layer output of rank 4 [batch, tokens, streams, hidden] is reshaped to [batch, tokens,
streams*hidden], the full concatenated state of crossmodel_suite.summarize (Qwen3.8-Flash-Next's output already is).
One sequence per forward, attention mask of ones, no generation, logits_to_keep=1 (as the pilot). Resumable: one json + npz per
body, written atomically; DONE when every selected body is on disk. Stop: --stop-file (checked before every body), --min-free-gib.
usage: words_along_20261110.py --model DIR --tmpl-kwargs JSON --bodies FILE --words words_v2.json --out DIR
       [--lang en --bodies-en FILE] [--order natural --request TEXT] [--roles] [--pairs new|all] [--split confirmation]
       [--gate-pilot DIR | --gate-g DIR [--gate-tmpl-kwargs JSON]] [--gate-n 2] [--limit N] [--dry-run] [--tokenizer DIR]"""
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
import crossmodel_suite as cs  # noqa: E402
from crossmodel_suite import atomic, checked_tokenizer, common_body, find_layers, load  # noqa: E402
import persistence_readout_20261106 as pr  # noqa: E402
from persistence_readout_20261106 import KEEP_FRACS, ROLES, RPAIRS, msgs_for, natural_system_ok, render  # noqa: E402

GATE_COS, GATE_REL = 0.999, 0.03


def load_words(path, lang, which):
    W = json.loads(Path(path).read_text(encoding='utf-8'))
    pairs = []
    for p in W['pairs']:
        first = p['id'].endswith('1')                      # the first pair of its category (stage 1 of GLM-5.3-Flash)
        if (which == 'new' and p['existing']) or (which == 'first' and not first) or (which == 'rest' and first):
            continue
        frame = W['frame_zh'] if lang == 'zh' else p['frame_en']
        pairs.append({'id': p['id'], 'category': p['category'], 'existing': p['existing'], 'w': p[lang], 'frame': frame})
    # the existing pairs must be exactly the pilot's (words and frame before the text)
    pilot = {w[0]: w for w in pr.WORDS[lang]}
    for p in W['pairs']:
        if p['existing']:
            q = pilot[p['category']]
            frame = W['frame_zh'] if lang == 'zh' else p['frame_en']
            assert (p[lang][0], p[lang][1], frame) == (q[1], q[2], q[3]), ('existing pair differs from the pilot', p['id'], q)
    return W, pairs


def select_bodies(args, path=None, path_en=None):
    path, path_en = path or args.bodies, path_en or args.bodies_en
    bodies = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    if args.lang == 'en':
        en = json.loads(Path(path_en).read_text(encoding='utf-8'))
        bodies = [{**b, 'body': en[b['id']]} for b in bodies]
    if args.split != 'all':
        bodies = [b for b in bodies if b.get('split') == args.split]
    assert bodies, 'no body in the selected split'
    return bodies


def plan(tok, kw, body, order, q, request, sys_ok, pairs, with_roles):
    """renderings of one body: roles (full body) and the two renderings of every word pair; common body spans"""
    text = body.rstrip()
    R, spans = {}, {}
    if with_roles:
        for ri, (rn, role) in enumerate(ROLES):
            if order == 'natural' and role == 'system' and not sys_ok:
                continue
            R[rn] = render(tok, kw, msgs_for(order, role, text, q, request), text)
        spans['roles'] = common_body(R)
    W = {}
    for p in pairs:
        r2 = {f'w{i + 1}': render(tok, kw, msgs_for(order, 'user', p['frame'].format(w=w) + text, q, request), text)
              for i, w in enumerate(p['w'])}
        W[p['id']] = r2
        spans[p['id']] = common_body(r2)
    return text, R, W, spans


def rows_for(with_roles, pairs):
    return (['A-U', 'A-S', 'U-S'] if with_roles else []) + [p['id'] for p in pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tmpl-kwargs', default='{}')
    ap.add_argument('--bodies', required=True)
    ap.add_argument('--words', required=True)
    ap.add_argument('--lang', choices=('zh', 'en'), default='zh')
    ap.add_argument('--bodies-en', default=None)
    ap.add_argument('--order', choices=('main', 'natural'), default='main')
    ap.add_argument('--request', default=None)
    ap.add_argument('--roles', action='store_true', help='also measure the three role pairs (settings without a pilot run)')
    ap.add_argument('--pairs', choices=('new', 'all', 'first', 'rest'), default='all',
                    help='new: not in the pilot; first / rest: the first pair of each category / the other 24 (GLM-5.3 stages)')
    ap.add_argument('--split', default='confirmation', help="body split to run ('all' for every body)")
    ap.add_argument('--out', required=True)
    ap.add_argument('--gate-pilot', default=None)
    ap.add_argument('--gate-g', default=None)
    ap.add_argument('--gate-sub', default=None, help='subdirectory of --gate-g with the npz (default G in main order, N in natural)')
    ap.add_argument('--gate-tmpl-kwargs', default=None)
    ap.add_argument('--gate-n', type=int, default=2)
    ap.add_argument('--gate-bodies', default=None, help='body file of the gate (default --bodies), e.g. the 150 bodies of the paper G run')
    ap.add_argument('--gate-bodies-en', default=None)
    ap.add_argument('--no-gate-reason', default=None, help='run without a gate; the reason is written to identity.json')
    ap.add_argument('--max-memory-gib', type=float, default=44)
    ap.add_argument('--cpu-offload-gib', type=float, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('--min-free-gib', type=float, default=8.0)
    ap.add_argument('--stop-file', default=None)
    ap.add_argument('--store-f32', action='store_true')
    ap.add_argument('--fixed-date', default=None)
    args = ap.parse_args()
    VDT = np.float32 if args.store_f32 else np.float16
    kw = json.loads(args.tmpl_kwargs)
    gkw = json.loads(args.gate_tmpl_kwargs) if args.gate_tmpl_kwargs else None
    if args.fixed_date:
        import datetime as _dt
        _day = _dt.date.fromisoformat(args.fixed_date)
        fix = lambda fmt: _dt.datetime(_day.year, _day.month, _day.day).strftime(fmt)  # noqa: E731
        kw['strftime_now'] = fix
        if gkw is not None:
            gkw['strftime_now'] = fix
    q = cs.Q_REPLY_EN if args.lang == 'en' else cs.Q_REPLY
    if args.order == 'natural':
        assert args.request, '--request is required in natural order'
    W, pairs = load_words(args.words, args.lang, args.pairs)
    bodies = select_bodies(args)
    gate_bodies = (select_bodies(args, args.gate_bodies, args.gate_bodies_en) if args.gate_bodies else bodies)[:args.gate_n]
    if args.limit:
        bodies = bodies[:args.limit]
    rows = rows_for(args.roles, pairs)
    out = Path(args.out)
    n_gates = sum(x is not None for x in (args.gate_pilot, args.gate_g, args.no_gate_reason))
    assert n_gates == 1 or args.dry_run, 'give exactly one of --gate-pilot, --gate-g, --no-gate-reason'

    if args.dry_run:
        from transformers import AutoTokenizer
        tdir = args.tokenizer or args.model
        tok = checked_tokenizer(AutoTokenizer.from_pretrained(tdir, trust_remote_code=True), tdir)
        sys_ok = natural_system_ok(tok, kw, q, args.request, bodies[0]['body']) if args.order == 'natural' else True
        unequal, wlen, ncommon, nfw = {}, {}, [], 0
        for p in pairs:
            wlen[p['id']] = [len(cs.ids_of(tok(w, add_special_tokens=False))) for w in p['w']]
        for b in bodies:
            text, R, Wr, spans = plan(tok, kw, b['body'], args.order, q, args.request, sys_ok, pairs, args.roles)
            nfw += len(R) + 2 * len(Wr)
            for pid, r2 in Wr.items():
                if len(r2['w1'][0]) != len(r2['w2'][0]):
                    unequal[pid] = unequal.get(pid, 0) + 1
            ncommon += [len(s[next(iter(s))]) for s in spans.values()]
        print(json.dumps({'bodies': len(bodies), 'order': args.order, 'lang': args.lang, 'natural_system_ok': sys_ok,
                          'rows': rows, 'forwards': nfw, 'pairs_unequal_rendering_length': unequal,
                          'word_token_counts': wlen, 'common_body_tokens_min_max': [min(ncommon), max(ncommon)],
                          'tokenizer': str(tdir)}, ensure_ascii=False))
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
    ident = {'experiment': 'P2_words_along_20261110', 'script_sha256': hashlib.sha256(me.read_bytes()).hexdigest(),
             'suite_sha256': hashlib.sha256((me.parent / 'crossmodel_suite.py').read_bytes()).hexdigest(),
             'pilot_script_sha256': hashlib.sha256((me.parent / 'persistence_readout_20261106.py').read_bytes()).hexdigest(),
             'words_sha256': hashlib.sha256(Path(args.words).read_bytes()).hexdigest(), 'words_version': W['version'],
             'model': args.model, 'model_class': cls, 'layers_module': lname, 'n_layers': L, 'keep_layers': keep,
             'tmpl_kwargs': json.loads(args.tmpl_kwargs), 'fixed_date': args.fixed_date, 'order': args.order, 'request': args.request,
             'reply_question': q, 'lang': args.lang, 'natural_system_ok': sys_ok, 'rows': rows, 'pairs': args.pairs,
             'pair_ids': [p['id'] for p in pairs], 'split': args.split, 'n_bodies': len(bodies), 'bodies_file': args.bodies,
             'bodies_sha256': hashlib.sha256(Path(args.bodies).read_bytes()).hexdigest(),
             'bodies_en_sha256': hashlib.sha256(Path(args.bodies_en).read_bytes()).hexdigest() if args.bodies_en else None,
             'gate_pilot': args.gate_pilot, 'gate_g': args.gate_g, 'gate_bodies': args.gate_bodies or args.bodies,
             'gate_body_ids': [b['id'] for b in gate_bodies], 'gate_tmpl_kwargs': json.loads(args.gate_tmpl_kwargs) if args.gate_tmpl_kwargs else None,
             'no_gate_reason': args.no_gate_reason, 'device_map': placement, 'torch': torch.__version__,
             'transformers': __import__('transformers').__version__, 'host': socket.gethostname(),
             'tokenizer_rebuilt_from': getattr(tok, '_p2_rebuilt_from', None), 'cpu_offload_gib': args.cpu_offload_gib,
             'store': 'float32' if VDT == np.float32 else 'float16'}
    dev0 = model.get_input_embeddings().weight.device
    TDT = torch.float32 if VDT == np.float32 else torch.float16
    st = {'span': None, 'rank': set()}
    cap_last, cap_span = {}, {}

    def hook(i):
        def h(mod, inp, outp):
            x = outp[0] if isinstance(outp, tuple) else outp
            st['rank'].add(x.dim())
            if x.dim() == 4:                                    # [batch, tokens, streams, hidden] -> concatenated streams
                x = x.reshape(x.shape[0], x.shape[1], -1)
            assert x.dim() == 3, f'layer output of rank {x.dim()}'
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

    def cmp(ours, ref):
        ref = torch.as_tensor(np.asarray(ref, np.float32), device=dev0).reshape(ours.shape[0], -1)
        ours = ours.reshape(ref.shape)
        cosv = torch.nn.functional.cosine_similarity(ours, ref, dim=-1)
        rel = (ours - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)
        return float(cosv.min()), float(rel.max())

    def measure(b, pairs_here, with_roles, kw_here):
        """forwards of one body; returns readout states (every layer) per role, decomposition per row, token counts"""
        text, R, Wr, spans = plan(tok, kw_here, b['body'], args.order, q, args.request, sys_ok, pairs_here, with_roles)
        H, dec, tc, ncom, uneq = {}, {}, {}, {}, []
        for rn, (ids, _) in R.items():
            H[rn] = fwd(ids, spans['roles'][rn])
            tc[rn] = len(ids)
        for pi, (i1, i2) in enumerate(RPAIRS):
            a, c = ROLES[i1][0], ROLES[i2][0]
            if a in H and c in H:
                dec[f'{a}-{c}'] = decomp(H[a][1], H[c][1])
                ncom[f'{a}-{c}'] = int(H[a][1].shape[1])
        for pid, r2 in Wr.items():
            h1 = fwd(r2['w1'][0], spans[pid]['w1'])
            h2 = fwd(r2['w2'][0], spans[pid]['w2'])
            dec[pid] = decomp(h1[1], h2[1])
            ncom[pid] = int(h1[1].shape[1])
            tc[pid] = [len(r2['w1'][0]), len(r2['w2'][0])]
            if tc[pid][0] != tc[pid][1]:
                uneq.append(pid)
            del h1, h2
        return H, dec, tc, ncom, uneq

    # ---------------- gate ----------------
    gate = {'criterion': {'cos_min': GATE_COS, 'rel_max': GATE_REL}, 'bodies': [], 'ok': True}
    if args.gate_pilot:
        gp = Path(args.gate_pilot)
        pid_ident = json.loads((gp / 'identity.json').read_text(encoding='utf-8'))
        for k in ('order', 'lang', 'tmpl_kwargs', 'fixed_date'):
            assert pid_ident.get(k) == ident[k] or (k == 'tmpl_kwargs' and pid_ident.get(k) == json.loads(args.tmpl_kwargs)), \
                ('pilot run differs', k, pid_ident.get(k), ident[k])
        assert pid_ident['keep_layers'] == keep, ('kept layers differ from the pilot', pid_ident['keep_layers'], keep)
        own = 'rows' in pid_ident                           # an earlier words_along run (GLM-5.3 stage 1) instead of the pilot
        all_pairs = load_words(args.words, args.lang, 'all')[1]
        if own:
            assert pid_ident['rows'][:3] == ['A-U', 'A-S', 'U-S'], 'the earlier run has no role rows'
            ref_pairs = [p for p in all_pairs if p['id'] in pid_ident['rows'][3:5]]   # recompute two of its word pairs
        else:
            ref_pairs = [p for p in all_pairs if p['existing']]
        for b in gate_bodies:
            with np.load(gp / f'{b["id"]}.npz') as z:
                Z = {k: z[k] for k in z.files}
            states = Z['role_state'] if own else Z['full_state']
            with_words = own or (args.order == 'main' and Z['dec_mF'].shape[0] >= 3 + len(ref_pairs))
            H, dec, _, _, _ = measure(b, ref_pairs if with_words else [], True, kw)
            rec = {'id': b['id'], 'checks': {}}
            for ri, (rn, _) in enumerate(ROLES):
                if rn in H and np.isfinite(states[ri]).all():
                    rec['checks'][f'state_{rn}'] = cmp(H[rn][0][keep], states[ri])
            names = pid_ident['rows'] if own else ['A-U', 'A-S', 'U-S'] + ([p['id'] for p in ref_pairs] if with_words else [])
            for qi, nm in enumerate(names):
                if nm in dec and np.isfinite(Z['dec_mF'][qi]).all():
                    mF, mL, eF, eL = dec[nm]
                    c1, r1 = cmp(mF, Z['dec_mF'][qi])
                    c2, r2 = cmp(mL, Z['dec_mL'][qi])
                    re = float(max(np.max(np.abs(eF.cpu().numpy() / Z['dec_eF'][qi] - 1)), np.max(np.abs(eL.cpu().numpy() / Z['dec_eL'][qi] - 1))))
                    rec['checks'][f'dec_{nm}'] = (min(c1, c2), max(r1, r2, re))
            rec['ok'] = bool(rec['checks']) and all(c >= GATE_COS and r <= GATE_REL for c, r in rec['checks'].values())
            gate['bodies'].append(rec)
            gate['ok'] &= rec['ok']
            del H, dec
    elif args.gate_g:
        sub = args.gate_sub or ('G' if args.order == 'main' else 'N')
        gd = Path(args.gate_g) / sub
        for b in gate_bodies:
            H, _, _, _, _ = measure(b, [], True, gkw if gkw is not None else kw)
            rec = {'id': b['id'], 'checks': {}}
            with np.load(gd / f'{b["id"]}.npz') as z:
                for rn, role in ROLES:
                    if rn in H and f'onset_{role}' in z.files:
                        rec['checks'][f'onset_{rn}'] = cmp(H[rn][0], z[f'onset_{role}'])
            rec['ok'] = bool(rec['checks']) and all(c >= GATE_COS and r <= GATE_REL for c, r in rec['checks'].values())
            gate['bodies'].append(rec)
            gate['ok'] &= rec['ok']
            del H
    else:
        gate = {'none': args.no_gate_reason, 'ok': True}
    ident['layer_output_ranks'] = sorted(st['rank'])
    ident['gate'] = gate
    atomic(out / 'identity.json', ident)
    print('gate:', json.dumps(gate), flush=True)
    if not gate['ok']:
        sys.exit(f'GATE FAILED: {json.dumps(gate)}')

    # ---------------- measurement ----------------
    t0, done = time.time(), 0
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
        H, dec, tc, ncom, uneq = measure(b, pairs, args.roles, kw)
        d = next(iter(dec.values()))[0].shape[-1]
        R = len(rows)
        arrs = {'dec_mF': np.full((R, K, d), np.nan, VDT), 'dec_mL': np.full((R, K, d), np.nan, VDT),
                'dec_eF': np.full((R, K), np.nan, np.float32), 'dec_eL': np.full((R, K), np.nan, np.float32)}
        for qi, nm in enumerate(rows):
            if nm in dec:
                mF, mL, eF, eL = dec[nm]
                arrs['dec_mF'][qi], arrs['dec_mL'][qi] = mF.to(TDT).cpu().numpy(), mL.to(TDT).cpu().numpy()
                arrs['dec_eF'][qi], arrs['dec_eL'][qi] = eF.cpu().numpy(), eL.cpu().numpy()
        if args.roles:
            arrs['role_state'] = np.full((3, K, d), np.nan, VDT)
            for ri, (rn, _) in enumerate(ROLES):
                if rn in H:
                    arrs['role_state'][ri] = H[rn][0][keep].to(TDT).cpu().numpy()
        for kk, v in arrs.items():
            assert not np.isinf(v).any(), f'{kk} has inf in {b["id"]} (float16 overflow? rerun with --store-f32)'
        part = out / f'{b["id"]}.npz.part'
        with open(part, 'wb') as fh:
            np.savez(fh, **arrs)
        part.replace(out / f'{b["id"]}.npz')
        atomic(out / f'{b["id"]}.json', {'id': b['id'], 'split': b.get('split'), 'category': b.get('category'),
                                          'token_counts': tc, 'n_common': ncom, 'unequal_pairs': uneq,
                                          'seconds': round(time.time() - tb, 1)})
        done += 1
        del H, dec
        if done % 10 == 0 or done == 1:
            print(f'{done} bodies, {time.time() - t0:.0f}s, last {b["id"]} ({time.time() - tb:.1f}s), {free:.0f} GiB free', flush=True)
    for hd in handles:
        hd.remove()
    if sorted(st['rank']) != ident['layer_output_ranks']:
        ident['layer_output_ranks'] = sorted(st['rank'])
        atomic(out / 'identity.json', ident)
    n_all = sum(1 for b in bodies if (out / f'{b["id"]}.json').exists())
    print(f'finished: {n_all}/{len(bodies)} bodies on disk', flush=True)
    if n_all == len(bodies):
        (out / 'DONE').write_text(time.strftime('%F %T') + '\n')


if __name__ == '__main__':
    main()
