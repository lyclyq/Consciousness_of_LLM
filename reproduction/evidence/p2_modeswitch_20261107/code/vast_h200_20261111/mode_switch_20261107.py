"""P2 mode-switch test (11-07, evidence/p2_modeswitch_20261107/PREREG_zh.md).

On the body tokens of a message, does the model predict the text as its own generation only when the message is its own? For one
model setting, every body (150, both splits) and both orders, the message is rendered with the model's own chat template:
  main     [ROLE: body][user: Q]
  natural  [user: REQUEST][ROLE: body][user: Q]   (System only where the template keeps it in place)
One teacher-forced forward per rendering (one sequence, attention mask of ones, as phase G of crossmodel_suite.py); the logits
are the model's own. On the body positions t (predicting body token t+1; the last body token is excluded):
  p_close  probability of the token the template writes right after this body (the end of this message), median over t
  ent, top1  entropy and top-1 probability of the next-token distribution, median over t
  lp_next, acc  mean log-probability and top-1 accuracy of the actual next body token
Per pair of roles (A-U, A-S, U-S): KL(X || Y) on the body positions (median over t) and at the reply position (last token).
Gate (main order, models whose layer outputs have rank 3): last-token states at every layer against phase G onset_<role>.
Per body one json in OUT/bodies (resumable); OUT/summary.json at the end (held-out bodies, 15 families, 2,000 resamples).
usage: mode_switch_20261107.py --model DIR --tmpl-kwargs JSON --bodies role_bodies.jsonl --out DIR [--lang en --bodies-en F]
       [--gate-g DIR] [--fixed-date D] [--max-memory-gib G] [--cpu-offload-gib G] [--limit N] [--dry-run] [--summary-only]"""
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
from crossmodel_suite import Renderer, atomic, checked_tokenizer, common_body, find_layers, load  # noqa: E402
from persistence_readout_20261106 import msgs_for, natural_system_ok  # noqa: E402

ROLES = (('A', 'assistant'), ('U', 'user'), ('S', 'system'))
PAIRS = (('A', 'U'), ('A', 'S'), ('U', 'S'))
REQ = {'zh': '请写一段简短的描述。', 'en': 'Please write a short description.'}
ROLE_KEYS = ('p_close', 'p_close_last', 'ent', 'top1', 'lp_next', 'acc')
PAIR_KEYS = ('kl_body', 'kl_body_first', 'kl_body_last', 'kl_reply')
NB = 2000


def renderings(tok, kw, body, lang, sys_ok):
    q = cs.Q_REPLY_EN if lang == 'en' else cs.Q_REPLY
    out = {}
    for order in ('main', 'natural'):
        r = {}
        for rn, role in ROLES:
            if order == 'natural' and role == 'system' and not sys_ok:
                continue
            r[rn] = Renderer(tok, kw).render(msgs_for(order, role, body, q, REQ[lang]), body)
        com = common_body(r)
        out[order] = {rn: {'ids': ids, 'span': span, 'common': com[rn], 'close': ids[span[-1] + 1]} for rn, (ids, span) in r.items()}
    return out


def summarize(out):
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'bodies').glob('*.json'))]
    held = [r for r in rows if r['split'] == 'confirmation']
    disc = [r for r in rows if r['split'] != 'confirmation']
    fams = sorted({r['category'] for r in held})
    rng = np.random.default_rng(20261107)
    draws = [rng.integers(0, len(fams), len(fams)) for _ in range(NB)]
    fidx = np.array([fams.index(r['category']) for r in held])

    def stat(get, rs=held):
        v = np.array([get(r) for r in rs], float)
        return v

    def mean_ci(v):
        fm = np.array([v[fidx == f].sum() for f in range(len(fams))])
        fn = np.array([(fidx == f).sum() for f in range(len(fams))])
        bs = [fm[d].sum() / fn[d].sum() for d in draws]
        return {'mean': float(v.mean()), 'ci95': [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}

    def ratio_ci(a, b):
        fa = np.array([a[fidx == f].sum() for f in range(len(fams))])
        fb = np.array([b[fidx == f].sum() for f in range(len(fams))])
        bs = [fa[d].sum() / fb[d].sum() for d in draws]
        return {'ratio': float(a.sum() / b.sum()), 'ci95': [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
    res = {'n_held_out': len(held), 'n_discovery': len(disc), 'families': len(fams), 'orders': {}}
    for order in ('main', 'natural'):
        roles = [rn for rn, _ in ROLES if rn in held[0][order]['roles']]
        o = {'roles_present': roles, 'roles': {}, 'pairs': {}, 'ratios': {}}
        for rn in roles:
            o['roles'][rn] = {k: {**mean_ci(stat(lambda r: r[order]['roles'][rn][k])),
                                  'discovery': float(np.mean([r[order]['roles'][rn][k] for r in disc])) if disc else None}
                              for k in ROLE_KEYS}
        for pn, d in held[0][order]['pairs'].items():
            o['pairs'][pn] = {k: {**mean_ci(stat(lambda r: r[order]['pairs'][pn][k])),
                                  'discovery': float(np.mean([r[order]['pairs'][pn][k] for r in disc])) if disc else None}
                              for k in PAIR_KEYS}
        if 'U-S' in o['pairs']:
            for k in ('kl_body', 'kl_reply'):
                us = stat(lambda r: r[order]['pairs']['U-S'][k])
                for pn in ('A-U', 'A-S'):
                    o['ratios'][f'{pn}/U-S_{k}'] = ratio_ci(stat(lambda r: r[order]['pairs'][pn][k]), us)
            lo = [o['ratios'][f'{pn}/U-S_kl_body']['ci95'][0] for pn in ('A-U', 'A-S')]
            o['mode_switch'] = bool(min(lo) > 10)
        res['orders'][order] = o
    res['gate'] = {'checked': sum(1 for r in rows if r.get('gate')), 'all_ok': all(r['gate']['ok'] for r in rows if r.get('gate'))}
    atomic(out / 'summary.json', res)
    m = res['orders']['main']
    print('summary:', json.dumps({'mode_switch_main': m.get('mode_switch'), 'ratios_main': {k: round(v['ratio'], 2) for k, v in m['ratios'].items()},
                                  'p_close_main': {rn: round(v['p_close']['mean'], 4) for rn, v in m['roles'].items()},
                                  'kl_body_main': {pn: round(v['kl_body']['mean'], 4) for pn, v in m['pairs'].items()},
                                  'gate': res['gate']}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tmpl-kwargs', default='{}')
    ap.add_argument('--bodies', required=True)
    ap.add_argument('--lang', choices=('zh', 'en'), default='zh')
    ap.add_argument('--bodies-en', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gate-g', default=None)
    ap.add_argument('--fixed-date', default=None)
    ap.add_argument('--max-memory-gib', type=float, default=44)
    ap.add_argument('--cpu-offload-gib', type=float, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--min-free-gib', type=float, default=4.0)
    ap.add_argument('--stop-file', default=None)
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('--dry-run', action='store_true', help='render and check; no model, no GPU')
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    if args.summary_only:
        summarize(out)
        return
    kw = json.loads(args.tmpl_kwargs)
    if args.fixed_date:
        import datetime as _dt
        _day = _dt.date.fromisoformat(args.fixed_date)
        kw['strftime_now'] = lambda fmt: _dt.datetime(_day.year, _day.month, _day.day).strftime(fmt)
    bodies = [json.loads(l) for l in open(args.bodies, encoding='utf-8') if l.strip()]
    if args.lang == 'en':
        en = json.loads(Path(args.bodies_en).read_text(encoding='utf-8'))
        bodies = [{**b, 'body': en[b['id']]} for b in bodies]
    if args.limit:
        bodies = bodies[:args.limit]
    q = cs.Q_REPLY_EN if args.lang == 'en' else cs.Q_REPLY

    if args.dry_run:
        from transformers import AutoTokenizer
        tdir = args.tokenizer or args.model
        tok = checked_tokenizer(AutoTokenizer.from_pretrained(tdir, trust_remote_code=True), tdir)
        sys_ok = natural_system_ok(tok, kw, q, REQ[args.lang], bodies[0]['body'])
        ntok, pref = [], {}
        for b in bodies:
            R = renderings(tok, kw, b['body'], args.lang, sys_ok)
            ntok.append(len(R['main']['A']['common']))
        R = renderings(tok, kw, bodies[0]['body'], args.lang, sys_ok)
        for order, rr in R.items():
            for rn, v in rr.items():
                pref[f'{order}_{rn}'] = {'close': tok.decode([v['close']]),
                                         'prefix_to_close_sha': hashlib.sha256(json.dumps(v['ids'][:v['span'][-1] + 2]).encode()).hexdigest()[:12],
                                         'tail': tok.decode(v['ids'][v['span'][-1] + 1:])}
        print(json.dumps({'bodies': len(bodies), 'natural_system_ok': sys_ok, 'common_body_tokens_min_max': [min(ntok), max(ntok)],
                          'body0': pref}, ensure_ascii=False, indent=1))
        print('DRY RUN OK')
        return

    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: this script does not fall back to the CPU')
    (out / 'bodies').mkdir(parents=True, exist_ok=True)
    largs = argparse.Namespace(model=args.model, max_memory_gib=args.max_memory_gib, cpu_offload_gib=args.cpu_offload_gib)
    torch, tok, model, cfg, placement, cls = load(largs)
    lname, layers = find_layers(model)
    L = len(layers)
    sys_ok = natural_system_ok(tok, kw, q, REQ[args.lang], bodies[0]['body'])
    me = Path(__file__).resolve()
    ident = {'experiment': 'P2_modeswitch_20261107', 'script_sha256': hashlib.sha256(me.read_bytes()).hexdigest(),
             'suite_sha256': hashlib.sha256((me.parent / 'crossmodel_suite.py').read_bytes()).hexdigest(),
             'model': args.model, 'model_class': cls, 'n_layers': L, 'tmpl_kwargs': json.loads(args.tmpl_kwargs),
             'fixed_date': args.fixed_date, 'lang': args.lang, 'reply_question': q, 'request': REQ[args.lang],
             'natural_system_ok': sys_ok, 'device_map': placement, 'torch': torch.__version__,
             'transformers': __import__('transformers').__version__, 'host': socket.gethostname(), 'gate_g': args.gate_g,
             'cpu_offload_gib': args.cpu_offload_gib, 'n_bodies': len(bodies),
             'bodies_sha256': hashlib.sha256(Path(args.bodies).read_bytes()).hexdigest(),
             'bodies_en_sha256': hashlib.sha256(Path(args.bodies_en).read_bytes()).hexdigest() if args.bodies_en else None}
    atomic(out / 'identity.json', ident)
    dev0 = model.get_input_embeddings().weight.device
    gate_dir = Path(args.gate_g) if args.gate_g else None
    cap, st = {}, {'on': False, 'rank_ok': True}

    def hook(i):
        def h(mod, inp, outp):
            if not st['on']:
                return
            x = outp[0] if isinstance(outp, tuple) else outp
            if x.dim() != 3:
                st['rank_ok'] = False
                return
            cap[i] = x[0, -1].detach()
        return h
    handles = [layers[i].register_forward_hook(hook(i)) for i in range(L)] if gate_dir else []

    def med(x):
        return float(x.median())
    t0, done = time.time(), 0
    for b in bodies:
        f = out / 'bodies' / f'{b["id"]}.json'
        if f.exists():
            continue
        if args.stop_file and Path(args.stop_file).exists():
            print(f'[stop] {args.stop_file} exists: stopping before {b["id"]} (rerun resumes)', flush=True)
            break
        free = shutil.disk_usage(out).free / 2 ** 30
        if free < args.min_free_gib:
            print(f'[stop] {free:.1f} GiB free < {args.min_free_gib}: stopping before {b["id"]} (rerun resumes)', flush=True)
            break
        tb = time.time()
        R = renderings(tok, kw, b['body'], args.lang, sys_ok)
        rec = {'id': b['id'], 'split': b.get('split'), 'category': b.get('category'), 'gate': None}
        gate = {}
        for order, rr in R.items():
            lps, ro = {}, {}
            for rn, v in rr.items():
                x = torch.tensor([v['ids']], dtype=torch.long, device=dev0)
                cap.clear()
                st['on'] = gate_dir is not None and order == 'main'
                with torch.no_grad():
                    o = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=0)
                st['on'] = False
                lg = o.logits[0].float()
                c = v['common']
                pos = torch.tensor(c[:-1], device=lg.device)
                nxt = torch.tensor([v['ids'][i + 1] for i in c[:-1]], device=lg.device)
                lpb = torch.log_softmax(lg[pos], -1)
                pb = lpb.exp()
                ar = torch.arange(len(c) - 1, device=lg.device)
                ro[rn] = {'p_close': med(pb[:, v['close']]),
                          'p_close_last': float(torch.softmax(lg[c[-1]], -1)[v['close']]),
                          'ent': med(-(pb * lpb).sum(-1)), 'top1': med(pb.max(-1).values),
                          'lp_next': float(lpb[ar, nxt].mean()), 'acc': float((pb.argmax(-1) == nxt).float().mean()),
                          'n_body_tokens': len(c), 'close_token': v['close']}
                lps[rn] = (lpb, torch.log_softmax(lg[-1], -1))
                if gate_dir is not None and order == 'main' and st['rank_ok'] and len(cap) == L:
                    g = gate_dir / 'G' / f'{b["id"]}.npz'
                    role = dict(ROLES)[rn]
                    if g.exists():
                        with np.load(g) as z:
                            if f'onset_{role}' in z.files:
                                ref = torch.tensor(z[f'onset_{role}'].astype(np.float32), device=dev0).reshape(L, -1)
                                ours = torch.stack([cap[i].float().to(dev0) for i in range(L)])
                                cosv = torch.nn.functional.cosine_similarity(ours, ref, dim=-1)
                                rel = (ours - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)
                                gate[rn] = {'min_cos': float(cosv.min()), 'max_rel': float(rel.max())}
                del o, lg
            pr = {}
            for X, Y in PAIRS:
                if X in lps and Y in lps:
                    lx, ly = lps[X][0], lps[Y][0]
                    kb = (lx.exp() * (lx - ly)).sum(-1)
                    a3 = max(1, len(kb) // 3)
                    rx, ry = lps[X][1], lps[Y][1]
                    pr[f'{X}-{Y}'] = {'kl_body': med(kb), 'kl_body_first': med(kb[:a3]), 'kl_body_last': med(kb[-a3:]),
                                      'kl_reply': float((rx.exp() * (rx - ry)).sum())}
            rec[order] = {'roles': ro, 'pairs': pr}
            del lps
        if gate:
            gate['ok'] = all(v['min_cos'] >= 0.999 and v['max_rel'] <= 0.03 for v in gate.values())
            rec['gate'] = gate
        rec['seconds'] = round(time.time() - tb, 1)
        atomic(f, rec)
        done += 1
        if done % 10 == 0 or done == 1:
            m = rec['main']
            print(f'{done} bodies, {time.time() - t0:.0f}s, last {b["id"]}: p_close ' +
                  ' '.join(f'{rn} {v["p_close"]:.3f}' for rn, v in m['roles'].items()) + ' | kl_body ' +
                  ' '.join(f'{pn} {v["kl_body"]:.4f}' for pn, v in m['pairs'].items()) + (f' | gate {gate}' if gate else ''), flush=True)
    for hd in handles:
        hd.remove()
    n_all = sum(1 for b in bodies if (out / 'bodies' / f'{b["id"]}.json').exists())
    print(f'finished: {n_all}/{len(bodies)} bodies on disk', flush=True)
    if n_all == len(bodies):
        summarize(out)
        (out / 'DONE').write_text(time.strftime('%F %T') + '\n')


if __name__ == '__main__':
    main()
