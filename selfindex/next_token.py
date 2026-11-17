"""Next-token prediction on the message under test: does the model predict the text of a message differently when the message is
its own? (Section "Next-Token Prediction" of the paper; Tables tab:self_kl and tab:self_kl_natural.)

No question is asked about the source. The message is rendered with the model's own chat template, followed by the reply request
(items/templates.json, layout next_token):
  main     [ROLE: body] [user: reply request]
  natural  [user: request] [ROLE: body] [user: reply request]   (System only where the template keeps it in place after a user turn)
  natural3 the natural order of the templates that do not accept the main probe (selfindex.natural --roles 3):
           [user: request] [assistant | user | system: body] [user: reply request], the body checked to lie between the request
           and the question; Mistral checkpoints with the corrected tokenizer (fix_mistral_regex), as in that run
ROLE is assistant (A), user (U) or system (S). One teacher-forced forward per rendering (one sequence, attention mask of ones,
use_cache=False); the logits are the model's own. On the body positions t shared by the renderings (render.common_body; all but
the last body token, each predicting body token t+1):
--measure full (orders main and natural together):
  p_close       probability of the token the template writes right after this body (the end of this message), median over t;
                p_close_last at the last body token
  ent, top1     entropy and top-1 probability of the next-token distribution, median over t
  lp_next, acc  mean log-probability and top-1 accuracy of the actual next body token
  per pair of roles (A-U, A-S, U-S): KL(X || Y), median over t (kl_body; kl_body_first, kl_body_last over the first and last third
  of the positions) and at the last position (kl_reply)
  --gate-g DIR (main order, models whose layer outputs have rank 3): the last-token state at every layer against onset_<role> of
  the main probe's G/<body>.npz in DIR (min cosine >= 0.999, max relative difference <= 0.03)
--measure split (one order, --order main | natural | natural3): every next-token distribution split into control tokens (every
  special id and added token of the tokenizer: chat markup, end tokens, think and channel tags; plus logit rows beyond the
  tokenizer) and content tokens (the rest):
  p_ctrl        probability mass on control tokens, median over t (p_close and lp_next as above)
  kl_content    KL(X || Y) between the content parts, each renormalised to 1, median over t
  kl_body       the full KL, as in --measure full
Summary (summary.json): the held-out (confirmation) bodies, 15 wording families resampled 2,000 times (seed 20261107): means with
95% intervals, and the ratios KL(A||U)/KL(U||S) and KL(A||S)/KL(U||S) (sums over bodies) with intervals; the paper reports the
content ratios of --measure split. The summary of --measure full also keeps the flag mode_switch of the first runs (both lower
bounds above 10), which the paper does not use.
Outputs in --out: identity.json, bodies/<body>.json (one per body; a run resumes from the bodies on disk), summary.json and DONE
when every body is on disk. --dry-run renders every body with the tokenizer only and prints the token counts and the closing tokens.
usage: python -m selfindex.next_token --model DIR --tmpl-kwargs JSON --out DIR [--measure full|split] [--order main|natural|natural3]
         [--lang zh|en] [--fixed-date D] [--gate-g DIR] [--limit N] [--stop-file F] [--max-memory-gib G] [--cpu-offload-gib C]
       python -m selfindex.next_token --dry-run --model DIR --tmpl-kwargs JSON [--measure M] [--order O] [--lang L]
       python -m selfindex.next_token --summarize-only --out DIR --measure full|split
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from . import __version__
from . import items as I
from . import natural as N
from .chat_order import msgs_for, natural_system_ok
from .io import atomic
from .load import checked_tokenizer, find_layers, load_model
from .render import Renderer, common_body, template_kwargs

ROLES = (('A', 'assistant'), ('U', 'user'), ('S', 'system'))
PAIRS = (('A', 'U'), ('A', 'S'), ('U', 'S'))
ROLE_KEYS = ('p_close', 'p_close_last', 'ent', 'top1', 'lp_next', 'acc')  # --measure full
PAIR_KEYS = ('kl_body', 'kl_body_first', 'kl_body_last', 'kl_reply')
SPLIT_ROLE_KEYS = ('p_ctrl', 'p_close', 'lp_next')                        # --measure split
SPLIT_PAIR_KEYS = ('kl_body', 'kl_content')
NB = 2000
SEED = 20261107
EXPERIMENT = {'full': 'P2_modeswitch_20261107', ('split', 'main'): 'P2_modeswitch_ctrl_20261107',
              ('split', 'natural'): 'P2_modeswitch_nat_20261108', ('split', 'natural3'): 'P2_modeswitch_nat_20261108'}


def entry(ids, span, com):
    return {'ids': ids, 'span': span, 'common': com, 'close': ids[span[-1] + 1]}


def renderings(tok, kw, body, lang, sys_ok, orders=('main', 'natural')):
    """{order: {role: {ids, span, common, close}}} for the main and the natural order."""
    q, req = I.reply_request(lang), I.natural_order_request(lang)
    out = {}
    for order in orders:
        r = {}
        for rn, role in ROLES:
            if order == 'natural' and role == 'system' and not sys_ok:
                continue
            r[rn] = Renderer(tok, kw).render(msgs_for(order, role, body, q, req), body)
        com = common_body(r)
        out[order] = {rn: entry(ids, span, com[rn]) for rn, (ids, span) in r.items()}
    return out


def render_order(order, tok, kw, body, lang, sys_ok):
    """{role: {ids, span, common, close}} of one order (--measure split)."""
    if order != 'natural3':
        return renderings(tok, kw, body, lang, sys_ok, (order,))[order]
    q, req = I.reply_request(lang), I.natural_order_request(lang)
    sep = I.templates()['natural_order']['merged_separator']
    R = Renderer(tok, kw)
    r = {rn: N.render(R, cond, req, body, q, sep) for rn, cond in ROLES}
    com = common_body(r)
    return {rn: entry(ids, span, com[rn]) for rn, (ids, span) in r.items()}


def control_ids(tok):
    return sorted(set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}).keys()))


def control_mask(torch, tok, ctrl, n_rows, device):
    mask = torch.zeros(n_rows, dtype=torch.bool, device=device)
    mask[[i for i in ctrl if i < n_rows]] = True
    mask[len(tok):] = True
    return mask


# ---------------------------------------------------------------------------------------------------------------------------
# summaries

def _family_tools(held, seed=SEED):
    fams = sorted({r['category'] for r in held})
    fidx = np.array([fams.index(r['category']) for r in held])
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(fams), len(fams)) for _ in range(NB)]

    def fsum(v):
        return np.array([v[fidx == f].sum() for f in range(len(fams))]), np.array([(fidx == f).sum() for f in range(len(fams))])

    def mean_ci(v):
        fm, fn = fsum(v)
        bs = [fm[d].sum() / fn[d].sum() for d in draws]
        return {'mean': float(v.mean()), 'ci95': [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}

    def ratio_ci(a, b):
        fa, _ = fsum(a)
        fb, _ = fsum(b)
        bs = [fa[d].sum() / fb[d].sum() for d in draws]
        return {'ratio': float(a.sum() / b.sum()), 'ci95': [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
    return fams, mean_ci, ratio_ci


def summarize_full(out):
    out = Path(out)
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'bodies').glob('*.json'))]
    held = [r for r in rows if r['split'] == 'confirmation']
    disc = [r for r in rows if r['split'] != 'confirmation']
    fams, mean_ci, ratio_ci = _family_tools(held)

    def stat(get, rs=held):
        return np.array([get(r) for r in rs], float)
    res = {'n_held_out': len(held), 'n_discovery': len(disc), 'families': len(fams), 'orders': {}}
    for order in ('main', 'natural'):
        roles = [rn for rn, _ in ROLES if rn in held[0][order]['roles']]
        o = {'roles_present': roles, 'roles': {}, 'pairs': {}, 'ratios': {}}
        for rn in roles:
            o['roles'][rn] = {k: {**mean_ci(stat(lambda r: r[order]['roles'][rn][k])),
                                  'discovery': float(np.mean([r[order]['roles'][rn][k] for r in disc])) if disc else None}
                              for k in ROLE_KEYS}
        for pn in held[0][order]['pairs']:
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
    return res


def summarize_split(out):
    out = Path(out)
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'bodies').glob('*.json'))]
    held = [r for r in rows if r['split'] == 'confirmation']
    _, mean_ci, ratio_ci = _family_tools(held)
    res = {'n_held_out': len(held), 'roles': {}, 'pairs': {}, 'ratios': {}}
    for rn in held[0]['roles']:
        res['roles'][rn] = {k: mean_ci(np.array([r['roles'][rn][k] for r in held])) for k in SPLIT_ROLE_KEYS}
    for pn in held[0]['pairs']:
        res['pairs'][pn] = {k: mean_ci(np.array([r['pairs'][pn][k] for r in held])) for k in SPLIT_PAIR_KEYS}
    if 'U-S' in res['pairs']:
        for k in SPLIT_PAIR_KEYS:
            us = np.array([r['pairs']['U-S'][k] for r in held])
            for pn in ('A-U', 'A-S'):
                res['ratios'][f'{pn}/U-S_{k}'] = ratio_ci(np.array([r['pairs'][pn][k] for r in held]), us)
    atomic(out / 'summary.json', res)
    print('summary:', json.dumps({'p_ctrl': {rn: round(v['p_ctrl']['mean'], 4) for rn, v in res['roles'].items()},
                                  'kl': {pn: {k: round(v[k]['mean'], 4) for k in SPLIT_PAIR_KEYS} for pn, v in res['pairs'].items()},
                                  'ratios': {k: [round(v['ratio'], 2)] + [round(x, 2) for x in v['ci95']] for k, v in res['ratios'].items()}},
                                 ensure_ascii=False), flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------------------
# the measurement

def bodies_of(a):
    bodies = list(I.bodies(a.lang))
    return bodies[:a.limit] if a.limit else bodies


def bodies_hashes(lang):
    d = I.items_dir() / 'bodies'
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    return {'bodies_sha256': h(d / 'role_bodies_zh.jsonl'), 'bodies_en_sha256': h(d / 'bodies_en_150.json') if lang == 'en' else None}


def dry_run(a):
    from transformers import AutoConfig, AutoTokenizer
    kw = template_kwargs(a.tmpl_kwargs, a.fixed_date)
    tok = checked_tokenizer(AutoTokenizer.from_pretrained(a.model, trust_remote_code=True), a.model)
    if a.measure == 'split' and a.order == 'natural3':
        tok, _ = N.mistral_tokenizer(tok, AutoConfig.from_pretrained(a.model, trust_remote_code=True), a.model)
    bodies = bodies_of(a)
    q, req = I.reply_request(a.lang), I.natural_order_request(a.lang)
    sys_ok = natural_system_ok(tok, kw, q, req, bodies[0]['body'])
    orders = ('main', 'natural') if a.measure == 'full' else (a.order,)
    n, pref = [], {}
    for b in bodies:
        R = {o: render_order(o, tok, kw, b['body'], a.lang, sys_ok) for o in orders}
        n.append(len(next(iter(R[orders[0]].values()))['common']))
    for o in orders:
        for rn, v in render_order(o, tok, kw, bodies[0]['body'], a.lang, sys_ok).items():
            pref[f'{o}_{rn}'] = {'close': tok.decode([v['close']]), 'tail': tok.decode(v['ids'][v['span'][-1] + 1:])}
    print(json.dumps({'bodies': len(bodies), 'orders': list(orders), 'natural_system_ok': sys_ok, 'tokenizer': type(tok).__name__,
                      'common_body_tokens_min_max': [min(n), max(n)], 'n_control_ids': len(control_ids(tok)), 'body0': pref},
                     ensure_ascii=False, indent=1))
    print('DRY RUN OK')


def run(a, loaded=None):
    out = Path(a.out)
    (out / 'bodies').mkdir(parents=True, exist_ok=True)
    kw = template_kwargs(a.tmpl_kwargs, a.fixed_date)
    L_ = loaded or load_model(a.model, max_memory_gib=a.max_memory_gib, cpu_offload_gib=a.cpu_offload_gib)
    torch, tok, model = L_.torch, L_.tok, L_.model
    full = a.measure == 'full'
    tok_note = None
    if not full and a.order == 'natural3':
        tok, tok_note = N.mistral_tokenizer(tok, L_.cfg, a.model)
    _, layers = find_layers(model)
    L = len(layers)
    bodies = bodies_of(a)
    q, req = I.reply_request(a.lang), I.natural_order_request(a.lang)
    sys_ok = natural_system_ok(tok, kw, q, req, bodies[0]['body']) if full or a.order != 'natural3' else None
    ctrl = control_ids(tok)
    ident = {'experiment': EXPERIMENT['full' if full else ('split', a.order)], 'selfindex_version': __version__, 'measure': a.measure,
             'order': 'main+natural' if full else a.order, 'model': a.model, 'model_class': L_.model_class, 'n_layers': L,
             'tmpl_kwargs': json.loads(a.tmpl_kwargs), 'fixed_date': a.fixed_date, 'lang': a.lang, 'reply_question': q, 'request': req,
             'natural_system_ok': sys_ok, 'tokenizer_note': tok_note, 'device_map': L_.placement, 'torch': torch.__version__,
             'transformers': __import__('transformers').__version__, 'cpu_offload_gib': a.cpu_offload_gib, 'n_bodies': len(bodies),
             **bodies_hashes(a.lang)}
    if full:
        ident['gate_g'] = a.gate_g
    else:
        ident.update(n_control_ids=len(ctrl), len_tokenizer=len(tok), control_tokens_sample=[tok.convert_ids_to_tokens(i) for i in ctrl[:40]])
    atomic(out / 'identity.json', ident)
    dev0 = model.get_input_embeddings().weight.device
    gate_dir = Path(a.gate_g) if (full and a.gate_g) else None
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

    def forward(ids, capture=False):
        x = torch.tensor([ids], dtype=torch.long, device=dev0)
        cap.clear()
        st['on'] = capture
        with torch.no_grad():
            o = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=0)
        st['on'] = False
        return o.logits[0].float()

    mask = None
    t0, done = time.time(), 0
    try:
        for b in bodies:
            f = out / 'bodies' / f'{b["id"]}.json'
            if f.exists():
                continue
            if a.stop_file and Path(a.stop_file).exists():
                print(f'[stop] {a.stop_file} exists: stopping before {b["id"]} (rerun resumes)', flush=True)
                break
            tb = time.time()
            if full:
                R = renderings(tok, kw, b['body'], a.lang, sys_ok)
                rec = {'id': b['id'], 'split': b.get('split'), 'category': b.get('category'), 'gate': None}
                gate = {}
                for order, rr in R.items():
                    lps, ro = {}, {}
                    for rn, v in rr.items():
                        lg = forward(v['ids'], capture=gate_dir is not None and order == 'main')
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
                        del lg
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
            else:
                R = render_order(a.order, tok, kw, b['body'], a.lang, sys_ok)
                rec = {'id': b['id'], 'split': b.get('split'), 'category': b.get('category'), 'roles': {}, 'pairs': {}}
                lps = {}
                for rn, v in R.items():
                    lg = forward(v['ids'])
                    if mask is None:
                        mask = control_mask(torch, tok, ctrl, lg.shape[-1], lg.device)
                    c = v['common']
                    pos = torch.tensor(c[:-1], device=lg.device)
                    nxt = torch.tensor([v['ids'][i + 1] for i in c[:-1]], device=lg.device)
                    lp = torch.log_softmax(lg[pos], -1)
                    p = lp.exp()
                    lc = lp[:, ~mask]
                    lc = lc - torch.logsumexp(lc, -1, keepdim=True)
                    rec['roles'][rn] = {'p_ctrl': float(p[:, mask].sum(-1).median()), 'p_close': float(p[:, v['close']].median()),
                                        'lp_next': float(lp[torch.arange(len(nxt), device=lg.device), nxt].mean())}
                    lps[rn] = (lp, lc)
                    del lg
                for X, Y in PAIRS:
                    if X in lps and Y in lps:
                        (lx, cx), (ly, cy) = lps[X], lps[Y]
                        rec['pairs'][f'{X}-{Y}'] = {'kl_body': float((lx.exp() * (lx - ly)).sum(-1).median()),
                                                    'kl_content': float((cx.exp() * (cx - cy)).sum(-1).median())}
                del lps
            atomic(f, rec)
            done += 1
            if full and (done % 10 == 0 or done == 1):
                m = rec['main']
                print(f'{done} bodies, {time.time() - t0:.0f}s, last {b["id"]}: p_close ' +
                      ' '.join(f'{rn} {v["p_close"]:.3f}' for rn, v in m['roles'].items()) + ' | kl_body ' +
                      ' '.join(f'{pn} {v["kl_body"]:.4f}' for pn, v in m['pairs'].items()) +
                      (f' | gate {rec["gate"]}' if rec['gate'] else ''), flush=True)
            elif not full and (done % 25 == 0 or done == 1):
                print(f'{done} bodies, {time.time() - t0:.0f}s: ' + json.dumps(rec['pairs']), flush=True)
    finally:
        for hd in handles:
            hd.remove()
    n_all = sum(1 for b in bodies if (out / 'bodies' / f'{b["id"]}.json').exists())
    print(f'finished: {n_all}/{len(bodies)} bodies on disk', flush=True)
    if n_all == len(bodies):
        (summarize_full if full else summarize_split)(out)
        (out / 'DONE').write_text(time.strftime('%F %T') + '\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model')
    p.add_argument('--tmpl-kwargs', default='{}', help='chat-template keyword arguments of the rendering (JSON)')
    p.add_argument('--out')
    p.add_argument('--measure', choices=('full', 'split'), default='split',
                   help='full: orders main and natural, full distributions; split: one order, control and content tokens apart')
    p.add_argument('--order', choices=('main', 'natural', 'natural3'), default='main', help='--measure split: the order')
    p.add_argument('--lang', choices=('zh', 'en'), default='zh')
    p.add_argument('--fixed-date', default=None, help='YYYY-MM-DD: templates that write the current date render this date')
    p.add_argument('--gate-g', default=None, help='--measure full: a main-probe run folder whose G/<body>.npz the states are checked against')
    p.add_argument('--limit', type=int, default=0, help='only the first N bodies')
    p.add_argument('--stop-file', default=None, help='stop before the next body if this file exists (a rerun resumes)')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    p.add_argument('--cpu-offload-gib', type=float, default=0, help='allow weights beyond the GPU budget to stay in CPU memory')
    p.add_argument('--dry-run', action='store_true', help='render and check with the tokenizer only; no model, no GPU')
    p.add_argument('--summarize-only', action='store_true', help='rewrite summary.json from the bodies in --out')
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.measure == 'full' and a.order != 'main':
        p.error('--measure full measures the main and the natural order together; --order is for --measure split')
    if a.summarize_only:
        if not a.out:
            p.error('--summarize-only needs --out')
        (summarize_full if a.measure == 'full' else summarize_split)(Path(a.out))
        return
    if not a.model:
        p.error('--model is required')
    if a.dry_run:
        dry_run(a)
        return
    if not a.out:
        p.error('--out is required')
    try:
        run(a)
    except Exception as exc:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        atomic(Path(a.out) / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
