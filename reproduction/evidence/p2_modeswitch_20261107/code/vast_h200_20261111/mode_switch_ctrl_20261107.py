"""Follow-up to mode_switch_20261107.py (user 11-07: is the effect just each model's prediction of the end token?).
Same renderings and forwards, main order only, no gate. On the body positions t (predicting body token t+1), each next-token
distribution is split into control tokens (every added token of the tokenizer and every special id: chat markup, end tokens,
think and channel tags; plus logit rows beyond the tokenizer) and content tokens (the rest):
  p_ctrl      probability mass on control tokens, median over t (p_close: the template's next token after this body, as before)
  kl_content  KL(X || Y) between the content parts, each renormalised to 1, median over t
  kl_body     the full KL as in mode_switch_20261107.py (consistency check)
  lp_next     mean log-probability of the actual next body token
Per body one json in OUT/bodies; OUT/summary.json (held-out bodies, 15 families, 2,000 resamples).
usage: mode_switch_ctrl_20261107.py --model DIR --tmpl-kwargs JSON --bodies role_bodies.jsonl --out DIR [--lang en --bodies-en F]
       [--fixed-date D] [--max-memory-gib G] [--limit N]"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402
from crossmodel_suite import atomic, load  # noqa: E402
from persistence_readout_20261106 import natural_system_ok  # noqa: E402
from mode_switch_20261107 import PAIRS, REQ, renderings  # noqa: E402

NB = 2000
RK = ('p_ctrl', 'p_close', 'lp_next')
PK = ('kl_body', 'kl_content')


def summarize(out):
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'bodies').glob('*.json'))]
    held = [r for r in rows if r['split'] == 'confirmation']
    fams = sorted({r['category'] for r in held})
    fidx = np.array([fams.index(r['category']) for r in held])
    rng = np.random.default_rng(20261107)
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
    res = {'n_held_out': len(held), 'roles': {}, 'pairs': {}, 'ratios': {}}
    for rn in held[0]['roles']:
        res['roles'][rn] = {k: mean_ci(np.array([r['roles'][rn][k] for r in held])) for k in RK}
    for pn in held[0]['pairs']:
        res['pairs'][pn] = {k: mean_ci(np.array([r['pairs'][pn][k] for r in held])) for k in PK}
    if 'U-S' in res['pairs']:
        for k in PK:
            us = np.array([r['pairs']['U-S'][k] for r in held])
            for pn in ('A-U', 'A-S'):
                res['ratios'][f'{pn}/U-S_{k}'] = ratio_ci(np.array([r['pairs'][pn][k] for r in held]), us)
    atomic(out / 'summary.json', res)
    print('summary:', json.dumps({'p_ctrl': {rn: round(v['p_ctrl']['mean'], 4) for rn, v in res['roles'].items()},
                                  'kl': {pn: {k: round(v[k]['mean'], 4) for k in PK} for pn, v in res['pairs'].items()},
                                  'ratios': {k: [round(v['ratio'], 2)] + [round(x, 2) for x in v['ci95']] for k, v in res['ratios'].items()}},
                                 ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tmpl-kwargs', default='{}')
    ap.add_argument('--bodies', required=True)
    ap.add_argument('--lang', choices=('zh', 'en'), default='zh')
    ap.add_argument('--bodies-en', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fixed-date', default=None)
    ap.add_argument('--max-memory-gib', type=float, default=44)
    ap.add_argument('--cpu-offload-gib', type=float, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stop-file', default=None)
    args = ap.parse_args()
    out = Path(args.out)
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
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: this script does not fall back to the CPU')
    (out / 'bodies').mkdir(parents=True, exist_ok=True)
    largs = argparse.Namespace(model=args.model, max_memory_gib=args.max_memory_gib, cpu_offload_gib=args.cpu_offload_gib)
    torch, tok, model, cfg, placement, cls = load(largs)
    sys_ok = natural_system_ok(tok, kw, q, REQ[args.lang], bodies[0]['body'])
    dev0 = model.get_input_embeddings().weight.device
    ctrl_ids = sorted(set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}).keys()))
    me = Path(__file__).resolve()
    atomic(out / 'identity.json', {'experiment': 'P2_modeswitch_ctrl_20261107', 'script_sha256': hashlib.sha256(me.read_bytes()).hexdigest(),
                                   'model': args.model, 'tmpl_kwargs': json.loads(args.tmpl_kwargs), 'fixed_date': args.fixed_date,
                                   'lang': args.lang, 'n_control_ids': len(ctrl_ids), 'len_tokenizer': len(tok),
                                   'control_tokens_sample': [tok.convert_ids_to_tokens(i) for i in ctrl_ids[:40]]})
    mask = None
    t0, done = time.time(), 0
    for b in bodies:
        f = out / 'bodies' / f'{b["id"]}.json'
        if f.exists():
            continue
        if args.stop_file and Path(args.stop_file).exists():
            print(f'[stop] {args.stop_file} exists: stopping before {b["id"]}', flush=True)
            break
        R = renderings(tok, kw, b['body'], args.lang, sys_ok)['main']
        rec = {'id': b['id'], 'split': b.get('split'), 'category': b.get('category'), 'roles': {}, 'pairs': {}}
        lps = {}
        for rn, v in R.items():
            x = torch.tensor([v['ids']], dtype=torch.long, device=dev0)
            with torch.no_grad():
                o = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=0)
            lg = o.logits[0].float()
            if mask is None:
                mask = torch.zeros(lg.shape[-1], dtype=torch.bool, device=lg.device)
                mask[[i for i in ctrl_ids if i < lg.shape[-1]]] = True
                mask[len(tok):] = True
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
            del o, lg
        for X, Y in PAIRS:
            if X in lps and Y in lps:
                (lx, cx), (ly, cy) = lps[X], lps[Y]
                rec['pairs'][f'{X}-{Y}'] = {'kl_body': float((lx.exp() * (lx - ly)).sum(-1).median()),
                                            'kl_content': float((cx.exp() * (cx - cy)).sum(-1).median())}
        del lps
        atomic(f, rec)
        done += 1
        if done % 25 == 0 or done == 1:
            print(f'{done} bodies, {time.time() - t0:.0f}s: ' + json.dumps(rec['pairs']), flush=True)
    n_all = sum(1 for b in bodies if (out / 'bodies' / f'{b["id"]}.json').exists())
    print(f'finished: {n_all}/{len(bodies)}', flush=True)
    if n_all == len(bodies):
        summarize(out)
        (out / 'DONE').write_text(time.strftime('%F %T') + '\n')


if __name__ == '__main__':
    main()
