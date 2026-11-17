"""P2 mode switch, natural-order runs (11-08, evidence/p2_modeswitch_20261107/README_zh.md; user: finish the KL effect on every
model of the protocol first). The measurement is that of mode_switch_ctrl_20261107.py. On the body positions t (predicting body
token t+1), it computes:
  p_ctrl, p_close, lp_next              as in the ctrl script
  kl_body                               full KL(X || Y), median over t, as in mode_switch_20261107.py
  kl_content                            KL between the content parts (control tokens removed, each renormalised), median over t
It uses held-out bodies, 15 families and 2,000 resamples. The orders are:
  natural   [user: REQ][ROLE: body][user: Q]. Rendered by mode_switch_20261107.renderings()['natural'] (System only where the
            template keeps it in place). This is the natural-order control-token split, which the ctrl script (main order only)
            did not do.
  natural3  The paper's natural-order run for the templates that do not accept the main probe (crossmodel_natural3.py,
            83bff549), with its messages and its body-between-request-and-question check:
              A  [user: REQ][assistant: body][user: Q]
              U  [user: REQ][user: body][user: Q]     (the template's own rendering; Mistral merges it into one user turn)
              S  [user: REQ][system: body][user: Q]
            For Mistral checkpoints it uses the fix_mistral_regex tokenizer, as that run did.
usage: mode_switch_nat_20261108.py --order natural|natural3 --model DIR --tmpl-kwargs JSON --bodies F --out DIR
       [--lang en --bodies-en F] [--fixed-date D] [--max-memory-gib G] [--cpu-offload-gib C] [--limit N] [--stop-file F]
       [--dry-run]"""
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
from crossmodel_suite import Renderer, atomic, checked_tokenizer, common_body, load  # noqa: E402
from persistence_readout_20261106 import natural_system_ok  # noqa: E402
from mode_switch_20261107 import PAIRS, REQ, renderings  # noqa: E402
from mode_switch_ctrl_20261107 import summarize  # noqa: E402
import crossmodel_natural3 as n3  # noqa: E402

N3 = (('A', 'assistant'), ('U', 'user'), ('S', 'system'))


def render_order(order, tok, kw, body, lang, sys_ok):
    q = cs.Q_REPLY_EN if lang == 'en' else cs.Q_REPLY
    if order == 'natural':
        return renderings(tok, kw, body, lang, sys_ok)['natural']
    R = Renderer(tok, kw)
    r = {rn: n3.render(R, cond, REQ[lang], body, q) for rn, cond in N3}
    com = common_body(r)
    return {rn: {'ids': ids, 'span': span, 'common': com[rn], 'close': ids[span[-1] + 1]} for rn, (ids, span) in r.items()}


def mistral_tok(model_dir, cfg=None):
    if 'mistral' in str(getattr(cfg, 'model_type', '')).lower() or 'mistral' in model_dir.lower():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--order', choices=('natural', 'natural3'), required=True)
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
    ap.add_argument('--dry-run', action='store_true', help='render and check; no model, no GPU')
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

    if args.dry_run:
        from transformers import AutoConfig, AutoTokenizer
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        tok = mistral_tok(args.model, cfg) or checked_tokenizer(AutoTokenizer.from_pretrained(args.model, trust_remote_code=True), args.model)
        sys_ok = natural_system_ok(tok, kw, q, REQ[args.lang], bodies[0]['body']) if args.order == 'natural' else None
        n = []
        for b in bodies:
            R = render_order(args.order, tok, kw, b['body'], args.lang, sys_ok)
            n.append(len(next(iter(R.values()))['common']))
        R = render_order(args.order, tok, kw, bodies[0]['body'], args.lang, sys_ok)
        print(json.dumps({'order': args.order, 'bodies': len(bodies), 'natural_system_ok': sys_ok, 'roles': list(R),
                          'tokenizer': type(tok).__name__, 'common_body_tokens_min_max': [min(n), max(n)],
                          'close': {rn: tok.decode([v['close']]) for rn, v in R.items()}}, ensure_ascii=False, indent=1))
        print('DRY RUN OK')
        return

    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: this script does not fall back to the CPU')
    (out / 'bodies').mkdir(parents=True, exist_ok=True)
    largs = argparse.Namespace(model=args.model, max_memory_gib=args.max_memory_gib, cpu_offload_gib=args.cpu_offload_gib)
    torch, tok, model, cfg, placement, cls = load(largs)
    tok_note = None
    t2 = mistral_tok(args.model, cfg) if args.order == 'natural3' else None
    if t2 is not None:
        tok_note = {'fix_mistral_regex': True, 'replaced': type(tok).__name__}
        tok = t2
    sys_ok = natural_system_ok(tok, kw, q, REQ[args.lang], bodies[0]['body']) if args.order == 'natural' else None
    dev0 = model.get_input_embeddings().weight.device
    ctrl_ids = sorted(set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}).keys()))
    me = Path(__file__).resolve()
    atomic(out / 'identity.json', {'experiment': 'P2_modeswitch_nat_20261108', 'script_sha256': hashlib.sha256(me.read_bytes()).hexdigest(),
                                   'natural3_sha256': hashlib.sha256(Path(n3.__file__).read_bytes()).hexdigest(),
                                   'order': args.order, 'model': args.model, 'model_class': cls, 'tmpl_kwargs': json.loads(args.tmpl_kwargs),
                                   'fixed_date': args.fixed_date, 'lang': args.lang, 'natural_system_ok': sys_ok, 'tokenizer_note': tok_note,
                                   'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                                   'device_map': placement, 'cpu_offload_gib': args.cpu_offload_gib,
                                   'n_control_ids': len(ctrl_ids), 'len_tokenizer': len(tok),
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
        R = render_order(args.order, tok, kw, b['body'], args.lang, sys_ok)
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
