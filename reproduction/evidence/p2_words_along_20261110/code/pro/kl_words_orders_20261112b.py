"""P2 ordinary words in next-token prediction, both orders in one load (11-11, user: "按照自然索引的附表规格做", "一次就把没跑的
模型的两个一次跑完", "不重叠"). Exploratory and internal, like the rest of this archive: not for the release or the data bundle.
The measurement of kl_words_20261111.py (renderings, one forward per rendering, content mask, KL, pairs, check, summary), run for
each order in --orders with one model load:
  main     roles mode_switch_20261107.renderings()['main'], words words_along_20261110.plan(order 'main'); check against the
           paper's main ctrl run (--check-ctrl, pairs[pn][kl_body|kl_content]), as kl_words_20261111.py
  natural  roles renderings()['natural'] ([user: REQ][ROLE: body][user: Q]; S left out where natural_system_ok is false), words
           plan(order 'natural', request REQ) = the words-along natural runs; check against the paper's natural KL of this setting
           (--check-natural): kind 'nat' = mode_switch_nat_20261108.py records (pairs[pn][kl_body|kl_content]), kind 'full' =
           mode_switch_20261107.py records (natural.pairs[pn].kl_body, full vocabulary only)
  natural3 (11-12, the templates that accept only the natural order: Ministral-3-14B Instruct and Reasoning) roles as the paper's
           natural3 KL run (mode_switch_nat_20261108.render_order('natural3'), crossmodel_natural3.render: A [user: REQ][assistant:
           body][user: Q], U the template's own merge of [user: REQ][user: body][user: Q], S [user: REQ][system: body][user: Q]),
           with the fix_mistral_regex tokenizer of that run; words plan(order 'natural', request REQ) = the words-along natural
           runs; check against modeswitch/nat3/<tag> (--check-natural, kind 'nat'). kl_words_orders_20261111.py cannot run these
           cells: its renderings() also renders the main probe, which the Mistral template refuses. On the 75 held-out bodies
           the natural3 role renderings, the words-along role and word renderings and the control-token sets are identical under
           the two tokenizers (test_nat3_ministral_render_20261112.py, RENDER CHECK OK, 11-12).
Check: role KLs of the first 2 held-out bodies, relative difference <= 0.02 each; an order is not measured on a failure. Every
body records the comparison. Without U-S (natural order, no System role), the summary gives ratios to KL(A||U) only.
usage: kl_words_orders_20261112.py --model DIR --tmpl-kwargs JSON --bodies role_bodies.jsonl --words words_v2.json
       --orders main,natural | natural3
       --out-main DIR --check-ctrl modeswitch/ctrl/<tag> --out-natural DIR --check-natural DIR --check-natural-kind nat|full
       [--lang en --bodies-en F] [--fixed-date D] [--max-memory-gib G] [--cpu-offload-gib G] [--stop-file F]
Otherwise identical to kl_words_orders_20261111.py (2c9d0922).
20261112b (11-12, user: R1-Distill-Llama-70B moved to server3, "准了"; the user's rule of 11-12: an environment difference is
not a reason to re-run): kl_words_orders_20261112.py (7772e4d6) plus --check-info-only. With it, the check against the
reference run (made on PRO) is computed, logged and stored per body as before, but a difference above 0.02 does not stop the
order (numerics of another GPU type and placement). Without the flag the script behaves exactly as 7772e4d6."""
from __future__ import annotations
import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402
from crossmodel_suite import atomic, load  # noqa: E402
from persistence_readout_20261106 import natural_system_ok  # noqa: E402
from mode_switch_20261107 import PAIRS, REQ, renderings  # noqa: E402
from mode_switch_nat_20261108 import mistral_tok, render_order  # noqa: E402
from words_along_20261110 import load_words, plan  # noqa: E402

NB = 2000
CHECK_REL = 0.02


def summarize(out, pair_ids):
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'bodies').glob('*.json'))]
    fams = sorted({r['category'] for r in rows})
    fidx = np.array([fams.index(r['category']) for r in rows])
    rng = np.random.default_rng(20261111)
    draws = [rng.integers(0, len(fams), len(fams)) for _ in range(NB)]

    def fs(v):
        return np.array([v[fidx == f].sum() for f in range(len(fams))])

    def ratio(a, b):
        fa, fb = fs(a), fs(b)
        bs = np.array([fa[d].sum() / fb[d].sum() for d in draws])
        return {'ratio': float(a.sum() / b.sum()), 'ci95': [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}, bs
    res = {'n_bodies': len(rows), 'families': len(fams), 'roles_pairs': list(rows[0]['roles']), 'kinds': {}}
    bases = [b for b in ('U-S', 'A-U') if b in rows[0]['roles']]
    for k in ('kl_content', 'kl_body'):
        V = {pn: np.array([r['roles'][pn][k] for r in rows]) for pn in rows[0]['roles']}
        for pid in pair_ids:
            V[f'{pid}:12'] = np.array([r['words'][pid][f'{k}_12'] for r in rows])
            V[f'{pid}:21'] = np.array([r['words'][pid][f'{k}_21'] for r in rows])
        o = {'mean_per_body': {n: float(v.mean()) for n, v in V.items()}}
        bsd = {}
        for base in bases:
            key = 'to_US' if base == 'U-S' else 'to_AU'
            o[key], bsd[key] = {}, {}
            for n, v in V.items():
                o[key][n], bsd[key][n] = ratio(v, V[base])
        for d in ('12', '21'):
            names = [f'{pid}:{d}' for pid in pair_ids]
            for key in bsd:
                med = np.median(np.stack([bsd[key][n] for n in names]), 0)
                o[f'word_median_{key}_{d}'] = {'ratio': float(np.median([o[key][n]['ratio'] for n in names])),
                                               'ci95': [float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))],
                                               'max': float(max(o[key][n]['ratio'] for n in names)),
                                               'min': float(min(o[key][n]['ratio'] for n in names))}
        res['kinds'][k] = o
    res['check_max_rel'] = float(max(r['check']['max_rel'] for r in rows))
    atomic(out / 'summary.json', res)
    c = res['kinds']['kl_content']
    s = []
    if 'to_US' in c:
        s.append('A-U/U-S {:.2f}, A-S/U-S {:.2f}; words/U-S median {:.3f} [{:.3f}, {:.3f}] (max {:.3f})'.format(
            c['to_US']['A-U']['ratio'], c['to_US']['A-S']['ratio'], c['word_median_to_US_12']['ratio'],
            *c['word_median_to_US_12']['ci95'], c['word_median_to_US_12']['max']))
    s.append('words/A-U median {:.4f} [{:.4f}, {:.4f}] (max {:.4f})'.format(
        c['word_median_to_AU_12']['ratio'], *c['word_median_to_AU_12']['ci95'], c['word_median_to_AU_12']['max']))
    print(f'summary (content, {out.name}): ' + '; '.join(s) + '; check max rel {:.2e}'.format(res['check_max_rel']), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tmpl-kwargs', default='{}')
    ap.add_argument('--bodies', required=True)
    ap.add_argument('--words', required=True)
    ap.add_argument('--lang', choices=('zh', 'en'), default='zh')
    ap.add_argument('--bodies-en', default=None)
    ap.add_argument('--orders', required=True, help='main, natural, main,natural or natural3')
    ap.add_argument('--out-main', default=None)
    ap.add_argument('--check-ctrl', default=None)
    ap.add_argument('--out-natural', default=None)
    ap.add_argument('--check-natural', default=None)
    ap.add_argument('--check-natural-kind', choices=('nat', 'full'), default=None)
    ap.add_argument('--fixed-date', default=None)
    ap.add_argument('--max-memory-gib', type=float, default=44)
    ap.add_argument('--cpu-offload-gib', type=float, default=0)
    ap.add_argument('--stop-file', default=None)
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--check-only', action='store_true', help='smoke: run the checks of every order, measure nothing')
    ap.add_argument('--check-info-only', action='store_true', help='log and store the check, never stop on it (another environment)')
    args = ap.parse_args()
    orders = args.orders.split(',')
    assert orders and (set(orders) <= {'main', 'natural'} or orders == ['natural3']) and len(set(orders)) == len(orders), orders
    O = {'main': (args.out_main, args.check_ctrl, 'nat'), 'natural': (args.out_natural, args.check_natural, args.check_natural_kind),
         'natural3': (args.out_natural, args.check_natural, args.check_natural_kind)}
    if orders == ['natural3']:
        assert args.check_natural_kind == 'nat', 'natural3 checks against mode_switch_nat_20261108.py records (kind nat)'
    for o in orders:
        assert O[o][0] and O[o][1] and O[o][2], f'--out/--check arguments missing for {o}'
    W, pairs = load_words(args.words, args.lang, 'all')
    pair_ids = [p['id'] for p in pairs]
    if args.summary_only:
        for o in orders:
            summarize(Path(O[o][0]), pair_ids)
        return
    kw = json.loads(args.tmpl_kwargs)
    if args.fixed_date:
        import datetime as _dt
        _day = _dt.date.fromisoformat(args.fixed_date)
        kw['strftime_now'] = lambda fmt: _dt.datetime(_day.year, _day.month, _day.day).strftime(fmt)
    allb = [json.loads(l) for l in open(args.bodies, encoding='utf-8') if l.strip()]
    if args.lang == 'en':
        en = json.loads(Path(args.bodies_en).read_text(encoding='utf-8'))
        allb = [{**b, 'body': en[b['id']]} for b in allb]
    bodies = [b for b in allb if b.get('split') == 'confirmation']
    assert len(bodies) == 75, len(bodies)
    q = cs.Q_REPLY_EN if args.lang == 'en' else cs.Q_REPLY
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: this script does not fall back to the CPU')
    largs = argparse.Namespace(model=args.model, max_memory_gib=args.max_memory_gib, cpu_offload_gib=args.cpu_offload_gib)
    torch, tok, model, cfg, placement, cls = load(largs)
    tok_note = None
    if orders == ['natural3']:
        t2 = mistral_tok(args.model, cfg)   # as mode_switch_nat_20261108.py --order natural3
        if t2 is not None:
            tok_note = {'fix_mistral_regex': True, 'replaced': type(tok).__name__}
            tok = t2
    sys_ok = natural_system_ok(tok, kw, q, REQ[args.lang], allb[0]['body'])   # as mode_switch_ctrl_20261107.py
    dev0 = model.get_input_embeddings().weight.device
    ctrl_ids = sorted(set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}).keys()))
    me = Path(__file__).resolve()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    mask = None

    def lps_of(ids, c):
        nonlocal mask
        x = torch.tensor([ids], dtype=torch.long, device=dev0)
        with torch.no_grad():
            o = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=0)
        lg = o.logits[0].float()
        if mask is None:
            mask = torch.zeros(lg.shape[-1], dtype=torch.bool, device=lg.device)
            mask[[i for i in ctrl_ids if i < lg.shape[-1]]] = True
            mask[len(tok):] = True
        pos = torch.tensor(c[:-1], device=lg.device)
        lp = torch.log_softmax(lg[pos], -1)
        lc = lp[:, ~mask]
        lc = lc - torch.logsumexp(lc, -1, keepdim=True)
        del o, lg
        return lp, lc

    def kl(a, b):
        return float((a.exp() * (a - b)).sum(-1).median())

    def reference(order, ref_dir, kind, bid):
        r = json.loads((ref_dir / 'bodies' / f'{bid}.json').read_text(encoding='utf-8'))
        if kind == 'full':
            return {pn: {'kl_body': v['kl_body']} for pn, v in r['natural']['pairs'].items()}
        return {pn: {k: v[k] for k in ('kl_body', 'kl_content')} for pn, v in r['pairs'].items()}

    def body_record(b, order, ref_dir, kind):
        if order == 'natural3':
            R = render_order('natural3', tok, kw, b['body'], args.lang, None)
        else:
            R = renderings(tok, kw, b['body'], args.lang, sys_ok)[order]
        rec = {'id': b['id'], 'split': b['split'], 'category': b['category'], 'order': order, 'roles': {}, 'words': {}}
        L = {rn: lps_of(v['ids'], v['common']) for rn, v in R.items()}
        for X, Y in PAIRS:
            if X in L and Y in L:
                rec['roles'][f'{X}-{Y}'] = {'kl_body': kl(L[X][0], L[Y][0]), 'kl_content': kl(L[X][1], L[Y][1])}
        del L
        ref = reference(order, ref_dir, kind, b['id'])
        assert set(ref) == set(rec['roles']), ('role pairs differ from the reference', sorted(ref), sorted(rec['roles']))
        rels = [abs(rec['roles'][pn][k] - ref[pn][k]) / max(abs(ref[pn][k]), 1e-12) for pn in ref for k in ref[pn]]
        rec['check'] = {'max_rel': float(max(rels)), 'ok': bool(max(rels) <= CHECK_REL), 'kinds': sorted(next(iter(ref.values())))}
        _, _, Wr, spans = plan(tok, kw, b['body'], 'main' if order == 'main' else 'natural', q, REQ[args.lang], sys_ok, pairs, False)
        for pid in pair_ids:
            l1 = lps_of(Wr[pid]['w1'][0], spans[pid]['w1'])
            l2 = lps_of(Wr[pid]['w2'][0], spans[pid]['w2'])
            rec['words'][pid] = {'kl_body_12': kl(l1[0], l2[0]), 'kl_body_21': kl(l2[0], l1[0]),
                                 'kl_content_12': kl(l1[1], l2[1]), 'kl_content_21': kl(l2[1], l1[1]),
                                 'n_positions': len(spans[pid]['w1']) - 1,
                                 'token_counts': [len(Wr[pid]['w1'][0]), len(Wr[pid]['w2'][0])]}
            del l1, l2
        return rec

    failed = []
    for order in orders:
        out, ref_dir, kind = Path(O[order][0]), Path(O[order][1]), O[order][2]
        (out / 'bodies').mkdir(parents=True, exist_ok=True)
        if (out / 'DONE').exists():
            print(f'{order}: already DONE', flush=True)
            continue
        rid = json.loads((ref_dir / 'identity.json').read_text(encoding='utf-8'))
        if 'n_control_ids' in rid:
            assert rid['n_control_ids'] == len(ctrl_ids), ('control ids differ from the reference run', rid['n_control_ids'], len(ctrl_ids))
        atomic(out / 'identity.json', {'experiment': 'P2_kl_words_orders_20261111', 'order': order, 'script_sha256': sha(me),
                                       'deps_sha256': {n: sha(me.parent / n) for n in ('crossmodel_suite.py', 'persistence_readout_20261106.py',
                                                                                      'mode_switch_20261107.py', 'words_along_20261110.py',
                                                                                      'mode_switch_nat_20261108.py', 'crossmodel_natural3.py')},
                                       'model': args.model, 'model_class': cls, 'tmpl_kwargs': json.loads(args.tmpl_kwargs),
                                       'fixed_date': args.fixed_date, 'lang': args.lang, 'natural_system_ok': sys_ok,
                                       'tokenizer_note': tok_note,
                                       'request': REQ[args.lang] if order != 'main' else None,
                                       'words_sha256': sha(args.words), 'words_version': W['version'], 'pair_ids': pair_ids,
                                       'n_bodies': len(bodies), 'bodies_sha256': sha(args.bodies), 'check_ref': str(ref_dir),
                                       'check_kind': kind, 'check_info_only': bool(args.check_info_only), 'n_control_ids': len(ctrl_ids),
                                       'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                                       'host': socket.gethostname(), 'device_map': placement})
        chk = []
        for b in bodies[:2]:
            f = out / 'bodies' / f'{b["id"]}.json'
            rec = json.loads(f.read_text(encoding='utf-8')) if f.exists() else body_record(b, order, ref_dir, kind)
            chk.append({'id': b['id'], **rec['check'], 'ours': rec['roles']})
            if (rec['check']['ok'] or args.check_info_only) and not f.exists():
                atomic(f, rec)
        atomic(out / 'check.json', {'criterion_rel': CHECK_REL, 'info_only': bool(args.check_info_only), 'bodies': chk})
        print(f'check ({order}{", info only" if args.check_info_only else ""}):',
              json.dumps([{k: c[k] for k in ('id', 'max_rel', 'ok')} for c in chk]), flush=True)
        if not all(c['ok'] for c in chk) and not args.check_info_only:
            print(f'CHECK FAILED ({order}): the role KLs differ from the reference run', flush=True)
            failed.append(order)
            continue
        if args.check_only:
            continue
        t0, done = time.time(), 0
        for b in bodies:
            f = out / 'bodies' / f'{b["id"]}.json'
            if f.exists():
                continue
            if args.stop_file and Path(args.stop_file).exists():
                print(f'[stop] {args.stop_file} exists: stopping before {order} {b["id"]} (rerun resumes)', flush=True)
                sys.exit(2)
            tb = time.time()
            rec = body_record(b, order, ref_dir, kind)
            atomic(f, rec)
            done += 1
            if done % 10 == 0 or done == 1:
                print(f'{order}: {done} bodies, {time.time() - t0:.0f}s, last {b["id"]} ({time.time() - tb:.1f}s), '
                      f'check max rel {rec["check"]["max_rel"]:.2e}', flush=True)
        n_all = sum(1 for b in bodies if (out / 'bodies' / f'{b["id"]}.json').exists())
        print(f'{order}: finished {n_all}/{len(bodies)} bodies on disk', flush=True)
        if n_all == len(bodies):
            summarize(out, pair_ids)
            (out / 'DONE').write_text(time.strftime('%F %T') + '\n')
    if failed:
        raise SystemExit(f'CHECK FAILED for {failed}')


if __name__ == '__main__':
    main()
