"""P2 ordinary words in next-token prediction, depth part (11-11, user: "机制组 KL 可以补 200"). Exploratory and internal, like the
rest of this archive: not for the release or the data bundle.
The same measurement as kl_words_20261111.py (same renderings, forwards, content mask, KL and pairs; lps_of, kl and the body
record are copied from it, the record with the per-body check made optional), on the 200 held-out phase-3 bodies (phase3_bodies_zh.jsonl split confirmation; English from
phase3_bodies_en.json), the body set of the words-along depth cells (p3_main).
Check, before any measurement: the role KLs of the first 2 held-out bodies of the paper's 150 (--bodies, split confirmation) against
the paper's ctrl run of this setting (--check-ctrl), relative difference <= 0.02 each, as kl_words_20261111.py; the run refuses to
measure on a failure. Those 2 check records go to check_bodies/, not into the measured set. The phase-3 bodies have no paper ctrl
record, so their records carry no per-body check.
Summary: as kl_words_20261111.py, with every phase-3 body its own family (200 resampling units, as the words-along p3 cells).
usage: kl_words_p3_20261111.py --model DIR --tmpl-kwargs JSON --bodies role_bodies.jsonl --p3-bodies phase3_bodies_zh.jsonl
       --words words_v2.json --out DIR --check-ctrl modeswitch/ctrl/<tag> [--lang en --bodies-en F --p3-bodies-en F]
       [--fixed-date D] [--max-memory-gib G] [--stop-file F]"""
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
from words_along_20261110 import load_words, plan  # noqa: E402

NB = 2000
CHECK_REL = 0.02
N_P3 = 200


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
    res = {'n_bodies': len(rows), 'families': len(fams), 'kinds': {}}
    for k in ('kl_content', 'kl_body'):
        V = {pn: np.array([r['roles'][pn][k] for r in rows]) for pn in rows[0]['roles']}
        for pid in pair_ids:
            V[f'{pid}:12'] = np.array([r['words'][pid][f'{k}_12'] for r in rows])
            V[f'{pid}:21'] = np.array([r['words'][pid][f'{k}_21'] for r in rows])
        o = {'mean_per_body': {n: float(v.mean()) for n, v in V.items()}, 'to_US': {}, 'to_AU': {}}
        bsU, bsA = {}, {}
        for n, v in V.items():
            o['to_US'][n], bsU[n] = ratio(v, V['U-S'])
            o['to_AU'][n], bsA[n] = ratio(v, V['A-U'])
        for d in ('12', '21'):
            names = [f'{pid}:{d}' for pid in pair_ids]
            for base, bsd, key in (('to_US', bsU, 'U-S'), ('to_AU', bsA, 'A-U')):
                med = np.median(np.stack([bsd[n] for n in names]), 0)
                o[f'word_median_{base}_{d}'] = {'ratio': float(np.median([o[base][n]['ratio'] for n in names])),
                                                'ci95': [float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))],
                                                'max': float(max(o[base][n]['ratio'] for n in names)),
                                                'min': float(min(o[base][n]['ratio'] for n in names))}
        res['kinds'][k] = o
    chk = json.loads((out / 'check.json').read_text(encoding='utf-8'))
    res['check_max_rel'] = float(max(c['max_rel'] for c in chk['bodies']))
    atomic(out / 'summary.json', res)
    c = res['kinds']['kl_content']
    print('summary (content): A-U/U-S {:.2f}, A-S/U-S {:.2f}; words/U-S median {:.3f} [{:.3f}, {:.3f}] (max {:.3f}); words/A-U median {:.4f}; '
          'check max rel {:.2e}'.format(c['to_US']['A-U']['ratio'], c['to_US']['A-S']['ratio'], c['word_median_to_US_12']['ratio'],
                                        *c['word_median_to_US_12']['ci95'], c['word_median_to_US_12']['max'],
                                        c['word_median_to_AU_12']['ratio'], res['check_max_rel']), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tmpl-kwargs', default='{}')
    ap.add_argument('--bodies', required=True)
    ap.add_argument('--p3-bodies', required=True)
    ap.add_argument('--words', required=True)
    ap.add_argument('--lang', choices=('zh', 'en'), default='zh')
    ap.add_argument('--bodies-en', default=None)
    ap.add_argument('--p3-bodies-en', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--check-ctrl', required=True)
    ap.add_argument('--fixed-date', default=None)
    ap.add_argument('--max-memory-gib', type=float, default=44)
    ap.add_argument('--cpu-offload-gib', type=float, default=0)
    ap.add_argument('--stop-file', default=None)
    ap.add_argument('--max-bodies', type=int, default=None, help='smoke only: stop after this many phase-3 bodies, no summary')
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    W, pairs = load_words(args.words, args.lang, 'all')
    pair_ids = [p['id'] for p in pairs]
    if args.summary_only:
        summarize(out, pair_ids)
        return
    kw = json.loads(args.tmpl_kwargs)
    if args.fixed_date:
        import datetime as _dt
        _day = _dt.date.fromisoformat(args.fixed_date)
        kw['strftime_now'] = lambda fmt: _dt.datetime(_day.year, _day.month, _day.day).strftime(fmt)
    allb = [json.loads(l) for l in open(args.bodies, encoding='utf-8') if l.strip()]
    p3all = [json.loads(l) for l in open(args.p3_bodies, encoding='utf-8') if l.strip()]
    if args.lang == 'en':
        en = json.loads(Path(args.bodies_en).read_text(encoding='utf-8'))
        allb = [{**b, 'body': en[b['id']]} for b in allb]
        en3 = json.loads(Path(args.p3_bodies_en).read_text(encoding='utf-8'))
        p3all = [{**b, 'body': en3[b['id']]} for b in p3all]
    bodies = [b for b in allb if b.get('split') == 'confirmation']
    assert len(bodies) == 75, len(bodies)
    p3 = [b for b in p3all if b.get('split') == 'confirmation']
    assert len(p3) == N_P3 and len({b['category'] for b in p3}) == N_P3, (len(p3), len({b['category'] for b in p3}))
    q = cs.Q_REPLY_EN if args.lang == 'en' else cs.Q_REPLY
    ctrl = Path(args.check_ctrl)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: this script does not fall back to the CPU')
    (out / 'bodies').mkdir(parents=True, exist_ok=True)
    (out / 'check_bodies').mkdir(parents=True, exist_ok=True)
    largs = argparse.Namespace(model=args.model, max_memory_gib=args.max_memory_gib, cpu_offload_gib=args.cpu_offload_gib)
    torch, tok, model, cfg, placement, cls = load(largs)
    sys_ok = natural_system_ok(tok, kw, q, REQ[args.lang], allb[0]['body'])   # as mode_switch_ctrl_20261107.py and kl_words_20261111.py
    dev0 = model.get_input_embeddings().weight.device
    ctrl_ids = sorted(set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}).keys()))
    cid = json.loads((ctrl / 'identity.json').read_text(encoding='utf-8'))
    assert cid['n_control_ids'] == len(ctrl_ids), ('control ids differ from the paper ctrl run', cid['n_control_ids'], len(ctrl_ids))
    me = Path(__file__).resolve()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    atomic(out / 'identity.json', {'experiment': 'P2_kl_words_p3_20261111', 'script_sha256': sha(me),
                                   'deps_sha256': {n: sha(me.parent / n) for n in ('crossmodel_suite.py', 'persistence_readout_20261106.py',
                                                                                  'mode_switch_20261107.py', 'words_along_20261110.py')},
                                   'model': args.model, 'model_class': cls, 'tmpl_kwargs': json.loads(args.tmpl_kwargs),
                                   'fixed_date': args.fixed_date, 'lang': args.lang, 'words_sha256': sha(args.words),
                                   'words_version': W['version'], 'pair_ids': pair_ids, 'n_bodies': len(p3),
                                   'bodies_sha256': sha(args.bodies), 'p3_bodies_sha256': sha(args.p3_bodies),
                                   'p3_bodies_en_sha256': sha(args.p3_bodies_en) if args.p3_bodies_en else None,
                                   'check_ctrl': str(ctrl), 'n_control_ids': len(ctrl_ids),
                                   'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                                   'host': socket.gethostname(), 'device_map': placement})
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

    def body_record(b, check):
        R = renderings(tok, kw, b['body'], args.lang, sys_ok)['main']
        rec = {'id': b['id'], 'split': b['split'], 'category': b['category'], 'roles': {}, 'words': {}}
        L = {rn: lps_of(v['ids'], v['common']) for rn, v in R.items()}
        for X, Y in PAIRS:
            rec['roles'][f'{X}-{Y}'] = {'kl_body': kl(L[X][0], L[Y][0]), 'kl_content': kl(L[X][1], L[Y][1])}
        del L
        if check:
            ref = json.loads((ctrl / 'bodies' / f'{b["id"]}.json').read_text(encoding='utf-8'))['pairs']
            rels = [abs(rec['roles'][pn][k] - ref[pn][k]) / max(abs(ref[pn][k]), 1e-12) for pn in rec['roles'] for k in ('kl_body', 'kl_content')]
            rec['check'] = {'max_rel': float(max(rels)), 'ok': bool(max(rels) <= CHECK_REL)}
        else:
            rec['check'] = None
        _, _, Wr, spans = plan(tok, kw, b['body'], 'main', q, REQ[args.lang], sys_ok, pairs, False)
        for pid in pair_ids:
            l1 = lps_of(Wr[pid]['w1'][0], spans[pid]['w1'])
            l2 = lps_of(Wr[pid]['w2'][0], spans[pid]['w2'])
            rec['words'][pid] = {'kl_body_12': kl(l1[0], l2[0]), 'kl_body_21': kl(l2[0], l1[0]),
                                 'kl_content_12': kl(l1[1], l2[1]), 'kl_content_21': kl(l2[1], l1[1]),
                                 'n_positions': len(spans[pid]['w1']) - 1,
                                 'token_counts': [len(Wr[pid]['w1'][0]), len(Wr[pid]['w2'][0])]}
            del l1, l2
        return rec

    # check on the first 2 held-out bodies of the paper's 150 before any measurement
    chk = []
    for b in bodies[:2]:
        f = out / 'check_bodies' / f'{b["id"]}.json'
        rec = json.loads(f.read_text(encoding='utf-8')) if f.exists() else body_record(b, True)
        chk.append({'id': b['id'], **rec['check'], 'ours': rec['roles']})
        if rec['check']['ok'] and not f.exists():
            atomic(f, rec)
    atomic(out / 'check.json', {'criterion_rel': CHECK_REL, 'bodies': chk})
    print('check:', json.dumps([{k: c[k] for k in ('id', 'max_rel', 'ok')} for c in chk]), flush=True)
    if not all(c['ok'] for c in chk):
        raise SystemExit('CHECK FAILED: the role KLs differ from the paper ctrl run')
    t0, done = time.time(), 0
    for b in p3:
        f = out / 'bodies' / f'{b["id"]}.json'
        if f.exists():
            continue
        if args.stop_file and Path(args.stop_file).exists():
            print(f'[stop] {args.stop_file} exists: stopping before {b["id"]} (rerun resumes)', flush=True)
            break
        if args.max_bodies is not None and done >= args.max_bodies:
            print(f'[smoke] {done} phase-3 bodies measured: stopping', flush=True)
            return
        tb = time.time()
        rec = body_record(b, False)
        atomic(f, rec)
        done += 1
        if done % 20 == 0 or done == 1:
            print(f'{done} bodies, {time.time() - t0:.0f}s, last {b["id"]} ({time.time() - tb:.1f}s)', flush=True)
    n_all = sum(1 for b in p3 if (out / 'bodies' / f'{b["id"]}.json').exists())
    print(f'finished: {n_all}/{len(p3)} bodies on disk', flush=True)
    if n_all == len(p3):
        summarize(out, pair_ids)
        (out / 'DONE').write_text(time.strftime('%F %T') + '\n')


if __name__ == '__main__':
    main()
