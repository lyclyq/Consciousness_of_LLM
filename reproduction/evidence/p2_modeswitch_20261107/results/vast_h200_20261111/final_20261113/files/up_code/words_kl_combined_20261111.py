"""Ordinary words: retention along the message and next-token KL from the same forwards (11-11, user: GLM-5.3-Flash "同时记录 KL 普通词和
保持度 ... 成本最低", option A). Exploratory and internal: not for the release or the data bundle.
For each of the 75 held-out bodies, main order:
  retention  as words_along_20261110.py (583d3fff) with --roles --pairs all: renderings of words_along_20261110.plan, layer outputs at
             the KEEP_FRACS layers on the common body tokens (hyper-connection outputs reshaped to the concatenated streams), the
             first/last-third decomposition per row; per-body npz + json in words_along's format (rows: A-U, A-S, U-S, 32 pairs)
  KL         as kl_words_20261111.py (f297e130) with --pairs all: role renderings of mode_switch_20261107.renderings()['main'], word
             renderings of the same plan; content and full KL, median over the common body positions but the last
One forward per rendering serves both readouts: the forward hooks read the layer outputs and the logits of all positions are kept
(logits_to_keep=0, which does not change the layer outputs). A role rendering is shared only where mode_switch and words_along give
the same ids and common positions; otherwise the KL rendering gets its own forward.
Checks before any measurement, on the first 2 held-out bodies (their records are then kept):
  KL         role KLs against the paper's ctrl run (--check-ctrl), relative difference <= 0.02 (kl_words_20261111.py)
  retention  role states and every row present in the earlier run (--gate-pilot: GLM-5.3 stage 1, roles + the first pair of each
             category) at the kept layers, cosine >= 0.999 and relative error <= 0.03 (words_along_20261110.py, gate criterion)
usage: words_kl_combined_20261111.py --model DIR --tmpl-kwargs JSON --bodies role_bodies.jsonl --words words_v2.json --lang zh
       --check-ctrl modeswitch/ctrl/<tag> --gate-pilot words_along/b150_main/<tag> --out-words DIR --out-kl DIR
       [--max-memory-gib G] [--cpu-offload-gib G] [--stop-file F]"""
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
from crossmodel_suite import atomic, find_layers, load  # noqa: E402
from persistence_readout_20261106 import KEEP_FRACS, ROLES, RPAIRS, natural_system_ok  # noqa: E402
from mode_switch_20261107 import PAIRS, REQ, renderings  # noqa: E402
from words_along_20261110 import load_words, plan, rows_for  # noqa: E402
from kl_words_20261111 import summarize as kl_summarize  # noqa: E402

GATE_COS, GATE_REL, CHECK_REL = 0.999, 0.03, 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tmpl-kwargs', default='{}')
    ap.add_argument('--bodies', required=True)
    ap.add_argument('--words', required=True)
    ap.add_argument('--lang', choices=('zh', 'en'), default='zh')
    ap.add_argument('--check-ctrl', required=True)
    ap.add_argument('--gate-pilot', required=True)
    ap.add_argument('--out-words', required=True)
    ap.add_argument('--out-kl', required=True)
    ap.add_argument('--max-memory-gib', type=float, default=44)
    ap.add_argument('--cpu-offload-gib', type=float, default=0)
    ap.add_argument('--stop-file', default=None)
    ap.add_argument('--min-free-gib', type=float, default=2.0)
    ap.add_argument('--check-only', action='store_true', help='run the checks on the first 2 bodies (records kept) and stop')
    ap.add_argument('--gate-info-only', action='store_true', help='GLM-5.3 re-run in a new environment (11-11): record the comparison with --gate-pilot, gate on the KL check against --check-ctrl (the ctrl run of this same environment) only')
    args = ap.parse_args()
    assert args.lang == 'zh', 'written for the Chinese items of GLM-5.3-Flash'
    VDT, TDT_np = np.float16, 'float16'
    kw = json.loads(args.tmpl_kwargs)
    q = cs.Q_REPLY
    W, pairs = load_words(args.words, args.lang, 'all')
    pair_ids = [p['id'] for p in pairs]
    rows = rows_for(True, pairs)
    allb = [json.loads(l) for l in open(args.bodies, encoding='utf-8') if l.strip()]
    bodies = [b for b in allb if b.get('split') == 'confirmation']
    assert len(bodies) == 75, len(bodies)
    ow, ok_ = Path(args.out_words), Path(args.out_kl)
    ow.mkdir(parents=True, exist_ok=True)
    (ok_ / 'bodies').mkdir(parents=True, exist_ok=True)
    ctrl, gp = Path(args.check_ctrl), Path(args.gate_pilot)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: this script does not fall back to the CPU')
    largs = argparse.Namespace(model=args.model, max_memory_gib=args.max_memory_gib, cpu_offload_gib=args.cpu_offload_gib)
    torch, tok, model, cfg, placement, cls = load(largs)
    lname, layers = find_layers(model)
    L = len(layers)
    keep = sorted({round(f * (L - 1)) for f in KEEP_FRACS})
    K = len(keep)
    kpos = {l: i for i, l in enumerate(keep)}
    sys_ok_kl = natural_system_ok(tok, kw, q, REQ[args.lang], allb[0]['body'])   # as kl_words_20261111.py
    dev0 = model.get_input_embeddings().weight.device
    TDT = torch.float16
    ctrl_ids = sorted(set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}).keys()))
    cid = json.loads((ctrl / 'identity.json').read_text(encoding='utf-8'))
    if 'n_control_ids' in cid:
        assert cid['n_control_ids'] == len(ctrl_ids), ('control ids differ from the paper ctrl run', cid['n_control_ids'], len(ctrl_ids))
    pid_ident = json.loads((gp / 'identity.json').read_text(encoding='utf-8'))
    for k in ('order', 'lang', 'fixed_date'):
        assert pid_ident.get(k) == {'order': 'main', 'lang': args.lang, 'fixed_date': None}[k], ('earlier run differs', k)
    assert pid_ident.get('tmpl_kwargs') == json.loads(args.tmpl_kwargs) and pid_ident['keep_layers'] == keep
    st = {'span': None, 'rank': set()}
    cap_last, cap_span = {}, {}

    def hook(i):
        def h(mod, inp, outp):
            x = outp[0] if isinstance(outp, tuple) else outp
            st['rank'].add(x.dim())
            if x.dim() == 4:
                x = x.reshape(x.shape[0], x.shape[1], -1)
            assert x.dim() == 3
            cap_last[i] = x[0, -1].detach()
            if st['span'] is not None and i in kpos:
                a, e = st['span']
                cap_span[i] = x[0, a:e].detach()
        return h
    handles = [layers[i].register_forward_hook(hook(i)) for i in range(L)]
    mask = None

    def fwd(ids, span=None, kl_pos=None):
        """words_along's forward (layer outputs), with the logits of every position kept for the KL at kl_pos[:-1]"""
        nonlocal mask
        cap_last.clear()
        cap_span.clear()
        st['span'] = None if span is None else (span[0], span[-1] + 1)
        x = torch.tensor([ids], dtype=torch.long, device=dev0)
        with torch.no_grad():
            o = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=0)
        st['span'] = None
        last = torch.stack([cap_last[i].float().to(dev0) for i in range(L)])
        sp = torch.stack([cap_span[l].float().to(dev0) for l in keep]) if span is not None else None
        lps = None
        if kl_pos is not None:
            lg = o.logits[0].float()
            if mask is None:
                mask = torch.zeros(lg.shape[-1], dtype=torch.bool, device=lg.device)
                mask[[i for i in ctrl_ids if i < lg.shape[-1]]] = True
                mask[len(tok):] = True
            pos = torch.tensor(kl_pos[:-1], device=lg.device)
            lp = torch.log_softmax(lg[pos], -1)
            lc = lp[:, ~mask]
            lc = lc - torch.logsumexp(lc, -1, keepdim=True)
            lps = (lp, lc)
            del lg
        del o
        return last, sp, lps

    def decomp(x1, x2):
        D = x1 - x2
        T = D.shape[1]
        p = torch.arange(T, dtype=torch.float32, device=D.device) / T
        f, la = p < 1 / 3, p >= 2 / 3
        return (D[:, f].mean(1), D[:, la].mean(1), (D[:, f] ** 2).sum(-1).mean(1), (D[:, la] ** 2).sum(-1).mean(1))

    def cmp(ours, ref):
        ref = torch.as_tensor(np.asarray(ref, np.float32), device=ours.device).reshape(ours.shape[0], -1)
        ours = ours.reshape(ref.shape)
        cosv = torch.nn.functional.cosine_similarity(ours, ref, dim=-1)
        rel = (ours - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)
        return float(cosv.min()), float(rel.max())

    def kl(a, b):
        return float((a.exp() * (a - b)).sum(-1).median())

    def measure(b):
        text, R, Wr, spans = plan(tok, kw, b['body'], 'main', q, None, True, pairs, True)   # words_along: main order, roles + pairs
        Rk = renderings(tok, kw, b['body'], args.lang, sys_ok_kl)['main']                  # kl_words: the paper's KL renderings
        H, dec, tc, ncom, uneq, Lk, shared = {}, {}, {}, {}, [], {}, []
        for rn, (ids, _) in R.items():
            same = rn in Rk and Rk[rn]['ids'] == ids and list(Rk[rn]['common']) == list(spans['roles'][rn])
            last, sp, lps = fwd(ids, spans['roles'][rn], Rk[rn]['common'] if same else None)
            H[rn] = (last, sp)
            tc[rn] = len(ids)
            if same:
                Lk[rn] = lps
                shared.append(rn)
        for rn, v in Rk.items():
            if rn not in Lk:
                Lk[rn] = fwd(v['ids'], None, v['common'])[2]
        kr = {}
        for X, Y in PAIRS:
            if X in Lk and Y in Lk:
                kr[f'{X}-{Y}'] = {'kl_body': kl(Lk[X][0], Lk[Y][0]), 'kl_content': kl(Lk[X][1], Lk[Y][1])}
        del Lk
        for i1, i2 in RPAIRS:
            a, c = ROLES[i1][0], ROLES[i2][0]
            if a in H and c in H:
                dec[f'{a}-{c}'] = decomp(H[a][1], H[c][1])
                ncom[f'{a}-{c}'] = int(H[a][1].shape[1])
        kw_ = {}
        for pid, r2 in Wr.items():
            l1 = fwd(r2['w1'][0], spans[pid]['w1'], spans[pid]['w1'])
            l2 = fwd(r2['w2'][0], spans[pid]['w2'], spans[pid]['w2'])
            dec[pid] = decomp(l1[1], l2[1])
            ncom[pid] = int(l1[1].shape[1])
            tc[pid] = [len(r2['w1'][0]), len(r2['w2'][0])]
            if tc[pid][0] != tc[pid][1]:
                uneq.append(pid)
            (a1, c1), (a2, c2) = l1[2], l2[2]
            kw_[pid] = {'kl_body_12': kl(a1, a2), 'kl_body_21': kl(a2, a1), 'kl_content_12': kl(c1, c2), 'kl_content_21': kl(c2, c1),
                        'n_positions': len(spans[pid]['w1']) - 1, 'token_counts': tc[pid]}
            del l1, l2
        return H, dec, tc, ncom, uneq, kr, kw_, shared

    def kl_check(b, kr):
        ref = json.loads((ctrl / 'bodies' / f'{b["id"]}.json').read_text(encoding='utf-8'))['pairs']
        rels = [abs(kr[pn][k] - ref[pn][k]) / max(abs(ref[pn][k]), 1e-12) for pn in kr for k in ('kl_body', 'kl_content')]
        return {'max_rel': float(max(rels)), 'ok': bool(max(rels) <= CHECK_REL)}

    def write(b, H, dec, tc, ncom, uneq, kr, kw_, shared, chk, secs):
        d = next(iter(dec.values()))[0].shape[-1]
        R_ = len(rows)
        arrs = {'dec_mF': np.full((R_, K, d), np.nan, VDT), 'dec_mL': np.full((R_, K, d), np.nan, VDT),
                'dec_eF': np.full((R_, K), np.nan, np.float32), 'dec_eL': np.full((R_, K), np.nan, np.float32),
                'role_state': np.full((3, K, d), np.nan, VDT)}
        for qi, nm in enumerate(rows):
            if nm in dec:
                mF, mL, eF, eL = dec[nm]
                arrs['dec_mF'][qi], arrs['dec_mL'][qi] = mF.to(TDT).cpu().numpy(), mL.to(TDT).cpu().numpy()
                arrs['dec_eF'][qi], arrs['dec_eL'][qi] = eF.cpu().numpy(), eL.cpu().numpy()
        for ri, (rn, _) in enumerate(ROLES):
            if rn in H:
                arrs['role_state'][ri] = H[rn][0][keep].to(TDT).cpu().numpy()
        for kk, v in arrs.items():
            assert not np.isinf(v).any(), f'{kk} has inf in {b["id"]}'
        part = ow / f'{b["id"]}.npz.part'
        with open(part, 'wb') as fh:
            np.savez(fh, **arrs)
        part.replace(ow / f'{b["id"]}.npz')
        atomic(ow / f'{b["id"]}.json', {'id': b['id'], 'split': b.get('split'), 'category': b.get('category'), 'token_counts': tc,
                                         'n_common': ncom, 'unequal_pairs': uneq, 'seconds': secs, 'kl_shared_roles': shared})
        atomic(ok_ / 'bodies' / f'{b["id"]}.json', {'id': b['id'], 'split': b['split'], 'category': b['category'], 'roles': kr,
                                                    'words': kw_, 'check': chk})

    me = Path(__file__).resolve()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    base = {'script_sha256': sha(me), 'deps_sha256': {n: sha(me.parent / n) for n in ('crossmodel_suite.py', 'persistence_readout_20261106.py',
            'mode_switch_20261107.py', 'words_along_20261110.py', 'kl_words_20261111.py')}, 'model': args.model, 'model_class': cls,
            'tmpl_kwargs': json.loads(args.tmpl_kwargs), 'fixed_date': None, 'lang': args.lang, 'order': 'main', 'words_sha256': sha(args.words),
            'words_version': W['version'], 'pair_ids': pair_ids, 'n_bodies': len(bodies), 'bodies_sha256': sha(args.bodies),
            'torch': torch.__version__, 'transformers': __import__('transformers').__version__, 'host': socket.gethostname(),
            'device_map': placement, 'cpu_offload_gib': args.cpu_offload_gib}

    # ---------------- checks on the first 2 held-out bodies ----------------
    gate = {'criterion': {'cos_min': GATE_COS, 'rel_max': GATE_REL, 'kl_rel_max': CHECK_REL}, 'bodies': [], 'ok': True}
    for b in bodies[:2]:
        done = (ok_ / 'bodies' / f'{b["id"]}.json').exists() and (ow / f'{b["id"]}.json').exists()
        if done:
            rec = json.loads((ok_ / 'bodies' / f'{b["id"]}.json').read_text(encoding='utf-8'))
            gate['bodies'].append({'id': b['id'], 'kl': rec['check'], 'resumed': True})
            continue
        tb = time.time()
        H, dec, tc, ncom, uneq, kr, kw_, shared = measure(b)
        chk = kl_check(b, kr)
        g = {'id': b['id'], 'kl': chk, 'checks': {}}
        with np.load(gp / f'{b["id"]}.npz') as z:
            Z = {k: z[k] for k in z.files}
        for ri, (rn, _) in enumerate(ROLES):
            if rn in H and np.isfinite(Z['role_state'][ri]).all():
                g['checks'][f'state_{rn}'] = cmp(H[rn][0][keep], Z['role_state'][ri])
        for qi, nm in enumerate(pid_ident['rows']):
            if nm in dec and np.isfinite(Z['dec_mF'][qi]).all():
                mF, mL, eF, eL = dec[nm]
                c1, r1 = cmp(mF, Z['dec_mF'][qi])
                c2, r2 = cmp(mL, Z['dec_mL'][qi])
                re_ = float(max(np.max(np.abs(eF.cpu().numpy() / Z['dec_eF'][qi] - 1)), np.max(np.abs(eL.cpu().numpy() / Z['dec_eL'][qi] - 1))))
                g['checks'][f'dec_{nm}'] = (min(c1, c2), max(r1, r2, re_))
        g['ok'] = chk['ok'] and bool(g['checks']) and all(c >= GATE_COS and r <= GATE_REL for c, r in g['checks'].values())
        if args.gate_info_only:
            g['stage1_within_criterion'] = g['ok']
            g['ok'] = chk['ok']
        gate['bodies'].append(g)
        gate['ok'] &= g['ok']
        if g['ok']:
            write(b, H, dec, tc, ncom, uneq, kr, kw_, shared, chk, round(time.time() - tb, 1))
        del H, dec
    ident_w = {**base, 'experiment': 'P2_words_along_20261110', 'combined_with': 'P2_kl_words (words_kl_combined_20261111.py)',
               'n_layers': L, 'layers_module': lname, 'keep_layers': keep, 'rows': rows, 'pairs': 'all', 'split': 'confirmation',
               'request': None, 'reply_question': q, 'natural_system_ok': True, 'gate_pilot': str(gp), 'gate': gate,
               'layer_output_ranks': sorted(st['rank']), 'store': TDT_np}
    ident_k = {**base, 'experiment': 'P2_kl_words_20261111', 'combined_with': 'P2_words_along (words_kl_combined_20261111.py)',
               'check_ctrl': str(ctrl), 'n_control_ids': len(ctrl_ids), 'natural_system_ok': sys_ok_kl}
    atomic(ow / 'identity.json', ident_w)
    atomic(ok_ / 'identity.json', ident_k)
    atomic(ok_ / 'check.json', {'criterion_rel': CHECK_REL, 'bodies': [{'id': g['id'], **g['kl']} for g in gate['bodies']]})
    print('gate:', json.dumps(gate), flush=True)
    if not gate['ok']:
        raise SystemExit(f'GATE FAILED: {json.dumps(gate)}')
    if args.check_only:
        print('CHECK ONLY: checks passed, records of the 2 bodies kept', flush=True)
        return

    # ---------------- measurement ----------------
    t0, n = time.time(), 0
    for b in bodies:
        if (ok_ / 'bodies' / f'{b["id"]}.json').exists() and (ow / f'{b["id"]}.json').exists():
            continue
        if args.stop_file and Path(args.stop_file).exists():
            print(f'[stop] {args.stop_file} exists: stopping before {b["id"]} (rerun resumes)', flush=True)
            break
        free = shutil.disk_usage(ow).free / 2 ** 30
        if free < args.min_free_gib:
            print(f'[stop] {free:.1f} GiB free: stopping before {b["id"]}', flush=True)
            break
        tb = time.time()
        H, dec, tc, ncom, uneq, kr, kw_, shared = measure(b)
        write(b, H, dec, tc, ncom, uneq, kr, kw_, shared, kl_check(b, kr), round(time.time() - tb, 1))
        n += 1
        del H, dec
        if n % 10 == 0 or n == 1:
            print(f'{n} bodies, {time.time() - t0:.0f}s, last {b["id"]} ({time.time() - tb:.1f}s)', flush=True)
    for hd in handles:
        hd.remove()
    done = [b for b in bodies if (ok_ / 'bodies' / f'{b["id"]}.json').exists() and (ow / f'{b["id"]}.json').exists()]
    print(f'finished: {len(done)}/{len(bodies)} bodies on disk', flush=True)
    if len(done) == len(bodies):
        kl_summarize(ok_, pair_ids)
        stamp = time.strftime('%F %T') + '\n'
        (ow / 'DONE').write_text(stamp)
        (ok_ / 'DONE').write_text(stamp)


if __name__ == '__main__':
    main()
