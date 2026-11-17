"""mode_switch_swap_hc_20261111.py: mode_switch_swap_20261108.py with hc_edit_20261111 (turn-start tokens of every template; one subspace per stream on four-stream
models). Identical on single-stream models with <|start|> / <|im_start|> templates. Original docstring follows.
P2 mode switch under role-coordinate replacement (user 11-08: KL replacement ablation on the mechanism group, gpt-oss first).
Question: is the asymmetry of the next-token predictions on the body tokens carried by the fitted role subspace of the paper's
causal tests? The asymmetry is KL(A||U) and KL(A||S) >> KL(U||S), as in mode_switch_20261107.py.

Replacement follows crossmodel_role_swap_r05.py and its framework (user 10-23):
  - The same text is rendered under the donor role and run unedited.
  - At every layer of --band, its states at the aligned body positions (align() of that script) are projected on the k leading
    directions of V_l, or of the random R_l. Both come from r05r06_fits/<model>/fit_m.pt: fitted on the 75 discovery bodies,
    body and end of message only, base seed 20261022.
  - The recipient rendering is then run with those coordinates written into its own states at the same positions:
    x <- x + (c_donor - x B) B^T.
  - Region: the body. An edit after the body cannot change a prediction on the body.
Rendering: the mode-switch main probe [ROLE: body][user: neutral reply request] (mode_switch_20261107.renderings).
Evaluation: held-out bodies only (75 bodies, 15 families).
Conditions (k = 8, every layer of the band):
  base:         A, U, S (unedited; donors are captured from these same forwards)
  replacement:  A_U (own <- User) and A_S (own <- System)
  sufficiency:  U_A (User <- own) and S_A (System <- own)
  random 8-dim: A_Ur and U_Ar
Per condition X and unedited reference Y in {A, U, S}, on the common body positions (predicting body token t+1, median over t):
  - KL(X || Y), full;
  - KL(X || Y), content (control tokens removed, each part renormalised);
  - lp_next of the actual next token;
  - per edited condition, the size of the edit: summed squared change and summed squared state over layers and positions.
The summary has family-bootstrap CIs (2,000 resamples) for:
  - left:  KL(A_U||U)/KL(A||U), KL(A_S||S)/KL(A||S), KL(A_Ur||U)/KL(A||U)
           (the share of the own-vs-external difference left after the edit);
  - moved: 1 - KL(U_A||A)/KL(U||A), 1 - KL(S_A||A)/KL(S||A), 1 - KL(U_Ar||A)/KL(U||A)
           (the share of the gap to the own message closed);
  - the mean of every KL, full and content.
usage: mode_switch_swap_20261108.py --model DIR --tmpl-kwargs JSON --bodies role_bodies.jsonl --fit fit_m.pt --out DIR
       [--lang en --bodies-en F] [--band A-B] [--k 8] [--fixed-date D] [--max-memory-gib G] [--cpu-offload-gib C] [--limit N]
       [--stop-file F] [--dry-run]"""
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
from crossmodel_suite import atomic, checked_tokenizer, find_layers, load  # noqa: E402
from hc_edit_20261111 import turn_start_ids, proj, back, gram, top_eig, random_basis, share as hc_share, overlap as hc_overlap  # noqa: E402
from persistence_readout_20261106 import natural_system_ok  # noqa: E402
from mode_switch_20261107 import REQ, renderings  # noqa: E402
from crossmodel_role_swap_r05 import align  # noqa: E402

NB = 2000
CONDS = (('A_U', 'A', 'U', 'V'), ('A_S', 'A', 'S', 'V'), ('U_A', 'U', 'A', 'V'), ('S_A', 'S', 'A', 'V'),
         ('A_Ur', 'A', 'U', 'R'), ('U_Ar', 'U', 'A', 'R'))
LEFT = (('A_U', 'A', 'U'), ('A_S', 'A', 'S'), ('A_Ur', 'A', 'U'))      # KL(X||ref) / KL(base||ref)
MOVED = (('U_A', 'U', 'A'), ('S_A', 'S', 'A'), ('U_Ar', 'U', 'A'))     # 1 - KL(X||A) / KL(base||A)


def tuples(tok, R):
    """(ids, body start, body end, generation-prompt start) per role, as crossmodel_role_swap_r05.render returns them."""
    cands = turn_start_ids(tok)
    out = {}
    for rn, v in R.items():
        ids, sp = v['ids'], v['span']
        g0 = max(i for i in range(len(ids)) if ids[i] in cands)
        assert g0 > sp[-1], 'no generation prompt after the body'
        out[rn] = (ids, sp[0], sp[-1] + 1, g0)
    return out


def body_pairs(r, d):
    ri, di = align(r, d)
    keep = [j for j, i in enumerate(ri) if r[1] <= i < r[2]]
    assert keep, 'no aligned body position'
    return [ri[j] for j in keep], [di[j] for j in keep]


def summarize(out):
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'bodies').glob('*.json'))]
    fams = sorted({r['category'] for r in rows})
    fidx = np.array([fams.index(r['category']) for r in rows])
    rng = np.random.default_rng(20261108)
    draws = [rng.integers(0, len(fams), len(fams)) for _ in range(NB)]

    def fs(v):
        return np.array([v[fidx == f].sum() for f in range(len(fams))]), np.array([(fidx == f).sum() for f in range(len(fams))])

    def mean_ci(v):
        fm, fn = fs(v)
        bs = [fm[d].sum() / fn[d].sum() for d in draws]
        return {'mean': float(v.mean()), 'ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}

    def ratio_ci(a, b, one_minus=False):
        fa, _ = fs(a)
        fb, _ = fs(b)
        f = (lambda x: 1 - x) if one_minus else (lambda x: x)
        bs = [f(fa[d].sum() / fb[d].sum()) for d in draws]
        return {'value': float(f(a.sum() / b.sum())), 'ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}

    def kl(c, y, part):
        return np.array([r['conds'][c]['kl'][y][part] for r in rows])
    res = {'n_held_out': len(rows), 'families': len(fams), 'kl': {}, 'left': {}, 'moved': {}, 'asym': {}, 'dose': {}}
    names = list(rows[0]['conds'])
    for c in names:
        res['kl'][c] = {y: {p: mean_ci(kl(c, y, p)) for p in ('full', 'content')} for y in rows[0]['conds'][c]['kl']}
    for part in ('full', 'content'):
        res['left'][part] = {x: ratio_ci(kl(x, ref, part), kl(base, ref, part)) for x, base, ref in LEFT}
        res['moved'][part] = {x: ratio_ci(kl(x, 'A', part), kl(base, 'A', part), one_minus=True) for x, base, ref in MOVED}
        us = kl('U', 'S', part)
        res['asym'][part] = {f'{x}||{y}/U||S': ratio_ci(kl(x, y, part), us) for x, y in (('A', 'U'), ('A', 'S'), ('A_U', 'U'), ('A_S', 'S'), ('A_Ur', 'U'))}
    for c in names:
        if 'dose' in rows[0]['conds'][c]:
            e2 = np.array([r['conds'][c]['dose']['edit2'] for r in rows])
            s2 = np.array([r['conds'][c]['dose']['state2'] for r in rows])
            res['dose'][c] = {'rel_size': float(np.sqrt(e2.sum() / s2.sum()))}
    atomic(out / 'summary.json', res)
    short = {part: {**{k: round(v['value'], 3) for k, v in res['left'][part].items()}, **{k: round(v['value'], 3) for k, v in res['moved'][part].items()}}
             for part in ('full', 'content')}
    print('summary:', json.dumps({'left_moved': short, 'dose': {k: round(v['rel_size'], 4) for k, v in res['dose'].items()}}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tmpl-kwargs', default='{}')
    ap.add_argument('--bodies', required=True)
    ap.add_argument('--lang', choices=('zh', 'en'), default='zh')
    ap.add_argument('--bodies-en', default=None)
    ap.add_argument('--fit', required=True)
    ap.add_argument('--band', default='')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fixed-date', default=None)
    ap.add_argument('--max-memory-gib', type=float, default=44)
    ap.add_argument('--cpu-offload-gib', type=float, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stop-file', default=None)
    ap.add_argument('--dry-run', action='store_true', help='render and align every held-out body; no model, no GPU')
    a = ap.parse_args()
    out = Path(a.out)
    kw = json.loads(a.tmpl_kwargs)
    if a.fixed_date:
        import datetime as _dt
        _day = _dt.date.fromisoformat(a.fixed_date)
        kw['strftime_now'] = lambda fmt: _dt.datetime(_day.year, _day.month, _day.day).strftime(fmt)
    bodies = [json.loads(l) for l in open(a.bodies, encoding='utf-8') if l.strip()]
    if a.lang == 'en':
        en = json.loads(Path(a.bodies_en).read_text(encoding='utf-8'))
        bodies = [{**b, 'body': en[b['id']]} for b in bodies]
    held = [b for b in bodies if b.get('split') == 'confirmation']
    if a.limit:
        held = held[:a.limit]
    q = cs.Q_REPLY_EN if a.lang == 'en' else cs.Q_REPLY

    if a.dry_run:
        from transformers import AutoTokenizer
        tok = checked_tokenizer(AutoTokenizer.from_pretrained(a.model, trust_remote_code=True), a.model)
        sys_ok = natural_system_ok(tok, kw, q, REQ[a.lang], held[0]['body'])
        cov = {f'{x}<-{d}': [] for _, x, d, _ in CONDS}
        for b in held:
            R = renderings(tok, kw, b['body'], a.lang, sys_ok)['main']
            T = tuples(tok, R)
            for _, x, d, _ in CONDS:
                ri, _ = body_pairs(T[x], T[d])
                cov[f'{x}<-{d}'].append(len(ri) / (T[x][2] - T[x][1]))
        print(json.dumps({'held_out': len(held), 'aligned_body_share_min': {k: round(min(v), 3) for k, v in cov.items()}}, indent=1))
        print('DRY RUN OK')
        return

    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: this script does not fall back to the CPU')
    (out / 'bodies').mkdir(parents=True, exist_ok=True)
    largs = argparse.Namespace(model=a.model, max_memory_gib=a.max_memory_gib, cpu_offload_gib=a.cpu_offload_gib)
    torch, tok, model, cfg, placement, cls = load(largs)
    _, layers = find_layers(model)
    nL = len(layers)
    b0, b1 = (int(x) for x in a.band.split('-')) if a.band else (0, nL - 1)
    band = tuple(range(b0, b1 + 1))
    F = torch.load(a.fit)
    assert all(l in F['V'] and l in F['R'] for l in band), 'the fit does not cover the band'
    Bs = {w: {l: F[w][l][..., :a.k].contiguous().float() for l in band} for w in ('V', 'R')}
    sys_ok = natural_system_ok(tok, kw, q, REQ[a.lang], held[0]['body'])
    dev0 = model.get_input_embeddings().weight.device
    ctrl_ids = sorted(set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}).keys()))
    st = {'mode': None}

    def make_hook(l):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if st['mode'] is None:
                return output
            if st['mode'] == 'capture':
                pos = torch.tensor(st['pos'], dtype=torch.long, device=h.device)
                x = h[0, pos].float()
                for w in ('V', 'R'):
                    st['coords'][w][l] = proj(x, Bs[w][l].to(h.device)).cpu()
            else:
                B = Bs[st['w']][l].to(h.device)
                ri = torch.tensor(st['ri'], dtype=torch.long, device=h.device)
                c = st['c'][l][st['di_rows']].to(h.device)
                x = h[0, ri].float()
                delta = back(c - proj(x, B), B)
                st['edit2'] += float((delta * delta).sum())
                st['state2'] += float((x * x).sum())
                h[0, ri] = (x + delta).to(h.dtype)
            return output
        return hook
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in band]
    me = Path(__file__).resolve()
    atomic(out / 'identity.json', {'experiment': 'P2_modeswitch_swap_20261108', 'script_sha256': hashlib.sha256(me.read_bytes()).hexdigest(),
                                   'fit': a.fit, 'fit_sha256': hashlib.sha256(Path(a.fit).read_bytes()).hexdigest(), 'k': a.k,
                                   'band': [b0, b1], 'n_layers': nL, 'model': a.model, 'model_class': cls, 'tmpl_kwargs': json.loads(a.tmpl_kwargs),
                                   'fixed_date': a.fixed_date, 'lang': a.lang, 'n_held_out': len(held), 'conditions': [c[0] for c in CONDS],
                                   'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                                   'device_map': placement, 'n_control_ids': len(ctrl_ids)})
    mask = None

    def fwd(ids):
        nonlocal mask
        x = torch.tensor([ids], dtype=torch.long, device=dev0)
        with torch.no_grad():
            lg = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=0).logits[0].float()
        if mask is None:
            mask = torch.zeros(lg.shape[-1], dtype=torch.bool, device=lg.device)
            mask[[i for i in ctrl_ids if i < lg.shape[-1]]] = True
            mask[len(tok):] = True
        return lg

    def dists(lg, v):
        c = v['common']
        lp = torch.log_softmax(lg[torch.tensor(c[:-1], device=lg.device)], -1)
        lc = lp[:, ~mask]
        lc = lc - torch.logsumexp(lc, -1, keepdim=True)
        nxt = torch.tensor([v['ids'][i + 1] for i in c[:-1]], device=lg.device)
        return lp, lc, float(lp[torch.arange(len(nxt), device=lg.device), nxt].mean())

    def klmed(p, q_):
        return float((p.exp() * (p - q_)).sum(-1).median())
    t0, done = time.time(), 0
    for b in held:
        f = out / 'bodies' / f'{b["id"]}.json'
        if f.exists():
            continue
        if a.stop_file and Path(a.stop_file).exists():
            print(f'[stop] {a.stop_file} exists: stopping before {b["id"]}', flush=True)
            break
        R = renderings(tok, kw, b['body'], a.lang, sys_ok)['main']
        T = tuples(tok, R)
        base, coords = {}, {}
        for rn in ('A', 'U', 'S'):
            st.update(mode='capture', pos=list(range(T[rn][1], T[rn][2])), coords={'V': {}, 'R': {}})
            lg = fwd(R[rn]['ids'])
            st['mode'] = None
            coords[rn] = st['coords']
            base[rn] = dists(lg, R[rn])
            del lg
        rec = {'id': b['id'], 'split': b['split'], 'category': b['category'], 'conds': {}}
        for rn in ('A', 'U', 'S'):
            rec['conds'][rn] = {'lp_next': base[rn][2], 'kl': {y: {'full': klmed(base[rn][0], base[y][0]), 'content': klmed(base[rn][1], base[y][1])}
                                                            for y in ('A', 'U', 'S') if y != rn}}
        for name, x, d, w in CONDS:
            ri, di = body_pairs(T[x], T[d])
            st.update(mode='write', w=w, ri=ri, di_rows=[i - T[d][1] for i in di], c=coords[d][w], edit2=0.0, state2=0.0)
            lg = fwd(R[x]['ids'])
            st['mode'] = None
            lp, lc, lpn = dists(lg, R[x])
            del lg
            rec['conds'][name] = {'lp_next': lpn, 'kl': {y: {'full': klmed(lp, base[y][0]), 'content': klmed(lc, base[y][1])} for y in ('A', 'U', 'S')},
                                  'dose': {'edit2': st['edit2'], 'state2': st['state2'], 'n_pos': len(ri)}}
        atomic(f, rec)
        done += 1
        if done % 25 == 0 or done == 1:
            print(f'{done} bodies, {time.time() - t0:.0f}s: A_U||U {rec["conds"]["A_U"]["kl"]["U"]["full"]:.4f} vs A||U '
                  f'{rec["conds"]["A"]["kl"]["U"]["full"]:.4f}; U_A||A {rec["conds"]["U_A"]["kl"]["A"]["full"]:.4f} vs U||A '
                  f'{rec["conds"]["U"]["kl"]["A"]["full"]:.4f}', flush=True)
    for hd in handles:
        hd.remove()
    n_all = sum(1 for b in held if (out / 'bodies' / f'{b["id"]}.json').exists())
    print(f'finished: {n_all}/{len(held)}', flush=True)
    if n_all == len(held):
        summarize(out)
        (out / 'DONE').write_text(time.strftime('%F %T') + '\n')


if __name__ == '__main__':
    main()
