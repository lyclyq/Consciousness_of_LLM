"""Next-token prediction under the replacement of the self-index coordinates (Appendix "Next-Token Prediction", Table
tab:self_kl_swap): is the difference between the predictions on the model's own message and on a User or System message carried by
the fitted subspace of the causal tests?

Rendering: the main order of selfindex.next_token, [ROLE: body] [user: reply request]. Replacement as in selfindex.role_swap:
  - the same text is rendered under the donor role and run unedited (one sequence, no padding);
  - at every layer of --band, its states at the aligned body positions (render.align, restricted to the body) are read in the k
    leading directions of V_l, or of the random R_l, of --fit (fit_m.pt of selfindex.role_swap --make-fits: fitted on the
    discovery bodies, body and end of the message only);
  - the recipient rendering is run with those coordinates written into its own states at the same positions,
    x <- x + (c_donor - x B) B^T. Only the body is edited: an edit after the body cannot change a prediction on the body.
Evaluation: the held-out (confirmation) bodies. Conditions (k = --k, every layer of the band):
  A, U, S       unedited (the donors are captured in these forwards)
  A_U, A_S      the own message given the User / System coordinates (removal)
  U_A, S_A      a User / System message given the own coordinates (installation)
  A_Ur, U_Ar    the same as A_U and U_A in the random subspace
Per condition X and unedited reference Y in {A, U, S}, on the common body positions (predicting body token t+1, median over t):
KL(X || Y), full and content (control tokens removed, each part renormalised, as next_token --measure split); lp_next of the
actual next token; per edited condition the size of the edit (summed squared change and summed squared state over layers and
positions, 'dose').
Summary (summary.json; 15 families resampled 2,000 times, seed 20261108):
  left   KL(A_U||U)/KL(A||U), KL(A_S||S)/KL(A||S), KL(A_Ur||U)/KL(A||U): the share of the own-versus-external difference left
  moved  1 - KL(U_A||A)/KL(U||A), 1 - KL(S_A||A)/KL(S||A), 1 - KL(U_Ar||A)/KL(U||A): the share of the gap to the own message closed
  asym   KL(X||Y)/KL(U||S) for A||U, A||S, A_U||U, A_S||S, A_Ur||U; kl: the mean of every KL; dose: relative size of each edit
Outputs in --out: identity.json, bodies/<body>.json (resumable), summary.json and DONE when every held-out body is on disk.
usage: python -m selfindex.next_token_swap --model DIR --tmpl-kwargs JSON --fit FIT --out DIR [--band A-B] [--k 8] [--lang zh|en]
         [--fixed-date D] [--limit N] [--stop-file F] [--max-memory-gib G] [--cpu-offload-gib C]
       python -m selfindex.next_token_swap --dry-run --model DIR --tmpl-kwargs JSON [--lang L]   (alignment coverage, tokenizer only)
       python -m selfindex.next_token_swap --summarize-only --out DIR
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
from .io import atomic
from .load import checked_tokenizer, find_layers, load_model
from .chat_order import natural_system_ok
from .next_token import control_ids, control_mask, renderings
from .render import align, template_kwargs
from .role_swap import TURN_STARTS

NB = 2000
SEED = 20261108
CONDS = (('A_U', 'A', 'U', 'V'), ('A_S', 'A', 'S', 'V'), ('U_A', 'U', 'A', 'V'), ('S_A', 'S', 'A', 'V'),
         ('A_Ur', 'A', 'U', 'R'), ('U_Ar', 'U', 'A', 'R'))
LEFT = (('A_U', 'A', 'U'), ('A_S', 'A', 'S'), ('A_Ur', 'A', 'U'))      # KL(X||ref) / KL(base||ref)
MOVED = (('U_A', 'U', 'A'), ('S_A', 'S', 'A'), ('U_Ar', 'U', 'A'))     # 1 - KL(X||A) / KL(base||A)


def tuples(tok, R):
    """(ids, body start, body end, generation-prompt start) per role, the form render.align takes."""
    cands = [t for t in (tok.convert_tokens_to_ids(x) for x in TURN_STARTS) if isinstance(t, int) and t >= 0]
    out = {}
    for rn, v in R.items():
        ids, sp = v['ids'], v['span']
        g0 = max(i for i in range(len(ids)) if ids[i] in cands)
        assert g0 > sp[-1], 'no generation prompt after the body'
        out[rn] = (ids, sp[0], sp[-1] + 1, g0)
    return out


def body_pairs(r, d):
    """The aligned recipient and donor positions that lie in the recipient's body."""
    ri, di = align(r, d)
    keep = [j for j, i in enumerate(ri) if r[1] <= i < r[2]]
    assert keep, 'no aligned body position'
    return [ri[j] for j in keep], [di[j] for j in keep]


def summarize(out):
    out = Path(out)
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'bodies').glob('*.json'))]
    fams = sorted({r['category'] for r in rows})
    fidx = np.array([fams.index(r['category']) for r in rows])
    rng = np.random.default_rng(SEED)
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
        res['asym'][part] = {f'{x}||{y}/U||S': ratio_ci(kl(x, y, part), us)
                             for x, y in (('A', 'U'), ('A', 'S'), ('A_U', 'U'), ('A_S', 'S'), ('A_Ur', 'U'))}
    for c in names:
        if 'dose' in rows[0]['conds'][c]:
            e2 = np.array([r['conds'][c]['dose']['edit2'] for r in rows])
            s2 = np.array([r['conds'][c]['dose']['state2'] for r in rows])
            res['dose'][c] = {'rel_size': float(np.sqrt(e2.sum() / s2.sum()))}
    atomic(out / 'summary.json', res)
    short = {part: {**{k: round(v['value'], 3) for k, v in res['left'][part].items()}, **{k: round(v['value'], 3) for k, v in res['moved'][part].items()}}
             for part in ('full', 'content')}
    print('summary:', json.dumps({'left_moved': short, 'dose': {k: round(v['rel_size'], 4) for k, v in res['dose'].items()}}), flush=True)
    return res


def held_out(a):
    held = [b for b in I.bodies(a.lang) if b.get('split') == 'confirmation']
    return held[:a.limit] if a.limit else held


def dry_run(a):
    from transformers import AutoTokenizer
    kw = template_kwargs(a.tmpl_kwargs, a.fixed_date)
    tok = checked_tokenizer(AutoTokenizer.from_pretrained(a.model, trust_remote_code=True), a.model)
    held = held_out(a)
    sys_ok = natural_system_ok(tok, kw, I.reply_request(a.lang), I.natural_order_request(a.lang), held[0]['body'])
    cov = {f'{x}<-{d}': [] for _, x, d, _ in CONDS}
    for b in held:
        T = tuples(tok, renderings(tok, kw, b['body'], a.lang, sys_ok, ('main',))['main'])
        for _, x, d, _ in CONDS:
            ri, _ = body_pairs(T[x], T[d])
            cov[f'{x}<-{d}'].append(len(ri) / (T[x][2] - T[x][1]))
    print(json.dumps({'held_out': len(held), 'aligned_body_share_min': {k: round(min(v), 3) for k, v in cov.items()}}, indent=1))
    print('DRY RUN OK')


def run(a, loaded=None):
    out = Path(a.out)
    (out / 'bodies').mkdir(parents=True, exist_ok=True)
    kw = template_kwargs(a.tmpl_kwargs, a.fixed_date)
    held = held_out(a)
    L_ = loaded or load_model(a.model, max_memory_gib=a.max_memory_gib, cpu_offload_gib=a.cpu_offload_gib)
    torch, tok, model = L_.torch, L_.tok, L_.model
    _, layers = find_layers(model)
    nL = len(layers)
    b0, b1 = (int(x) for x in a.band.split('-')) if a.band else (0, nL - 1)
    band = tuple(range(b0, b1 + 1))
    F = torch.load(a.fit, weights_only=True)
    assert all(l in F['V'] and l in F['R'] for l in band), 'the fit does not cover the band'
    Bs = {w: {l: F[w][l][:, :a.k].contiguous().float() for l in band} for w in ('V', 'R')}
    sys_ok = natural_system_ok(tok, kw, I.reply_request(a.lang), I.natural_order_request(a.lang), held[0]['body'])
    dev0 = model.get_input_embeddings().weight.device
    ctrl = control_ids(tok)
    st = {'mode': None}

    def make_hook(l):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if st['mode'] is None:
                return output
            if st['mode'] == 'capture':  # the donor run: its coordinates in both bases
                pos = torch.tensor(st['pos'], dtype=torch.long, device=h.device)
                x = h[0, pos].float()
                for w in ('V', 'R'):
                    st['coords'][w][l] = (x @ Bs[w][l].to(h.device)).cpu()
            else:
                B = Bs[st['w']][l].to(h.device)
                ri = torch.tensor(st['ri'], dtype=torch.long, device=h.device)
                c = st['c'][l][st['di_rows']].to(h.device)
                x = h[0, ri].float()
                delta = (c - x @ B) @ B.T
                st['edit2'] += float((delta * delta).sum())
                st['state2'] += float((x * x).sum())
                h[0, ri] = (x + delta).to(h.dtype)
            return output
        return hook
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in band]
    atomic(out / 'identity.json', {'experiment': 'P2_modeswitch_swap_20261108', 'selfindex_version': __version__, 'fit': a.fit,
                                   'fit_sha256': hashlib.sha256(Path(a.fit).read_bytes()).hexdigest(), 'k': a.k, 'band': [b0, b1],
                                   'n_layers': nL, 'model': a.model, 'model_class': L_.model_class, 'tmpl_kwargs': json.loads(a.tmpl_kwargs),
                                   'fixed_date': a.fixed_date, 'lang': a.lang, 'n_held_out': len(held), 'conditions': [c[0] for c in CONDS],
                                   'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                                   'device_map': L_.placement, 'n_control_ids': len(ctrl)})
    mask = None

    def fwd(ids):
        nonlocal mask
        x = torch.tensor([ids], dtype=torch.long, device=dev0)
        with torch.no_grad():
            lg = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=0).logits[0].float()
        if mask is None:
            mask = control_mask(torch, tok, ctrl, lg.shape[-1], lg.device)
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
    try:
        for b in held:
            f = out / 'bodies' / f'{b["id"]}.json'
            if f.exists():
                continue
            if a.stop_file and Path(a.stop_file).exists():
                print(f'[stop] {a.stop_file} exists: stopping before {b["id"]}', flush=True)
                break
            R = renderings(tok, kw, b['body'], a.lang, sys_ok, ('main',))['main']
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
    finally:
        st['mode'] = None
        for hd in handles:
            hd.remove()
    n_all = sum(1 for b in held if (out / 'bodies' / f'{b["id"]}.json').exists())
    print(f'finished: {n_all}/{len(held)}', flush=True)
    if n_all == len(held):
        summarize(out)
        (out / 'DONE').write_text(time.strftime('%F %T') + '\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model')
    p.add_argument('--tmpl-kwargs', default='{}', help='chat-template keyword arguments of the rendering (JSON)')
    p.add_argument('--fit', help='fit file with the subspaces V and R (fit_m.pt of selfindex.role_swap --make-fits)')
    p.add_argument('--out')
    p.add_argument('--band', default='', help='first-last layer edited, e.g. 0-63 (default: every layer)')
    p.add_argument('--k', type=int, default=8, help='number of leading directions')
    p.add_argument('--lang', choices=('zh', 'en'), default='zh')
    p.add_argument('--fixed-date', default=None, help='YYYY-MM-DD: templates that write the current date render this date')
    p.add_argument('--limit', type=int, default=0, help='only the first N held-out bodies')
    p.add_argument('--stop-file', default=None, help='stop before the next body if this file exists (a rerun resumes)')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    p.add_argument('--cpu-offload-gib', type=float, default=0, help='allow weights beyond the GPU budget to stay in CPU memory')
    p.add_argument('--dry-run', action='store_true', help='render and align every held-out body with the tokenizer only')
    p.add_argument('--summarize-only', action='store_true', help='rewrite summary.json from the bodies in --out')
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.summarize_only:
        if not a.out:
            p.error('--summarize-only needs --out')
        summarize(Path(a.out))
        return
    if not a.model:
        p.error('--model is required')
    if a.dry_run:
        dry_run(a)
        return
    missing = [k for k in ('fit', 'out') if not getattr(a, k)]
    if missing:
        p.error('missing: ' + ', '.join('--' + k for k in missing))
    try:
        run(a)
    except Exception as exc:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        atomic(Path(a.out) / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
