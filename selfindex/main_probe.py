"""The main probe: geometry and self-reference of one model in one template setting (template-native renderings only).

Every input is the model's own chat-template rendering in the given setting (--tmpl-kwargs); each rendering is checked against
apply_chat_template(tokenize=True). One teacher-forced forward per condition, use_cache=False. Phases (each resumable, results
per item on disk):
  G  geometry and self-reference, all bodies x 3 roles: [ROLE: body][user: reply request] -> residual of every decoder layer,
     mean over the body tokens (identical ids across roles) and at the response onset; [ROLE: body][user: attribution question]
     in three cyclic option orders -> the probabilities of the three option letters at the response onset (after at most two
     format tokens that the model writes before its letter).
  M  reasoning over own records (the 120 two-step problems): [user: problem][ROLE: step][user: final-answer question], ROLE
     assistant or user, step correct or wrong; the correct answer and the error-propagating answer are scored by the summed
     log-probability of their tokens after the generation prompt; adoption index P_adopt / (P_adopt + P_correct).
--geometry-only runs G without the attribution forwards (thinking-mode renderings).
Outputs: identity.json, G/<body>.json + G/<body>.npz (float16 arrays content_<role>, onset_<role>: layers x hidden), M/<problem>.json,
status.json, summary.json (distance ratios per layer, the share of held-out bodies in which the own message is singled out,
direction consistency, mean answer probabilities, reasoning adoption).
usage: python -m selfindex.main_probe --model DIR --tmpl-kwargs JSON --out DIR [--lang zh|en] [--phases GM] [--geometry-only]
       [--max-memory-gib G] [--cpu-offload-gib C] [--disk-offload-dir D] [--sim-fp8 PRESET] [--g-batch N]
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from . import __version__
from . import items as I
from .io import atomic
from .load import find_layers, load_model
from .render import Renderer, common_body

ROLES = I.ROLES
LETTERS = ('A', 'B', 'C')


def run(args, loaded=None):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = json.loads(args.tmpl_kwargs)
    L_ = loaded or load_model(args.model, max_memory_gib=args.max_memory_gib, cpu_offload_gib=args.cpu_offload_gib,
                              disk_offload_dir=args.disk_offload_dir, sim_fp8=args.sim_fp8)
    torch, tok, model, cfg = L_.torch, L_.tok, L_.model, L_.cfg
    lname, layers = find_layers(model)
    L = len(layers)
    R = Renderer(tok, kw)
    letters = {}
    for x in LETTERS:
        t = tok.encode(x, add_special_tokens=False)
        assert len(t) == 1, (x, t)
        letters[x] = t[0]
    lang = args.lang
    bodies = list(I.bodies(lang))
    problems = I.problems(lang)
    orders = I.orders()
    q_reply, q_final = I.reply_request(lang), I.final_answer_question(lang)
    ident = {'experiment': 'P2_crossmodel_suite', 'selfindex_version': __version__, 'model': args.model, 'model_class': L_.model_class,
             'architectures': cfg.architectures, 'layers_module': lname, 'n_layers': L, 'tmpl_kwargs': kw, 'device_map': L_.placement,
             'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
             'cpu_offload_gib': args.cpu_offload_gib, 'geometry_only': args.geometry_only,
             'tokenizer_rebuilt_from': getattr(tok, '_p2_rebuilt_from', None), 'fp8_backend': L_.fp8_backend, 'sim_fp8': L_.sim_fp8,
             'disk_offload_dir': args.disk_offload_dir, 'g_batch': args.g_batch, 'n_bodies': len(bodies), 'n_problems': len(problems),
             'phases': args.phases, 'lang': lang, 'attr_question_example': I.attr_question(orders[0], lang), 'reply_question': q_reply,
             'final_question': q_final, 'problem_example': problems[0]['problem']}
    atomic(out / 'identity.json', ident)
    print(json.dumps({k: ident[k] for k in ('model_class', 'layers_module', 'n_layers')}), flush=True)
    dev0 = model.get_input_embeddings().weight.device
    st = {'mode': None}

    def hook(i):
        def h(mod, inp, outp):
            m = st['mode']
            if m is None:
                return None
            x = outp[0] if isinstance(outp, tuple) else outp
            if m == 'capture':
                st['cap'][i] = x[0].detach()
            elif m == 'capture_batch':
                st['cap'][i] = x.detach()
            return None
        return h
    handles = [layers[i].register_forward_hook(hook(i)) for i in range(L)]

    def fwd(ids, mode=None, keep=1):
        st.clear()
        st.update(mode=mode, cap={})
        x = torch.tensor([ids], dtype=torch.long, device=dev0)
        with torch.no_grad():
            o = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=keep)
        cap = st.get('cap', {})
        st['mode'] = None
        return torch.log_softmax(o.logits[0].float(), -1), cap

    def letter_p(lp, order):
        return {k: float(lp[-1, letters[LETTERS[i]]].exp()) for i, k in enumerate(order)}

    def answer_prefix(ids):
        # format tokens the model writes before its answer letter (greedy, probability > 0.5, at most two)
        pre = []
        for _ in range(2):
            lp, _ = fwd(ids + pre)
            mass = sum(float(lp[-1, letters[x]].exp()) for x in LETTERS)
            top = int(lp[-1].argmax())
            if mass >= 0.5 or float(lp[-1, top].exp()) < 0.5 or top in letters.values():
                return pre, lp
            pre.append(top)
        lp, _ = fwd(ids + pre)
        return pre, lp

    def fwd_capture_batch(seqs):
        # right padding: with causal attention the real tokens never see the pads, so their states equal unpadded ones
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        T = max(len(s) for s in seqs)
        x = torch.tensor([s + [pad] * (T - len(s)) for s in seqs], dtype=torch.long, device=dev0)
        am = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs], dtype=torch.long, device=dev0)
        st.clear()
        st.update(mode='capture_batch', cap={})
        with torch.no_grad():
            model(input_ids=x, attention_mask=am, use_cache=False, return_dict=True, logits_to_keep=1)
        cap = st.get('cap', {})
        st['mode'] = None
        return cap

    t0 = time.time()
    gb = int(args.g_batch or 1)
    if 'G' in args.phases and args.geometry_only and gb > 1:
        # several bodies x 3 roles per forward
        gdir = out / 'G'
        gdir.mkdir(exist_ok=True)
        todo = [b for b in bodies if not (gdir / f'{b["id"]}.json').exists()]
        for c0 in range(0, len(todo), gb):
            group = todo[c0:c0 + gb]
            seqs, meta = [], []
            for b in group:
                geo = {r: R.render([{'role': r, 'content': b['body']}, {'role': 'user', 'content': q_reply}], b['body']) for r in ROLES}
                spans = common_body(geo)
                for r in ROLES:
                    meta.append((b['id'], r, spans[r], len(geo[r][0])))
                    seqs.append(geo[r][0])
            cap = fwd_capture_batch(seqs)
            for b in group:
                arrs, rec = {}, {'id': b['id'], 'split': b['split'], 'category': b['category'], 'roles': {}, 'g_batch': gb}
                for j, (bid, r, pos, n) in enumerate(meta):
                    if bid != b['id']:
                        continue
                    arrs[f'content_{r}'] = torch.stack([cap[i][j, pos].float().mean(0).to(dev0) for i in range(L)]).half().cpu().numpy()
                    arrs[f'onset_{r}'] = torch.stack([cap[i][j, n - 1].float().to(dev0) for i in range(L)]).half().cpu().numpy()
                    rec['roles'][r] = {'n_body_tokens': len(pos), 'n_tokens': n}
                np.savez_compressed(gdir / f'{b["id"]}.npz', **arrs)
                atomic(gdir / f'{b["id"]}.json', rec)
            print(f'G {c0 + len(group)}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    elif 'G' in args.phases:
        gdir = out / 'G'
        gdir.mkdir(exist_ok=True)
        for bi, b in enumerate(bodies):
            done = gdir / f'{b["id"]}.json'
            if done.exists():
                continue
            geo = {r: R.render([{'role': r, 'content': b['body']}, {'role': 'user', 'content': q_reply}], b['body']) for r in ROLES}
            spans = common_body(geo)
            arrs, rec = {}, {'id': b['id'], 'split': b['split'], 'category': b['category'], 'roles': {}}
            for r in ROLES:
                ids = geo[r][0]
                _, cap = fwd(ids, 'capture')
                pos = spans[r]
                arrs[f'content_{r}'] = torch.stack([cap[i][pos].float().mean(0).to(dev0) for i in range(L)]).half().cpu().numpy()
                arrs[f'onset_{r}'] = torch.stack([cap[i][len(ids) - 1].float().to(dev0) for i in range(L)]).half().cpu().numpy()
                if args.geometry_only:
                    rec['roles'][r] = {'n_body_tokens': len(pos), 'n_tokens': len(ids)}
                    continue
                ps = []
                for order in orders:
                    aid, _ = R.render([{'role': r, 'content': b['body']}, {'role': 'user', 'content': I.attr_question(order, lang)}], b['body'])
                    if args.readout == 'v1':   # the first version: letters read at the response onset, no format tokens
                        lp, _ = fwd(aid)
                        ps.append(letter_p(lp, order))
                        continue
                    pre, lp = answer_prefix(aid)
                    ps.append({**letter_p(lp, order), 'prefix': tok.decode(pre)})
                rec['roles'][r] = {'by_order': ps, 'mean': {k: float(np.mean([p[k] for p in ps])) for k in ('self', 'user', 'system')},
                                   'letter_mass_min': float(min(sum(p[k] for k in ('self', 'user', 'system')) for p in ps)), 'n_body_tokens': len(pos)}
            np.savez_compressed(gdir / f'{b["id"]}.npz', **arrs)
            atomic(done, rec)
            if bi % 10 == 0:
                print(f'G {bi + 1}/{len(bodies)} {time.time() - t0:.0f}s', flush=True)
    if 'M' in args.phases:
        mdir = out / 'M'
        mdir.mkdir(exist_ok=True)

        def digit_prefix(ids):
            pre = []
            for _ in range(2):
                lp, _ = fwd(ids + pre)
                top = int(lp[-1].argmax())
                if float(lp[-1, top].exp()) < 0.5 or tok.decode([top]).strip()[:1].isdigit():
                    break
                pre.append(top)
            return pre

        def score(ids, cand):
            cid = tok.encode(cand, add_special_tokens=False)
            lp, _ = fwd(ids + cid, keep=len(cid) + 1)
            return float(sum(lp[j, cid[j]] for j in range(len(cid))))
        for pi, p in enumerate(problems):
            done = mdir / f'{p["id"]}.json'
            if done.exists():
                continue
            rec = {'id': p['id'], 'family': p['family'], 'split': p['split'], 'results': []}
            for which in ('correct', 'wrong'):
                for r in ('assistant', 'user'):
                    step = p[f'step_{which}']
                    ids, _ = R.render([{'role': 'user', 'content': p['problem']}, {'role': r, 'content': step}, {'role': 'user', 'content': q_final}], step)
                    if args.readout != 'v1':
                        ids = ids + digit_prefix(ids)
                    lc, la = score(ids, str(p['answer'])), score(ids, str(p['answer_if_adopted']))
                    rec['results'].append({'step': which, 'role': r, 'lp_correct': lc, 'lp_adopt': la,
                                           'adoption_index': 1 / (1 + math.exp(lc - la))})
            atomic(done, rec)
            if pi % 20 == 0:
                print(f'M {pi + 1}/{len(problems)} {time.time() - t0:.0f}s', flush=True)
    for h in handles:
        h.remove()
    atomic(out / 'status.json', {'state': 'complete', 'elapsed_s': time.time() - t0})
    summarize(out, L)


def summarize(out, L):
    S = {'n_layers': L}
    gdir = out / 'G'
    if gdir.exists():
        recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted(gdir.glob('*.json'))]
        # hyper-connection models (GLM-5.3-Flash) store each layer as [streams, hidden]: the full concatenated state is used
        V = {}
        for r in recs:
            with np.load(gdir / f'{r["id"]}.npz') as z:
                V[r['id']] = {k: (z[k].reshape(z[k].shape[0], -1) if z[k].ndim == 3 else z[k]) for k in z.files}
        held = [r for r in recs if r['split'] == 'confirmation']
        disc = [r for r in recs if r['split'] == 'discovery']
        for kind in ('content', 'onset') if held and disc else ():
            Rs, cos = [], []
            dirs = np.mean([V[r['id']][f'{kind}_assistant'].astype(np.float32) - (V[r['id']][f'{kind}_user'].astype(np.float32) + V[r['id']][f'{kind}_system'].astype(np.float32)) / 2 for r in disc], 0)
            for r in held:
                a, u, s = (V[r['id']][f'{kind}_{x}'].astype(np.float32) for x in ROLES)
                au, as_, us = np.linalg.norm(a - u, axis=-1), np.linalg.norm(a - s, axis=-1), np.linalg.norm(u - s, axis=-1)
                Rs.append([us / ((au + as_) / 2), as_ / ((au + us) / 2), au / ((as_ + us) / 2)])
                off = a - (u + s) / 2
                cos.append((off * dirs).sum(-1) / (np.linalg.norm(off, axis=-1) * np.linalg.norm(dirs, axis=-1) + 1e-9))
            Rs = np.array(Rs)
            low = Rs.argmin(1)
            S[kind] = {'mean_R': Rs.mean(0).tolist(), 'share_A_lowest': (low == 0).mean(0).tolist(),
                       'layers_A_lowest_all_bodies': int(((low == 0).mean(0) == 1).sum()), 'layers_A_lowest_80': int(((low == 0).mean(0) >= 0.8).sum()),
                       'direction_cos_mean': np.mean(cos, 0).tolist()}
        if held and disc and all('mean' in r['roles'][r0] for r in recs for r0 in ROLES):
            S['attr'] = {r0: {k: float(np.mean([r['roles'][r0]['mean'][k] for r in recs])) for k in ('self', 'user', 'system')} for r0 in ROLES}
            S['attr_top_share'] = {r0: float(np.mean([max(r['roles'][r0]['mean'], key=r['roles'][r0]['mean'].get) == {'assistant': 'self', 'user': 'user', 'system': 'system'}[r0] for r in recs])) for r0 in ROLES}
            S['letter_mass_min'] = float(min(r['roles'][r0]['letter_mass_min'] for r in recs for r0 in ROLES))
    mdir = out / 'M'
    if mdir.exists() and any(mdir.glob('*.json')):
        rs = [x for f in sorted(mdir.glob('*.json')) for x in json.loads(f.read_text(encoding='utf-8'))['results']]
        S['reasoning'] = {f'{w}_{r}': float(np.mean([x['adoption_index'] for x in rs if x['step'] == w and x['role'] == r])) for w in ('correct', 'wrong') for r in ('assistant', 'user')}
    (out / 'summary.json').write_text(json.dumps(S, indent=1), encoding='utf-8')
    q = [0, L // 4, L // 2, 3 * L // 4, L - 1]
    for kind in ('content', 'onset'):
        if kind in S:
            print(f'{kind}: R_A at {q}', [round(S[kind]["mean_R"][0][i], 3) for i in q], '| layers A lowest in all held-out bodies',
                  S[kind]['layers_A_lowest_all_bodies'], '/ >=80%', S[kind]['layers_A_lowest_80'], '| dir cos', [round(S[kind]['direction_cos_mean'][i], 2) for i in q])
    if 'attr' in S:
        print('answers:', {r: {k: round(v, 3) for k, v in d.items()} for r, d in S['attr'].items()}, '| top-answer correct', {r: round(v, 3) for r, v in S['attr_top_share'].items()})
    if 'reasoning' in S:
        print('reasoning adoption:', {k: round(v, 3) for k, v in S['reasoning'].items()})


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', required=True)
    p.add_argument('--tmpl-kwargs', default='{}')
    p.add_argument('--out', required=True)
    p.add_argument('--lang', choices=('zh', 'en'), default='zh')
    p.add_argument('--phases', default='GM')
    p.add_argument('--geometry-only', action='store_true', help='phase G without the attribution forwards (thinking-mode renderings)')
    p.add_argument('--g-batch', type=int, default=1, help='bodies per forward in a geometry-only G phase')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    p.add_argument('--cpu-offload-gib', type=float, default=0, help='allow weights beyond the GPU budget to stay in CPU memory')
    p.add_argument('--disk-offload-dir', default=None, help='allow accelerate disk offload into this folder')
    p.add_argument('--sim-fp8', default=None, help='simulate FP8 W8A8 inference on a BF16 model for this layer preset (patches.fp8_fakequant)')
    p.add_argument('--readout', choices=('prefix', 'v1'), default='prefix',
                   help="prefix: the option letters (and the digits of the reasoning problems) are read after at most two format "
                        "tokens that the model writes first; v1: read directly at the response onset, as the first version of the "
                        "measurement (the earliest runs of the cross-model comparison)")
    p.add_argument('--summarize-only', type=int, default=0, help='rewrite summary.json from the outputs in --out for this many layers')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.summarize_only:
        summarize(Path(args.out), args.summarize_only)
        return
    try:
        run(args)
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        atomic(Path(args.out) / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
