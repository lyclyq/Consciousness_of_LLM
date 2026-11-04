"""The fits of the causal tests (fit.pt: the self-index subspaces with the User means) and the mean-replacement ablations run with
them (template-native renderings only).

Fit, both modes, on the discovery bodies of selfindex.items (the first two option orders, attribution question): the own message
(Assistant) and the same body as a User message are run unedited; at every fitted layer (the band; in thinking mode also
--main-layer) D = h^A - h^U at the positions from the body start to the second-to-last token (body, end of the message, question
and generation prompt but its last token; the two renderings agree token by token from the body on). V_l = the 64 leading
eigenvectors of D^T D (float32 states, float64 accumulation), R_l = QR of a d x 64 Gaussian (CPU generator, seed 20261022 + l): the
V and R of `selfindex.role_swap --make-fits` fit_q.pt. m_U = the mean of the User states over the edited positions of the User
renderings (from the turn-start token of the message, never position 0, to the turn-start token of the generation prompt,
excluded; thinking mode with --edit-end prompt: to the end of the prompt), and its coordinates mV_l = m_U V_l, mR_l = m_U R_l.
fit.pt holds V, R, mV, mR, n_user_positions and, in thinking mode, n_rows (the rows of D); with --fit-from these are loaded
instead (thinking mode writes them to fit.pt again). --a-mean also writes fit_a_mean.pt: the Assistant mean over the same
positions of the Assistant renderings, mVA and mRA. --mask-extra masks, in the fit forwards, the header tokens that only the
Assistant rendering has (render: the tokens after the role word that the User rendering of the same body does not have, e.g.
gpt-oss <|channel|>final) out of attention, with position ids that follow the mask (cumulative sum minus one), in both
renderings. --fit-chunk N passes the discovery bodies once per group of N layers (host memory) and computes the
eigendecomposition on the device of the input embeddings, as the fits that used it did. --fit-only stops after the fit.
Edit (conditions name:role:V|R|VA|RA|-:k): mean replacement x <- x + (m - x B) B^T in the k leading directions B of V (V, VA) or R
(R, RA), m the User mean (V, R) or the Assistant mean (VA, RA), at every token from the turn-start token of the message under
test (never position 0) to the generation prompt (excluded; thinking --edit-end prompt: to the end of the prompt), prompt only
(thinking --gen-edit: also every generated token).
--mode single (the edit at every layer of --band; --edit-mode remove_all: x <- x - x B B^T at every prompt position instead):
the probabilities of the three option letters at the response onset, both option orders, on the held-out bodies; conditions
base_A, rm64_A, rnd64_A, base_U, rm64_U, rnd64_U, rm8_A, rm1_A (rm: V, rnd: R, the number the rank; _A the own message, _U a User
message). --question content: base_A, rm64_A, rnd64_A with the content question (the last number in the message), greedy, 12
new tokens decoded without special tokens, correct if the first number equals the last number of the body. Outputs: identity.json,
fit.pt, items.json (every item), summary.json (per condition the mean P(the letter of "you") and the smallest letter mass, for an
edited condition the paired difference to base_A / base_U with a 95% bootstrap interval, 5000 resamples, seed 0; or the accuracy).
--mode thinking (greedy generation, --batch items per forward, left padding, at most --max-new new tokens, the answer read after
the last end-of-reasoning marker: the first option letter; the content suite: the first number): --suite smoke: base_A, base_U
and, for the band and for --main-layer, rm64_A, rnd64_A, rnd64_U (names band_* and main_*), on --limit-bodies held-out bodies;
--suite full (the layer set of --layers): base_A, rm64_A, rnd64_A, base_U, rm64_U, rnd64_U on the held-out bodies, then rm8_A and
rm1_A on the first 30 (--secondary-all: on all; the rank-8 and rank-1 conditions and their rank-matched controls rnd8_A, rnd1_A,
rm8_U, rm1_U share these bodies); --suite content: base_A, rm64_A, rnd64_A with the content question, one order. Outputs:
identity.json, fit.pt, items/<cond>__<body id>__o<order>.json (a run resumes from the items on disk), summary.json (per condition
the answers, the share answering "you", the median number of new tokens, the content accuracy; for an edited condition the
share of items answered as in base_A / base_U and the paired difference in "you" answers with a 95% bootstrap interval).
--conds replaces the condition list (thinking: on the layers of --layers). Bodies: those of selfindex.items in list order
(--body-ids FILE: only those listed there), held-out bodies after at most --limit-bodies.
usage: python -m selfindex.fits --mode single --model DIR --lang zh|en --band A-B --out DIR [--on-kwargs JSON] [--question attr|content]
         [--edit-mode mean_msg|remove_all] [--conds ...] [--a-mean] [--fit-from FIT] [--fit-only] [--mask-extra] [--fit-chunk N]
       python -m selfindex.fits --mode thinking --model DIR --lang zh|en --band A-B --main-layer L --out DIR [--on-kwargs JSON]
         [--suite smoke|full|content] [--layers band|main] [--conds ...] [--secondary-all] [--a-mean] [--fit-from FIT] [--fit-only]
         [--edit-end before_gen|prompt] [--gen-edit] [--mask-extra] [--fit-chunk N] [--limit-bodies N] [--batch N] [--max-new N]
         [--save-gen]
       python -m selfindex.fits --summarize-only --out DIR
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import traceback
from collections import Counter
from pathlib import Path

from . import __version__
from . import items as I
from .io import atomic
from .load import find_layers, load_model
from .render import CLOSE, Renderer
from .role_swap import selected_bodies, unpaired

TURN_STARTS = ('<|start|>', '<|im_start|>')
K_MAX = 64
SEED0 = 20261022  # random subspace of layer l: seed SEED0 + l
LETTERS = ('A', 'B', 'C')
SECONDARY = ('rm8_A', 'rm1_A', 'rnd8_A', 'rnd1_A', 'rm8_U', 'rm1_U')  # thinking: on the first 30 held-out bodies
DEFAULT_KWARGS = {'single': '{"enable_thinking": false}', 'thinking': '{}'}


def thinking_conditions(suite, layers_used, band, main):
    if suite == 'content':
        ls = band if layers_used == 'band' else (main,)
        return [('base_A', 'assistant', None), ('rm64_A', 'assistant', ('V', 64, ls)), ('rnd64_A', 'assistant', ('R', 64, ls))]
    if suite == 'smoke':
        c = [('base_A', 'assistant', None), ('base_U', 'user', None)]
        for tag, ls in (('band', band), ('main', (main,))):
            c += [(f'{tag}_rm64_A', 'assistant', ('V', 64, ls)), (f'{tag}_rnd64_A', 'assistant', ('R', 64, ls)),
                  (f'{tag}_rnd64_U', 'user', ('R', 64, ls))]
        return c
    ls = band if layers_used == 'band' else (main,)
    return [('base_A', 'assistant', None), ('rm64_A', 'assistant', ('V', 64, ls)), ('rnd64_A', 'assistant', ('R', 64, ls)),
            ('base_U', 'user', None), ('rm64_U', 'user', ('V', 64, ls)), ('rnd64_U', 'user', ('R', 64, ls)),
            ('rm8_A', 'assistant', ('V', 8, ls)), ('rm1_A', 'assistant', ('V', 1, ls))]


def single_conditions(content):
    if content:
        return [('base_A', 'assistant', None), ('rm64_A', 'assistant', ('V', 64)), ('rnd64_A', 'assistant', ('R', 64))]
    return [('base_A', 'assistant', None), ('rm64_A', 'assistant', ('V', 64)), ('rnd64_A', 'assistant', ('R', 64)),
            ('base_U', 'user', None), ('rm64_U', 'user', ('V', 64)), ('rnd64_U', 'user', ('R', 64)),
            ('rm8_A', 'assistant', ('V', 8)), ('rm1_A', 'assistant', ('V', 1))]


def run(a, loaded=None):
    out = Path(a.out)
    thinking = a.mode == 'thinking'
    (out / 'items').mkdir(parents=True, exist_ok=True) if thinking else out.mkdir(parents=True, exist_ok=True)
    assert not (a.mask_extra and a.a_mean), '--mask-extra applies to the fit forwards only; no --a-mean with it'
    L_ = loaded or load_model(a.model, max_memory_gib=a.max_memory_gib, cpu_offload_gib=a.cpu_offload_gib)
    torch, tok, model = L_.torch, L_.tok, L_.model
    _, layers = find_layers(model)
    nL = len(layers)
    b0, b1 = (int(x) for x in a.band.split('-'))
    band = tuple(range(b0, b1 + 1))
    assert b1 < nL and (not thinking or a.main_layer < nL)
    hook_layers = sorted(set(band) | {a.main_layer}) if thinking else list(band)  # the fitted layers
    lang = a.lang
    R = Renderer(tok, json.loads(a.on_kwargs))
    ORDERS = I.orders()
    rows = selected_bodies(a)
    disc = [b for b in rows if b['split'] == 'discovery']
    held = [b for b in rows if b['split'] == 'confirmation']
    if a.limit_bodies:
        held = held[:a.limit_bodies]
    dev0 = model.get_input_embeddings().weight.device
    cands = [tok.convert_tokens_to_ids(t) for t in TURN_STARTS]
    content = (a.suite == 'content') if thinking else (a.question == 'content')
    end_at_prompt = thinking and a.edit_end == 'prompt'

    def render(role, b, oi, fit=False):
        q = I.content_question(lang) if (content and not fit) else I.attr_question(ORDERS[oi], lang)
        ids, span = R.render([{'role': role, 'content': b['body']}, {'role': 'user', 'content': q}], b['body'])
        starts = [i for i in range(span[0] + 1) if ids[i] in cands]
        assert starts, 'no turn-start token before the body'
        m0 = max(starts[-1], 1)  # the turn-start token of the message under test; position 0 (the sink token) is never edited
        g0 = max(i for i in range(len(ids)) if ids[i] in cands)  # the generation prompt's turn-start token
        assert g0 > span[-1], 'no generation prompt after the body'
        return ids, m0, span[0], (len(ids) if end_at_prompt else g0)  # edited range [m0, end)

    st = {'cap': None, 'edit': None, 'rows': None, 'rng': None}

    def make_hook(l):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if st['cap'] is not None and l in st['cap']:
                st['cap'][l] = h[0].float().cpu()
            e = st['edit']
            if e is None or l not in e:
                return output
            B, m = e[l]
            if not thinking:
                if h.shape[1] > 1:
                    if a.edit_mode == 'remove_all':
                        x = h[0].float()
                        h[0] = (x - (x @ B.to(h.device)) @ B.to(h.device).T).to(h.dtype)
                    else:
                        s, t = st['rng']
                        x = h[0, s:t].float()
                        h[0, s:t] = (x + (m.to(h.device) - x @ B.to(h.device)) @ B.to(h.device).T).to(h.dtype)
                return output
            B, m = B.to(h.device), m.to(h.device)
            if h.shape[1] > 1:  # prefill: from the message start of each row to the end of its edited range
                Lmax = h.shape[1]
                for row, (n, m0, ee) in enumerate(st['rows']):
                    sl = slice(Lmax - n + m0, Lmax - n + ee)
                    x = h[row, sl].float()
                    h[row, sl] = (x + (m - x @ B) @ B.T).to(h.dtype)
            elif a.gen_edit:  # every generated token
                x = h.float()
                h.copy_((x + (m - x @ B) @ B.T).to(h.dtype))
            return output
        return hook
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in hook_layers]

    def fit():
        """V, R, mV, mR, n_rows, n_user_positions on the discovery bodies (see the module docstring)."""
        groups = [hook_layers[c0:c0 + a.fit_chunk] for c0 in range(0, len(hook_layers), a.fit_chunk)] if a.fit_chunk else [hook_layers]
        V, Rr, MV, MR = {}, {}, {}, {}
        for chunk in groups:
            G, SU, NU, n_rows = {}, {}, 0, 0
            with torch.no_grad():
                for b in disc:
                    for oi in (0, 1):
                        caps = {}
                        for role in ('assistant', 'user'):
                            ids, m0, s0, ee = render(role, b, oi, fit=True)
                            st['cap'] = {l: None for l in chunk}
                            x = torch.tensor([ids], device=dev0)
                            am, kw = torch.ones_like(x), {}
                            if a.mask_extra:  # the Assistant-only header tokens out of attention, positions follow the mask
                                if role == 'assistant':
                                    u = render('user', b, oi, fit=True)
                                    z = unpaired((ids, s0), (u[0], u[2]))
                                    assert z, 'no Assistant-only header token to mask'
                                    am[0, z] = 0
                                kw['position_ids'] = (am.cumsum(-1) - 1).clamp(min=0)
                            model(input_ids=x, attention_mask=am, use_cache=False, logits_to_keep=1, **kw)
                            caps[role] = (st['cap'], ids, m0, s0, ee)
                            st['cap'] = None
                        (ca, ia, ma, sa, _), (cu, iu, mu_, su, eu) = caps['assistant'], caps['user']
                        assert ia[sa:] == iu[su:], 'body and later tokens differ between roles'
                        for l in chunk:
                            D = (ca[l][sa:len(ia) - 1] - cu[l][su:len(iu) - 1]).double()
                            G[l] = G.get(l, 0) + D.T @ D
                            SU[l] = SU.get(l, 0) + cu[l][mu_:eu].double().sum(0)
                        n_rows += len(ia) - 1 - sa
                        NU += eu - mu_
            for l in chunk:
                w, E = torch.linalg.eigh(G[l].to(dev0) if a.fit_chunk else G[l])
                V[l] = E[:, -K_MAX:].flip(-1).float().cpu().contiguous()
                del w, E
                g = torch.Generator().manual_seed(SEED0 + l)
                Rr[l] = torch.linalg.qr(torch.randn(G[l].shape[0], K_MAX, generator=g)).Q.float().contiguous()
                mean_u = (SU[l] / NU).float()
                MV[l], MR[l] = mean_u @ V[l], mean_u @ Rr[l]
            if a.fit_chunk:
                del G, SU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f'fit layers {chunk[0]}-{chunk[-1]} done {time.time() - t0:.0f}s', flush=True)
        return V, Rr, MV, MR, n_rows, NU

    t0 = time.time()
    try:
        if a.fit_from:  # the subspaces and User means of an earlier fit (the specificity controls use the main run's)
            F = torch.load(a.fit_from, weights_only=True)
            V, Rr, MV, MR = F['V'], F['R'], F['mV'], F['mR']
            n_rows, NU = (F['n_rows'], F['n_user_positions']) if thinking else (None, None)
            assert all(l in V for l in hook_layers), 'the loaded fit does not cover the layers'
        else:
            V, Rr, MV, MR, n_rows, NU = fit()
            if not thinking:
                torch.save({'V': V, 'R': Rr, 'mV': MV, 'mR': MR, 'n_user_positions': NU}, out / 'fit.pt')
        MVA = MRA = None
        if a.a_mean:  # the Assistant mean over the edited positions (same positions and layers as the User mean)
            SA, NA = {}, 0
            with torch.no_grad():
                for b in disc:
                    for oi in (0, 1):
                        ids, m0, s0, ee = render('assistant', b, oi, fit=True)
                        st['cap'] = {l: None for l in hook_layers}
                        x = torch.tensor([ids], device=dev0)
                        model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
                        for l in hook_layers:
                            SA[l] = SA.get(l, 0) + st['cap'][l][m0:ee].double().sum(0)
                        NA += ee - m0
                        st['cap'] = None
            MVA = {l: (SA[l] / NA).float() @ V[l] for l in hook_layers}
            MRA = {l: (SA[l] / NA).float() @ Rr[l] for l in hook_layers}
            torch.save({'mVA': MVA, 'mRA': MRA, 'n_assistant_positions': NA}, out / 'fit_a_mean.pt')
        if thinking:
            torch.save({'V': V, 'R': Rr, 'mV': MV, 'mR': MR, 'n_rows': n_rows, 'n_user_positions': NU}, out / 'fit.pt')
        means = {'V': MV, 'R': MR, 'VA': MVA, 'RA': MRA}

        def parse_conds(ls):
            conds = []
            for c in a.conds.split(','):
                n, r, w, k = c.split(':')
                assert w in ('-', 'V', 'R', 'VA', 'RA'), f'{n}: unknown basis {w}'
                assert w not in ('VA', 'RA') or a.a_mean, f'{n}: bases VA / RA need --a-mean'
                conds.append((n, r, None if w == '-' else ((w, int(k), ls) if thinking else (w, int(k)))))
            return conds
        if thinking:
            ls = band if a.layers == 'band' else (a.main_layer,)
            conds = parse_conds(ls) if a.conds else thinking_conditions(a.suite, a.layers, band, a.main_layer)
            ident = {'experiment': f'P2_thinking_ablate2_{a.suite}', 'selfindex_version': __version__, 'model': a.model, 'n_layers': nL,
                     'band': [b0, b1], 'main_layer': a.main_layer, 'fit_from': a.fit_from, 'suite': a.suite,
                     'edit_generated_tokens': a.gen_edit, 'edit_end': a.edit_end, 'layers_used': a.layers if a.suite == 'full' else 'both',
                     'lang': lang, 'k_max': K_MAX, 'fit_rows': n_rows, 'n_fit': len(disc), 'n_eval': len(held),
                     'conditions': [c[0] for c in conds], 'batch': a.batch, 'max_new': a.max_new, 'torch': torch.__version__,
                     'transformers': __import__('transformers').__version__, 'fit_s': time.time() - t0, 'on_kwargs': a.on_kwargs,
                     'mask_extra': a.mask_extra, 'fit_chunk': a.fit_chunk, 'body_ids': a.body_ids}
        else:
            conds = parse_conds(None) if a.conds else single_conditions(content)
            ident = {'experiment': 'P2_singlepass_ablate', 'selfindex_version': __version__, 'model': a.model, 'on_kwargs': a.on_kwargs,
                     'band': [b0, b1], 'edit_mode': a.edit_mode, 'question': a.question, 'fit_from': a.fit_from,
                     'conditions': [c[0] for c in conds], 'n_eval': len(held), 'fit_s': time.time() - t0, 'lang': lang,
                     'mask_extra': a.mask_extra, 'fit_chunk': a.fit_chunk, 'body_ids': a.body_ids}
        (out / 'identity.json').write_text(json.dumps(ident, indent=1), encoding='utf-8')
        if thinking:
            print(json.dumps(ident), flush=True)
        if a.fit_only:
            return
        if thinking:
            run_thinking(a, L_, conds, held, render, st, V, Rr, means, ORDERS, out)
        else:
            run_single(a, L_, conds, held, render, st, V, Rr, means, ORDERS, out, content, band)
    finally:
        for hd in handles:
            hd.remove()


def run_single(a, L_, conds, held, render, st, V, Rr, means, ORDERS, out, content, band):
    torch, tok, model = L_.torch, L_.tok, L_.model
    dev0 = model.get_input_embeddings().weight.device
    letters = {x: tok.encode(x, add_special_tokens=False) for x in LETTERS}
    assert all(len(v) == 1 for v in letters.values())
    letters = {x: v[0] for x, v in letters.items()}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0 = time.time()
    recs = []
    for name, role, spec in conds:
        if spec is None:
            st['edit'] = None
        else:
            basis = V if spec[0] in ('V', 'VA') else Rr
            st['edit'] = {l: (basis[l][:, :spec[1]], means[spec[0]][l][:spec[1]]) for l in band}
        for b in held:
            for oi in ((0,) if content else (0, 1)):
                ids, m0, s0, g0 = render(role, b, oi)
                st['rng'] = (m0, g0)
                x = torch.tensor([ids], device=dev0)
                with torch.no_grad():
                    if content:
                        g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=12, do_sample=False, pad_token_id=pad)
                        text = tok.decode(g[0, len(ids):], skip_special_tokens=True)
                        nums = re.findall(r'\d+', text.replace(',', ''))
                        exp = re.findall(r'\d+', b['body'])[-1]
                        rec = {'cond': name, 'id': b['id'], 'text': text, 'expected': exp, 'correct': bool(nums) and nums[0] == exp}
                    else:
                        lp = torch.log_softmax(model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                                                     logits_to_keep=1).logits[0, -1].float(), -1)
                        p = {k: float(lp[letters[LETTERS[i]]].exp()) for i, k in enumerate(ORDERS[oi])}
                        rec = {'cond': name, 'id': b['id'], 'role': role, 'order': list(ORDERS[oi]), 'p': p, 'mass': sum(p.values())}
                recs.append(rec)
        st['edit'] = None
        print(f'{name} done {time.time() - t0:.0f}s', flush=True)
    (out / 'items.json').write_text(json.dumps(recs, ensure_ascii=False), encoding='utf-8')
    summarize_single(out, recs)


def summarize_single(out, recs):
    """summary.json of a single-pass run, the conditions in the order of the items."""
    S = {}
    rng = random.Random(0)
    names = list(dict.fromkeys(r['cond'] for r in recs))
    for name in names:
        rs = [r for r in recs if r['cond'] == name]
        if 'correct' in rs[0]:
            S[name] = {'n': len(rs), 'accuracy': round(sum(r['correct'] for r in rs) / len(rs), 4)}
            continue
        S[name] = {'n': len(rs), 'p_you_mean': round(sum(r['p']['self'] for r in rs) / len(rs), 4),
                   'mass_min': round(min(r['mass'] for r in rs), 4)}
        base = 'base_A' if name.endswith('_A') else 'base_U'
        if name != base and any(r['cond'] == base for r in recs):
            bb = {(r['id'], tuple(r['order'])): r for r in recs if r['cond'] == base}
            d = [r['p']['self'] - bb[(r['id'], tuple(r['order']))]['p']['self'] for r in rs]
            bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
            S[name]['minus_base'] = {'mean': round(sum(d) / len(d), 4), 'ci95': [round(bs[124], 4), round(bs[4874], 4)]}
    (Path(out) / 'summary.json').write_text(json.dumps(S, indent=1), encoding='utf-8')
    print(json.dumps(S), flush=True)


def run_thinking(a, L_, conds, held, render, st, V, Rr, means, ORDERS, out):
    torch, tok, model = L_.torch, L_.tok, L_.model
    dev0 = model.get_input_embeddings().weight.device
    content = a.suite == 'content'

    def edit_for(spec):
        if spec is None:
            return None
        which, k, ls = spec
        basis = V if which in ('V', 'VA') else Rr
        return {l: (basis[l][:, :k], means[which][l][:k]) for l in ls}

    held_sec = held if a.secondary_all else held[:30]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = {tok.eos_token_id} | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    eos.discard(None)
    t0 = time.time()
    for name, role, spec in conds:
        todo = [(b, oi) for b in (held_sec if name in SECONDARY else held) for oi in ((0,) if content else (0, 1))
                if not (out / 'items' / f'{name}__{b["id"]}__o{oi}.json').exists()]
        st['edit'] = edit_for(spec)
        for s0 in range(0, len(todo), a.batch):
            chunk = todo[s0:s0 + a.batch]
            rend = [render(role, b, oi) for b, oi in chunk]
            seqs = [r[0] for r in rend]
            st['rows'] = [(len(r[0]), r[1], r[3]) for r in rend]
            Lmax = max(len(s) for s in seqs)
            x = torch.tensor([[pad] * (Lmax - len(s)) + s for s in seqs], dtype=torch.long, device=dev0)
            am = torch.tensor([[0] * (Lmax - len(s)) + [1] * len(s) for s in seqs], dtype=torch.long, device=dev0)
            with torch.no_grad():
                g = model.generate(input_ids=x, attention_mask=am, max_new_tokens=a.max_new, do_sample=False, pad_token_id=pad)
            for k, (b, oi) in enumerate(chunk):
                new = g[k, Lmax:].tolist()
                n_new = len(new)
                for j, t_ in enumerate(new):
                    if t_ in eos:
                        n_new = j + 1
                        break
                gen = tok.decode(new[:n_new], skip_special_tokens=False)
                tail = gen
                for cl in CLOSE:
                    if cl in gen:
                        tail = gen.split(cl)[-1]
                mm = re.search(r'([ABC])', tail.replace('**', ''))
                letter = mm.group(1) if mm else None
                if content:
                    nums = re.findall(r'\d+', tail.replace(',', ''))
                    expected = re.findall(r'\d+', b['body'])[-1]
                    cres = {'expected': expected, 'given': nums[0] if nums else None, 'correct': bool(nums) and nums[0] == expected}
                else:
                    cres = None
                rec = {'cond': name, 'id': b['id'], 'role': role, 'order': list(ORDERS[oi]), 'spec': spec, 'content': cres,
                       'on_letter': letter, 'on_answer': ORDERS[oi][LETTERS.index(letter)] if letter else None,
                       'finished': any(cl in gen for cl in CLOSE), 'n_new': int(n_new), 'tail': tail[-300:],
                       'gen': gen if a.save_gen else None}
                atomic(out / 'items' / f'{name}__{b["id"]}__o{oi}.json', rec)
            print(f'{name} {s0 + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
        st['edit'] = None
        if todo:
            summarize(out)
    summarize(out)


def summarize(out):
    """summary.json of a thinking-mode run from its items (a single-pass run: from items.json)."""
    out = Path(out)
    if (out / 'items.json').exists() and not (out / 'items').is_dir():
        summarize_single(out, json.loads((out / 'items.json').read_text(encoding='utf-8')))
        return
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    S = {}
    for c in sorted({r['cond'] for r in recs}):
        rs = [r for r in recs if r['cond'] == c]
        fin = [r for r in rs if r['finished']]
        cnt = Counter(r['on_answer'] for r in fin)
        S[c] = {'n': len(rs), 'finished': len(fin), 'answers': dict(cnt), 'you_share': round(cnt.get('self', 0) / max(1, len(fin)), 4),
                'median_new_tokens': sorted(r['n_new'] for r in rs)[len(rs) // 2]}
        if rs and rs[0].get('content'):
            S[c]['content_accuracy'] = round(sum(r['content']['correct'] for r in fin) / max(1, len(fin)), 4)
    by = {}
    for r in recs:
        by.setdefault((r['id'], tuple(r['order'])), {})[r['cond']] = r
    rng = random.Random(0)
    for c in S:
        base = 'base_A' if c.endswith('_A') and c != 'base_A' else ('base_U' if c.endswith('_U') and c != 'base_U' else None)
        if base is None or base not in S:
            continue
        pairs = [(v[c], v[base]) for v in by.values() if c in v and base in v]
        same = sum(x['on_answer'] == y['on_answer'] for x, y in pairs)
        d = [(x['on_answer'] == 'self') - (y['on_answer'] == 'self') for x, y in pairs if x['finished'] and y['finished']]
        S[c]['same_answer_as_base'] = round(same / max(1, len(pairs)), 4)
        if d:
            bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
            S[c]['you_minus_base'] = {'n_pairs': len(d), 'mean': round(sum(d) / len(d), 4), 'ci95': [bs[124], bs[4874]]}
    (out / 'summary.json').write_text(json.dumps(S, indent=1), encoding='utf-8')
    print(json.dumps(S), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--mode', choices=('single', 'thinking'))
    p.add_argument('--model')
    p.add_argument('--on-kwargs', default=None, help='chat-template keyword arguments (JSON; default: {"enable_thinking": false} '
                                                      'in single mode, {} in thinking mode)')
    p.add_argument('--lang', choices=('zh', 'en'))
    p.add_argument('--band', default='', help='first-last layer of the fit and the edits, e.g. 0-63')
    p.add_argument('--out', default='')
    p.add_argument('--fit-from', default='', help='load V, R and the User means from an earlier fit.pt')
    p.add_argument('--fit-only', action='store_true', help='write identity.json and fit.pt (and fit_a_mean.pt) and stop')
    p.add_argument('--conds', default='', help='name:role:V|R|VA|RA|-:k,... replaces the condition list')
    p.add_argument('--a-mean', action='store_true', help='also the Assistant mean (conditions VA / RA: move toward the own message)')
    p.add_argument('--mask-extra', action='store_true', help='fit forwards: mask the Assistant-only header tokens (positions follow)')
    p.add_argument('--fit-chunk', type=int, default=0, help='fit: one pass over the discovery bodies per group of N layers')
    p.add_argument('--body-ids', default='', help='JSON list of body ids (or objects with an id): only these bodies, in this order')
    p.add_argument('--limit-bodies', type=int, default=0, help='the first N held-out bodies only')
    p.add_argument('--question', choices=('attr', 'content'), default='attr', help='single mode: the attribution or the content question')
    p.add_argument('--edit-mode', choices=('mean_msg', 'remove_all'), default='mean_msg',
                   help='single mode: mean replacement on the message and question, or removal of the subspace at every prompt position')
    p.add_argument('--suite', choices=('smoke', 'full', 'content'), default='full', help='thinking mode: the condition list')
    p.add_argument('--main-layer', type=int, default=None, help='thinking mode: the single layer of the main_* conditions')
    p.add_argument('--layers', choices=('band', 'main'), default='band', help='thinking mode: the layer set of the full and content suites')
    p.add_argument('--secondary-all', action='store_true', help='thinking mode: rank-8 and rank-1 conditions on all held-out bodies')
    p.add_argument('--edit-end', choices=('before_gen', 'prompt'), default='before_gen',
                   help='thinking mode: the edited range ends before the generation prompt, or at the end of the prompt')
    p.add_argument('--gen-edit', action='store_true', help='thinking mode: also edit every generated token')
    p.add_argument('--batch', type=int, default=16, help='thinking mode: items per forward')
    p.add_argument('--max-new', type=int, default=3072, help='thinking mode: new tokens at most')
    p.add_argument('--save-gen', action='store_true', help='thinking mode: keep the whole generated text in the item files')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    p.add_argument('--cpu-offload-gib', type=float, default=0, help='allow weights beyond the GPU budget to stay in CPU memory')
    p.add_argument('--summarize-only', action='store_true', help='rewrite summary.json from the items in --out')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.summarize_only:
        if not args.out:
            p.error('--summarize-only needs --out')
        summarize(Path(args.out))
        return
    missing = [k for k in ('mode', 'model', 'lang', 'band', 'out') if not getattr(args, k)]
    if args.mode == 'thinking' and args.main_layer is None:
        missing.append('main_layer')
    if missing:
        p.error('missing: ' + ', '.join('--' + k.replace('_', '-') for k in missing))
    if args.on_kwargs is None:
        args.on_kwargs = DEFAULT_KWARGS[args.mode]
    if args.mode == 'single':
        args.main_layer = None
    try:
        run(args)
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        atomic(Path(args.out) / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
