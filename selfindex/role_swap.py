"""Role-coordinate replacement: the causal tests of the self-index (template-native renderings only).

Per item, the message body is first rendered under the DONOR role and run unedited, one sequence at a time (no padding, so its
positions match the recipient's); at every layer of --band its states at the aligned positions (the body and the template tokens
after it up to the generation prompt, matched token by token, render.align) are read in a basis B of k directions. The RECIPIENT
rendering is then run with those coordinates written into its own states at the same aligned positions, x <- x + (c_donor - x B) B^T,
at every layer of the band, prompt only: the role header, the generation prompt and, in thinking mode, the generated tokens are
never edited. Every position receives the donor's coordinates at that position (one average written at every position would
flatten the position-specific index instead of writing it). Removal acts on the model's own message; a User or System message is
only given the own coordinates (sufficiency), never "removed": there is no self-index there.
Conditions (--conds, comma-separated name:recipient:donor:basis:k[:TARGET]), e.g.
  base_A:assistant:-:-:0      the own message, unedited (base_U, base_S: the User and System messages)
  swU8_A:assistant:user:V:8   the own message given the User coordinates (removal; inside/outside: compare swS8_A)
  swS8_A:assistant:system:V:8 the own message given the System coordinates
  swA8_U:user:assistant:V:8   a User message given the own coordinates (installation); swA8_S the same for a System message
  swS8_U:user:system:V:8      a User message given the System coordinates (one external source told from the other)
  rswU8_A:assistant:user:R:8  the same copy in a random 8-dim subspace (control)
Bases:
  V     the k leading directions of the fitted self-index subspace V_l of --fit-from
  R     the k leading directions of the random subspace R_l of the same file
  C     the complement of the subspace: the recipient state x takes the donor state y outside the k directions of V and keeps its
        own coordinates inside them, x <- y + (xB - yB)B^T; if the readout of the source went through the residual outside the
        subspace rather than through it, this edit would move the claim
  F     the full state, x <- y (all coordinates), the reference for V and C
  P, M  one fixed unit direction U_l per layer from --content-fit (a combination of the leading principal directions of the message
        states, orthogonal to the self-index subspace), added at every edited position, +U_l for P and -U_l for M, with a length
        that makes its realized relative size at each layer equal to that of the V edit of condition TARGET on the same item in
        --dose-floor (name:recipient:donor:P:8:TARGET; the donor role only aligns the positions): an edit of the same size along
        a relevant direction that is not the source
  An R condition with a TARGET has its random edit scaled up, layer by layer and item by item, to at least the realized relative
  size of the V edit of TARGET on the same item in --dose-floor DIR (a run with doses): with e2 = sum ||delta||^2 and
  s2 = sum ||x||^2 over the edited positions, delta <- max(1, sqrt(rho * s2 / e2)) * delta, rho = edit2 / state2 of TARGET at that
  layer. The direction and the positions are those of the random edit; only its size changes, and never downward. --floor-order O
  takes the TARGET item of option order O, and --floor-ratio-only matches the relative size without requiring the same number of
  edited positions (a question rendered differently from TARGET's).
Every edited item records, per layer of the band, the summed squared change and the summed squared state over the edited positions
('dose', the realized size of the edit), and a scaled item its factor per layer ('floor'; for P and M the length). --no-dose leaves
the dose out, as in the item files of the first runs of the causal tests.
Readouts (--mode): single: one forward per item, the probabilities of the answer tokens at the response onset; thinking: greedy
generation (--batch items per forward, left padding, at most --max-new new tokens), the answer read after the last end-of-reasoning
marker. Questions (--question, built by selfindex.items):
  attr        who wrote the message before this question (three options, two cyclic option orders)
  attr_alt    the same in another wording, same options, letters and orders (held-out wording)
  attr_num    the options numbered 1-3 in the third cyclic order, which the causal tests never used otherwise (held-out mapping)
  yn_self, yn_user  "did you / did the user write the message above?", answered yes or no; the two orders of "yes or no" are the
              two orders (polarity). Single pass: the summed probability of the single-token yes and no words
  content     the last number in the message (specificity: a question that does not concern the source); arith: an unrelated
              two-digit multiplication; both one order, scored as a number answer (generated, 12 new tokens in single mode)
Regions (--region): all aligned positions, or those in the body, in the template tokens between the body and the question text
(end of the message), in the question text up to the generation prompt, or in body and end together (message: then neither the
fit fit_m.pt nor the edit sees the question).
--mask-extra masks the Assistant-only header tokens of the message (the header tokens that the User rendering of the same body does
not have) out of attention, in the donor and the recipient run; position ids follow the mask, so their position offset disappears
too. --pos-compress (single pass, unedited Assistant conditions) keeps those tokens visible but gives them the position of the
token before them and moves every later token back by their number, so the body has the User rendering's positions.
--make-fits DIR fits the subspaces on the discovery bodies (attribution question, both option orders; float32 states, float64
accumulation): V_l = the 64 leading eigenvectors of D^T D, D = h^assistant - h^user at aligned positions, at every layer of --band,
for two position sets: fit_q.pt from the body start to the second-to-last token (body, end of message, question and generation
prompt but its last token: the positions of the paper's fits) and fit_m.pt from the body start to the token before the question
text (body and end of message only). Random subspaces R_l = QR of a d x 64 Gaussian (CPU generator, seed S + l) for five
consecutive base seeds S = 20261022 (the paper's) to 20261026; fit_q.pt and fit_m.pt carry S = 20261022, fit_seed1.pt ..
fit_seed4.pt the V of fit_q.pt and S = 20261022 + j. Consecutive bases share draws across layers (layer l of S+1 uses the matrix of
layer l+1 of S); at any one layer the five draws are distinct. fit_diag.json: rows, the energy share of the 8 leading directions in
each fit, the overlap of the 8-dim spaces of fit_q and fit_m (mean squared cosine of the principal angles), and with --compare-fit
FIT the same overlap with an earlier fit and max |R - R_earlier|.
--check-align (tokenizer only) prints the alignment coverage of every held-out body under every recipient/donor pair of --conds.
Bodies: the held-out (confirmation) bodies of selfindex.items, in list order (--body-ids FILE: only the bodies listed there, in
its order), after --exclude-first, at most --limit-bodies.
Outputs in --out: identity.json, items/<cond>__<body id>__o<order>.json (one per item; a run resumes from the items on disk),
summary.json (per condition: the mean answer probabilities, or the answer counts, the share answering "you" / "yes" and the median
number of new tokens, or the content accuracy; for an edited condition the paired difference to the unedited condition of the same
recipient, base_A / base_U / base_S, with a 95% bootstrap interval, 5000 resamples, seed 0).
usage: python -m selfindex.role_swap --mode single|thinking --model DIR --on-kwargs JSON --lang zh|en --fit-from FIT --band A-B
         --conds ... --out DIR [--question Q] [--region R] [--limit-bodies N] [--batch N] [--max-new N] [--save-gen]
         [--dose-floor DIR [--floor-order O] [--floor-ratio-only]] [--content-fit FILE] [--mask-extra | --pos-compress] [--no-dose]
       python -m selfindex.role_swap --make-fits DIR --mode M --model DIR --on-kwargs JSON --lang zh|en --band A-B [--compare-fit FIT]
       python -m selfindex.role_swap --check-align --mode M --model DIR --on-kwargs JSON --lang zh|en --conds ...
       python -m selfindex.role_swap --summarize-only --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import traceback
from collections import Counter
from pathlib import Path

from . import __version__
from . import items as I
from .io import atomic
from .load import checked_tokenizer, find_layers, load_model
from .render import CLOSE, Renderer, align

QUESTIONS = ('attr', 'content', 'arith', 'attr_alt', 'attr_num', 'yn_self', 'yn_user')
ATTR_KIND = {'attr': 'attribution', 'attr_alt': 'attribution_other_wording', 'attr_num': 'attribution_numbered'}
REGIONS = ('all', 'body', 'end', 'question', 'message')
BASES = ('V', 'R', 'C', 'F', 'P', 'M')
TURN_STARTS = ('<|start|>', '<|im_start|>')  # the generation prompt begins at the last of these tokens
K_MAX = 64
SEED0 = 20261022  # base seed of the random subspaces


def parse_yn(tail, lang):
    """The first yes or no in a generated answer."""
    t = tail.replace('**', '')
    if lang == 'zh':
        hits = [(t.find(w), v) for w, v in (('不是', 'no'), ('否', 'no'), ('是', 'yes')) if t.find(w) >= 0]
        return min(hits)[1] if hits else None
    m = re.search(r'\b(yes|no)\b', t, re.I)
    return m.group(1).lower() if m else None


def assistant_only_header(r, d):
    """(start, number) of the header tokens of rendering r (the body under the Assistant role) that rendering d (the same body under
    the User role) does not have: after the role word, minus the tokens both headers share at their start and end. r and d are
    (ids, body start, ...)."""
    ri, rs, di, ds = r[0], r[1], d[0], d[1]
    p = 0
    while p < min(rs, ds) and ri[p] == di[p]:
        p += 1
    ah, uh = ri[p + 1:rs], di[p + 1:ds]
    i = 0
    while i < min(len(ah), len(uh)) and ah[i] == uh[i]:
        i += 1
    j = 0
    while j < min(len(ah), len(uh)) - i and ah[-1 - j] == uh[-1 - j]:
        j += 1
    return p + 1 + i, len(ah) - j - i


def unpaired(r, d):
    """--mask-extra: the positions of the Assistant-only header tokens of r."""
    z0, k = assistant_only_header(r, d)
    return list(range(z0, z0 + k))


def compressed(r, d):
    """--pos-compress: position ids of r with its Assistant-only header tokens at the position of the token before them and every
    later token moved back by their number."""
    z0, k = assistant_only_header(r, d)
    assert k > 0, 'no Assistant-only header token'
    pos = [t if t < z0 else (z0 - 1 if t < z0 + k else t - k) for t in range(len(r[0]))]
    assert pos[r[1]] == d[1], 'compressed body start differs from the User rendering'
    return pos


def selected_bodies(a):
    """The bodies of the run in list order, or those of --body-ids in its order (a JSON list of ids or of objects with an id)."""
    rows = I.bodies(a.lang)
    if not getattr(a, 'body_ids', ''):
        return list(rows)
    by = {b['id']: b for b in rows}
    sel = []
    for x in json.loads(Path(a.body_ids).read_text(encoding='utf-8')):
        b = by[x['id'] if isinstance(x, dict) else x]
        assert not isinstance(x, dict) or 'body' not in x or x['body'] == b['body'], f'{b["id"]}: body text differs from the items'
        sel.append(b)
    return sel


def held_out(a):
    held = [b for b in selected_bodies(a) if b['split'] == 'confirmation']
    if a.exclude_first:
        held = held[a.exclude_first:]
    if a.limit_bodies:
        held = held[:a.limit_bodies]
    return held


def parse_conds(a, has_content_fit):
    conds = []
    for c in a.conds.split(','):
        f = c.split(':')
        n, rr, dr, w, k = f[:5]
        tgt = f[5] if len(f) > 5 else None
        assert w == '-' or w in BASES, f'{n}: unknown basis {w}'
        assert tgt is None or (w in ('R', 'P', 'M') and a.dose_floor), 'a dose-floor target needs basis R, P or M and --dose-floor'
        assert w not in ('P', 'M') or (tgt and has_content_fit), 'basis P/M needs a TARGET and --content-fit'
        conds.append((n, rr, None if dr == '-' else dr, None if w == '-' else w, int(k), tgt))
    return conds


def run(a, loaded=None):
    out = Path(a.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    assert not (a.mask_extra and a.pos_compress), '--mask-extra and --pos-compress exclude each other'
    assert not a.pos_compress or a.mode == 'single', '--pos-compress: single pass only'
    L_ = loaded or load_model(a.model, max_memory_gib=a.max_memory_gib, cpu_offload_gib=a.cpu_offload_gib)
    torch, tok, model = L_.torch, L_.tok, L_.model
    _, layers = find_layers(model)
    nL = len(layers)
    b0, b1 = (int(x) for x in a.band.split('-'))
    band = tuple(range(b0, b1 + 1))
    assert b1 < nL
    lang, kind = a.lang, a.question
    R = Renderer(tok, json.loads(a.on_kwargs))
    held = held_out(a)
    dev0 = model.get_input_embeddings().weight.device
    cands = [tok.convert_tokens_to_ids(t) for t in TURN_STARTS]
    T = I.templates()
    ORDERS = I.orders()
    content = kind in ('content', 'arith')  # both are scored as a number answer, one option order
    yn = kind in ('yn_self', 'yn_user')
    labels = tuple(T['attribution_numbered']['digits'] if kind == 'attr_num' else T['attribution']['letters'])
    label_re = '([' + ''.join(labels) + '])'
    if a.mode == 'single' and not content:
        if yn:
            enc = lambda v: tok.encode(v, add_special_tokens=False)  # noqa: E731
            yn_ids = {k: sorted({enc(v)[0] for v in vs if len(enc(v)) == 1}) for k, vs in T['yes_no']['answer_tokens'][lang].items()}
            assert all(yn_ids.values()), 'single-token yes/no answers needed'
        else:
            letters = {x: tok.encode(x, add_special_tokens=False) for x in labels}
            assert all(len(v) == 1 for v in letters.values())
            letters = {x: v[0] for x, v in letters.items()}

    def question_text(b, oi):
        if kind == 'arith':
            return I.arithmetic_question(b['id'], lang)[0]
        if kind == 'content':
            return I.content_question(lang)
        if yn:
            return I.yes_no_question(kind, lang, oi)
        return I.attr_question(ORDERS[oi], lang, ATTR_KIND[kind])

    def render(role, b, oi):
        q = question_text(b, oi)
        msgs = [{'role': role, 'content': b['body']}, {'role': 'user', 'content': q}]
        ids, span = R.render(msgs, b['body'])
        g0 = max(i for i in range(len(ids)) if ids[i] in cands)  # the generation prompt's turn-start token
        assert g0 > span[-1], 'no generation prompt after the body'
        qs = R.render(msgs, q)[1][0] if a.region != 'all' else None  # first token of the question text
        assert qs is None or span[-1] < qs < g0
        return ids, span[0], span[-1] + 1, g0, qs  # body [s, e); the edited range ends before the generation prompt g0

    def in_region(i, r):
        s, e, g0, qs = r[1], r[2], r[3], r[4]
        return {'all': True, 'body': s <= i < e, 'end': e <= i < qs, 'question': qs is not None and qs <= i < g0,
                'message': s <= i < qs}[a.region]

    F = torch.load(a.fit_from, weights_only=True)
    V, Rr = F['V'], F['R']
    assert all(l in V for l in band), 'the loaded fit does not cover the band'
    U = torch.load(a.content_fit, weights_only=True)['U'] if a.content_fit else None
    st = {'mode': None, 'B': None, 'rows': None, 'coords': None, 'dose': None, 'floor': None, 'scale': None, 'basis': None}

    def make_hook(l):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if st['mode'] is None or h.shape[1] == 1 or l not in st['B']:
                return output
            B = st['B'][l].to(h.device)
            Lmax = h.shape[1]
            basis = st['basis']
            for row, (n, pos) in enumerate(st['rows']):
                sl = torch.tensor(pos, dtype=torch.long, device=h.device) + (Lmax - n)
                x = h[row, sl].float()
                if st['mode'] == 'capture':  # the donor run: its coordinates in B (C and F: its whole state, model dtype)
                    st['coords'][row][l] = h[row, sl].clone() if basis in ('C', 'F') else (x @ B).clone()
                    continue
                if basis in ('P', 'M'):  # a fixed direction at the realized relative size of the TARGET V edit
                    amp = math.sqrt(st['floor'][row][l] * float((x * x).sum()) / x.shape[0])
                    delta = (amp if basis == 'P' else -amp) * B[:, 0].float().expand_as(x)
                    st['scale'][row][l] = amp
                else:
                    c = st['coords'][row][l].to(h.device)
                    if basis == 'F':  # the donor state, all coordinates
                        delta = c.float() - x
                    elif basis == 'C':  # the donor state outside the subspace, the recipient's coordinates inside it
                        e = c.float() - x
                        delta = e - (e @ B) @ B.T
                    else:
                        delta = (c - x @ B) @ B.T
                    if st['floor'] is not None and l in st['floor'][row]:  # --dose-floor: at least the V edit's size here
                        e2, s2 = float((delta * delta).sum()), float((x * x).sum())
                        f = max(1.0, math.sqrt(st['floor'][row][l] * s2 / e2)) if e2 > 0 else 1.0
                        st['scale'][row][l] = f
                        delta = delta * f
                if st['dose'] is not None:  # the realized size of the edit at this layer
                    st['dose'][row][l] = (float((delta * delta).sum()), float((x * x).sum()))
                h[row, sl] = (x + delta).to(h.dtype)
            return output
        return hook

    conds = parse_conds(a, U is not None)

    def floor_of(tgt, b, oi, n_pos):  # relative size per layer of the V edit of condition tgt on this item, in --dose-floor
        ot = oi if a.floor_order < 0 else a.floor_order
        rec = json.loads((Path(a.dose_floor) / 'items' / f'{tgt}__{b["id"]}__o{ot}.json').read_text(encoding='utf-8'))
        assert 'dose' in rec, f'{tgt} {b["id"]} o{ot}: no dose recorded in --dose-floor (a run with --no-dose?)'
        D = rec['dose']
        assert a.floor_ratio_only or D['n_pos'] == n_pos, f'{tgt} {b["id"]} o{ot}: {D["n_pos"]} edited positions there, {n_pos} here'
        return {l: e / s for l, e, s in zip(D['layers'], D['edit2'], D['state2'])}
    ident = {'experiment': f'P2_role_swap_{a.mode}', 'selfindex_version': __version__, 'model': a.model, 'on_kwargs': a.on_kwargs,
             'n_layers': nL, 'band': [b0, b1], 'fit_from': a.fit_from, 'question': a.question, 'region': a.region, 'lang': lang,
             'n_eval': len(held), 'exclude_first': a.exclude_first, 'body_ids': a.body_ids, 'conditions': [c[0] for c in conds],
             'batch': a.batch, 'dose_floor': a.dose_floor, 'floor_targets': {c[0]: c[5] for c in conds if c[5]},
             'content_fit': a.content_fit, 'floor_order': a.floor_order, 'floor_ratio_only': a.floor_ratio_only,
             'record_dose': not a.no_dose, 'mask_extra': a.mask_extra, 'pos_compress': a.pos_compress, 'max_new': a.max_new,
             'torch': torch.__version__, 'transformers': __import__('transformers').__version__}
    (out / 'identity.json').write_text(json.dumps(ident, indent=1), encoding='utf-8')
    print(json.dumps(ident), flush=True)

    def donor_coords(drole, b, oi, Bk, r):
        d = render(drole, b, oi)
        ri, di = align(r[:4], d[:4])
        if a.region != 'all':  # keep the aligned pairs whose recipient position lies in the region
            keep = [j for j, i in enumerate(ri) if in_region(i, r)]
            assert keep, f'no aligned position in region {a.region}'
            ri, di = [ri[j] for j in keep], [di[j] for j in keep]
        if st['basis'] in ('P', 'M'):  # no coordinates are copied: the donor only aligns the positions
            return {}, ri
        st.update(mode='capture', B=Bk, rows=[(len(d[0]), di)], coords=[{}])
        x = torch.tensor([d[0]], device=dev0)
        amd = torch.ones_like(x)
        kw = {}
        if a.mask_extra:
            if drole == 'assistant':
                amd[0, unpaired(d, render('user', b, oi))] = 0
            kw['position_ids'] = (amd.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            model(input_ids=x, attention_mask=amd, use_cache=False, logits_to_keep=1, **kw)
        cd = st['coords'][0]
        st.update(mode=None, coords=None)
        return cd, ri

    def single_pass(x, am, chunk, rend, rrole, Bk):
        kw = {}
        if a.pos_compress and rrole == 'assistant':
            assert len(chunk) == 1 and Bk is None, 'pos-compress: unedited conditions, one item per forward'
            (b_, oi_), r_ = chunk[0], rend[0]
            pt = torch.tensor([compressed(r_, render('user', b_, oi_))], dtype=torch.long, device=dev0)
            if getattr(model.config, 'model_type', '') in ('qwen3_5', 'qwen3_5_moe'):  # text + three rotary position rows
                pt = pt.view(1, 1, -1).expand(4, 1, -1)
            kw['position_ids'] = pt
        elif a.mask_extra:
            kw['position_ids'] = (am.cumsum(-1) - 1).clamp(min=0)
        o = model(input_ids=x, attention_mask=am, use_cache=False, logits_to_keep=1, **kw)
        return torch.log_softmax(o.logits[:, -1].float(), -1)

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = {tok.eos_token_id} | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    eos.discard(None)
    if content:
        orders = (0,)
    elif kind == 'attr_num':
        orders = (ORDERS.index(tuple(T['attribution_numbered']['order'])),)
    else:
        orders = (0, 1)
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in band]
    t0 = time.time()
    try:
        for name, rrole, drole, which, k, tgt in conds:
            if which in ('P', 'M'):
                Bk = {l: U[l].float().reshape(-1, 1).contiguous() for l in band}
            elif which is None:
                Bk = None
            else:
                Bk = {l: (Rr if which == 'R' else V)[l][:, :k].contiguous() for l in band}
            st['basis'] = which
            todo = [(b, oi) for b in held for oi in orders if not (out / 'items' / f'{name}__{b["id"]}__o{oi}.json').exists()]
            step = a.batch if a.mode == 'thinking' else 1
            for s0 in range(0, len(todo), step):
                chunk = todo[s0:s0 + step]
                rend = [render(rrole, b, oi) for b, oi in chunk]
                coords = rows = None
                if Bk is not None:
                    got = [donor_coords(drole, b, oi, Bk, r) for (b, oi), r in zip(chunk, rend)]
                    coords = [c for c, _ in got]
                    rows = [(len(r[0]), ri) for r, (_, ri) in zip(rend, got)]  # the recipient's aligned positions
                seqs = [r[0] for r in rend]
                Lmax = max(len(q) for q in seqs)
                x = torch.tensor([[pad] * (Lmax - len(q)) + q for q in seqs], dtype=torch.long, device=dev0)
                am = torch.tensor([[0] * (Lmax - len(q)) + [1] * len(q) for q in seqs], dtype=torch.long, device=dev0)
                masked = None
                if a.mask_extra:
                    masked = [unpaired(r, render('user', b, oi)) if rrole == 'assistant' else [] for (b, oi), r in zip(chunk, rend)]
                    for row, (q, mp) in enumerate(zip(seqs, masked)):
                        assert mp or rrole != 'assistant', 'no Assistant-only header token to mask'
                        am[row, [Lmax - len(q) + i for i in mp]] = 0
                if Bk is not None:
                    floor = [floor_of(tgt, b, oi, len(ri)) for (b, oi), (_, ri) in zip(chunk, rows)] if tgt else None
                    st.update(mode='write', B=Bk, rows=rows, coords=coords, dose=None if a.no_dose else [{} for _ in rows],
                              floor=floor, scale=[{} for _ in rows] if tgt else None)
                with torch.no_grad():
                    if a.mode == 'thinking' or content:
                        g = model.generate(input_ids=x, attention_mask=am, max_new_tokens=(a.max_new if a.mode == 'thinking' else 12),
                                           do_sample=False, pad_token_id=pad)
                    else:
                        lp = single_pass(x, am, chunk, rend, rrole, Bk)
                dose, scale = st['dose'], st['scale']
                st.update(mode=None, coords=None, dose=None, floor=None, scale=None)
                for j, (b, oi) in enumerate(chunk):
                    rec = {'cond': name, 'id': b['id'], 'recipient': rrole, 'donor': drole, 'basis': which, 'k': k, 'order': list(ORDERS[oi])}
                    if masked is not None:
                        rec.update(mask_extra=True, masked_positions=masked[j])
                    if dose is not None:
                        rec['dose'] = {'layers': [l for l in band if l in dose[j]], 'n_pos': len(rows[j][1]),
                                       'edit2': [dose[j][l][0] for l in band if l in dose[j]],
                                       'state2': [dose[j][l][1] for l in band if l in dose[j]]}
                    if scale is not None:
                        rec['floor'] = {'target': tgt, 'layers': [l for l in band if l in scale[j]], 'scale': [scale[j][l] for l in band if l in scale[j]]}
                    if a.mode == 'single' and not content and yn:
                        p = {kk: float(sum(lp[j, i].exp() for i in ids)) for kk, ids in yn_ids.items()}
                        rec.update(p=p, mass=sum(p.values()))
                    elif a.mode == 'single' and not content:
                        p = {kk: float(lp[j, letters[labels[i]]].exp()) for i, kk in enumerate(ORDERS[oi])}
                        rec.update(p=p, mass=sum(p.values()))
                    else:
                        new = g[j, Lmax:].tolist()
                        n_new = len(new)
                        for jj, t_ in enumerate(new):
                            if t_ in eos:
                                n_new = jj + 1
                                break
                        gen = tok.decode(new[:n_new], skip_special_tokens=False)
                        tail = gen
                        for cl in CLOSE:
                            if cl in gen:
                                tail = gen.split(cl)[-1]
                        if content:
                            nums = re.findall(r'\d+', tail.replace(',', ''))
                            expected = I.arithmetic_question(b['id'], lang)[1] if kind == 'arith' else re.findall(r'\d+', b['body'])[-1]
                            rec['content'] = {'expected': expected, 'given': nums[0] if nums else None, 'correct': bool(nums) and nums[0] == expected}
                        elif yn:
                            rec.update(yn=parse_yn(tail, lang))
                        else:
                            mm = re.search(label_re, tail.replace('**', ''))
                            letter = mm.group(1) if mm else None
                            rec.update(on_letter=letter, on_answer=ORDERS[oi][labels.index(letter)] if letter else None)
                        rec.update(finished=(a.mode == 'single') or any(cl in gen for cl in CLOSE), n_new=int(n_new), tail=tail[-300:],
                                   gen=gen if a.save_gen else None)
                    atomic(out / 'items' / f'{name}__{b["id"]}__o{oi}.json', rec)
                print(f'{name} {s0 + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
            summarize(out)
    finally:
        for hd in handles:
            hd.remove()
    summarize(out)


def summarize(out):
    out = Path(out)
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    S = {}
    rng = random.Random(0)
    by = {}
    for r in recs:
        by.setdefault((r['id'], tuple(r['order'])), {})[r['cond']] = r
    for c in sorted({r['cond'] for r in recs}):
        rs = [r for r in recs if r['cond'] == c]
        if 'p' in rs[0] and 'yes' in rs[0]['p']:  # yes/no question, single pass
            S[c] = {'n': len(rs), 'p_yes': round(sum(r['p']['yes'] for r in rs) / len(rs), 4), 'mass_min': round(min(r['mass'] for r in rs), 4)}
            val = lambda r: r['p']['yes']  # noqa: E731
        elif 'yn' in rs[0]:  # yes/no question, thinking
            fin = [r for r in rs if r['finished']]
            cnt = Counter(r['yn'] for r in fin)
            S[c] = {'n': len(rs), 'finished': len(fin), 'answers': dict(cnt), 'yes_share': round(cnt.get('yes', 0) / max(1, len(fin)), 4),
                    'median_new_tokens': sorted(r['n_new'] for r in rs)[len(rs) // 2]}
            val = lambda r: float(r['yn'] == 'yes')  # noqa: E731
        elif 'p' in rs[0]:
            S[c] = {'n': len(rs), **{f'p_{k}': round(sum(r['p'][k] for r in rs) / len(rs), 4) for k in ('self', 'user', 'system')},
                    'mass_min': round(min(r['mass'] for r in rs), 4)}
            val = lambda r: r['p']['self']  # noqa: E731
        elif rs[0].get('content'):
            fin = [r for r in rs if r['finished']]
            S[c] = {'n': len(rs), 'finished': len(fin), 'content_accuracy': round(sum(r['content']['correct'] for r in fin) / max(1, len(fin)), 4)}
            continue
        else:
            fin = [r for r in rs if r['finished']]
            cnt = Counter(r['on_answer'] for r in fin)
            S[c] = {'n': len(rs), 'finished': len(fin), 'answers': dict(cnt), 'you_share': round(cnt.get('self', 0) / max(1, len(fin)), 4),
                    'median_new_tokens': sorted(r['n_new'] for r in rs)[len(rs) // 2]}
            val = lambda r: float(r['on_answer'] == 'self')  # noqa: E731
        base = {'assistant': 'base_A', 'user': 'base_U', 'system': 'base_S'}[rs[0]['recipient']]
        if c == base or base not in {r['cond'] for r in recs}:
            continue
        d = [val(v[c]) - val(v[base]) for v in by.values() if c in v and base in v and v[c].get('finished', True) and v[base].get('finished', True)]
        if d:
            bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
            S[c]['you_minus_base'] = {'n_pairs': len(d), 'mean': round(sum(d) / len(d), 4), 'ci95': [round(bs[124], 4), round(bs[4874], 4)]}
    (out / 'summary.json').write_text(json.dumps(S, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(S, ensure_ascii=False), flush=True)


def make_fits(a, loaded=None):
    """fit_q.pt, fit_m.pt, fit_seed1..4.pt and fit_diag.json in --make-fits (see the module docstring)."""
    out = Path(a.make_fits)
    out.mkdir(parents=True, exist_ok=True)
    L_ = loaded or load_model(a.model, max_memory_gib=a.max_memory_gib, cpu_offload_gib=a.cpu_offload_gib)
    torch, tok, model = L_.torch, L_.tok, L_.model
    _, layers = find_layers(model)
    nL = len(layers)
    b0, b1 = (int(x) for x in a.band.split('-'))
    band = tuple(range(b0, b1 + 1))
    assert b1 < nL
    R = Renderer(tok, json.loads(a.on_kwargs))
    ORDERS = I.orders()
    disc = [b for b in selected_bodies(a) if b['split'] == 'discovery']
    dev0 = model.get_input_embeddings().weight.device
    cap = {}

    def make_cap(l):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if l in cap:
                cap[l] = h[0].float().cpu()
            return output
        return hook

    def render_fit(role, b, oi):
        q = I.attr_question(ORDERS[oi], a.lang)
        msgs = [{'role': role, 'content': b['body']}, {'role': 'user', 'content': q}]
        ids, span = R.render(msgs, b['body'])
        qs = R.render(msgs, q)[1][0]  # first token of the question text
        assert span[-1] < qs < len(ids) - 1
        return ids, span[0], qs

    handles = [layers[l].register_forward_hook(make_cap(l)) for l in band]
    t0 = time.time()
    Gq, Gm, nq, nm = {}, {}, 0, 0
    try:
        with torch.no_grad():
            for b in disc:
                for oi in (0, 1):
                    caps = {}
                    for role in ('assistant', 'user'):
                        ids, s0, qs = render_fit(role, b, oi)
                        cap.clear()
                        cap.update({l: None for l in band})
                        x = torch.tensor([ids], device=dev0)
                        model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
                        caps[role] = (dict(cap), ids, s0, qs)
                    (ca, ia, sa, qa), (cu, iu, su, qu) = caps['assistant'], caps['user']
                    assert ia[sa:] == iu[su:], 'body and later tokens differ between roles'
                    assert qa - sa == qu - su
                    for l in band:
                        Dq = (ca[l][sa:len(ia) - 1] - cu[l][su:len(iu) - 1]).double()
                        Dm = (ca[l][sa:qa] - cu[l][su:qu]).double()
                        Gq[l] = Gq.get(l, 0) + Dq.T @ Dq
                        Gm[l] = Gm.get(l, 0) + Dm.T @ Dm
                    nq += len(ia) - 1 - sa
                    nm += qa - sa
            print(f'fit passes {len(disc)} bodies x 2 orders x 2 roles: {time.time() - t0:.0f}s, rows q {nq} m {nm}', flush=True)
    finally:
        for hd in handles:
            hd.remove()
    Vq, Vm, Rs = {}, {}, {j: {} for j in range(5)}
    diag = {'band': [b0, b1], 'n_fit': len(disc), 'rows_q': nq, 'rows_m': nm, 'seeds': [SEED0 + j for j in range(5)], 'layers': {}}

    def share(G, V8):  # fraction of the trace of D^T D in the 8-dim space
        return float((V8.double().T @ G @ V8.double()).trace() / G.trace())
    for l in band:
        for G, V in ((Gq[l], Vq), (Gm[l], Vm)):
            w, E = torch.linalg.eigh(G)
            V[l] = E[:, -K_MAX:].flip(-1).float().contiguous()
        d = Gq[l].shape[0]
        for j in range(5):
            g = torch.Generator().manual_seed(SEED0 + j + l)
            Rs[j][l] = torch.linalg.qr(torch.randn(d, K_MAX, generator=g)).Q.float().contiguous()
        q8, m8 = Vq[l][:, :8], Vm[l][:, :8]
        diag['layers'][l] = {'share_q_in_q': round(share(Gq[l], q8), 4), 'share_m_in_m': round(share(Gm[l], m8), 4),
                             'share_m_in_q': round(share(Gq[l], m8), 4), 'share_q_in_m': round(share(Gm[l], q8), 4),
                             'overlap_q_m': round(float((torch.linalg.svdvals(q8.T @ m8) ** 2).mean()), 4),
                             'share_random_in_q': round(share(Gq[l], Rs[0][l][:, :8]), 5)}
    if a.compare_fit:
        F = torch.load(a.compare_fit, weights_only=True)
        for l in band:
            if l in F['V']:
                diag['layers'][l]['overlap_q_earlier'] = round(float((torch.linalg.svdvals(Vq[l][:, :8].T @ F['V'][l][:, :8].float()) ** 2).mean()), 4)
                diag['layers'][l]['maxabs_R_earlier'] = float((Rs[0][l] - F['R'][l].float()).abs().max())
        diag['compare_fit'] = a.compare_fit
    meta = {'selfindex_version': __version__, 'model': a.model, 'on_kwargs': a.on_kwargs, 'lang': a.lang,
            'bodies': getattr(a, 'body_ids', '') or f'selfindex.items.bodies({a.lang!r})', 'torch': str(torch.__version__),
            'transformers': str(__import__('transformers').__version__)}
    # plain strings only: torch >= 2.6 loads with weights_only=True, which refuses a TorchVersion object
    torch.save({'V': Vq, 'R': Rs[0], 'positions': 'body to second-to-last token', 'n_rows': nq, 'seed': SEED0, **meta}, out / 'fit_q.pt')
    torch.save({'V': Vm, 'R': Rs[0], 'positions': 'body to the token before the question text', 'n_rows': nm, 'seed': SEED0, **meta},
               out / 'fit_m.pt')
    for j in range(1, 5):
        torch.save({'V': Vq, 'R': Rs[j], 'positions': 'V of fit_q.pt', 'n_rows': nq, 'seed': SEED0 + j, **meta}, out / f'fit_seed{j}.pt')
    (out / 'fit_diag.json').write_text(json.dumps({**diag, **meta}, indent=1), encoding='utf-8')
    L = diag['layers']
    print('fits written; median over layers: share q', sorted(v['share_q_in_q'] for v in L.values())[len(L) // 2],
          'share m', sorted(v['share_m_in_m'] for v in L.values())[len(L) // 2],
          'overlap q-m', sorted(v['overlap_q_m'] for v in L.values())[len(L) // 2],
          *(['overlap q-earlier min', min(v.get('overlap_q_earlier', 1) for v in L.values()),
             'max|R-R_earlier|', max(v.get('maxabs_R_earlier', 0) for v in L.values())] if a.compare_fit else []), flush=True)


def check_align(a):
    """Tokenizer only: align every held-out body under every recipient/donor pair of the conditions (attribution question, both
    option orders); print the coverage."""
    from transformers import AutoTokenizer
    tok = checked_tokenizer(AutoTokenizer.from_pretrained(a.model, trust_remote_code=True), a.model)
    R = Renderer(tok, json.loads(a.on_kwargs))
    ORDERS = I.orders()
    cands = [tok.convert_tokens_to_ids(t) for t in TURN_STARTS]
    held = [b for b in selected_bodies(a) if b['split'] == 'confirmation']
    pairs = sorted({(c.split(':')[1], c.split(':')[2]) for c in a.conds.split(',') if c.split(':')[2] != '-'})
    for rr, dr in pairs:
        cov, reg = [], {'body': [], 'end': [], 'question': []}
        for b in held:
            for oi in (0, 1):
                q = I.attr_question(ORDERS[oi], a.lang)
                out = []
                for role in (rr, dr):
                    msgs = [{'role': role, 'content': b['body']}, {'role': 'user', 'content': q}]
                    ids, span = R.render(msgs, b['body'])
                    out.append((ids, span[0], span[-1] + 1, max(i for i in range(len(ids)) if ids[i] in cands), R.render(msgs, q)[1][0]))
                ri, di = align(out[0][:4], out[1][:4])
                r = out[0]
                cov.append(len(ri) / ((r[2] - r[1]) + (r[3] - r[2])))
                reg['body'].append(sum(r[1] <= i < r[2] for i in ri))
                reg['end'].append(sum(r[2] <= i < r[4] for i in ri))
                reg['question'].append(sum(r[4] <= i < r[3] for i in ri))
        print(f'{rr} <- {dr}: {len(cov)} items, aligned share min {min(cov):.3f} median {sorted(cov)[len(cov) // 2]:.3f}; aligned positions '
              + ', '.join(f'{k} min {min(v)} median {sorted(v)[len(v) // 2]}' for k, v in reg.items()), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--mode', choices=('thinking', 'single'))
    p.add_argument('--model')
    p.add_argument('--on-kwargs', default='{}', help='chat-template keyword arguments of the rendering (JSON)')
    p.add_argument('--lang', choices=('zh', 'en'))
    p.add_argument('--fit-from', default='', help='fit file with the subspaces V and R (fit_q.pt / fit_m.pt of --make-fits)')
    p.add_argument('--band', default='', help='first-last layer edited, e.g. 0-63 or 39-39')
    p.add_argument('--conds', default='', help='name:recipient:donor:V|R|C|F|P|M|-:k[:TARGET],...')
    p.add_argument('--out', default='')
    p.add_argument('--question', choices=QUESTIONS, default='attr',
                   help='attr: who wrote the message; content: its last number; arith: an unrelated two-digit multiplication; '
                        'attr_alt: another wording; attr_num: numbered options, third order; yn_self / yn_user: yes/no questions')
    p.add_argument('--region', choices=REGIONS, default='all',
                   help='replace only at aligned positions in the body, in the template tokens between the body and the question text '
                        '(the end of the message), in the question text up to the generation prompt, or in body and end together '
                        '(message)')
    p.add_argument('--dose-floor', default='', help='run folder with per-item doses of the TARGET conditions')
    p.add_argument('--floor-order', type=int, default=-1, help='option order of the --dose-floor TARGET item (default: the same)')
    p.add_argument('--floor-ratio-only', action='store_true', help='match the relative size without equal numbers of positions')
    p.add_argument('--content-fit', default='', help='file with the unit directions U of bases P and M')
    p.add_argument('--no-dose', action='store_true', help='do not record the per-layer size of the edit in the item files')
    p.add_argument('--mask-extra', action='store_true', help='mask the Assistant-only header tokens of the message out of attention')
    p.add_argument('--pos-compress', action='store_true', help='single pass: compress the positions of the Assistant-only header tokens')
    p.add_argument('--body-ids', default='', help='JSON list of body ids (or objects with an id): only these bodies, in this order')
    p.add_argument('--limit-bodies', type=int, default=0)
    p.add_argument('--exclude-first', type=int, default=0)
    p.add_argument('--batch', type=int, default=16, help='items per forward in thinking mode')
    p.add_argument('--max-new', type=int, default=3072, help='new tokens at most in thinking mode')
    p.add_argument('--save-gen', action='store_true', help='keep the whole generated text in the item files')
    p.add_argument('--make-fits', default='', help='write fit_q.pt, fit_m.pt, fit_seed1..4.pt, fit_diag.json in this folder and stop')
    p.add_argument('--compare-fit', default='', help='with --make-fits: an earlier fit file to compare fit_q.pt with')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    p.add_argument('--cpu-offload-gib', type=float, default=0, help='allow weights beyond the GPU budget to stay in CPU memory')
    p.add_argument('--summarize-only', action='store_true', help='rewrite summary.json from the items in --out')
    p.add_argument('--check-align', action='store_true', help='tokenizer only: alignment coverage for every held-out body')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.summarize_only:
        if not args.out:
            p.error('--summarize-only needs --out')
        summarize(Path(args.out))
        return
    missing = [k for k in ('mode', 'model', 'lang') if not getattr(args, k)]
    if args.check_align:
        missing += [] if args.conds else ['conds']
    elif args.make_fits:
        missing += [] if args.band else ['band']
    else:
        missing += [k for k in ('fit_from', 'band', 'conds', 'out') if not getattr(args, k)]
    if missing:
        p.error('missing: ' + ', '.join('--' + k.replace('_', '-') for k in missing))
    if args.check_align:
        check_align(args)
        return
    target = Path(args.make_fits or args.out)
    try:
        make_fits(args) if args.make_fits else run(args)
    except Exception as exc:
        target.mkdir(parents=True, exist_ok=True)
        atomic(target / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
