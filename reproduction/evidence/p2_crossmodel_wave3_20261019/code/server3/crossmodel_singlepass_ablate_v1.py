"""The fourth thinking design applied WITHOUT thinking (user 10-22: with an all-layer intervention under thinking, the single-pass
evidence should use the same intervention, so that both modes have one method and two readouts). Qwen3.8-27B, the template's
non-thinking rendering (--on-kwargs '{"enable_thinking": false}'), the main probe, as its single-pass results.
Fit (as crossmodel_thinking_ablate2.py, on this rendering): V_l = top-k eigenvectors of D^T D, D = h^A - h^U at aligned body and
post-body positions of the 75 discovery bodies (both orders), every layer; R_l random (QR, seed 20261022 + l); m_U = the mean of
the User states over the edited positions, projected on V_l / R_l.
Edit: mean replacement x <- x + (m_U - x B) B^T, at every layer of --band (default all) and every token from the turn-start token
of the message under test (never position 0) to the generation prompt (excluded, as in the thinking design).
Readout (as the paper's single-pass self-reference): the probabilities of the three option letters at the response onset, one
forward, two option orders; P(the letter of "you").
Conditions (held-out 75 bodies x 2 orders): base_A, rm64_A, rnd64_A, base_U, rm64_U, rnd64_U, rm8_A, rm1_A.
--question content: the specificity control, base_A / rm64_A / rnd64_A with "In the message before this question, what is the
last number that appears? Output only the number." (one per body); greedy decoding of up to 12 tokens, correct if the first
number equals the last number of the body.
usage: crossmodel_singlepass_ablate.py --model DIR --bodies F --splits F --out DIR --lang zh [--band 0-63] [--question content]
                                       [--fit-from fit.pt] [--max-memory-gib 44]"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402
from crossmodel_suite import Renderer, load, attr_question, ORDERS, LETTERS  # noqa: E402

K_MAX = 64
Q_CONTENT = {'zh': '这个问题之前的那条消息里，最后出现的数字是多少？只回答数字。',
             'en': 'In the message before this question, what is the last number that appears? Output only the number.'}


def run(a):
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.lang == 'en':
        cs.Q_ATTR = cs.Q_ATTR_EN
    torch, tok, model, cfg, placement, cls = load(a)
    _, layers = cs.find_layers(model)
    nL = len(layers)
    b0, b1 = (int(x) for x in a.band.split('-'))
    band = tuple(range(b0, b1 + 1))
    R = Renderer(tok, json.loads(a.on_kwargs))
    splits = {json.loads(l)['id']: json.loads(l)['split'] for l in open(a.splits, encoding='utf-8') if l.strip()}
    bodies = json.loads(Path(a.bodies).read_text(encoding='utf-8'))
    disc = [b for b in bodies if splits[b['id']] == 'discovery']
    held = [b for b in bodies if splits[b['id']] == 'confirmation']
    dev0 = model.get_input_embeddings().weight.device
    cands = [tok.convert_tokens_to_ids(t) for t in ('<|start|>', '<|im_start|>')]
    letters = {x: tok.encode(x, add_special_tokens=False) for x in LETTERS}
    assert all(len(v) == 1 for v in letters.values())
    letters = {x: v[0] for x, v in letters.items()}

    def render(role, b, oi, content=False):
        q = Q_CONTENT[a.lang] if content else attr_question(ORDERS[oi])
        ids, span = R.render([{'role': role, 'content': b['body']}, {'role': 'user', 'content': q}], b['body'])
        starts = [i for i in range(span[0] + 1) if ids[i] in cands]
        g0 = max(i for i in range(len(ids)) if ids[i] in cands)
        return ids, max(starts[-1], 1), span[0], g0

    state = {'cap': None, 'edit': None, 'rng': None}

    def make_hook(l):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if state['cap'] is not None and l in state['cap']:
                state['cap'][l] = h[0].float().cpu()
            e = state['edit']
            if e is not None and l in e and h.shape[1] > 1:
                B, m = e[l]
                s, t = state['rng']
                x = h[0, s:t].float()
                h[0, s:t] = (x + (m.to(h.device) - x @ B.to(h.device)) @ B.to(h.device).T).to(h.dtype)
            return output
        return hook
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in band]

    t0 = time.time()
    if a.fit_from:
        F = torch.load(a.fit_from)
        V, Rr, MV, MR = F['V'], F['R'], F['mV'], F['mR']
    else:
        G, SU, NU = {}, {}, 0
        with torch.no_grad():
            for b in disc:
                for oi in (0, 1):
                    caps = {}
                    for role in ('assistant', 'user'):
                        ids, m0, s0, g0 = render(role, b, oi)
                        state['cap'] = {l: None for l in band}
                        x = torch.tensor([ids], device=dev0)
                        model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
                        caps[role] = (state['cap'], ids, m0, s0, g0)
                        state['cap'] = None
                    (ca, ia, ma, sa, ga), (cu, iu, mu_, su, gu) = caps['assistant'], caps['user']
                    assert ia[sa:] == iu[su:], 'body and later tokens differ between roles'
                    for l in band:
                        D = (ca[l][sa:len(ia) - 1] - cu[l][su:len(iu) - 1]).double()
                        G[l] = G.get(l, 0) + D.T @ D
                        SU[l] = SU.get(l, 0) + cu[l][mu_:gu].double().sum(0)
                    NU += gu - mu_
        V, Rr, MV, MR = {}, {}, {}, {}
        for l in band:
            w, U = torch.linalg.eigh(G[l])
            V[l] = U[:, -K_MAX:].flip(-1).float().contiguous()
            g = torch.Generator().manual_seed(20261022 + l)
            Rr[l] = torch.linalg.qr(torch.randn(G[l].shape[0], K_MAX, generator=g)).Q.float().contiguous()
            mean_u = (SU[l] / NU).float()
            MV[l], MR[l] = mean_u @ V[l], mean_u @ Rr[l]
        torch.save({'V': V, 'R': Rr, 'mV': MV, 'mR': MR, 'n_user_positions': NU}, out / 'fit.pt')
    content = a.question == 'content'
    conds = ([('base_A', 'assistant', None), ('rm64_A', 'assistant', ('V', 64)), ('rnd64_A', 'assistant', ('R', 64))] if content else
             [('base_A', 'assistant', None), ('rm64_A', 'assistant', ('V', 64)), ('rnd64_A', 'assistant', ('R', 64)),
              ('base_U', 'user', None), ('rm64_U', 'user', ('V', 64)), ('rnd64_U', 'user', ('R', 64)),
              ('rm8_A', 'assistant', ('V', 8)), ('rm1_A', 'assistant', ('V', 1))])
    ident = {'experiment': 'P2_singlepass_ablate', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'model': a.model, 'on_kwargs': a.on_kwargs, 'band': [b0, b1], 'question': a.question, 'fit_from': a.fit_from,
             'conditions': [c[0] for c in conds], 'n_eval': len(held), 'fit_s': time.time() - t0}
    (out / 'identity.json').write_text(json.dumps(ident, indent=1), encoding='utf-8')
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    recs = []
    for name, role, spec in conds:
        state['edit'] = None if spec is None else {l: ((V if spec[0] == 'V' else Rr)[l][:, :spec[1]], (MV if spec[0] == 'V' else MR)[l][:spec[1]]) for l in band}
        for b in held:
            for oi in ((0,) if content else (0, 1)):
                ids, m0, s0, g0 = render(role, b, oi, content)
                state['rng'] = (m0, g0)
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
        state['edit'] = None
        print(f'{name} done {time.time() - t0:.0f}s', flush=True)
    for hd in handles:
        hd.remove()
    (out / 'items.json').write_text(json.dumps(recs, ensure_ascii=False), encoding='utf-8')
    S = {}
    rng = random.Random(0)
    for name, role, spec in conds:
        rs = [r for r in recs if r['cond'] == name]
        if content:
            S[name] = {'n': len(rs), 'accuracy': round(sum(r['correct'] for r in rs) / len(rs), 4)}
            continue
        S[name] = {'n': len(rs), 'p_you_mean': round(sum(r['p']['self'] for r in rs) / len(rs), 4),
                   'mass_min': round(min(r['mass'] for r in rs), 4)}
        base = 'base_A' if name.endswith('_A') else 'base_U'
        if name != base:
            bb = {(r['id'], tuple(r['order'])): r for r in recs if r['cond'] == base}
            d = [r['p']['self'] - bb[(r['id'], tuple(r['order']))]['p']['self'] for r in rs]
            bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
            S[name]['minus_base'] = {'mean': round(sum(d) / len(d), 4), 'ci95': [round(bs[124], 4), round(bs[4874], 4)]}
    (out / 'summary.json').write_text(json.dumps(S, indent=1), encoding='utf-8')
    print(json.dumps(S), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--on-kwargs', default='{"enable_thinking": false}')
    p.add_argument('--bodies', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--lang', choices=('en', 'zh'), required=True)
    p.add_argument('--band', default='0-63')
    p.add_argument('--question', choices=('attr', 'content'), default='attr')
    p.add_argument('--fit-from', default='')
    p.add_argument('--max-memory-gib', type=float, default=44)
    p.add_argument('--cpu-offload-gib', type=float, default=0)
    run(p.parse_args())


if __name__ == '__main__':
    main()
