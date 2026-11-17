"""Causal test of the self-index under THINKING, every token, second design (user 10-22). The first design
(crossmodel_thinking_ablate.py: the subspace set to zero at every layer and every position, the whole prompt included) broke
gpt-oss-20b in its smoke, the random-subspace control as well (unfinished garbage), so it tests nothing. Changes:
  * positions: from the start of the message under test (its turn-start token) to the end of the prompt, and EVERY generated
    token at every decoding step; the tokens before the message (system prompt) and the first token of the sequence (the
    attention sink; in Qwen3.8-27B the message's own turn-start token) are not edited;
  * mean replacement instead of removal: inside the k-dimensional subspace B the coordinates are set to their mean over User
    messages, x <- x + (m_U - x B) B^T, which moves an own message toward a User one without leaving the range of states the
    model produces (m_U = mean of x B over the edited positions of the User renderings of the discovery bodies);
  * a band of layers, not all of them.
Subspace V_l: the top-k eigenvectors of D^T D, D = h^A - h^U at aligned body and post-body positions of the discovery bodies
(75, both orders), as before; random control R_l: QR of a Gaussian, seed 20261022 + l, its own m_U.
--suite smoke: base_A, base_U and, for each of two layer sets fixed in advance (band = --band, main = --main-layer),
rm64_A, rnd64_A, rnd64_U, on --limit-bodies held-out bodies. Choice rule, fixed before any result: the full run uses the band if
its random controls leave the behaviour intact (rnd64_A and rnd64_U finished >= 90% and answering like base_A / base_U in
>= 90% of items), otherwise the main layer if it does, otherwise nothing. The choice looks only at the random controls.
--suite full --layers band|main: base_A, rm64_A, rnd64_A, base_U, rm64_U, rnd64_U on all 75 held-out bodies, then rm8_A,
rm1_A on the first 30; waits for <out>.GO before loading the model (exits on <out>.SKIP); summary.json after every condition.
--no-gen-edit (10-22, third design, after the second failed its smoke: editing the generated tokens disrupts the model's own
generation, since on the tokens it writes the same subspace marks that the assistant is speaking): only the prompt is edited,
every token from the message under test to the end of the prompt and at every layer of the set (e.g. --band 0-63, all layers),
so the reasoning reads a message whose states have been moved toward a User message at every token and layer; the tokens the
model writes are left alone. The choice rule and the conditions are unchanged.
--edit-end before_gen (10-22, fourth design, after the third stopped at once when the message was edited at all layers: the
edited range then included the generation prompt, the assistant's own turn-start tokens, which carry "the assistant speaks
next"): the edit ends before the generation prompt (the last turn-start token), so the message under test and the following
User question are edited at every token and layer, and neither the generation prompt nor the generated tokens are.
--suite content (10-22, user: the specificity control, the thinking counterpart of "the same displacement changes the output
only on questions about the self"): the same edits (--fit-from the full run's fit.pt, the same layers and positions), but the
question asks for the content of the message instead of its author: "In the message before this question, what is the last
number that appears? Output only the number." (Chinese / English as the model's runs; the answer is unique: the last number of
the body, distinct across the 150 bodies). Conditions base_A, rm64_A, rnd64_A on the 75 held-out bodies (one question each);
the answer is the first number after the reasoning, correct if it equals the expected one.
Probe, decoding (greedy, up to 3072 new tokens, batch 16, left padding) and parsing as the model's thinking runs.
usage: crossmodel_thinking_ablate2.py --suite smoke|full --model DIR --bodies F --splits F --out DIR --lang en|zh
        --main-layer L --band A-B [--layers band|main] [--limit-bodies N] [--batch 16] [--max-new 3072] [--max-memory-gib 44]"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402
from crossmodel_suite import Renderer, load, attr_question, ORDERS, LETTERS  # noqa: E402

CLOSE = ('</think>', '<channel|>', '</seed:think>', '<|channel|>final<|message|>')
K_MAX = 64
SECONDARY = ('rm8_A', 'rm1_A', 'rnd8_A', 'rnd1_A', 'rm8_U', 'rm1_U')  # 10-23: the low-rank conditions and their rank-matched controls share the first 30 bodies


Q_CONTENT = {'zh': '这个问题之前的那条消息里，最后出现的数字是多少？只回答数字。',
             'en': 'In the message before this question, what is the last number that appears? Output only the number.'}


def conditions(suite, layers_used, band, main):
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


def run(a):
    import torch
    out = Path(a.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    if a.suite in ('full', 'content'):
        go, skip = Path(str(out) + '.GO'), Path(str(out) + '.SKIP')
        t_wait = time.time()
        while not go.exists():
            if skip.exists():
                print(f'{skip} present: full run skipped', flush=True)
                return
            if int(time.time() - t_wait) % 600 < 60:
                print(f'waiting for {go}', flush=True)
            time.sleep(60)
    if a.lang == 'en':
        cs.Q_ATTR = cs.Q_ATTR_EN
    torch, tok, model, cfg, placement, cls = load(a)
    _, layers = cs.find_layers(model)
    nL = len(layers)
    b0, b1 = (int(x) for x in a.band.split('-'))
    band = tuple(range(b0, b1 + 1))
    assert b1 < nL and a.main_layer < nL
    fit_layers = sorted(set(band) | {a.main_layer})
    R = Renderer(tok, json.loads(a.on_kwargs))
    splits = {json.loads(l)['id']: json.loads(l)['split'] for l in open(a.splits, encoding='utf-8') if l.strip()}
    bodies = json.loads(Path(a.bodies).read_text(encoding='utf-8'))
    disc = [b for b in bodies if splits[b['id']] == 'discovery']
    held = [b for b in bodies if splits[b['id']] == 'confirmation']
    if a.limit_bodies:
        held = held[:a.limit_bodies]
    dev0 = model.get_input_embeddings().weight.device
    cands = [tok.convert_tokens_to_ids(t) for t in ('<|start|>', '<|im_start|>')]

    def question(oi):
        return Q_CONTENT[a.lang] if a.suite == 'content' else attr_question(ORDERS[oi])

    def render(role, b, oi, fit=False):
        q = attr_question(ORDERS[oi]) if fit else question(oi)
        ids, span = R.render([{'role': role, 'content': b['body']}, {'role': 'user', 'content': q}], b['body'])
        s0 = span[0]
        starts = [i for i in range(s0 + 1) if ids[i] in cands]
        assert starts, 'no turn-start token before the body'
        m0 = max(starts[-1], 1)  # the turn-start token of the message under test; position 0 (the sink token) is never edited
        g0 = max(i for i in range(len(ids)) if ids[i] in cands)  # the generation prompt's turn-start token
        assert g0 > span[-1], 'no generation prompt after the body'
        return ids, m0, s0, span[-1] + 1, (g0 if a.edit_end == 'before_gen' else len(ids))

    state = {'cap': None, 'edit': None, 'rows': None}

    def make_hook(l):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if state['cap'] is not None and l in state['cap']:
                state['cap'][l] = h[0].float().cpu()
            e = state['edit']
            if e is not None and l in e:
                B, m = e[l]
                B, m = B.to(h.device), m.to(h.device)
                if h.shape[1] > 1:  # prefill: from the message start of each row to the end of its prompt
                    Lmax = h.shape[1]
                    for row, (n, m0, ee) in enumerate(state['rows']):
                        sl = slice(Lmax - n + m0, Lmax - n + ee)
                        x = h[row, sl].float()
                        h[row, sl] = (x + (m - x @ B) @ B.T).to(h.dtype)
                elif not a.no_gen_edit:  # every generated token
                    x = h.float()
                    h.copy_((x + (m - x @ B) @ B.T).to(h.dtype))
            return output
        return hook
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in fit_layers]

    if a.fit_from:  # the same subspaces and User means as an earlier run (the specificity control uses the full run's)
        t0 = time.time()
        F = torch.load(a.fit_from)
        V, Rr, MV, MR, n_rows, NU = F['V'], F['R'], F['mV'], F['mR'], F['n_rows'], F['n_user_positions']
        assert all(l in V for l in fit_layers), 'the loaded fit does not cover the layers'
    else:
        # fit: subspace of the aligned Assistant-User differences; mean of User states over the edited positions
        t0 = time.time()
        G, SU, NU, n_rows = {}, {}, 0, 0
        with torch.no_grad():
            for b in disc:
                for oi in (0, 1):
                    caps = {}
                    for role in ('assistant', 'user'):
                        ids, m0, s0, e, ee = render(role, b, oi, fit=True)
                        state['cap'] = {l: None for l in fit_layers}
                        x = torch.tensor([ids], device=dev0)
                        model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
                        caps[role] = (state['cap'], ids, m0, s0, ee)
                        state['cap'] = None
                    (ca, ia, ma, sa, _), (cu, iu, mu_, su, eu) = caps['assistant'], caps['user']
                    assert ia[sa:] == iu[su:], 'body and later tokens differ between roles'
                    for l in fit_layers:
                        D = (ca[l][sa:len(ia) - 1] - cu[l][su:len(iu) - 1]).double()
                        G[l] = G.get(l, 0) + D.T @ D
                        SU[l] = SU.get(l, 0) + cu[l][mu_:eu].double().sum(0)
                    n_rows += len(ia) - 1 - sa
                    NU += eu - mu_
        V, Rr, MV, MR = {}, {}, {}, {}
        for l in fit_layers:
            w, U = torch.linalg.eigh(G[l])
            V[l] = U[:, -K_MAX:].flip(-1).float().contiguous()
            g = torch.Generator().manual_seed(20261022 + l)
            Rr[l] = torch.linalg.qr(torch.randn(G[l].shape[0], K_MAX, generator=g)).Q.float().contiguous()
            mean_u = (SU[l] / NU).float()
            MV[l], MR[l] = mean_u @ V[l], mean_u @ Rr[l]
    torch.save({'V': V, 'R': Rr, 'mV': MV, 'mR': MR, 'n_rows': n_rows, 'n_user_positions': NU}, out / 'fit.pt')
    conds = conditions(a.suite, a.layers, band, a.main_layer)
    if a.conds:  # 10-23: explicit low-rank conditions (rank-matched random controls, content controls at k = 8 and 1)
        ls = band if a.layers == 'band' else (a.main_layer,)
        conds = [(n, r, None if w == '-' else (w, int(k), ls)) for n, r, w, k in (c.split(':') for c in a.conds.split(','))]
    ident = {'experiment': f'P2_thinking_ablate2_{a.suite}', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'suite_sha256': hashlib.sha256(Path(cs.__file__).read_bytes()).hexdigest(), 'model': a.model, 'n_layers': nL,
             'band': [b0, b1], 'main_layer': a.main_layer, 'fit_from': a.fit_from, 'suite': a.suite, 'edit_generated_tokens': not a.no_gen_edit, 'edit_end': a.edit_end, 'layers_used': a.layers if a.suite == 'full' else 'both', 'lang': a.lang,
             'k_max': K_MAX, 'fit_rows': n_rows, 'n_fit': len(disc), 'n_eval': len(held), 'conditions': [c[0] for c in conds],
             'batch': a.batch, 'max_new': a.max_new, 'torch': torch.__version__,
             'transformers': __import__('transformers').__version__, 'fit_s': time.time() - t0}
    (out / 'identity.json').write_text(json.dumps(ident, indent=1), encoding='utf-8')
    print(json.dumps(ident), flush=True)

    def edit_for(spec):
        if spec is None:
            return None
        which, k, ls = spec
        return {l: ((V if which == 'V' else Rr)[l][:, :k], (MV if which == 'V' else MR)[l][:k]) for l in ls}

    held_sec = held if a.secondary_all else held[:30]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = {tok.eos_token_id} | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    eos.discard(None)
    t0 = time.time()
    for name, role, spec in conds:
        items = [(b, oi) for b in (held_sec if name in SECONDARY else held) for oi in ((0,) if a.suite == 'content' else (0, 1))
                 if not (out / 'items' / f'{name}__{b["id"]}__o{oi}.json').exists()]
        state['edit'] = edit_for(spec)
        for s0 in range(0, len(items), a.batch):
            chunk = items[s0:s0 + a.batch]
            rend = [render(role, b, oi) for b, oi in chunk]
            seqs = [r[0] for r in rend]
            state['rows'] = [(len(r[0]), r[1], r[4]) for r in rend]
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
                if a.suite == 'content':
                    nums = re.findall(r'\d+', tail.replace(',', ''))
                    expected = re.findall(r'\d+', b['body'])[-1]
                    content = {'expected': expected, 'given': nums[0] if nums else None, 'correct': bool(nums) and nums[0] == expected}
                else:
                    content = None
                rec = {'cond': name, 'id': b['id'], 'role': role, 'order': list(ORDERS[oi]), 'spec': spec, 'content': content,
                       'on_letter': letter, 'on_answer': ORDERS[oi][LETTERS.index(letter)] if letter else None,
                       'finished': any(cl in gen for cl in CLOSE), 'n_new': int(n_new), 'tail': tail[-300:],
                       'gen': gen if a.save_gen else None}
                (out / 'items' / f'{name}__{b["id"]}__o{oi}.json').write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
            print(f'{name} {s0 + len(chunk)}/{len(items)} {time.time() - t0:.0f}s', flush=True)
        state['edit'] = None
        if items:
            summarize(out)
    for hd in handles:
        hd.remove()
    summarize(out)


def summarize(out):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', choices=('smoke', 'full', 'content'), required=True)
    p.add_argument('--fit-from', default='', help='load V, R and the User means from an earlier fit.pt of this script')
    p.add_argument('--conds', default='', help='name:role:V|R|-:k,... replaces the condition list (layers from --layers)')
    p.add_argument('--secondary-all', action='store_true', help='rm8_A / rm1_A on all held-out bodies, not the first 30')
    p.add_argument('--model', required=True)
    p.add_argument('--on-kwargs', default='{}')
    p.add_argument('--bodies', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--lang', choices=('en', 'zh'), required=True)
    p.add_argument('--main-layer', type=int, required=True)
    p.add_argument('--band', required=True, help='first-last layer of the band, e.g. 15-47')
    p.add_argument('--layers', choices=('band', 'main'), default='band', help='full run: the layer set chosen after the smoke')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--max-new', type=int, default=3072)
    p.add_argument('--max-memory-gib', type=float, default=44)
    p.add_argument('--cpu-offload-gib', type=float, default=0)
    p.add_argument('--limit-bodies', type=int, default=0)
    p.add_argument('--save-gen', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--no-gen-edit', action='store_true', help='edit the prompt only (from the message start), not the generated tokens')
    p.add_argument('--edit-end', choices=('prompt', 'before_gen'), default='prompt', help='end of the edited prompt range')
    a = p.parse_args()
    summarize(Path(a.out)) if a.summarize_only else run(a)


if __name__ == '__main__':
    main()
