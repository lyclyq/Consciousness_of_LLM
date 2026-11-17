"""R03 of the supervisor's review, part B (user 10-31: "r03 还是可以机制组的四个模型排上"): projected sensitivity of the attribution
readout, by central differences (forward passes only, so the weights and kernels are those of the main runs: bf16 Qwen3.8-27B,
MXFP4 gpt-oss; the MXFP4 expert kernels have no backward pass).
Per held-out item (75 bodies x 2 option orders) the model's own message (A) and the same text as a User message (U) are rendered
as in crossmodel_role_swap_r05.py and aligned as there (body, end of the message, question); d_l = h^A_l - h^U_l at the aligned
positions of layer l. The readout f is the logit of the option letter for "you" minus that of the letter for "user" where the
answer letter is read: the response onset without thinking; with thinking, the answer letter after the model's own greedy
reasoning, teacher-forced on the generated prefix (as Section 4.1.6 reads letter probabilities after reasoning). At every layer l
of the band, five directions are added to the states of layer l at the aligned positions only (every other layer as computed):
  total  d_l
  V      P_V d_l, its part in the fitted subspace (the 8 leading directions of fit_q.pt)
  perp   d_l - P_V d_l, the rest (the residual outside the subspace)
  R      P_R d_l, its part in a random 8-dim subspace (R of fit_q.pt, base seed 20261022)
  zero   no change (two unperturbed rows: the numerical noise of the batch)
and D_v f = (f(h + eps v) - f(h - eps v)) / (2 eps), eps = 0.25, the first-order change of f along v, at the own message
(endpoint A: its states and reasoning) and at the User message (endpoint U: its states and reasoning). If the readout of the role
difference went through the residual outside the subspace, D along perp would carry it; if through the subspace, D along V. The ten
rows of one layer run as one batch of identical sequences.
Outputs: <out>/items/<id>__o<order>.json (per layer of --layer-step and endpoint: D for the five directions; the norms of the vectors and of the
states; f and the letter probabilities unperturbed), <out>/traces/ (thinking: the generated ids), identity.json, summary.json
(per layer, medians over items of D_V / D_total and D_perp / D_total at both endpoints). A run without --limit-bodies waits for
<out>.GO (exits on <out>.SKIP), as crossmodel_role_swap_r05.py.
usage: crossmodel_readout_fd_r03.py --mode thinking|single --model DIR --on-kwargs JSON --lang zh|en --bodies F --splits F
       --fit-from fit_q.pt --band A-B --out DIR [--limit-bodies 2] [--batch 16] [--max-new 3072] [--max-memory-gib G]"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_role_swap_r05 as rs  # noqa: E402  (align, CLOSE)
import crossmodel_suite as cs  # noqa: E402
from crossmodel_suite import Renderer, load, attr_question, ORDERS, LETTERS  # noqa: E402

EPS = 0.25
DIRS = ('total', 'V', 'perp', 'R', 'zero')


def run(a):
    import torch
    out = Path(a.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    (out / 'traces').mkdir(exist_ok=True)
    if not a.limit_bodies:
        go, skip = Path(str(out) + '.GO'), Path(str(out) + '.SKIP')
        t_wait = time.time()
        while not go.exists():
            if skip.exists():
                print(f'{skip} present: run skipped', flush=True)
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
    fd_layers = band[::a.layer_step]  # the layers at which the directions are applied (every state is captured)
    assert b1 < nL
    R = Renderer(tok, json.loads(a.on_kwargs))
    splits = {json.loads(l)['id']: json.loads(l)['split'] for l in open(a.splits, encoding='utf-8') if l.strip()}
    held = [b for b in json.loads(Path(a.bodies).read_text(encoding='utf-8')) if splits[b['id']] == 'confirmation']
    if a.limit_bodies:
        held = held[:a.limit_bodies]
    dev0 = model.get_input_embeddings().weight.device
    cands = [tok.convert_tokens_to_ids(t) for t in ('<|start|>', '<|im_start|>')]
    F = torch.load(a.fit_from)
    Bv = {l: F['V'][l][:, :8].float().contiguous() for l in band}
    Br = {l: F['R'][l][:, :8].float().contiguous() for l in band}
    sets = [{x: tok.encode(x, add_special_tokens=False) for x in LETTERS}, {x: tok.encode(' ' + x, add_special_tokens=False) for x in LETTERS}]
    sets = [{x: v[0] for x, v in s.items()} for s in sets if all(len(v) == 1 for v in s.values())]
    assert sets and len(sets[0]) == 3, 'single-token option letters needed'
    closes = [tok.encode(c, add_special_tokens=False) for c in rs.CLOSE]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = {tok.eos_token_id} | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    eos.discard(None)

    def render(role, b, oi):
        msgs = [{'role': role, 'content': b['body']}, {'role': 'user', 'content': attr_question(ORDERS[oi])}]
        ids, span = R.render(msgs, b['body'])
        g0 = max(i for i in range(len(ids)) if ids[i] in cands)
        assert g0 > span[-1], 'no generation prompt after the body'
        return ids, span[0], span[-1] + 1, g0

    st = {'mode': None, 'pos': None, 'layer': None, 'vecs': None, 'caps': None}

    def make_hook(l):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if st['mode'] is None or h.shape[1] == 1:
                return output
            sl = torch.tensor(st['pos'], dtype=torch.long, device=h.device)
            if st['mode'] == 'capture':
                st['caps'][l] = h[0, sl].float().clone()
            elif l == st['layer']:
                h[:, sl] = (h[:, sl].float() + st['vecs'].to(h.device)).to(h.dtype)
            return output
        return hook
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in band]

    def last_logits(seq, nrows):
        x = torch.tensor([seq] * nrows, dtype=torch.long, device=dev0)
        with torch.no_grad():
            return model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1).logits[:, -1].float()

    def generate(seqs):
        res = []
        for s0 in range(0, len(seqs), a.batch):
            chunk = seqs[s0:s0 + a.batch]
            Lmax = max(len(q) for q in chunk)
            x = torch.tensor([[pad] * (Lmax - len(q)) + q for q in chunk], dtype=torch.long, device=dev0)
            am = torch.tensor([[0] * (Lmax - len(q)) + [1] * len(q) for q in chunk], dtype=torch.long, device=dev0)
            with torch.no_grad():
                g = model.generate(input_ids=x, attention_mask=am, max_new_tokens=a.max_new, do_sample=False, pad_token_id=pad)
            for j in range(len(chunk)):
                new = g[j, Lmax:].tolist()
                n = next((i + 1 for i, t in enumerate(new) if t in eos), len(new))
                res.append(new[:n])
            print(f'generated {min(s0 + a.batch, len(seqs))}/{len(seqs)}', flush=True)
        return res

    def answer_at(gen):
        """index in gen of the answer letter after the last closing marker, and the letter set it belongs to"""
        start = None
        for c in closes:
            for i in range(len(gen) - len(c), -1, -1):
                if gen[i:i + len(c)] == c:
                    start = i + len(c) if start is None else max(start, i + len(c))
                    break
        if start is None:
            return None, None
        for j in range(start, len(gen)):
            for k, s in enumerate(sets):
                if gen[j] in s.values():
                    return j, k
        return None, None

    items = [(b, oi) for b in held for oi in (0, 1)]
    ident = {'experiment': 'P2_R03_readout_fd', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'swap_sha256': hashlib.sha256(Path(rs.__file__).read_bytes()).hexdigest(),
             'suite_sha256': hashlib.sha256(Path(cs.__file__).read_bytes()).hexdigest(), 'model': a.model, 'mode': a.mode,
             'on_kwargs': a.on_kwargs, 'n_layers': nL, 'band': [b0, b1], 'layer_step': a.layer_step, 'fit_from': a.fit_from, 'eps': EPS, 'directions': DIRS,
             'lang': a.lang, 'n_eval': len(held), 'batch': a.batch, 'max_new': a.max_new, 'placement': placement,
             'torch': str(torch.__version__), 'transformers': str(__import__('transformers').__version__)}
    (out / 'identity.json').write_text(json.dumps(ident, indent=1), encoding='utf-8')
    print(json.dumps(ident), flush=True)
    todo = [(b, oi) for b, oi in items if not (out / 'items' / f'{b["id"]}__o{oi}.json').exists()]
    rend = {(b['id'], oi, role): render(role, b, oi) for b, oi in todo for role in ('assistant', 'user')}
    traces = {}
    if a.mode == 'thinking':
        keys = [(b['id'], oi, role) for b, oi in todo for role in ('assistant', 'user')]
        need = [k for k in keys if not (out / 'traces' / f'{k[0]}__o{k[1]}__{k[2]}.json').exists()]
        for k, g in zip(need, generate([rend[k][0] for k in need])):
            (out / 'traces' / f'{k[0]}__o{k[1]}__{k[2]}.json').write_text(json.dumps({'gen': g, 'text': tok.decode(g)}, ensure_ascii=False),
                                                                          encoding='utf-8')
        for k in keys:
            traces[k] = json.loads((out / 'traces' / f'{k[0]}__o{k[1]}__{k[2]}.json').read_text(encoding='utf-8'))['gen']
    t0 = time.time()
    for n_done, (b, oi) in enumerate(todo):
        rA, rU = rend[(b['id'], oi, 'assistant')], rend[(b['id'], oi, 'user')]
        ri, di = rs.align(rA, rU)
        rec = {'id': b['id'], 'order': list(ORDERS[oi]), 'n_pos': len(ri), 'layers': list(fd_layers), 'eps': EPS, 'endpoints': {}}
        ends = {}
        for e, r_, pos, role in (('A', rA, ri, 'assistant'), ('U', rU, di, 'user')):
            seq, ls = r_[0], 0
            if a.mode == 'thinking':
                gen = traces[(b['id'], oi, role)]
                j, ls = answer_at(gen)
                if j is None:
                    rec['endpoints'][e] = {'skipped': 'no answer letter after the reasoning', 'n_new': len(gen)}
                    continue
                seq = r_[0] + gen[:j]
            if len(seq) > a.max_len:
                rec['endpoints'][e] = {'skipped': f'sequence of {len(seq)} tokens', 'n_tokens': len(seq)}
                continue
            ends[e] = (seq, pos, ls)
        if len(ends) < 2:
            (out / 'items' / f'{b["id"]}__o{oi}.json').write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
            continue
        caps = {}
        for e, (seq, pos, ls) in ends.items():
            st.update(mode='capture', pos=pos, caps={})
            lg = last_logits(seq, 1)[0]
            st.update(mode=None)
            caps[e] = st['caps']
            lt = sets[ls]
            ly, lu = lt[LETTERS[ORDERS[oi].index('self')]], lt[LETTERS[ORDERS[oi].index('user')]]
            p3 = torch.softmax(lg[[lt[x] for x in LETTERS]], -1).tolist()
            rec['endpoints'][e] = {'f0': float(lg[ly] - lg[lu]), 'p_letters': dict(zip(LETTERS, p3)), 'letter_set': ls,
                                   'n_tokens': len(seq), 'D': {k: [] for k in DIRS}}
            ends[e] = (seq, pos, ly, lu)
        norms = {k: [] for k in DIRS[:4]}
        rec['state_norm'] = {'A': [], 'U': []}
        for l in fd_layers:
            d = caps['A'][l] - caps['U'][l]
            vV = (d @ Bv[l].to(d.device)) @ Bv[l].to(d.device).T
            vR = (d @ Br[l].to(d.device)) @ Br[l].to(d.device).T
            vec = {'total': d, 'V': vV, 'perp': d - vV, 'R': vR, 'zero': torch.zeros_like(d)}
            for k in DIRS[:4]:
                norms[k].append(float(vec[k].norm()))
            for e in ('A', 'U'):
                rec['state_norm'][e].append(float(caps[e][l].norm()))
            rows = torch.stack([s * EPS * vec[k] for k in DIRS for s in (1, -1)])
            for e, (seq, pos, ly, lu) in ends.items():
                st.update(mode='perturb', pos=pos, layer=l, vecs=rows)
                lg = last_logits(seq, len(rows))
                st.update(mode=None, vecs=None)
                f = (lg[:, ly] - lg[:, lu]).tolist()
                for i, k in enumerate(DIRS):
                    rec['endpoints'][e]['D'][k].append((f[2 * i] - f[2 * i + 1]) / (2 * EPS))
        rec['norms'] = norms
        (out / 'items' / f'{b["id"]}__o{oi}.json').write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
        print(f'item {n_done + 1}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    for hd in handles:
        hd.remove()
    summarize(out)


def summarize(out):
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    ok = [r for r in recs if all('D' in r['endpoints'].get(e, {}) for e in ('A', 'U'))]
    S = {'n_items': len(recs), 'n_complete': len(ok), 'skipped': [[r['id'], r['order'], {e: v.get('skipped') for e, v in r['endpoints'].items()
                                                                                         if 'skipped' in v}] for r in recs if r not in ok]}
    if ok:
        L = ok[0]['layers']
        for e in ('A', 'U'):
            per = []
            for i, l in enumerate(L):
                tot = [r['endpoints'][e]['D']['total'][i] for r in ok]
                row = {'layer': l, 'median_total': statistics.median(tot)}
                for k in ('V', 'perp', 'R', 'zero'):
                    xs = [r['endpoints'][e]['D'][k][i] for r in ok]
                    row[f'median_{k}'] = statistics.median(xs)
                    row[f'median_{k}_over_total'] = statistics.median(x / t for x, t in zip(xs, tot) if t != 0) if any(tot) else None
                per.append(row)
            S[e] = per
    (out / 'summary.json').write_text(json.dumps(S, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in S.items() if k in ('n_items', 'n_complete')}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('thinking', 'single'), required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--on-kwargs', default='{}')
    p.add_argument('--lang', choices=('en', 'zh'), required=True)
    p.add_argument('--bodies', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--fit-from', required=True)
    p.add_argument('--band', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--limit-bodies', type=int, default=0)
    p.add_argument('--batch', type=int, default=16, help='generation batch (thinking)')
    p.add_argument('--max-new', type=int, default=3072)
    p.add_argument('--max-len', type=int, default=4096, help='longest teacher-forced sequence; longer items are recorded as skipped')
    p.add_argument('--layer-step', type=int, default=1, help='apply the directions at every n-th layer of the band, from its first')
    p.add_argument('--max-memory-gib', type=float, default=44)
    p.add_argument('--cpu-offload-gib', type=float, default=0)
    p.add_argument('--summarize-only', action='store_true')
    a = p.parse_args()
    summarize(Path(a.out)) if a.summarize_only else run(a)


if __name__ == '__main__':
    main()
