"""Role-coordinate replacement in interactive games (user 10-24): the answer depends on who wrote one message X, and the question
never asks who wrote it (items from make_games.py: split, score, secret; one question kind per run: main, main2, control).
The replacement is that of crossmodel_role_swap.py (same alignment, same hooks, same fitted subspace from --fit-from, fitted on the
neutral bodies, not on the games): the k coordinates of X and of every later prompt position up to the generation prompt are
written from the same conversation rendered with X under the donor role, x <- x + (c_donor - x B) B^T; generated tokens are not
edited. Conditions as there, with X as the recipient message: base_A (X the model's own), base_U (X the user's), swU8_A (own X with
User coordinates), swA8_U (User X with own coordinates), rswU8_A / rswA8_U (random 8-dim subspace).
Scoring: number games compare the first number of the answer with the answers expected if X is the model's own or the user's
(reads_as = assistant | user | None); the secret game classifies the answer as a setter's reply (too high / too low) or a guess (a
number); control questions are scored as correct or not. Every item is greedy, thinking or single pass by --mode.
Secret game v4 (kind role2, make_games_secret4.py, 10-26): the reply 'Go ahead' / 请猜 is the setter's, a number the guesser's.
A run without --limit-bodies waits for <out>.GO (exits on <out>.SKIP).
usage: crossmodel_games.py --mode thinking|single --model DIR --on-kwargs JSON --items games_zh.json --qkind main|main2|control
       --games split,score,secret --fit-from fit.pt --band A-B --conds ... --out DIR --lang zh|en [--limit-bodies 8] [--batch 16]"""
from __future__ import annotations

import argparse
import copy
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
from crossmodel_suite import Renderer, load  # noqa: E402
from crossmodel_role_swap import align, CLOSE  # noqa: E402

SETTER = re.compile(r'大了|小了|too high|too low', re.I)
GO_AHEAD = re.compile(r'请猜|go ahead', re.I)  # secret game v4: the setter's fixed reply


def load_items(a):
    items = [b for b in json.loads(Path(a.items).read_text(encoding='utf-8')) if b['game'] in a.games.split(',') and a.qkind in b['q']]
    if a.limit_bodies:  # the first --limit-bodies items of every game
        items = [b for g in a.games.split(',') for b in [x for x in items if x['game'] == g][:a.limit_bodies]]
    return items


def conversation(b, role, qkind):
    msgs = copy.deepcopy(b['msgs'])
    msgs[b['edit']]['role'] = role
    msgs.append({'role': 'user', 'content': b['q'][qkind]})
    return msgs, msgs[b['edit']]['content']


def make_render(R, cands, qkind):
    def render(role, b):
        msgs, target = conversation(b, role, qkind)
        ids, span = R.render(msgs, target)
        g0 = max(i for i in range(len(ids)) if ids[i] in cands)  # the generation prompt's turn-start token
        assert g0 > span[-1], 'no generation prompt after the edited message'
        return ids, span[0], span[-1] + 1, g0
    return render


def score_answer(b, qkind, tail):
    exp = b['expect'][qkind]
    t = tail.replace('**', '').replace(',', '')
    if b['kind'] == 'role2' and qkind == 'main':  # v4: 'Go ahead' = setter, a number = guesser (special tags removed first)
        t2 = re.sub(r'<[^<>]*>', ' ', t)
        typ = 'setter' if GO_AHEAD.search(t2) else ('guesser' if re.search(r'\d+', t2) else None)
        return {'type': typ, 'reads_as': {'setter': 'assistant', 'guesser': 'user'}.get(typ), 'setter_correct': None,
                'given': t2.strip()[:40]}
    if b['kind'] == 'role' and qkind == 'main':
        m = SETTER.search(t)
        if m:
            typ = 'setter'
        elif re.search(r'\d+', t):
            typ = 'guesser'
        else:
            typ = None
        return {'type': typ, 'reads_as': {'setter': 'assistant', 'guesser': 'user'}.get(typ),
                'setter_correct': bool(m) and m.group(0).lower() == exp['setter_answer'].lower(), 'given': (m.group(0) if m else None)}
    nums = re.findall(r'\d+', t)
    given = nums[0] if nums else None
    if exp['assistant'] == exp['user']:
        return {'given': given, 'correct': given == exp['assistant'], 'reads_as': None}
    return {'given': given, 'reads_as': 'assistant' if given == exp['assistant'] else ('user' if given == exp['user'] else None)}


def check_align(a):
    """Tokenizer only: render every item under every recipient/donor pair of the conditions and align them."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    R = Renderer(tok, json.loads(a.on_kwargs))
    cands = [tok.convert_tokens_to_ids(t) for t in ('<|start|>', '<|im_start|>')]
    render = make_render(R, cands, a.qkind)
    items = load_items(a)
    pairs = sorted({(c.split(':')[1], c.split(':')[2]) for c in a.conds.split(',') if c.split(':')[2] != '-'})
    for rr, dr in pairs:
        cov = {}
        for b in items:
            r, d = render(rr, b), render(dr, b)
            ri, di = align(r, d)
            cov.setdefault(b['game'], []).append(len(ri) / ((r[2] - r[1]) + (r[3] - r[2])))
        print(f'{rr} <- {dr}: ' + '; '.join(f'{g} {len(v)} items, aligned share min {min(v):.3f} median {sorted(v)[len(v) // 2]:.3f}'
                                           for g, v in cov.items()), flush=True)
    b = items[0]
    for role in ('assistant', 'user'):
        print(role, repr(tok.decode(render(role, b)[0]))[:600], flush=True)


def run(a):
    import torch
    out = Path(a.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
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
    torch, tok, model, cfg, placement, cls = load(a)
    _, layers = cs.find_layers(model)
    nL = len(layers)
    b0, b1 = (int(x) for x in a.band.split('-'))
    band = tuple(range(b0, b1 + 1))
    assert b1 < nL
    R = Renderer(tok, json.loads(a.on_kwargs))
    items = load_items(a)
    dev0 = model.get_input_embeddings().weight.device
    cands = [tok.convert_tokens_to_ids(t) for t in ('<|start|>', '<|im_start|>')]
    render = make_render(R, cands, a.qkind)

    F = torch.load(a.fit_from)
    V, Rr = F['V'], F['R']
    assert all(l in V for l in band), 'the loaded fit does not cover the band'
    state = {'mode': None, 'B': None, 'rows': None, 'coords': None}

    def make_hook(l):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if state['mode'] is None or h.shape[1] == 1 or l not in state['B']:
                return output
            B = state['B'][l].to(h.device)
            Lmax = h.shape[1]
            for row, (n, pos) in enumerate(state['rows']):
                sl = torch.tensor(pos, dtype=torch.long, device=h.device) + (Lmax - n)
                x = h[row, sl].float()
                if state['mode'] == 'capture':
                    state['coords'][row][l] = (x @ B).clone()
                else:
                    c = state['coords'][row][l].to(h.device)
                    h[row, sl] = (x + (c - x @ B) @ B.T).to(h.dtype)
            return output
        return hook
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in band]

    conds = []
    for c in a.conds.split(','):
        n, rr, dr, w, k = c.split(':')
        conds.append((n, rr, None if dr == '-' else dr, None if w == '-' else w, int(k)))
    ident = {'experiment': f'P2_games_{a.mode}', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'role_swap_sha256': hashlib.sha256((Path(__file__).parent / 'crossmodel_role_swap.py').read_bytes()).hexdigest(),
             'suite_sha256': hashlib.sha256(Path(cs.__file__).read_bytes()).hexdigest(), 'model': a.model, 'on_kwargs': a.on_kwargs,
             'items': a.items, 'items_sha256': hashlib.sha256(Path(a.items).read_bytes()).hexdigest(), 'games': a.games, 'qkind': a.qkind,
             'n_layers': nL, 'band': [b0, b1], 'fit_from': a.fit_from, 'lang': a.lang, 'n_items': len(items),
             'conditions': [c[0] for c in conds], 'batch': a.batch, 'max_new': a.max_new, 'torch': torch.__version__,
             'transformers': __import__('transformers').__version__}
    (out / 'identity.json').write_text(json.dumps(ident, indent=1), encoding='utf-8')
    print(json.dumps(ident), flush=True)

    def donor_coords(drole, b, Bk, r):
        d = render(drole, b)
        ri, di = align(r, d)
        state.update(mode='capture', B=Bk, rows=[(len(d[0]), di)], coords=[{}])
        x = torch.tensor([d[0]], device=dev0)
        with torch.no_grad():
            model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
        cd = state['coords'][0]
        state.update(mode=None, coords=None)
        return cd, ri

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = {tok.eos_token_id} | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    eos.discard(None)
    t0 = time.time()
    for name, rrole, drole, which, k in conds:
        Bk = None if which is None else {l: (V if which == 'V' else Rr)[l][:, :k].contiguous() for l in band}
        todo = [b for b in items if not (out / 'items' / f'{name}__{b["id"]}.json').exists()]
        step = a.batch if a.mode == 'thinking' else 1
        for s0 in range(0, len(todo), step):
            chunk = todo[s0:s0 + step]
            rend = [render(rrole, b) for b in chunk]
            coords = rows = None
            if Bk is not None:
                got = [donor_coords(drole, b, Bk, r) for b, r in zip(chunk, rend)]
                coords = [c for c, _ in got]
                rows = [(len(r[0]), ri) for r, (_, ri) in zip(rend, got)]
            seqs = [r[0] for r in rend]
            Lmax = max(len(q) for q in seqs)
            x = torch.tensor([[pad] * (Lmax - len(q)) + q for q in seqs], dtype=torch.long, device=dev0)
            am = torch.tensor([[0] * (Lmax - len(q)) + [1] * len(q) for q in seqs], dtype=torch.long, device=dev0)
            if Bk is not None:
                state.update(mode='write', B=Bk, rows=rows, coords=coords)
            with torch.no_grad():
                g = model.generate(input_ids=x, attention_mask=am, max_new_tokens=(a.max_new if a.mode == 'thinking' else 24),
                                   do_sample=False, pad_token_id=pad)
            state.update(mode=None, coords=None)
            for j, b in enumerate(chunk):
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
                rec = {'cond': name, 'id': b['id'], 'game': b['game'], 'qkind': a.qkind, 'recipient': rrole, 'donor': drole,
                       'basis': which, 'k': k, 'order': [], 'answer': score_answer(b, a.qkind, tail),
                       'finished': (a.mode == 'single') or any(cl in gen for cl in CLOSE), 'n_new': int(n_new), 'tail': tail[-300:],
                       'gen': gen if a.save_gen else None}
                (out / 'items' / f'{name}__{b["id"]}.json').write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
            print(f'{name} {s0 + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
        summarize(out)
    for hd in handles:
        hd.remove()
    summarize(out)


def summarize(out):
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    S = {}
    rng = random.Random(0)
    for game in sorted({r['game'] for r in recs}):
        G = [r for r in recs if r['game'] == game]
        by = {}
        for r in G:
            by.setdefault(r['id'], {})[r['cond']] = r
        S[game] = {}
        for c in sorted({r['cond'] for r in G}):
            rs = [r for r in G if r['cond'] == c]
            fin = [r for r in rs if r['finished']]
            s = {'n': len(rs), 'finished': len(fin), 'median_new_tokens': sorted(r['n_new'] for r in rs)[len(rs) // 2]}
            if 'correct' in rs[0]['answer']:
                s['accuracy'] = round(sum(r['answer']['correct'] for r in fin) / max(1, len(fin)), 4)
                S[game][c] = s
                continue
            cnt = Counter(r['answer']['reads_as'] for r in fin)
            s.update(reads_as=dict(cnt), own_share=round(cnt.get('assistant', 0) / max(1, len(fin)), 4))
            if game == 'secret':
                st = [r for r in fin if r['answer']['type'] == 'setter' and r['answer'].get('setter_correct') is not None]
                if st:  # v4 setters give a fixed reply, nothing to check
                    s['setter_correct'] = round(sum(r['answer']['setter_correct'] for r in st) / len(st), 4)
            base = {'assistant': 'base_A', 'user': 'base_U'}[rs[0]['recipient']]
            if c != base:
                val = lambda r: float(r['answer']['reads_as'] == 'assistant')  # noqa: E731
                d = [val(v[c]) - val(v[base]) for v in by.values() if c in v and base in v and v[c]['finished'] and v[base]['finished']]
                if d:
                    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
                    s['own_minus_base'] = {'n_pairs': len(d), 'mean': round(sum(d) / len(d), 4), 'ci95': [round(bs[124], 4), round(bs[4874], 4)]}
            S[game][c] = s
    (out / 'summary.json').write_text(json.dumps(S, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(S, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('thinking', 'single'), required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--on-kwargs', default='{}')
    p.add_argument('--items', required=True)
    p.add_argument('--qkind', choices=('main', 'main2', 'control'), required=True)
    p.add_argument('--games', default='split,score,secret')
    p.add_argument('--fit-from', default='')
    p.add_argument('--band', default='0-0', help='first-last layer edited, e.g. 0-63')
    p.add_argument('--conds', required=True, help='name:recipient:donor:V|R|-:k,...')
    p.add_argument('--out', default='')
    p.add_argument('--lang', choices=('en', 'zh'), required=True)
    p.add_argument('--limit-bodies', type=int, default=0, help='first N items of every game (smoke)')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--max-new', type=int, default=3072)
    p.add_argument('--max-memory-gib', type=float, default=44)
    p.add_argument('--cpu-offload-gib', type=float, default=0)
    p.add_argument('--save-gen', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--check-align', action='store_true', help='tokenizer only: alignment coverage for every item')
    a = p.parse_args()
    if a.check_align:
        return check_align(a)
    summarize(Path(a.out)) if a.summarize_only else run(a)


if __name__ == '__main__':
    main()
