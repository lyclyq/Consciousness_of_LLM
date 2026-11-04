"""The secret game: role-coordinate replacement where the answer depends on who wrote one message X and the question never asks
who wrote it (template-native renderings only).

Items (selfindex.items.games, items/games/games_<lang>_secret6.json; --items FILE: another file in that layout): 100 guessing
games. The System message gives the rules: the setter keeps a whole number secret and answers each guess only with too high, too
low or correct; on its turn the guesser replies only with a guess, and the setter of a game in which nobody has guessed yet only
with 'Go ahead' (请猜); a question that is not a move is answered directly. A User message opens the game, and message X says that
its writer is the setter and that the guesser gets m tries. X is rendered as the model's own message (assistant) or as a User
message; the last User message (--qkind) is the main question (your turn: guess if you are the guesser, say 'Go ahead' if you are
the setter) or the control question (how many tries does the guesser get?), which X answers whoever wrote it.
The replacement is that of selfindex.role_swap with X as the recipient message (the subspaces of --fit-from are fitted on the
neutral bodies, not on the games): the conversation is also rendered with X under the donor role and run unedited, one sequence
at a time; at every layer of --band its coordinates in the basis B (the k leading directions of V or R) at the aligned positions
(render.align: X's body and the template tokens after it up to the generation prompt; --region x: only X's body and the template
tokens up to the text of the last message, which then stays unedited) are written into the recipient run at the same positions,
x <- x + (c_donor - x B) B^T, prompt only; the generated tokens are never edited.
Conditions (--conds, comma-separated name:recipient:donor:V|R|-:k): base_A:assistant:-:-:0 (X the model's own), base_U:user:-:-:0
(X the user's), swU8_A:assistant:user:V:8 (an own X given the User coordinates), swA8_U:user:assistant:V:8 (a User X given the own
coordinates), rswU8_A / rswA8_U (the same copies in the random subspace R).
Readout: greedy generation (single: one game per forward, 24 new tokens; thinking: --batch games per forward, left padding, at most
--max-new new tokens), the answer read after the last end-of-reasoning marker. Main question: with '**' and commas removed and
special tags replaced by spaces, the reply is the setter's if it contains 'Go ahead' / 请猜, otherwise the guesser's if it contains a
number; reads_as is assistant for a setter (X read as the model's own) and user for a guesser. Control question: the first number
of the answer (commas and '**' removed), correct if it is m.
Outputs in --out: identity.json, items/<cond>__<game id>.json (a run resumes from the items on disk), summary.json (per game kind and
condition: the number of items and of finished ones, the median number of new tokens; main question: the answers by reads_as, the
share read as the model's own and, for an edited condition, the paired difference to the unedited condition of the same recipient
(base_A / base_U) with a 95% bootstrap interval, 5000 resamples, seed 0; control question: the accuracy).
--check-align (tokenizer only) prints the alignment coverage of every game under every recipient/donor pair of --conds and the
rendering of the first game under both roles.
usage: python -m selfindex.games --mode single|thinking --model DIR --on-kwargs JSON --lang zh|en --qkind main|control
         --fit-from FIT --band A-B --conds ... --out DIR [--region all|x] [--items FILE] [--limit-games N] [--batch N] [--max-new N]
         [--save-gen]
       python -m selfindex.games --check-align --model DIR --on-kwargs JSON --lang zh|en --qkind Q --conds ... [--items FILE]
       python -m selfindex.games --summarize-only --out DIR
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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
from .load import checked_tokenizer, find_layers, load_model
from .render import CLOSE, Renderer, align

TURN_STARTS = ('<|start|>', '<|im_start|>')  # the generation prompt begins at the last of these tokens
GO_AHEAD = re.compile(r'请猜|go ahead', re.I)  # the setter's fixed reply before any guess
QKINDS = ('main', 'control')


def items_file(a):
    return Path(a.items) if a.items else I.items_dir() / 'games' / f'games_{a.lang}_secret6.json'


def load_games(a):
    """The games of the run in file order (those that have the question --qkind), the first --limit-games of them."""
    games = [b for b in json.loads(items_file(a).read_text(encoding='utf-8')) if a.qkind in b['q']]
    for b in games:
        assert b['kind'] == 'role2' and b['expect']['control']['assistant'] == b['expect']['control']['user'], \
            f'{b["id"]}: not a secret game of the paper layout'
    return games[:a.limit_games] if a.limit_games else games


def conversation(b, role, qkind):
    """The messages of game b with X under `role` and the closing question; returns them and X's text."""
    msgs = copy.deepcopy(b['msgs'])
    msgs[b['edit']]['role'] = role
    msgs.append({'role': 'user', 'content': b['q'][qkind]})
    return msgs, msgs[b['edit']]['content']


def make_render(R, cands, qkind, region='all'):
    def render(role, b):
        msgs, target = conversation(b, role, qkind)
        ids, span = R.render(msgs, target)
        g0 = max(i for i in range(len(ids)) if ids[i] in cands)  # the generation prompt's turn-start token
        assert g0 > span[-1], 'no generation prompt after the edited message'
        qs = R.render(msgs, b['q'][qkind])[1][0] if region == 'x' else None  # first token of the last message's text
        assert qs is None or span[-1] < qs < g0
        return ids, span[0], span[-1] + 1, g0, qs  # X's body [s, e); the edited range ends before the generation prompt g0
    return render


def score_answer(b, qkind, tail):
    """The answer of one game (see the module docstring)."""
    t = tail.replace('**', '').replace(',', '')
    if qkind == 'main':
        t2 = re.sub(r'<[^<>]*>', ' ', t)
        typ = 'setter' if GO_AHEAD.search(t2) else ('guesser' if re.search(r'\d+', t2) else None)
        return {'type': typ, 'reads_as': {'setter': 'assistant', 'guesser': 'user'}.get(typ), 'setter_correct': None,
                'given': t2.strip()[:40]}
    nums = re.findall(r'\d+', t)
    given = nums[0] if nums else None
    return {'given': given, 'correct': given == b['expect'][qkind]['assistant'], 'reads_as': None}


def parse_conds(a):
    conds = []
    for c in a.conds.split(','):
        n, rr, dr, w, k = c.split(':')
        assert rr in ('assistant', 'user'), f'{n}: the recipient of X is assistant or user'
        assert w in ('-', 'V', 'R'), f'{n}: unknown basis {w}'
        conds.append((n, rr, None if dr == '-' else dr, None if w == '-' else w, int(k)))
    return conds


def run(a, loaded=None):
    out = Path(a.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    L_ = loaded or load_model(a.model, max_memory_gib=a.max_memory_gib, cpu_offload_gib=a.cpu_offload_gib)
    torch, tok, model = L_.torch, L_.tok, L_.model
    _, layers = find_layers(model)
    nL = len(layers)
    b0, b1 = (int(x) for x in a.band.split('-'))
    band = tuple(range(b0, b1 + 1))
    assert b1 < nL
    R = Renderer(tok, json.loads(a.on_kwargs))
    games = load_games(a)
    dev0 = model.get_input_embeddings().weight.device
    cands = [tok.convert_tokens_to_ids(t) for t in TURN_STARTS]
    render = make_render(R, cands, a.qkind, a.region)

    F = torch.load(a.fit_from, weights_only=True)
    V, Rr = F['V'], F['R']
    assert all(l in V for l in band), 'the loaded fit does not cover the band'
    st = {'mode': None, 'B': None, 'rows': None, 'coords': None}

    def make_hook(l):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if st['mode'] is None or h.shape[1] == 1 or l not in st['B']:
                return output
            B = st['B'][l].to(h.device)
            Lmax = h.shape[1]
            for row, (n, pos) in enumerate(st['rows']):
                sl = torch.tensor(pos, dtype=torch.long, device=h.device) + (Lmax - n)
                x = h[row, sl].float()
                if st['mode'] == 'capture':  # the donor run: its coordinates in B
                    st['coords'][row][l] = (x @ B).clone()
                else:
                    c = st['coords'][row][l].to(h.device)
                    h[row, sl] = (x + (c - x @ B) @ B.T).to(h.dtype)
            return output
        return hook

    conds = parse_conds(a)
    ipath = items_file(a)
    ident = {'experiment': f'P2_games_{a.mode}', 'selfindex_version': __version__, 'model': a.model, 'on_kwargs': a.on_kwargs,
             'items': str(a.items or ipath), 'items_sha256': hashlib.sha256(ipath.read_bytes()).hexdigest(),
             'games': ','.join(sorted({b['game'] for b in games})), 'qkind': a.qkind, 'n_layers': nL, 'band': [b0, b1],
             'fit_from': a.fit_from, 'region': a.region, 'lang': a.lang, 'n_items': len(games), 'conditions': [c[0] for c in conds],
             'batch': a.batch, 'max_new': a.max_new, 'torch': torch.__version__, 'transformers': __import__('transformers').__version__}
    (out / 'identity.json').write_text(json.dumps(ident, indent=1), encoding='utf-8')
    print(json.dumps(ident), flush=True)

    def donor_coords(drole, b, Bk, r):
        d = render(drole, b)
        ri, di = align(r[:4], d[:4])
        if a.region == 'x':  # the aligned pairs inside X: its body and the template tokens up to the last message's text
            keep = [j for j, i in enumerate(ri) if r[1] <= i < r[4]]
            assert keep, 'no aligned position in X'
            ri, di = [ri[j] for j in keep], [di[j] for j in keep]
        st.update(mode='capture', B=Bk, rows=[(len(d[0]), di)], coords=[{}])
        x = torch.tensor([d[0]], device=dev0)
        with torch.no_grad():
            model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
        cd = st['coords'][0]
        st.update(mode=None, coords=None)
        return cd, ri

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = {tok.eos_token_id} | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    eos.discard(None)
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in band]
    t0 = time.time()
    try:
        for name, rrole, drole, which, k in conds:
            Bk = None if which is None else {l: (V if which == 'V' else Rr)[l][:, :k].contiguous() for l in band}
            todo = [b for b in games if not (out / 'items' / f'{name}__{b["id"]}.json').exists()]
            step = a.batch if a.mode == 'thinking' else 1
            for s0 in range(0, len(todo), step):
                chunk = todo[s0:s0 + step]
                rend = [render(rrole, b) for b in chunk]
                coords = rows = None
                if Bk is not None:
                    got = [donor_coords(drole, b, Bk, r) for b, r in zip(chunk, rend)]
                    coords = [c for c, _ in got]
                    rows = [(len(r[0]), ri) for r, (_, ri) in zip(rend, got)]  # the recipient's aligned positions
                seqs = [r[0] for r in rend]
                Lmax = max(len(q) for q in seqs)
                x = torch.tensor([[pad] * (Lmax - len(q)) + q for q in seqs], dtype=torch.long, device=dev0)
                am = torch.tensor([[0] * (Lmax - len(q)) + [1] * len(q) for q in seqs], dtype=torch.long, device=dev0)
                if Bk is not None:
                    st.update(mode='write', B=Bk, rows=rows, coords=coords)
                with torch.no_grad():
                    g = model.generate(input_ids=x, attention_mask=am, max_new_tokens=(a.max_new if a.mode == 'thinking' else 24),
                                       do_sample=False, pad_token_id=pad)
                st.update(mode=None, coords=None)
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
                    atomic(out / 'items' / f'{name}__{b["id"]}.json', rec)
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
            if 'correct' in rs[0]['answer']:  # control question
                s['accuracy'] = round(sum(r['answer']['correct'] for r in fin) / max(1, len(fin)), 4)
                S[game][c] = s
                continue
            cnt = Counter(r['answer']['reads_as'] for r in fin)
            s.update(reads_as=dict(cnt), own_share=round(cnt.get('assistant', 0) / max(1, len(fin)), 4))
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


def check_align(a):
    """Tokenizer only: render every game under every recipient/donor pair of the conditions and align them; print the coverage and
    the rendering of the first game under both roles."""
    from transformers import AutoTokenizer
    tok = checked_tokenizer(AutoTokenizer.from_pretrained(a.model, trust_remote_code=True), a.model)
    R = Renderer(tok, json.loads(a.on_kwargs))
    cands = [tok.convert_tokens_to_ids(t) for t in TURN_STARTS]
    render = make_render(R, cands, a.qkind)
    games = load_games(a)
    pairs = sorted({(c.split(':')[1], c.split(':')[2]) for c in a.conds.split(',') if c.split(':')[2] != '-'})
    for rr, dr in pairs:
        cov = {}
        for b in games:
            r, d = render(rr, b)[:4], render(dr, b)[:4]
            ri, di = align(r, d)
            cov.setdefault(b['game'], []).append(len(ri) / ((r[2] - r[1]) + (r[3] - r[2])))
        print(f'{rr} <- {dr}: ' + '; '.join(f'{g} {len(v)} items, aligned share min {min(v):.3f} median {sorted(v)[len(v) // 2]:.3f}'
                                           for g, v in cov.items()), flush=True)
    b = games[0]
    for role in ('assistant', 'user'):
        print(role, repr(tok.decode(render(role, b)[0]))[:600], flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--mode', choices=('thinking', 'single'))
    p.add_argument('--model')
    p.add_argument('--on-kwargs', default='{}', help='chat-template keyword arguments of the rendering (JSON)')
    p.add_argument('--lang', choices=('zh', 'en'))
    p.add_argument('--qkind', choices=QKINDS, help='main: the next move of the game; control: how many tries the guesser gets')
    p.add_argument('--items', default='', help='games file in the layout of items/games (default: the secret game of --lang)')
    p.add_argument('--fit-from', default='', help='fit file with the subspaces V and R (fit.pt of selfindex.fits, fit_q.pt / fit_m.pt '
                                                   'of selfindex.role_swap --make-fits)')
    p.add_argument('--band', default='', help='first-last layer edited, e.g. 0-63')
    p.add_argument('--conds', default='', help='name:recipient:donor:V|R|-:k,...')
    p.add_argument('--region', choices=('all', 'x'), default='all',
                   help='all: X and every later prompt position up to the generation prompt; x: X only (its body and the template '
                        'tokens up to the text of the last message)')
    p.add_argument('--out', default='')
    p.add_argument('--limit-games', type=int, default=0, help='the first N games only')
    p.add_argument('--batch', type=int, default=16, help='games per forward in thinking mode')
    p.add_argument('--max-new', type=int, default=3072, help='new tokens at most in thinking mode')
    p.add_argument('--save-gen', action='store_true', help='keep the whole generated text in the item files')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    p.add_argument('--cpu-offload-gib', type=float, default=0, help='allow weights beyond the GPU budget to stay in CPU memory')
    p.add_argument('--summarize-only', action='store_true', help='rewrite summary.json from the items in --out')
    p.add_argument('--check-align', action='store_true', help='tokenizer only: alignment coverage for every game')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.summarize_only:
        if not args.out:
            p.error('--summarize-only needs --out')
        summarize(Path(args.out))
        return
    missing = [k for k in ('model', 'lang', 'qkind', 'conds') if not getattr(args, k)]
    if not args.check_align:
        missing += [k for k in ('mode', 'fit_from', 'band', 'out') if not getattr(args, k)]
    if missing:
        p.error('missing: ' + ', '.join('--' + k.replace('_', '-') for k in missing))
    if args.check_align:
        check_align(args)
        return
    try:
        run(args)
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        atomic(Path(args.out) / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
