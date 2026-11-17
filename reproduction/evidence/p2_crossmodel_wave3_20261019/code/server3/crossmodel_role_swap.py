"""Per-position role-coordinate replacement (user 10-23). Deletion and replacement both act on the model's own message; a User or
System message is only replaced, pulled toward the assistant (sufficiency); it is never "deleted", there is no self-index there.
The first replacement design (crossmodel_thinking_ablate2.py --a-mean: one average written at every position) flattened the
position-specific index instead of writing it and is not used.

Per item: the same text is first rendered under the DONOR role and run unedited, one sequence at a time (no padding, so the
positions match the recipient's); at every layer of --band its states at the aligned positions (the body and the template tokens
after it up to the generation prompt, matched token by token, see align()) are projected on the k leading directions of the fitted
subspace V_l (or of the random R_l). The RECIPIENT rendering is then run with those k coordinates written into its own states at
the same aligned positions, x <- x + (c_donor - x B) B^T, at every layer of the band, prompt only: the role header, the
generation prompt and, under thinking, the generated tokens are not edited.
Conditions --conds name:recipient:donor:V|R|-:k, e.g.
  base_A:assistant:-:-:0      unedited own message
  swU8_A:assistant:user:V:8   own message given the User coordinates     (inside/outside: compare with swS8_A)
  swS8_A:assistant:system:V:8 own message given the System coordinates
  swA8_U:user:assistant:V:8   User message given the own coordinates      (sufficiency)
  swA8_S:system:assistant:V:8 System message given the own coordinates
  rswU8_A:assistant:user:R:8  the same copy on a random 8-dim subspace (control)
--mode thinking: greedy generation and parsing as crossmodel_thinking_ablate2.py (batch --batch, left padding);
--mode single: the probabilities of the three option letters at the response onset, one forward per item, as
crossmodel_singlepass_ablate.py. --question content: the specificity control ("the last number in that message").
Subspace from --fit-from (fit.pt of crossmodel_thinking_ablate2.py or crossmodel_singlepass_ablate.py, on the same rendering).
A run without --limit-bodies waits for <out>.GO (exits on <out>.SKIP). Two option orders, one for the content question.
usage: crossmodel_role_swap.py --mode thinking|single --model DIR --on-kwargs JSON --bodies F --splits F --fit-from fit.pt
       --band A-B --conds ... --out DIR --lang zh|en [--limit-bodies 4] [--question attr|content] [--batch 16]"""
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
Q_CONTENT = {'zh': '这个问题之前的那条消息里，最后出现的数字是多少？只回答数字。',
             'en': 'In the message before this question, what is the last number that appears? Output only the number.'}


def arith_question(b, lang):
    """--question arith (user 10-24: a task that does not concern the message's source): one two-digit multiplication per body,
    fixed by the body id, asked after the message under test. Returns (question, expected answer)."""
    rng = random.Random(b['id'])
    x, y = rng.randint(13, 97), rng.randint(13, 97)
    q = f'请计算 {x}×{y} 等于多少？只回答数字。' if lang == 'zh' else f'What is {x} × {y}? Output only the number.'
    return q, str(x * y)


def align(r, d):
    """Aligned positions of a recipient and a donor rendering, each (ids, body start, body end, generation prompt start). Two
    segments, matched token by token: the body (the same text under every role; its first or last token may be merged with the
    template, so the common prefix and suffix of the two body token runs), and the template tokens after the body up to the
    generation prompt (their common suffix; gpt-oss renders a System message as a developer message with two extra newlines after
    the body, which stay unedited). Returns the recipient and donor position lists."""
    (rid, rs, re_, rg), (did, ds, de, dg) = r, d
    rb, db = rid[rs:re_], did[ds:de]
    p = 0
    while p < min(len(rb), len(db)) and rb[p] == db[p]:
        p += 1
    q = 0
    while q < min(len(rb), len(db)) - p and rb[-1 - q] == db[-1 - q]:
        q += 1
    ri = list(range(rs, rs + p)) + list(range(re_ - q, re_))
    di = list(range(ds, ds + p)) + list(range(de - q, de))
    assert len(rid) - rg == len(did) - dg, 'generation prompts differ'
    n = 0
    while n < min(len(rid) - re_, len(did) - de) and rid[-1 - n] == did[-1 - n]:
        n += 1
    ri += list(range(len(rid) - n, rg))
    di += list(range(len(did) - n, dg))
    assert [rid[i] for i in ri] == [did[i] for i in di], 'aligned tokens differ'
    assert len(ri) >= 0.8 * ((re_ - rs) + (rg - re_)) - 4, 'too few aligned positions'
    return ri, di


def check_align(a):
    """Tokenizer only: align every held-out body under every recipient/donor pair of the conditions; print the coverage."""
    from transformers import AutoTokenizer
    if a.lang == 'en':
        cs.Q_ATTR = cs.Q_ATTR_EN
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    R = Renderer(tok, json.loads(a.on_kwargs))
    cands = [tok.convert_tokens_to_ids(t) for t in ('<|start|>', '<|im_start|>')]
    splits = {json.loads(l)['id']: json.loads(l)['split'] for l in open(a.splits, encoding='utf-8') if l.strip()}
    held = [b for b in json.loads(Path(a.bodies).read_text(encoding='utf-8')) if splits[b['id']] == 'confirmation']
    pairs = sorted({(c.split(':')[1], c.split(':')[2]) for c in a.conds.split(',') if c.split(':')[2] != '-'})
    for rr, dr in pairs:
        cov, reg = [], {'body': [], 'end': [], 'question': []}
        for b in held:
            for oi in (0, 1):
                out = []
                for role in (rr, dr):
                    msgs = [{'role': role, 'content': b['body']}, {'role': 'user', 'content': attr_question(ORDERS[oi])}]
                    ids, span = R.render(msgs, b['body'])
                    out.append((ids, span[0], span[-1] + 1, max(i for i in range(len(ids)) if ids[i] in cands), R.render(msgs, attr_question(ORDERS[oi]))[1][0]))
                ri, di = align(out[0][:4], out[1][:4])
                r = out[0]
                cov.append(len(ri) / ((r[2] - r[1]) + (r[3] - r[2])))
                reg['body'].append(sum(r[1] <= i < r[2] for i in ri))
                reg['end'].append(sum(r[2] <= i < r[4] for i in ri))
                reg['question'].append(sum(r[4] <= i < r[3] for i in ri))
        print(f'{rr} <- {dr}: {len(cov)} items, aligned share min {min(cov):.3f} median {sorted(cov)[len(cov) // 2]:.3f}; aligned positions '
              + ', '.join(f'{k} min {min(v)} median {sorted(v)[len(v) // 2]}' for k, v in reg.items()), flush=True)


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
    if a.lang == 'en':
        cs.Q_ATTR = cs.Q_ATTR_EN
    torch, tok, model, cfg, placement, cls = load(a)
    _, layers = cs.find_layers(model)
    nL = len(layers)
    b0, b1 = (int(x) for x in a.band.split('-'))
    band = tuple(range(b0, b1 + 1))
    assert b1 < nL
    R = Renderer(tok, json.loads(a.on_kwargs))
    splits = {json.loads(l)['id']: json.loads(l)['split'] for l in open(a.splits, encoding='utf-8') if l.strip()}
    bodies = json.loads(Path(a.bodies).read_text(encoding='utf-8'))
    held = [b for b in bodies if splits[b['id']] == 'confirmation']
    if a.exclude_first:
        held = held[a.exclude_first:]
    if a.limit_bodies:
        held = held[:a.limit_bodies]
    dev0 = model.get_input_embeddings().weight.device
    cands = [tok.convert_tokens_to_ids(t) for t in ('<|start|>', '<|im_start|>')]
    content = a.question in ('content', 'arith')  # both are scored as a number answer, one option order
    if a.mode == 'single' and not content:
        letters = {x: tok.encode(x, add_special_tokens=False) for x in LETTERS}
        assert all(len(v) == 1 for v in letters.values())
        letters = {x: v[0] for x, v in letters.items()}

    def render(role, b, oi):
        q = (arith_question(b, a.lang)[0] if a.question == 'arith' else Q_CONTENT[a.lang]) if content else attr_question(ORDERS[oi])
        msgs = [{'role': role, 'content': b['body']}, {'role': 'user', 'content': q}]
        ids, span = R.render(msgs, b['body'])
        g0 = max(i for i in range(len(ids)) if ids[i] in cands)  # the generation prompt's turn-start token
        assert g0 > span[-1], 'no generation prompt after the body'
        qs = R.render(msgs, q)[1][0] if a.region != 'all' else None  # first token of the question text
        assert qs is None or span[-1] < qs < g0
        return ids, span[0], span[-1] + 1, g0, qs  # body [s, e); edited range ends before the generation prompt g0

    def in_region(i, r):
        s, e, g0, qs = r[1], r[2], r[3], r[4]
        return {'all': True, 'body': s <= i < e, 'end': e <= i < qs, 'question': qs is not None and qs <= i < g0}[a.region]

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
    ident = {'experiment': f'P2_role_swap_{a.mode}', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'suite_sha256': hashlib.sha256(Path(cs.__file__).read_bytes()).hexdigest(), 'model': a.model, 'on_kwargs': a.on_kwargs,
             'n_layers': nL, 'band': [b0, b1], 'fit_from': a.fit_from, 'question': a.question, 'region': a.region, 'lang': a.lang,
             'n_eval': len(held), 'exclude_first': a.exclude_first, 'conditions': [c[0] for c in conds], 'batch': a.batch,
             'max_new': a.max_new, 'torch': torch.__version__, 'transformers': __import__('transformers').__version__}
    (out / 'identity.json').write_text(json.dumps(ident, indent=1), encoding='utf-8')
    print(json.dumps(ident), flush=True)

    def donor_coords(drole, b, oi, Bk, r):
        d = render(drole, b, oi)
        ri, di = align(r[:4], d[:4])
        if a.region != 'all':  # --region: keep the aligned pairs whose recipient position lies in the region
            keep = [j for j, i in enumerate(ri) if in_region(i, r)]
            assert keep, f'no aligned position in region {a.region}'
            ri, di = [ri[j] for j in keep], [di[j] for j in keep]
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
    orders = (0,) if content else (0, 1)
    t0 = time.time()
    for name, rrole, drole, which, k in conds:
        Bk = None if which is None else {l: (V if which == 'V' else Rr)[l][:, :k].contiguous() for l in band}
        items = [(b, oi) for b in held for oi in orders if not (out / 'items' / f'{name}__{b["id"]}__o{oi}.json').exists()]
        step = a.batch if a.mode == 'thinking' else 1
        for s0 in range(0, len(items), step):
            chunk = items[s0:s0 + step]
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
            if Bk is not None:
                state.update(mode='write', B=Bk, rows=rows, coords=coords)
            with torch.no_grad():
                if a.mode == 'thinking' or content:
                    g = model.generate(input_ids=x, attention_mask=am, max_new_tokens=(a.max_new if a.mode == 'thinking' else 12),
                                       do_sample=False, pad_token_id=pad)
                else:
                    lp = torch.log_softmax(model(input_ids=x, attention_mask=am, use_cache=False, logits_to_keep=1).logits[:, -1].float(), -1)
            state.update(mode=None, coords=None)
            for j, (b, oi) in enumerate(chunk):
                rec = {'cond': name, 'id': b['id'], 'recipient': rrole, 'donor': drole, 'basis': which, 'k': k, 'order': list(ORDERS[oi])}
                if a.mode == 'single' and not content:
                    p = {kk: float(lp[j, letters[LETTERS[i]]].exp()) for i, kk in enumerate(ORDERS[oi])}
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
                        expected = arith_question(b, a.lang)[1] if a.question == 'arith' else re.findall(r'\d+', b['body'])[-1]
                        rec['content'] = {'expected': expected, 'given': nums[0] if nums else None, 'correct': bool(nums) and nums[0] == expected}
                    else:
                        mm = re.search(r'([ABC])', tail.replace('**', ''))
                        letter = mm.group(1) if mm else None
                        rec.update(on_letter=letter, on_answer=ORDERS[oi][LETTERS.index(letter)] if letter else None)
                    rec.update(finished=(a.mode == 'single') or any(cl in gen for cl in CLOSE), n_new=int(n_new), tail=tail[-300:],
                               gen=gen if a.save_gen else None)
                (out / 'items' / f'{name}__{b["id"]}__o{oi}.json').write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
            print(f'{name} {s0 + len(chunk)}/{len(items)} {time.time() - t0:.0f}s', flush=True)
        summarize(out)
    for hd in handles:
        hd.remove()
    summarize(out)


def summarize(out):
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    S = {}
    rng = random.Random(0)
    by = {}
    for r in recs:
        by.setdefault((r['id'], tuple(r['order'])), {})[r['cond']] = r
    for c in sorted({r['cond'] for r in recs}):
        rs = [r for r in recs if r['cond'] == c]
        if 'p' in rs[0]:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('thinking', 'single'), required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--on-kwargs', default='{}')
    p.add_argument('--bodies', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--fit-from', required=True)
    p.add_argument('--band', required=True, help='first-last layer edited, e.g. 0-63 or 39-39')
    p.add_argument('--conds', required=True, help='name:recipient:donor:V|R|-:k,...')
    p.add_argument('--out', required=True)
    p.add_argument('--lang', choices=('en', 'zh'), required=True)
    p.add_argument('--question', choices=('attr', 'content', 'arith'), default='attr',
                   help='attr: who wrote the message; content: its last number; arith: an unrelated two-digit multiplication')
    p.add_argument('--region', choices=('all', 'body', 'end', 'question'), default='all',
                   help='replace only at aligned positions in the body, in the template tokens between the body and the question text '
                        '(the end of the message), or in the question text up to the generation prompt')
    p.add_argument('--limit-bodies', type=int, default=0)
    p.add_argument('--exclude-first', type=int, default=0)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--max-new', type=int, default=3072)
    p.add_argument('--max-memory-gib', type=float, default=44)
    p.add_argument('--cpu-offload-gib', type=float, default=0)
    p.add_argument('--save-gen', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--check-align', action='store_true', help='tokenizer only: alignment coverage for every held-out body')
    a = p.parse_args()
    if a.check_align:
        return check_align(a)
    summarize(Path(a.out)) if a.summarize_only else run(a)


if __name__ == '__main__':
    main()
