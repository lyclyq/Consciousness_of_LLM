"""Natural-order attribution check (user-approved 2026-10-16 for Qwen3.8-Flash-Next).

The main probe opens the conversation with the message under test ([ROLE: body] [user: question]) because a System message must
be the first message in these templates. Here the message under test follows a user request, as an assistant reply normally does:
    [user: request] [ROLE: body] [user: who wrote the message before this question?]   ROLE = assistant / user
(System is left out: the templates refuse a System message after a user turn). Everything else is as in crossmodel_suite phase G:
the same 150 bodies, the three cyclic option orders, the model's own non-thinking rendering checked against tokenize=True, and the
native letter probabilities at the response onset.
Pre-specified comparison: mean P(you | own message) minus mean P(you | user message), body level (three orders averaged), with a
sign test over bodies; compared with the same model's main-probe values.
usage: crossmodel_natural.py --model DIR --tmpl-kwargs JSON --bodies role_bodies.jsonl --out DIR [--lang zh|en]
       [--max-memory-gib G] [--cpu-offload-gib C]"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402

REQ = {'zh': '请写一段简短的描述。', 'en': 'Please write a short description.'}
ROLES2 = ('assistant', 'user')


def sign_p(k, n):
    t = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * t)


def summarize(out):
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted(out.glob('role150_*.json'))]
    if not recs:
        return None
    pa = [r['roles']['assistant']['mean']['self'] for r in recs]
    pu = [r['roles']['user']['mean']['self'] for r in recs]
    d = [a - u for a, u in zip(pa, pu)]
    k = sum(x > 0 for x in d)
    top = {r0: sum(max(('self', 'user', 'system'), key=lambda q: r['roles'][r0]['mean'][q]) == t for r in recs) / len(recs)
           for r0, t in (('assistant', 'self'), ('user', 'user'))}
    S = {'n_bodies': len(recs), 'p_you_assistant': sum(pa) / len(pa), 'p_you_user': sum(pu) / len(pu),
         'diff_mean': sum(d) / len(d), 'diff_positive': k, 'sign_p': sign_p(k, len(d)), 'top_answer_correct': top,
         'answers': {r0: {q: sum(r['roles'][r0]['mean'][q] for r in recs) / len(recs) for q in ('self', 'user', 'system')} for r0 in ROLES2},
         'letter_mass_min': min(min(sum(p[q] for q in ('self', 'user', 'system')) for p in r['roles'][r0]['by_order']) for r in recs for r0 in ROLES2)}
    (out / 'summary.json').write_text(json.dumps(S, indent=1), encoding='utf-8')
    print(json.dumps(S), flush=True)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tmpl-kwargs', default='{"enable_thinking": false}')
    ap.add_argument('--bodies', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lang', choices=('zh', 'en'), default='zh')
    ap.add_argument('--max-memory-gib', type=float, default=44)
    ap.add_argument('--cpu-offload-gib', type=float, default=0)
    ap.add_argument('--bodies-en', default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.lang == 'en':
        cs.Q_ATTR = cs.Q_ATTR_EN
    torch, tok, model, cfg, placement, cls = cs.load(args)
    kw = json.loads(args.tmpl_kwargs)
    R = cs.Renderer(tok, kw)
    letters = {}
    for x in cs.LETTERS:
        t = tok.encode(x, add_special_tokens=False)
        assert len(t) == 1, (x, t)
        letters[x] = t[0]
    bodies = [json.loads(l) for l in open(args.bodies, encoding='utf-8') if l.strip()]
    if args.lang == 'en':
        en = json.loads(Path(args.bodies_en).read_text(encoding='utf-8'))
        bodies = [{**b, 'body': en[b['id']]} for b in bodies]
    ident = {'experiment': 'P2_crossmodel_natural_order', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'suite_sha256': hashlib.sha256(Path(cs.__file__).read_bytes()).hexdigest(), 'model': args.model, 'tmpl_kwargs': kw,
             'request': REQ[args.lang], 'example_question': cs.attr_question(cs.ORDERS[0]), 'device_map': placement,
             'cpu_offload_gib': args.cpu_offload_gib, 'n_bodies': len(bodies)}
    cs.atomic(out / 'identity.json', ident)
    dev0 = model.get_input_embeddings().weight.device
    t0 = time.time()
    for bi, b in enumerate(bodies):
        f = out / f'{b["id"]}.json'
        if f.exists():
            continue
        rec = {'id': b['id'], 'split': b['split'], 'category': b['category'], 'roles': {}}
        for r in ROLES2:
            ps = []
            for order in cs.ORDERS:
                msgs = [{'role': 'user', 'content': REQ[args.lang]}, {'role': r, 'content': b['body']},
                        {'role': 'user', 'content': cs.attr_question(order)}]
                ids, _ = R.render(msgs, b['body'])
                x = torch.tensor([ids], dtype=torch.long, device=dev0)
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                                                 logits_to_keep=1).logits[0, -1].float(), -1)
                p = {k: float(lp[letters[cs.LETTERS[i]]].exp()) for i, k in enumerate(order)}
                p['top_token'] = tok.decode([int(lp.argmax())])
                ps.append(p)
            rec['roles'][r] = {'by_order': ps, 'mean': {k: sum(q[k] for q in ps) / len(ps) for k in ('self', 'user', 'system')}}
        cs.atomic(f, rec)
        if bi % 10 == 0:
            print(f'{bi + 1}/{len(bodies)} {time.time() - t0:.0f}s', flush=True)
    summarize(out)


if __name__ == '__main__':
    main()
