"""Cross-model thinking pilot: does the model recognize its own messages when it reasons first?

Template-native renderings only. Non-thinking: native letter probabilities at the response onset (--off-kwargs). Thinking:
the template's thinking setting (--on-kwargs); greedy generation (up to --max-new tokens); the answer is the first option letter
written after the end of the reasoning (the text after the last reasoning-close marker, else the whole output).
Items: the 30 pilot bodies x 3 roles x the first two cyclic option orders.
usage: crossmodel_thinking.py --model DIR --off-kwargs JSON --on-kwargs JSON --bodies-zh F --out DIR [--max-new 3072]"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402
from crossmodel_suite import Renderer, load, attr_question, ORDERS, LETTERS, ROLES  # noqa: E402

CLOSE = ('</think>', '<channel|>', '</seed:think>')


def run(args):
    out = Path(args.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    if args.lang == 'en':
        cs.Q_ATTR = cs.Q_ATTR_EN  # attr_question reads the suite module's question
    torch, tok, model, cfg, placement, cls = load(args)
    Roff, Ron = Renderer(tok, json.loads(args.off_kwargs)), Renderer(tok, json.loads(args.on_kwargs))
    letters = {x: tok.encode(x, add_special_tokens=False)[0] for x in LETTERS}
    dev0 = model.get_input_embeddings().weight.device
    bodies = json.loads(Path(args.bodies_zh).read_text(encoding='utf-8'))
    t0 = time.time()
    for b in bodies:
        for oi, order in enumerate(ORDERS[:2]):
            for r in ROLES:
                f = out / 'items' / f'{b["id"]}__{r}__o{oi}.json'
                if f.exists():
                    continue
                msgs = [{'role': r, 'content': b['body']}, {'role': 'user', 'content': attr_question(order)}]
                ids, _ = Roff.render(msgs, b['body'])
                x = torch.tensor([ids], dtype=torch.long, device=dev0)
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1).logits[0, -1].float(), -1)
                off = {k: float(lp[letters[LETTERS[i]]].exp()) for i, k in enumerate(order)}
                ids_on, _ = Ron.render(msgs, b['body'])
                x = torch.tensor([ids_on], dtype=torch.long, device=dev0)
                with torch.no_grad():
                    g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=args.max_new, do_sample=False)
                gen = tok.decode(g[0, len(ids_on):], skip_special_tokens=False)
                tail = gen
                for c in CLOSE:
                    if c in gen:
                        tail = gen.split(c)[-1]
                m = re.search(r'\b([ABC])\b|([ABC])', tail.replace('**', ''))
                letter = (m.group(1) or m.group(2)) if m else None
                ans = order[LETTERS.index(letter)] if letter else None
                rec = {'id': b['id'], 'role': r, 'order': list(order), 'off': off, 'on_answer': ans, 'on_letter': letter,
                       'n_new': int(g.shape[1] - len(ids_on)), 'finished': any(c in gen for c in CLOSE), 'tail': tail[-300:],
                       'gen': gen if args.save_gen else None}
                f.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
        print(f'{b["id"]} {time.time() - t0:.0f}s', flush=True)
    summarize(out)


def summarize(out):
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    true = {'assistant': 'self', 'user': 'user', 'system': 'system'}
    S = {}
    for r in ROLES:
        rs = [x for x in recs if x['role'] == r]
        if not rs:
            continue
        off_top = np.mean([max(x['off'], key=x['off'].get) == true[r] for x in rs])
        on_ok = np.mean([x['on_answer'] == true[r] for x in rs])
        S[r] = {'n': len(rs), 'off_p_true': float(np.mean([x['off'][true[r]] for x in rs])), 'off_top_correct': float(off_top),
                'on_correct': float(on_ok), 'on_answers': {a: float(np.mean([x['on_answer'] == a for x in rs])) for a in ('self', 'user', 'system', None)},
                'median_new_tokens': float(np.median([x['n_new'] for x in rs])), 'unfinished': float(np.mean([not x['finished'] for x in rs]))}
        print(r, json.dumps(S[r]))
    (out / 'summary.json').write_text(json.dumps(S, indent=1), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--off-kwargs', default='{"enable_thinking": false}')
    p.add_argument('--on-kwargs', default='{}')
    p.add_argument('--bodies-zh', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-new', type=int, default=3072)
    p.add_argument('--max-memory-gib', type=float, default=44)
    p.add_argument('--save-gen', action='store_true', help='keep the full generated text')
    p.add_argument('--lang', choices=('zh', 'en'), default='zh', help='attribution question language; bodies come from --bodies-zh as given')
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
