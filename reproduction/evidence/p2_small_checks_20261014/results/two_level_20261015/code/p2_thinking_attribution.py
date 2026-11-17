"""P2 check N: does thinking make self-attribution more accurate?

Two official template modes, nothing else changed:
  off : enable_thinking=False (non-thinking mode). Readout = option-letter probabilities at the response onset (one
        teacher-forced forward, native logits), as in checks I-K.
  on  : template default (thinking on; the template adds its own reasoning-effort system text: a separate system message
        for Assistant/User conditions, merged into the system message for the System condition). The model thinks
        freely (greedy generation, labelled 'generation'); the readout is the option-letter distribution at the
        position where it writes its answer (teacher-forced forward over the model's own generated prefix).
Conversation: [ROLE: body] [user: attribution question (order q1 or q2)] + generation prompt; every sequence is asserted
to equal the template's tokenized output.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_null_pilot as base  # noqa: E402
from p2_attribution_patch import ORDERS, question, letter_ids  # noqa: E402
from p2_body_sets import all_bodies  # noqa: E402

THINK_END = '</think>'


def render(tok, role, body, order, thinking):
    msgs = [{'role': role, 'content': body}, {'role': 'user', 'content': question(order)}]
    kw = {} if thinking else {'enable_thinking': False}
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    ids = base.ids_of(tok(text, add_special_tokens=False))
    ref = base.ids_of(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw))
    assert ids == ref, 'text and token renderings disagree'
    return ids


def probs(lp, letters, order):
    p = {L: math.exp(float(lp[i])) for L, i in letters.items()}
    named = dict(zip(ORDERS[order], (p['A'], p['B'], p['C'])))
    return {'p_self': named['self'], 'p_user': named['user'], 'p_system': named['system'], 'letter_mass': sum(p.values())}


def run(args):
    out = Path(args.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    torch, tok, model, layers, norm, head, embed, placement, dtypes = base.load_model(args)
    letters = letter_ids(tok)
    think_end = tok.convert_tokens_to_ids(THINK_END)
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    bodies = all_bodies(args.bodies)
    roles = args.roles.split(',')
    items = [(b, r, o) for b in bodies[:args.n_bodies] for r in roles for o in ORDERS]
    k, n = (int(x) for x in args.shard.split('/'))
    items = [it for i, it in enumerate(items) if i % n == k]
    base.atomic(out / f'identity_{k}.json', {'experiment': 'P2_thinking_attribution_N', 'script_sha256': base.sha_file(__file__),
        'roles': roles, 'n_bodies': args.n_bodies, 'max_new_tokens': args.max_new_tokens, 'shard': args.shard,
        'device_map': placement, 'torch': torch.__version__, 'host': __import__('socket').gethostname(),
        'labels': {'off': 'teacher-forced prompt forward', 'on': 'greedy generation with thinking; answer-position readout teacher-forced on the generated prefix'}})
    dev0 = embed.weight.device

    def letter_lp(ids):
        x = torch.tensor([ids], dtype=torch.long, device=dev0)
        with torch.no_grad():
            o = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=1)
            return torch.log_softmax(o.logits[0, -1].float(), -1)

    t0 = time.time()
    for b, role, order in items:
        name = f"{b['id']}__{role}__{order}"
        done = out / 'items' / (name + '.json')
        if done.exists():
            continue
        rec = {'id': b['id'], 'split': b['split'], 'role': role, 'order': order}
        ids_off = render(tok, role, b['body'], order, False)
        lp = letter_lp(ids_off)
        rec['off'] = {**probs(lp, letters, order), 'top_id': int(lp.argmax())}
        ids_on = render(tok, role, b['body'], order, True)
        t1 = time.time()
        x = torch.tensor([ids_on], dtype=torch.long, device=dev0)
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = g[0, len(ids_on):].tolist()
        rec['on'] = {'gen_tokens': len(gen), 'gen_seconds': time.time() - t1, 'finished': im_end in gen,
                     'thinking_closed': think_end in gen, 'text': tok.decode(gen, skip_special_tokens=False)}
        if think_end in gen:
            t = gen.index(think_end)
            after = gen[t + 1:]
            ans = next((j for j, tid in enumerate(after) if tid in letters.values()), None)
            rec['on']['thinking_tokens'] = t
            if ans is not None:
                prefix = ids_on + gen[:t + 1 + ans]
                lp = letter_lp(prefix)
                rec['on'].update(probs(lp, letters, order))
                rec['on']['answer_letter'] = tok.decode([after[ans]])
                rec['on']['answer_offset_after_think'] = ans
        base.atomic(done, rec)
        base.atomic(out / f'status_{k}.json', {'state': 'running', 'done': len(list((out / 'items').glob('*.json'))),
                                              'planned_this_shard': len(items), 'elapsed_s': time.time() - t0})
    base.atomic(out / f'status_{k}.json', {'state': 'complete', 'planned_this_shard': len(items), 'elapsed_s': time.time() - t0})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--bodies', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--roles', default='assistant,user,system')
    p.add_argument('--n-bodies', type=int, default=150)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--max-memory-gib', type=float, default=38)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        k = args.shard.split('/')[0]
        base.atomic(Path(args.out) / f'status_{k}.json', {'state': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
