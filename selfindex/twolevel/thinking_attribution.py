"""Self-reference after thinking (run label N): does thinking make self-attribution more accurate?

Two official template modes, nothing else changed:
  off : enable_thinking=False (non-thinking mode). Readout = option-letter probabilities at the response onset (one teacher-forced
        forward, native logits).
  on  : template default (thinking on; the template adds its own reasoning-effort system text: a separate system message for
        Assistant/User conditions, merged into the system message for the System condition). The model thinks freely (greedy
        generation, labelled 'generation'); the readout is the option-letter distribution at the position where it writes its
        answer (teacher-forced forward over the model's own generated prefix up to the first option letter after '</think>').
Conversation: [ROLE: body] [user: attribution question (order q1 or q2)] + generation prompt; every sequence is checked to equal
the template's tokenized output. The paper reports the Assistant and User items. System items are run as well (default --roles)
but not reported: in thinking mode the System message also carries the template's reasoning-effort text, so its content differs
between the two modes.
Outputs: identity_<k>.json, items/<body>__<role>__<order>.json, status_<k>.json.
usage: python -m selfindex.twolevel.thinking_attribution --model DIR --out DIR [--roles assistant,user,system] [--n-bodies N]
       [--max-new-tokens T] [--shard K/N] [--max-memory-gib G]
"""
from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path

from .. import __version__
from ..render import ids_of
from . import base as B

ORDERS = B.ORDERS
THINK_END = '</think>'
LABELS = {'off': 'teacher-forced prompt forward',
          'on': 'greedy generation with thinking; answer-position readout teacher-forced on the generated prefix'}


def render(tok, role, body, order, thinking):
    msgs = [{'role': role, 'content': body}, {'role': 'user', 'content': B.question(order)}]
    kw = {} if thinking else {'enable_thinking': False}
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    ids = ids_of(tok(text, add_special_tokens=False))
    ref = ids_of(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw))
    assert ids == ref, 'text and token renderings disagree'
    return ids


def probs(lp, letters, order):
    p = {L: math.exp(float(lp[i])) for L, i in letters.items()}
    named = dict(zip(ORDERS[order], (p['A'], p['B'], p['C'])))
    return {'p_self': named['self'], 'p_user': named['user'], 'p_system': named['system'], 'letter_mass': sum(p.values())}


def run(args, loaded=None):
    out = Path(args.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    L_ = B.load(args, loaded)
    torch, tok, model = L_.torch, L_.tok, L_.model
    layers, norm, head, embed = B.model_parts(model)
    letters = B.letter_ids(tok)
    think_end = tok.convert_tokens_to_ids(THINK_END)
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    bodies = B.all_bodies()
    roles = args.roles.split(',')
    items = [(b, r, o) for b in bodies[:args.n_bodies] for r in roles for o in ORDERS]
    k, n = (int(x) for x in args.shard.split('/'))
    items = [it for i, it in enumerate(items) if i % n == k]
    B.write_json(out / f'identity_{k}.json', {'experiment': 'P2_thinking_attribution_N', 'selfindex_version': __version__,
        'roles': roles, 'n_bodies': args.n_bodies, 'max_new_tokens': args.max_new_tokens, 'shard': args.shard,
        'device_map': L_.placement, 'torch': torch.__version__, 'labels': LABELS})
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
        B.write_json(done, rec)
        B.write_json(out / f'status_{k}.json', {'state': 'running', 'done': len(list((out / 'items').glob('*.json'))),
                                                'planned_this_shard': len(items), 'elapsed_s': time.time() - t0})
    B.write_json(out / f'status_{k}.json', {'state': 'complete', 'planned_this_shard': len(items), 'elapsed_s': time.time() - t0})


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--roles', default='assistant,user,system')
    p.add_argument('--n-bodies', type=int, default=150, help='the first N bodies of the fixed order (default: all 150)')
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--shard', default='0/1', help='K/N: run the items whose index i (body, role, order) has i %% N == K')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        k = args.shard.split('/')[0]
        B.write_json(Path(args.out) / f'status_{k}.json', {'state': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
