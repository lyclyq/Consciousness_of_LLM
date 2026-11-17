"""P2 check M: does it matter for reasoning whether an intermediate step was written by the model itself?

Native conversations only (chat template default, thinking off):
  [user: two-step word problem] -> [ROLE: 'step 1: ... = s1'] -> [user: 'what is the final answer? digits only'] + generation prompt
ROLE is assistant (the model's own earlier turn; the template gives it its default history formatting) or user. The step
is either correct or wrong by delta in {±1, ±2, ±3}; problems are chosen so that the correct final answer f and the
answer that propagates the wrong step (f + delta) differ in their first digit. Readout at the response onset (native
logits, full-vocabulary softmax): P(first digit of f), P(first digit of f + delta), digit mass; greedy answer (generation,
6 tokens) for the unpatched baselines.
Patching (wrong-step conversations): assistant <- user and user <- assistant, region x layer, as in check I. Tokens before
the step message share positions; the step body and everything after align with a shift equal to the difference in
header length (the assistant formatting has no counterpart and is never patched). Regions: role word + newline of the
step message, step body, post (end of the step message, the final question, the generation prompt), body + post.
Single teacher-forced forward per condition, use_cache=False.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_null_pilot as base  # noqa: E402

FAMILIES = [
    ('inventory', '仓库里原有{a}箱货物，上午运走{b}箱，下午又运来{c}箱。现在仓库里有多少箱货物？', '第一步：上午运走后剩下 {a}-{b}={s} 箱。', '-', '+'),
    ('shelf', '书架上原有{a}本书，管理员又放上{b}本，随后借出{c}本。书架上现在有多少本书？', '第一步：放上之后共有 {a}+{b}={s} 本。', '+', '-'),
    ('class', '每组有{a}名学生，共{b}组，后来又转来{c}名学生。现在一共有多少名学生？', '第一步：原有学生 {a}×{b}={s} 名。', '*', '+'),
    ('tank', '水箱里有{a}升水，早上用掉{b}升，晚上又用掉{c}升。水箱里还剩多少升水？', '第一步：早上用掉后剩下 {a}-{b}={s} 升。', '-', '-'),
    ('station', '车站原有{a}辆车，上午开来{b}辆，下午又开来{c}辆。车站现在有多少辆车？', '第一步：上午开来后共有 {a}+{b}={s} 辆。', '+', '+'),
]
QUESTION = '这道题的最终答案是多少？只输出数字。'
ROLES = ('assistant', 'user')
PAIRS = (('assistant', 'user'), ('user', 'assistant'))


def op(x, y, o):
    return x - y if o == '-' else x + y if o == '+' else x * y


def make_problems(n, seed):
    rng = random.Random(seed)
    out, k = [], 0
    while len(out) < n:
        name, prob, step, o1, o2 = FAMILIES[k % len(FAMILIES)]
        if o1 == '*':
            a, b = rng.randint(3, 9), rng.randint(3, 9)
        else:
            a, b = rng.randint(20, 70), rng.randint(5, 30)
        s = op(a, b, o1)
        c = rng.randint(5, 40)
        f = op(s, c, o2)
        if not (s > 0 and 20 <= f <= 98):
            continue
        # step 2 is s + c or s - c, so a wrong step s + d propagates to the final answer f + d
        deltas = [d for d in (-3, -2, -1, 1, 2, 3) if s + d > 0 and 10 <= f + d <= 99 and str(f + d)[0] != str(f)[0]]
        if not deltas:
            continue
        d = rng.choice(deltas)
        fw = f + d
        out.append({'id': f'm{len(out):03d}_{name}', 'family': name, 'split': 'discovery' if len(out) % 2 == 0 else 'confirmation',
                    'problem': prob.format(a=a, b=b, c=c), 'step_correct': step.format(a=a, b=b, s=s),
                    'step_wrong': step.format(a=a, b=b, s=s + d), 'delta': d, 'answer': f, 'answer_if_adopted': fw})
        k += 1
    return out


def conversation(tok, p, role, which):
    msgs = [{'role': 'user', 'content': p['problem']}, {'role': role, 'content': p[f'step_{which}']},
            {'role': 'user', 'content': QUESTION}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = base.ids_of(enc)
    offsets = enc['offset_mapping']
    if offsets and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    step = p[f'step_{which}']
    s = text.index(step)
    assert text.count(step) == 1
    e = s + len(step)
    body = [i for i, (x, y) in enumerate(offsets) if y > s and x < e and y > x]
    assert body == list(range(body[0], body[-1] + 1))
    ref = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    assert base.ids_of(ref) == ids, 'text and token renderings disagree'
    starts = [i for i, t in enumerate(ids) if t == ids[0]]
    return ids, body, starts[1]


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    problems = json.loads(Path(args.problems).read_text(encoding='utf-8'))
    k, n = (int(x) for x in args.shard.split('/'))
    mine = [p for i, p in enumerate(problems) if i % n == k]
    torch, tok, model, layers, norm, head, embed, placement, dtypes = base.load_model(args)
    digits = {str(d): tok.encode(str(d), add_special_tokens=False) for d in range(10)}
    assert all(len(v) == 1 for v in digits.values())
    digit_ids = {d: v[0] for d, v in digits.items()}
    L = [int(x) for x in args.layers.split(',')]
    base.atomic(out / f'identity_{k}.json', {'experiment': 'P2_reasoning_ownership_M', 'script_sha256': base.sha_file(__file__),
        'problems': args.problems, 'shard': args.shard, 'layers': L, 'device_map': placement, 'torch': torch.__version__,
        'host': __import__('socket').gethostname(),
        'mode': 'native template default conversations; readout teacher-forced; baseline answers by greedy generation (labelled)'})
    dev0 = embed.weight.device
    state = {'mode': None}

    def hook_for(layer):
        def hook(module, inputs, output):
            mode = state['mode']
            if mode is None:
                return None
            h = output[0] if isinstance(output, tuple) else output
            if mode == 'capture':
                state['cap'][layer] = h[0].detach().clone()
                return None
            if mode == 'replace' and (state['layer'] == 'all' or state['layer'] == layer):
                h[0, state['rpos']] = state['donor'][layer][state['dpos']].to(h.device)
                return output
            return None
        return hook
    handles = [layers[l].register_forward_hook(hook_for(l)) for l in range(len(layers))]

    def readout(ids, p):
        x = torch.tensor([ids], dtype=torch.long, device=dev0)
        with torch.no_grad():
            o = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=1)
            lp = torch.log_softmax(o.logits[0, -1].float(), -1)
        dp = {d: float(lp[i].exp()) for d, i in digit_ids.items()}
        return {'p_correct_digit': dp[str(p['answer'])[0]], 'p_adopt_digit': dp[str(p['answer_if_adopted'])[0]],
                'digit_mass': sum(dp.values()), 'top_id': int(lp.argmax())}

    def greedy(ids):
        x = torch.tensor([ids], dtype=torch.long, device=dev0)
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=6, do_sample=False)
        return tok.decode(g[0, len(ids):], skip_special_tokens=True)

    t0 = time.time()
    for p in mine:
        done = out / 'problems' / (p['id'] + '.json')
        if done.exists():
            continue
        rec = {'id': p['id'], 'split': p['split'], 'family': p['family'], 'results': []}
        seqs = {}
        for which in ('correct', 'wrong'):
            for role in ROLES:
                ids, body, h = conversation(tok, p, role, which)
                seqs[(which, role)] = (ids, body, h)
                state['mode'] = None
                r = readout(ids, p)
                r['greedy'] = greedy(ids)
                rec['results'].append({'cond': 'baseline', 'step': which, 'role': role, **r})
        ia, ba, ha = seqs[('wrong', 'assistant')]
        iu, bu, hu = seqs[('wrong', 'user')]
        assert ha == hu and [ia[i] for i in ba] == [iu[i] for i in bu] and ia[:ha] == iu[:hu]
        shift = ba[0] - bu[0]
        assert ia[ba[-1] + 1:] == iu[bu[-1] + 1:] and len(ia) - len(iu) == shift
        regs = {'role_header': [ha + 1, ha + 2], 'body': ba, 'post': list(range(ba[-1] + 1, len(ia))),
                'body_post': list(range(ba[0], len(ia)))}

        def to_user(ps):
            return [q if q < ha + 3 else q - shift for q in ps]
        caps = {}
        for role, ids in (('assistant', ia), ('user', iu)):
            state.update(mode='capture', cap={})
            x = torch.tensor([ids], dtype=torch.long, device=dev0)
            with torch.no_grad():
                model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=1)
            caps[role] = state['cap']
            state['mode'] = None
        for recipient, donor in PAIRS:
            for region, pa in regs.items():
                pu = to_user(pa)
                rpos, dpos = (pa, pu) if recipient == 'assistant' else (pu, pa)
                for l in L + ['all']:
                    state.update(mode='replace', layer=l, rpos=rpos, dpos=dpos, donor=caps[donor])
                    r = readout(ia if recipient == 'assistant' else iu, p)
                    state['mode'] = None
                    rec['results'].append({'cond': 'patch', 'step': 'wrong', 'recipient': recipient, 'donor': donor,
                                           'region': region, 'layer': l, **r})
        caps.clear()
        base.atomic(done, rec)
        base.atomic(out / f'status_{k}.json', {'state': 'running', 'done': len(list((out / 'problems').glob('*.json'))),
                                              'planned': len(mine), 'elapsed_s': time.time() - t0})
    for h in handles:
        h.remove()
    base.atomic(out / f'status_{k}.json', {'state': 'complete', 'planned': len(mine), 'elapsed_s': time.time() - t0})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='')
    p.add_argument('--problems', required=True)
    p.add_argument('--out', default='')
    p.add_argument('--make', type=int, default=0, help='write N problems to --problems and exit')
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--check', action='store_true', help='tokenizer-only check of every conversation')
    p.add_argument('--layers', default='0,3,7,15,23,31,39,47,55,63')
    p.add_argument('--shard', default='0/1')
    p.add_argument('--max-memory-gib', type=float, default=38)
    args = p.parse_args()
    if args.make:
        probs = make_problems(args.make, args.seed)
        Path(args.problems).write_text(json.dumps(probs, ensure_ascii=False, indent=1), encoding='utf-8')
        print('wrote', len(probs), 'problems;', {f: sum(q['family'] == f for q in probs) for f, *_ in FAMILIES})
        return
    if args.check:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
        probs = json.loads(Path(args.problems).read_text(encoding='utf-8'))
        for q in probs:
            for which in ('correct', 'wrong'):
                ia, ba, ha = conversation(tok, q, 'assistant', which)
                iu, bu, hu = conversation(tok, q, 'user', which)
                assert ha == hu and ia[:ha] == iu[:hu] and [ia[i] for i in ba] == [iu[i] for i in bu]
                assert ia[ba[-1] + 1:] == iu[bu[-1] + 1:] and ba[0] - bu[0] == 4
        print('ok', len(probs), 'problems; example:', probs[0])
        return
    try:
        run(args)
    except Exception as exc:
        k = args.shard.split('/')[0]
        base.atomic(Path(args.out) / f'status_{k}.json', {'state': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
