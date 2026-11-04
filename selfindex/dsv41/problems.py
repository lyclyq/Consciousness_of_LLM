"""Trust in the model's own steps without thinking (phase M of selfindex.main_probe) for DeepSeek-V4.1-Flash-1017, with the model
code, kernels, weights and per-sequence layer-by-layer prefill of selfindex.dsv41.runtime.

The 120 two-step problems: [user: problem] [ROLE: step] [user: final-answer question], ROLE assistant or user, step correct or
wrong; up to two greedy format tokens before a digit (stop when the top token is a digit or has probability < 0.5); then the
log-probability of the correct answer and of the error-propagating answer, adoption index 1 / (1 + exp(lp_correct - lp_adopt)).
DeepSeek's encoder (chat mode) merges a user-written step with the problem and the question into one user turn, its native
rendering of user-supplied text. The readout is runtime.forward_seqs, log-probabilities at the last position: every candidate answer
is one token in this tokenizer (checked at start), so its score is the log-probability at the end of the prompt plus format prefix.
--shard k --n-shards n: one process per GPU on every n-th problem; then --finalize checks completeness and writes summary.json
(mean adoption index per step and role, and the own-minus-user difference on wrong steps with a bootstrap interval over problems).
usage: python -m selfindex.dsv41.problems --model-dir D --out DIR [--shard K --n-shards N] [--wave-seqs N] [--limit N] [--finalize]
       [--official-code DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .. import __version__
from .. import items as I
from . import kernel_torch as K
from .runtime import LANG, atomic, build, code_hashes, device_name, device_or_exit, forward_seqs, load_official, patch_runtime, sha


def problems_of(a):
    problems = I.problems(LANG)
    if a.limit:
        problems = problems[:a.limit]
    return problems


def msgs(p, which, role):
    return [{'role': 'user', 'content': p['problem']}, {'role': role, 'content': p[f'step_{which}']},
            {'role': 'user', 'content': I.final_answer_question(LANG)}]


def render(E, tok, p, which, role):
    text = E.encode_messages(msgs(p, which, role), thinking_mode='chat')
    step = p[f'step_{which}']
    assert text.count(step) == 1 and text.find(p['problem']) < text.find(step) < text.rfind(I.final_answer_question(LANG)), 'step not in place'
    return list(tok.backend_tokenizer.encode(text, add_special_tokens=False).ids), text


def run_waves(M, model, seqs, dev, n):
    """forward_seqs over groups of n sequences: last-position log-probabilities [vocab]."""
    res = []
    for i in range(0, len(seqs), n):
        res += [lp.reshape(-1) for lp, _ in forward_seqs(M, model, seqs[i:i + n], [None] * len(seqs[i:i + n]), dev)]
    return res


def run(a, allow_cpu=False):
    out = Path(a.out)
    (out / 'M').mkdir(parents=True, exist_ok=True)
    dev = device_or_exit(allow_cpu)
    M, E = load_official(a.official_code)
    model, tok, args, ck, loaded, n_host = build(M, a.model_dir, 1, a.max_seq, dev)
    torch.set_default_device(dev)  # the official code relies on the default device (get_window_topk_idxs)
    stats = {'expert_s': 0.0}
    patch_runtime(M, dev, stats)
    problems = problems_of(a)
    mine = problems[a.shard::a.n_shards]
    todo = [p for p in mine if not (out / 'M' / f"{p['id']}.json").exists()]
    ident = {'experiment': 'P2_M_dsv41', 'selfindex_version': __version__, 'script_sha256': sha(__file__), 'kernel_sha256': sha(K.__file__),
             **{k: v for k, v in code_hashes(M).items() if k != 'official_engram_py_sha256'},
             'tokenizer_json_sha256': sha(Path(a.model_dir) / 'tokenizer.json'), 'thinking_mode': 'chat', 'question': I.final_answer_question(LANG),
             'problems_sha256': sha(I.items_dir() / 'problems' / f'problems_{LANG}_120.json'), 'n_problems': len(problems),
             'shard': a.shard, 'n_shards': a.n_shards, 'torch': torch.__version__, 'gpu': device_name(dev)}
    atomic(out / f'identity_shard{a.shard}.json', ident)
    items = [(p, which, role) for p in todo for which in ('correct', 'wrong') for role in ('assistant', 'user')]
    base = [render(E, tok, p, w, r)[0] for p, w, r in items]
    print(f'shard {a.shard}: {len(todo)} problems, {len(items)} items', flush=True)
    t0 = time.time()
    cand = {}
    for p, w, r in items:
        for c in (str(p['answer']), str(p['answer_if_adopted'])):
            ids_c = tok.encode(c, add_special_tokens=False)
            assert len(ids_c) == 1, f'candidate {c} is {len(ids_c)} tokens'
            cand[c] = ids_c[0]
    # format tokens before the digit, at most two; the readout at the final prefix scores both candidates (one token each), so an
    # item extended twice needs one more forward
    pre = [[] for _ in items]
    final = [None] * len(items)
    open_ = list(range(len(items)))
    for rnd in range(3):
        if not open_:
            break
        lps = run_waves(M, model, [base[i] + pre[i] for i in open_], dev, a.wave_seqs)
        nxt = []
        for i, lp in zip(open_, lps):
            top = int(lp.argmax())
            if rnd == 2 or float(lp[top].exp()) < 0.5 or tok.decode([top]).strip()[:1].isdigit():
                final[i] = lp
                continue
            pre[i].append(top)
            nxt.append(i)
        open_ = nxt
        print(f'round {rnd}: {len(nxt)} items extended; {time.time() - t0:.0f}s; experts {stats["expert_s"]:.0f}s', flush=True)
    score = {i: [float(final[i][cand[str(p['answer'])]]), float(final[i][cand[str(p['answer_if_adopted'])]])]
             for i, (p, w, r) in enumerate(items)}
    by = {}
    for i, (p, w, r) in enumerate(items):
        lc, la = score[i]
        by.setdefault(p['id'], {'id': p['id'], 'family': p['family'], 'split': p['split'], 'results': []})['results'].append(
            {'step': w, 'role': r, 'lp_correct': lc, 'lp_adopt': la, 'adoption_index': 1 / (1 + math.exp(lc - la)),
             'prefix_ids': pre[i], 'n_tokens': len(base[i])})
    for pid, rec in by.items():
        atomic(out / 'M' / f'{pid}.json', rec)
    atomic(out / f'status_shard{a.shard}.json', {'state': 'complete', 'elapsed_s': time.time() - t0, 'expert_s': stats['expert_s']})


def finalize(a):
    out = Path(a.out)
    problems = problems_of(a)
    missing = [p['id'] for p in problems if not (out / 'M' / f"{p['id']}.json").exists()]
    assert not missing, f'{len(missing)} problems missing'
    recs = [json.loads((out / 'M' / f"{p['id']}.json").read_text(encoding='utf-8')) for p in problems]
    S = {'n_problems': len(recs)}
    for w in ('correct', 'wrong'):
        for r in ('assistant', 'user'):
            S[f'{w}_{r}_mean_adoption_index'] = float(np.mean([x['adoption_index'] for rec in recs for x in rec['results']
                                                               if x['step'] == w and x['role'] == r]))
    d = np.array([[x['adoption_index'] for x in rec['results'] if x['step'] == 'wrong' and x['role'] == 'assistant'][0] -
                  [x['adoption_index'] for x in rec['results'] if x['step'] == 'wrong' and x['role'] == 'user'][0] for rec in recs])
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    S['wrong_own_minus_user'] = {'mean': float(d.mean()), 'ci95': [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
    atomic(out / 'summary.json', S)
    print(json.dumps(S), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model-dir', required=True, help='the checkpoint folder (safetensors shards, tokenizer, inference/config.json)')
    p.add_argument('--out', required=True)
    p.add_argument('--official-code', default=None, help='folder with inference/ and encoding/ of the release (default: official/)')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=1)
    p.add_argument('--wave-seqs', type=int, default=64)
    p.add_argument('--max-seq', type=int, default=1024)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--finalize', action='store_true')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    finalize(a) if a.finalize else run(a)


if __name__ == '__main__':
    main()
