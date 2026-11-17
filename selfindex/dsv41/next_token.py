"""Next-token prediction for DeepSeek-V4.1-Flash-1017 in the natural order (Table tab:self_kl_natural, the template that accepts
only the natural order), with the official model code and the PyTorch kernels of selfindex.dsv41.runtime.

The measurement is that of selfindex.next_token --measure split --order natural3, which the other two models of that table ran:
on the body positions t (all common body tokens but the last, predicting token t+1)
  p_ctrl      probability mass on control tokens, median over t
  p_close     probability of the token the encoder writes right after this body, median over t
  lp_next     mean log-probability of the actual next body token
  kl_body     full KL(X || Y), median over t
  kl_content  KL between the content parts (control tokens removed, each renormalised), median over t
for A-U, A-S and U-S; summary on the held-out bodies, 15 families, 2,000 resamples (selfindex.next_token.summarize_split).
Conditions (chat mode, DeepSeek's own encoder; request and reply request of items/templates.json):
  A  [user: request] [assistant: body] [user: reply request]
  U  [user: request] [user: body] [user: reply request]   -> the encoder merges it into one user turn (checked per body)
  S  [user: request] [system: body] [user: reply request]
Body positions: the tokens common to the three renderings (render.common_body). Log-probabilities at every body position: the
model's own final step (hc_pre with the last layer's pre_mix, the norm, the head) applied at the body positions as well as at the
last one (head(..., full_logits=True)); hc_pre and the norm act position by position. Every sequence is computed on its own,
layer by layer over groups of --group-bodies bodies (runtime.forward_seqs). Control tokens: the tokenizer's special ids and added
tokens, and logit rows >= len(tokenizer).
--check: on the first group, the last-position log-probabilities computed here are compared with runtime.forward_seqs for the
first sequence (check_shard<k>.json). --shard k --n-shards n: one process per GPU on every n-th body; then --finalize (no model):
every body on disk, identity.json, summary, DONE. Resumable: one json per body in OUT/bodies. --stop-file F: stop before the next
group if F exists. --dry-run: render every body with the tokenizer and the encoder only.
usage: python -m selfindex.dsv41.next_token --model-dir D --out DIR [--shard k --n-shards n] [--group-bodies 21] [--limit N]
         [--check] [--stop-file F] [--official-code DIR]
       python -m selfindex.dsv41.next_token --model-dir D --out DIR --finalize
       python -m selfindex.dsv41.next_token --model-dir D --dry-run
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from .. import __version__
from .. import items as I
from ..next_token import PAIRS, control_ids, summarize_split
from ..render import common_body
from . import OFFICIAL
from . import kernel_torch as K
from .runtime import (LANG, Renderer, _load_file, _snap, atomic, build, code_hashes, device_name, device_or_exit, forward_seqs,
                      load_official, patch_runtime, sha, stage_layer, unstage_layer)

ROLES = (('A', 'assistant'), ('U', 'user'), ('S', 'system'))


def render_body(R, body):
    q = I.reply_request(LANG)
    r = {rn: R.render(cond, body, q) for rn, cond in ROLES}
    assert r['U'][0] == R.render('user_merged', body, q)[0], 'expected the encoder to merge the user turns'
    com = common_body({rn: (v[0], v[1]) for rn, v in r.items()})
    return {rn: {'ids': v[0], 'span': v[1], 'common': com[rn], 'close': v[0][v[1][-1] + 1]} for rn, v in r.items()}


@torch.inference_mode()
def forward_all(M, model, seqs, positions, dev):
    """runtime.forward_seqs with the final step applied at the given positions and at the last one: per sequence the
    log-probabilities [len(positions) + 1, vocab] (float32, on the host), the last row being the last position."""
    sa = M.shared_attn
    st = []
    for s in seqs:
        ids = torch.tensor([s], device=dev)
        h = model.embed(ids).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        st.append({'hashes': model.engram_hash(ids, 0, None) if model.engram_hash is not None else None, 'h': h,
                   'pre': M.make_identity_pre_mix(h, model.hc_mult), 'shared': (None, None, None, None)})
    for layer in model.layers:
        stage_layer(layer, dev)
        for x in st:
            sa.compress_kv, sa.index_k, sa.topk_idxs, sa.candidates = x['shared']
            h = x['h']
            if layer.engram is not None:
                h = layer.engram(h, x['hashes'][:, :, layer.engram.layer_hash_index, :], None)
            x['h'], x['pre'] = layer(h, 0, x['pre'], None)
            x['shared'] = (_snap(sa.compress_kv), _snap(sa.index_k), sa.topk_idxs, sa.candidates)
        unstage_layer(layer)
        if torch.device(dev).type == 'cuda':
            torch.cuda.empty_cache()
    out = []
    for x, pos in zip(st, positions):
        hl = layer.hc_pre(x['h'], x['pre'])  # [1, S, dim]: the four streams collapsed at every position
        p = torch.tensor(list(pos) + [x['h'].shape[1] - 1], device=dev)
        lg = model.head(model.norm(hl[:, p]), full_logits=True)[0].float()
        out.append(torch.log_softmax(lg, -1).cpu())
        del hl, lg
    return out


def body_record(b, R, lps, mask):
    """One body's record. Everything here is on the host, where forward_all returns the log-probabilities: run() sets the default
    device to the GPU, so every tensor made here names its device."""
    rec = {'id': b['id'], 'split': b.get('split'), 'category': b.get('category'), 'roles': {}, 'pairs': {}}
    parts = {}
    for rn, v in R.items():
        c = v['common']
        lp = lps[rn][:-1]
        nxt = torch.tensor([v['ids'][i + 1] for i in c[:-1]], device='cpu')
        p = lp.exp()
        lc = lp[:, ~mask]
        lc = lc - torch.logsumexp(lc, -1, keepdim=True)
        rec['roles'][rn] = {'p_ctrl': float(p[:, mask].sum(-1).median()), 'p_close': float(p[:, v['close']].median()),
                            'lp_next': float(lp[torch.arange(len(nxt), device='cpu'), nxt].mean()), 'n_body_tokens': len(c),
                            'n_tokens': len(v['ids']), 'close_token': v['close']}
        parts[rn] = (lp, lc)
    for X, Y in PAIRS:
        (lx, cx), (ly, cy) = parts[X], parts[Y]
        rec['pairs'][f'{X}-{Y}'] = {'kl_body': float((lx.exp() * (lx - ly)).sum(-1).median()),
                                    'kl_content': float((cx.exp() * (cx - cy)).sum(-1).median())}
    return rec


def bodies_of(a):
    bodies = list(I.bodies(LANG))
    return bodies[:a.limit] if a.limit else bodies


def encoder_and_tok(a):
    E = _load_file('ds_encoding', Path(a.official_code or OFFICIAL) / 'encoding' / 'encoding.py', register=False)
    from transformers import AutoTokenizer
    return E, AutoTokenizer.from_pretrained(a.model_dir)


def dry_run(a):
    E, tok = encoder_and_tok(a)
    R = Renderer(E, tok, False)
    bodies = bodies_of(a)
    n = [len(render_body(R, b['body'])['A']['common']) for b in bodies]
    r0 = render_body(R, bodies[0]['body'])
    print(json.dumps({'bodies': len(bodies), 'common_body_tokens_min_max': [min(n), max(n)], 'thinking_mode': R.kw,
                      'close': {rn: tok.decode([v['close']]) for rn, v in r0.items()},
                      'n_tokens_body0': {rn: len(v['ids']) for rn, v in r0.items()},
                      'n_control_ids': len(control_ids(tok)), 'len_tokenizer': len(tok),
                      'tokenizer_json_sha256': sha(Path(a.model_dir) / 'tokenizer.json')}, ensure_ascii=False, indent=1))
    for rn, v in r0.items():
        print(f'===== {rn}: {len(v["ids"])} tokens, body tokens {len(v["common"])}, close {tok.decode([v["close"]])!r}')
        print(tok.decode(v['ids']))
    print('DRY RUN OK')


def run(a, allow_cpu=False):
    out = Path(a.out)
    (out / 'bodies').mkdir(parents=True, exist_ok=True)
    dev = device_or_exit(allow_cpu)
    M, E = load_official(a.official_code)
    t_load = time.time()
    model, tok, args, ck, loaded, n_host = build(M, a.model_dir, 1, a.max_seq, dev)
    torch.set_default_device(dev)  # as selfindex.dsv41.natural: only after the model is built on the host
    stats = {'expert_s': 0.0}
    patch_runtime(M, dev, stats)
    R = Renderer(E, tok, False)
    ctrl = control_ids(tok)
    V = args.vocab_size
    mask = torch.zeros(V, dtype=torch.bool, device='cpu')  # on the host with the log-probabilities (see body_record)
    mask[torch.tensor([i for i in ctrl if i < V], dtype=torch.long, device='cpu')] = True
    mask[len(tok):] = True
    bodies = bodies_of(a)
    if a.n_shards > 1:
        bodies = bodies[a.shard::a.n_shards]
    ident = {'experiment': 'P2_modeswitch_nat3_dsv41_20261108', 'selfindex_version': __version__, 'kernel_sha256': sha(K.__file__),
             **code_hashes(M), 'tokenizer_json_sha256': sha(Path(a.model_dir) / 'tokenizer.json'), 'model_dir': a.model_dir,
             'config': {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}, **R.kw,
             'request': I.natural_order_request(LANG), 'reply_question': I.reply_request(LANG), 'n_control_ids': len(ctrl),
             'len_tokenizer': len(tok), 'vocab': V, 'control_tokens_sample': [tok.convert_ids_to_tokens(i) for i in ctrl[:40]],
             'n_host_memory_tensors': n_host, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': device_name(dev),
             'shard': a.shard, 'n_shards': a.n_shards, 'n_bodies': len(bodies), 'group_bodies': a.group_bodies,
             'load_s': time.time() - t_load}
    atomic(out / f'identity_shard{a.shard}.json', ident)
    print(json.dumps({k: ident[k] for k in ('n_bodies', 'n_control_ids', 'len_tokenizer', 'vocab', 'load_s')}), flush=True)
    todo = [b for b in bodies if not (out / 'bodies' / f'{b["id"]}.json').exists()]
    print(f'{len(bodies)} bodies in this shard, {len(todo)} to run', flush=True)
    t0, done, checked = time.time(), 0, not a.check
    for i in range(0, len(todo), a.group_bodies):
        if a.stop_file and Path(a.stop_file).exists():
            print(f'[stop] {a.stop_file} exists: stopping (rerun resumes)', flush=True)
            break
        grp = todo[i:i + a.group_bodies]
        Rs = [render_body(R, b['body']) for b in grp]
        seqs, poss = [], []
        for Rb in Rs:
            for rn, _ in ROLES:
                seqs.append(Rb[rn]['ids'])
                poss.append(Rb[rn]['common'][:-1])
        assert max(len(s) for s in seqs) + 2 <= a.max_seq, 'raise --max-seq'
        lps = forward_all(M, model, seqs, poss, dev)
        if not checked:
            ref = forward_seqs(M, model, [seqs[0]], [None], dev)[0][0]
            rep = {'body': grp[0]['id'], 'role': 'A', 'n_tokens': len(seqs[0]),
                   'max_abs_last_logprob_diff': float((ref - lps[0][-1]).abs().max()),
                   'same_argmax': int(ref.argmax()) == int(lps[0][-1].argmax())}
            atomic(out / f'check_shard{a.shard}.json', rep)
            print('check', rep, flush=True)
            checked = True
        for j, (b, Rb) in enumerate(zip(grp, Rs)):
            rec = body_record(b, Rb, {rn: lps[3 * j + k] for k, (rn, _) in enumerate(ROLES)}, mask)
            atomic(out / 'bodies' / f'{b["id"]}.json', rec)
            done += 1
        print(f'{done}/{len(todo)} bodies, {time.time() - t0:.0f}s, experts {stats["expert_s"]:.0f}s; last {grp[-1]["id"]}: '
              + json.dumps(rec['pairs']), flush=True)
    n_all = sum(1 for b in bodies if (out / 'bodies' / f'{b["id"]}.json').exists())
    atomic(out / f'status_shard{a.shard}.json', {'bodies_on_disk': n_all, 'n_bodies': len(bodies), 'elapsed_s': time.time() - t0,
                                                 'expert_s': stats['expert_s']})
    print(f'shard {a.shard}: {n_all}/{len(bodies)} bodies on disk', flush=True)


def finalize(a):
    out = Path(a.out)
    bodies = bodies_of(a)
    missing = [b['id'] for b in bodies if not (out / 'bodies' / f'{b["id"]}.json').exists()]
    assert not missing, f'{len(missing)} bodies missing, e.g. {missing[:3]}'
    idents = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(out.glob('identity_shard*.json'))]
    for other in idents[1:]:
        for k in ('selfindex_version', 'kernel_sha256', 'official_model_py_sha256', 'encoding_py_sha256', 'tokenizer_json_sha256',
                  'config', 'thinking_mode', 'n_control_ids'):
            assert other[k] == idents[0][k], f'shards differ in {k}'
    ident = {k: v for k, v in idents[0].items() if k not in ('shard', 'load_s', 'n_bodies', 'gpu')}
    ident.update(n_bodies=len(bodies), shards=[{k: d.get(k) for k in ('shard', 'gpu', 'n_bodies', 'load_s')} for d in idents])
    atomic(out / 'identity.json', ident)
    summarize_split(out)
    (out / 'DONE').write_text(time.strftime('%F %T') + '\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model-dir', required=True, help='the checkpoint folder (safetensors shards, tokenizer, inference/config.json)')
    p.add_argument('--out', default=None)
    p.add_argument('--official-code', default=None, help='folder with inference/ and encoding/ of the release (default: official/)')
    p.add_argument('--limit', type=int, default=0, help='only the first N bodies')
    p.add_argument('--group-bodies', type=int, default=21, help='bodies per group sharing the staging of the experts')
    p.add_argument('--max-seq', type=int, default=1024)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=1)
    p.add_argument('--check', action='store_true', help='compare the last-position log-probabilities with runtime.forward_seqs once')
    p.add_argument('--stop-file', default=None)
    p.add_argument('--dry-run', action='store_true', help='render and check with the tokenizer and encoder; no model, no GPU')
    p.add_argument('--finalize', action='store_true', help='after all shards: identity, summary and DONE (no model)')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.dry_run:
        dry_run(a)
        return
    assert a.out, '--out is required'
    if a.finalize:
        finalize(a)
        return
    try:
        run(a)
    except Exception as exc:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        atomic(Path(a.out) / f'failed_shard{a.shard}.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
