"""Natural conversation order for DeepSeek-V4.1-Flash-1017 (geometry and self-reference), with the official model code, the PyTorch
kernels and DeepSeek's own prompt encoder (see selfindex.dsv41.runtime). The DeepSeek-V4.1 format cannot render the main probe (an
opening Assistant message gets no role token), so the model is measured in the natural order only.

Conditions (the request, separator, questions and option orders of items/templates.json):
    assistant    [user: request] [assistant: body] [user: question]
    user         [user: request] [user: body] [user: question]   -> DeepSeek's encoder merges it into one user turn
    user_merged  [user: request + "\\n\\n" + body + "\\n\\n" + question]
    system       [user: request] [system: body] [user: question] (a mid-conversation System message, kept in place by the encoder)
user and user_merged render to the same tokens; each is recorded under its own name (user copied from user_merged).
Geometry: [cond: body] with the reply request as the question; the output of every block (4 hyper-connection streams x hidden),
mean over the body tokens common to all conditions and at the response onset, saved as [layer, stream, hidden].
Self-reference: the probabilities of the option letters at the answer position, three cyclic option orders, after at most two
greedy format tokens that the model writes before its letter.
Rendering: chat mode (the default), or --thinking: DeepSeek's thinking mode with reasoning effort "high" (the encoder default); with
this encoder the two renderings differ only in the "Reasoning Effort" line that opens the conversation and in the last token
(<think> instead of </think>). --geometry-only: the geometry sequences only.
Every sequence is computed on its own; --wave-seqs sequences share the staging of each layer's routed experts. --check-single
verifies on real inputs that this equals a plain forward of the sequence alone, bit for bit (single_check.json).
--shard k --n-shards n: one process per GPU, each on every n-th body; --finalize then renders every body again (tokenizer and
encoder only), checks that every sequence is on disk, assembles the per-body records and summarizes. Resumable: finished sequences
stay on disk (seq/).
Outputs: identity.json (or identity_shard<k>.json), seq/<body>__<cond>__<geo|o0|o1|o2>.json (+ .npz for geometry), N/<body>.json
+ N/<body>.npz (content_<cond>, onset_<cond>: layers x streams x hidden, float16), status.json, summary.json (distance ratios on
the held-out bodies per stream, for the concatenated state and for the mean of the per-stream ratios; mean answer probabilities).
usage: python -m selfindex.dsv41.natural --model-dir D --out DIR [--limit N] [--wave-seqs N] [--max-seq N] [--check-single]
       [--thinking] [--geometry-only] [--shard K --n-shards N] [--finalize] [--summarize-only] [--official-code DIR]
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .. import __version__
from .. import items as I
from ..render import common_body
from . import OFFICIAL
from . import kernel_torch as K
from .runtime import (CONDS, LANG, LETTERS, UNIQUE, Renderer, _load_file, atomic, build, code_hashes, device_name, device_or_exit,
                      forward_batch, forward_seqs, load_official, patch_runtime, sha, waves)

ORDERS = I.orders()


def bodies_of(a):
    bodies = list(I.bodies(LANG))
    if a.limit:
        bodies = bodies[:a.limit]
    return bodies


def run(a, allow_cpu=False):
    out = Path(a.out)
    (out / 'seq').mkdir(parents=True, exist_ok=True)
    (out / 'N').mkdir(exist_ok=True)
    dev = device_or_exit(allow_cpu)
    M, E = load_official(a.official_code)
    t_load = time.time()
    model, tok, args, ck, loaded, n_host = build(M, a.model_dir, 1, a.max_seq, dev)
    # the official code creates some index tensors without a device (get_window_topk_idxs) and relies on the default device
    # being the GPU, as generate.py sets it; the model is built on the host first, so set it only now
    torch.set_default_device(dev)
    stats = {'expert_s': 0.0}
    patch_runtime(M, dev, stats)
    letters = {}
    for x in LETTERS:
        t = tok.encode(x, add_special_tokens=False)
        assert len(t) == 1, (x, t)
        letters[x] = t[0]
    bodies = bodies_of(a)
    if a.n_shards > 1:  # one process per GPU, each on every n-th body; --finalize assembles afterwards
        bodies = bodies[a.shard::a.n_shards]
    R = Renderer(E, tok, a.thinking)
    q_reply = I.reply_request(LANG)
    ident = {'experiment': 'P2_crossmodel_natural3_dsv41', 'selfindex_version': __version__, 'script_sha256': sha(__file__),
             'kernel_sha256': sha(K.__file__), **code_hashes(M), 'tokenizer_json_sha256': sha(Path(a.model_dir) / 'tokenizer.json'),
             'model_dir': a.model_dir, 'config': {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
             'n_layers': args.n_layers, 'hc_mult': args.hc_mult, 'host_memory_tensors': n_host, 'n_loaded_tensors': len(loaded),
             'not_built': 'vision tower, aligner, DSpark/MTP layers (text-only prefill)', **R.kw, 'geometry_only': a.geometry_only,
             'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': device_name(dev),
             'request': I.natural_order_request(LANG), 'separator': I.templates()['natural_order']['merged_separator'],
             'reply_question': q_reply, 'attr_question_example': I.attr_question(ORDERS[0], LANG), 'letters': letters,
             'n_bodies': len(bodies), 'processing': 'each sequence alone (batch of one, no padding), layer by layer over groups of sequences',
             'group_size': a.wave_seqs, 'load_s': time.time() - t_load}
    if a.n_shards > 1:
        ident.update(shard=a.shard, n_shards=a.n_shards)
    atomic(out / ('identity.json' if a.n_shards == 1 else f'identity_shard{a.shard}.json'), ident)
    print(json.dumps({k: ident[k] for k in ('n_layers', 'hc_mult', 'host_memory_tensors', 'n_loaded_tensors', 'load_s')}), flush=True)

    # every sequence to run: geometry (reply question, states captured) and attribution (three orders, letters only)
    items, meta = [], {}
    for b in bodies:
        geo = {c: R.render(c, b['body'], q_reply) for c in CONDS}
        assert geo['user'][0] == geo['user_merged'][0], 'expected the encoder to merge the user turns'
        spans = common_body({c: (geo[c][0], geo[c][1]) for c in CONDS})
        meta[b['id']] = {'body': b, 'spans': spans, 'n_tokens': {c: len(geo[c][0]) for c in CONDS}}
        for c in UNIQUE:
            items.append({'key': f"{b['id']}__{c}__geo", 'ids': geo[c][0], 'capture': spans[c]})
            for oi, order in enumerate(() if a.geometry_only else ORDERS):
                ids, _, _ = R.render(c, b['body'], I.attr_question(order, LANG))
                items.append({'key': f"{b['id']}__{c}__o{oi}", 'ids': ids, 'capture': None})
    assert max(len(it['ids']) for it in items) + 2 <= a.max_seq, 'raise --max-seq'
    todo = [it for it in items if not (out / 'seq' / f"{it['key']}.json").exists()]
    print(f'{len(items)} sequences, {len(todo)} to run, max length {max(len(it["ids"]) for it in items)}', flush=True)

    if a.check_single:
        # the layer-by-layer group path must equal a plain forward of each sequence alone (expert-by-expert copies), bit for bit
        geo_items = [it for it in items if it['capture'] is not None]
        pick = [geo_items[0]] + [it for it in items if it['capture'] is None][:1]
        grouped = forward_seqs(M, model, [p['ids'] for p in pick], [p['capture'] for p in pick], dev)
        rep = {'lengths': [len(p['ids']) for p in pick], 'max_abs_logprob_diff': [], 'max_abs_state_diff': []}
        for j, p in enumerate(pick):
            lp_1, cap_1 = forward_batch(M, model, [p['ids']], 0, [p['capture']])
            rep['max_abs_logprob_diff'].append(float((lp_1[0] - grouped[j][0]).abs().max()))
            if p['capture'] is not None:
                rep['max_abs_state_diff'].append(float(max(np.abs(cap_1[0][k].astype(np.float32) - grouped[j][1][k].astype(np.float32)).max()
                                                           for k in (0, 1))))
        atomic(out / 'single_check.json', rep)
        print('single-sequence check', rep, flush=True)

    t0, done = time.time(), 0
    pending = list(todo)
    for rnd in range(3):  # greedy format tokens before the letter, at most two, as in the main probe
        nxt = []
        for wv in waves(pending, a.wave_seqs):
            res = forward_seqs(M, model, [it['ids'] for it in wv], [it['capture'] for it in wv], dev)
            for j, it in enumerate(wv):
                row, cap = res[j]
                mass = sum(float(row[letters[x]].exp()) for x in LETTERS)
                top = int(row.argmax())
                rec = {'key': it['key'], 'n_tokens': len(it['ids']), 'prefix': it.get('prefix', []),
                       'p_letters': {x: float(row[letters[x]].exp()) for x in LETTERS}, 'letter_mass': mass,
                       'top_id': top, 'top_p': float(row[top].exp())}
                if it['capture'] is not None:
                    np.savez_compressed(out / 'seq' / f"{it['key']}.npz", content=cap[0], onset=cap[1])
                elif mass < 0.5 and rec['top_p'] >= 0.5 and top not in letters.values() and rnd < 2:
                    nxt.append({**it, 'ids': it['ids'] + [top], 'prefix': it.get('prefix', []) + [top]})
                    continue
                atomic(out / 'seq' / f"{it['key']}.json", rec)
                done += 1
            el = time.time() - t0
            print(f'round {rnd} wave {len(wv)} seqs x {max(len(it["ids"]) for it in wv)} tok; done {done}/{len(todo)}; '
                  f'{el:.0f}s; experts {stats["expert_s"]:.0f}s', flush=True)
        pending = nxt
        if not pending:
            break
    if a.n_shards > 1:
        atomic(out / f'status_shard{a.shard}.json', {'state': 'complete', 'elapsed_s': time.time() - t0, 'expert_s': stats['expert_s'],
                                                      'n_bodies': len(bodies)})
        return
    assemble(out, meta, a.geometry_only)
    atomic(out / 'status.json', {'state': 'complete', 'elapsed_s': time.time() - t0, 'expert_s': stats['expert_s']})
    if any(m['body']['split'] == 'confirmation' for m in meta.values()):
        summarize(out)


def finalize(a):
    """After the shards: render every body again (tokenizer and encoder only), check that every sequence is on disk, assemble the
    per-body records and summarize."""
    out = Path(a.out)
    E = _load_file('ds_encoding', Path(a.official_code or OFFICIAL) / 'encoding' / 'encoding.py', register=False)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    R = Renderer(E, tok, a.thinking)
    q_reply = I.reply_request(LANG)
    bodies = bodies_of(a)
    meta, missing = {}, []
    for b in bodies:
        geo = {c: R.render(c, b['body'], q_reply) for c in CONDS}
        spans = common_body({c: (geo[c][0], geo[c][1]) for c in CONDS})
        meta[b['id']] = {'body': b, 'spans': spans, 'n_tokens': {c: len(geo[c][0]) for c in CONDS}}
        for c in UNIQUE:
            for k in ['geo'] + ([] if a.geometry_only else [f'o{i}' for i in range(len(ORDERS))]):
                if not (out / 'seq' / f"{b['id']}__{c}__{k}.json").exists():
                    missing.append(f"{b['id']}__{c}__{k}")
    assert not missing, f'{len(missing)} sequences missing, e.g. {missing[:3]}'
    shards = sorted(out.glob('identity_shard*.json'))
    idents = [json.loads(p.read_text(encoding='utf-8')) for p in shards]
    ident = {k: v for k, v in idents[0].items() if k not in ('shard', 'load_s', 'n_bodies')}
    for other in idents[1:]:
        for k in ('script_sha256', 'kernel_sha256', 'official_model_py_sha256', 'encoding_py_sha256', 'tokenizer_json_sha256', 'config',
                  'thinking_mode', 'geometry_only'):
            assert other[k] == ident[k], f'shards differ in {k}'
    ident.update(n_bodies=len(bodies), shards=[{k: d[k] for k in ('shard', 'gpu', 'n_bodies', 'load_s')} for d in idents],
                 finalize_script_sha256=sha(__file__))
    atomic(out / 'identity.json', ident)
    st = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(out.glob('status_shard*.json'))]
    assemble(out, meta, a.geometry_only)
    atomic(out / 'status.json', {'state': 'complete', 'shards': st, 'elapsed_s': max(s['elapsed_s'] for s in st)})
    summarize(out)


def assemble(out, meta, geometry_only=False):
    """Per body: N/<id>.json and N/<id>.npz (user copied from user_merged); with geometry_only the records carry no attribution."""
    for bid, m in meta.items():
        b = m['body']
        rec = {'id': bid, 'split': b['split'], 'category': b['category'], 'user_is_merged': True, 'conds': {}}
        arrs = {}
        for c in CONDS:
            src = 'user_merged' if c == 'user' else c
            z = np.load(out / 'seq' / f'{bid}__{src}__geo.npz')
            arrs[f'content_{c}'], arrs[f'onset_{c}'] = z['content'], z['onset']
            if geometry_only:
                rec['conds'][c] = {'n_body_tokens': len(m['spans'][c]), 'n_tokens': m['n_tokens'][c]}
                continue
            ps = []
            for oi, order in enumerate(ORDERS):
                r = json.loads((out / 'seq' / f'{bid}__{src}__o{oi}.json').read_text(encoding='utf-8'))
                p = {k: r['p_letters'][LETTERS[i]] for i, k in enumerate(order)}
                ps.append({**p, 'prefix_ids': r['prefix'], 'letter_mass': r['letter_mass']})
            rec['conds'][c] = {'n_body_tokens': len(m['spans'][c]), 'n_tokens': m['n_tokens'][c], 'by_order': ps,
                               'mean': {k: float(np.mean([p[k] for p in ps])) for k in ('self', 'user', 'system')},
                               'letter_mass_min': float(min(p['letter_mass'] for p in ps))}
        np.savez_compressed(out / 'N' / f'{bid}.npz', **arrs)
        atomic(out / 'N' / f'{bid}.json', rec)


TRIPLES = {'separate': ('assistant', 'user', 'system'), 'merged': ('assistant', 'user_merged', 'system')}


def _ratio_stats(Rs):
    low = Rs.argmin(1)
    share = (low == 0).mean(0)
    return {'R_A_by_layer': Rs[:, 0].mean(0).round(4).tolist(), 'share_A_lowest': share.round(4).tolist(),
            'layers_A_lowest_all_bodies': int((share == 1).sum()), 'layers_A_lowest_80': int((share >= 0.8).sum())}


def summarize(out):
    """Distance ratios on the held-out bodies, for the concatenated state and for each of the hc streams and the mean of the
    per-stream ratios (the paper's convention for four-stream models)."""
    gdir = out / 'N'
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted(gdir.glob('*.json'))]
    held = [r for r in recs if r['split'] == 'confirmation']
    V = {r['id']: np.load(gdir / f"{r['id']}.npz") for r in held}
    L, hc = V[held[0]['id']]['content_assistant'].shape[:2]
    S = {'n_layers': int(L), 'hc_mult': int(hc), 'n_bodies': len(recs), 'n_heldout': len(held), 'user_rendering_merged_by_template': True}
    for tname, (ra, ru, rs) in TRIPLES.items():
        for kind in ('content', 'onset'):
            per_stream = []
            for s in range(hc + 1):
                Rs = []
                for r in held:
                    a, u, y = (V[r['id']][f'{kind}_{x}'].astype(np.float32) for x in (ra, ru, rs))
                    a, u, y = ((x.reshape(L, -1) if s == hc else x[:, s]) for x in (a, u, y))
                    au, ay, uy = (np.linalg.norm(p - q, axis=-1) for p, q in ((a, u), (a, y), (u, y)))
                    Rs.append([uy / ((au + ay) / 2), ay / ((au + uy) / 2), au / ((ay + uy) / 2)])
                Rs = np.array(Rs)
                if s < hc:
                    per_stream.append(Rs)
                S[f'{tname}_{kind}_' + ('concat' if s == hc else f'stream{s}')] = _ratio_stats(Rs)
            S[f'{tname}_{kind}_mean_of_streams'] = _ratio_stats(np.mean(per_stream, 0))
    has_attr = all('mean' in r['conds'][c] for r in recs for c in CONDS)
    if has_attr:
        S['attr'] = {c: {k: float(np.mean([r['conds'][c]['mean'][k] for r in recs])) for k in ('self', 'user', 'system')} for c in CONDS}
        S['attr_bodies_own_above_external'] = int(sum(r['conds']['assistant']['mean']['self'] >
                                                      max(r['conds']['user']['mean']['self'], r['conds']['system']['mean']['self']) for r in recs))
        S['letter_mass_min'] = float(min(r['conds'][c]['letter_mass_min'] for r in recs for c in CONDS))
    atomic(out / 'summary.json', S)
    for tname in TRIPLES:
        for kind in ('content', 'onset'):
            for agg in ('mean_of_streams', 'concat') + tuple(f'stream{s}' for s in range(hc)):
                d = S[f'{tname}_{kind}_{agg}']
                print(f'{tname:8s} {kind:7s} {agg:15s} A singled out in >=80% of held-out bodies at {d["layers_A_lowest_80"]}/{L} '
                      f'layers, in all at {d["layers_A_lowest_all_bodies"]}/{L}', flush=True)
    if has_attr:
        print('answers:', {c: {k: round(v, 3) for k, v in d.items()} for c, d in S['attr'].items()}, flush=True)
    return S


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model-dir', required=True, help='the checkpoint folder (safetensors shards, tokenizer, inference/config.json)')
    p.add_argument('--out', required=True)
    p.add_argument('--official-code', default=None, help='folder with inference/ and encoding/ of the release (default: official/)')
    p.add_argument('--limit', type=int, default=0, help='only the first N bodies')
    p.add_argument('--wave-seqs', type=int, default=96, help='sequences per group sharing the staging of the experts')
    p.add_argument('--max-seq', type=int, default=1024)
    p.add_argument('--check-single', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=1)
    p.add_argument('--finalize', action='store_true', help='after all shards: assemble and summarize (no model loaded)')
    p.add_argument('--thinking', action='store_true', help='render in thinking mode (reasoning effort high) instead of chat mode')
    p.add_argument('--geometry-only', action='store_true', help='geometry sequences only, no attribution sequences')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.summarize_only:
        summarize(Path(a.out))
        return
    if a.finalize:
        finalize(a)
        return
    try:
        run(a)
    except Exception as exc:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        atomic(Path(a.out) / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
