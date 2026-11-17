"""P2 ordinary words along the message (11-10, ../../PROTOCOL_zh.md) for DeepSeek-V4.1-Flash-1017, natural order only (paper Table
tab:self_natural3), on the paper's DeepSeek runner dsv41_natural3.py (official inference code, pure-PyTorch kernels, release
weights read by memory map) with the grouped forward of the KL port dsv41_modeswitch_nat3_20261108.py (46e977c4): every sequence
on its own, layer by layer over a group, attention state handed down per sequence, routed experts staged once per layer.
Measurement identical to words_along_20261110.py (HF models):
  renderings in chat mode, natural order, Chinese items: [user: REQ][ROLE: body][user: Q_REPLY] for A, U, S (the encoder merges
  the user turns, as in the paper), and for a word pair the body as a User message preceded by the frame 以下是{w}的记录：.
  The body span is located in the rendered text by the body itself (dsv41_natural3.Renderer's rule), so the frame sentence is
  never part of it; the pair's common body tokens come from dsv41_natural3.common_body.
  State: the block output, the 4 hyper-connection streams concatenated (stream-major, [hc * dim]), the layout of the paper's
  natural3 features ([layer, stream, hidden]) flattened.
  At the KEEP_FRACS layers: D = x1 - x2 on the common body tokens; mean vector and mean squared difference over the first and the
  last third of them. Rows: A-U, A-S, U-S, then the 32 pairs of words_v2.json.
Gate (--gate, the smoke): A/U/S readout states at the last position, every layer, against the paper's natural3 features
(suite_nat3/dsv41_flash_1017/N/<id>.npz onset_<cond>) on the first 2 held-out bodies; cosine >= 0.999 and relative error <= 0.03.
A shard run requires the smoke's gate.json (ok, same script) via --gate-from.
Output: one npz + json per held-out body in OUT (the format of words_along_20261110.py, read by extract_gram_20261110.py),
meta/identity_shard<k>.json; --finalize (no model) writes identity.json and DONE when every body is on disk.
usage: dsv41_words_along_20261110.py --model-dir D --bodies role_bodies.jsonl --words words_v2.json --out DIR
       [--gate --paper-n DIR | --gate-from gate.json] [--shard k --n-shards n] [--group-bodies 2] [--limit N] [--stop-file F]
       [--dry-run] [--finalize]"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dsv41_natural3 as D  # noqa: E402

ROLES = (('A', 'assistant'), ('U', 'user'), ('S', 'system'))
RPAIRS = ((0, 1), (0, 2), (1, 2))
KEEP_FRACS = (0.375, 0.5, 0.625, 0.75, 0.875, 1.0)    # as persistence_readout_20261106 / words_along_20261110
GATE_COS, GATE_REL = 0.999, 0.03


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_words(path):
    W = json.loads(Path(path).read_text(encoding='utf-8'))
    return W, [{'id': p['id'], 'w': p['zh'], 'frame': W['frame_zh']} for p in W['pairs']]


def load_bodies(a):
    bodies = [json.loads(l) for l in open(a.bodies, encoding='utf-8') if l.strip()]
    bodies = [b for b in bodies if b.get('split') == 'confirmation']
    return bodies


def render(R, role, content, body):
    """[user: REQ][role: content][user: Q_REPLY] in chat mode; ids and the token span of `body` (located by itself)"""
    msgs = [{'role': 'user', 'content': D.REQ}, {'role': role, 'content': content}, {'role': 'user', 'content': D.Q_REPLY}]
    text = R.E.encode_messages(msgs, **R.kw)
    enc = R.backend.encode(text, add_special_tokens=False)
    ids, offs = list(enc.ids), list(enc.offsets)
    a, b, c = text.find(D.REQ), text.find(body), text.rfind(D.Q_REPLY)
    assert 0 <= a < b < c, f'{role}: the body is not between the request and the question'
    assert text.count(body) == 1
    e = b + len(body)
    span = [i for i, (x, y) in enumerate(offs) if y > b and x < e and y > x]
    assert span == list(range(span[0], span[-1] + 1))
    return ids, span


def plan(R, body, pairs, check_runner=False):
    """sequences of one body: the three roles, then the two renderings of every pair; common body positions per row"""
    body = body.rstrip()
    rr = {rn: render(R, role, body, body) for rn, role in ROLES}
    if check_runner:   # the roles must render exactly as the paper's runner renders them
        for rn, role in ROLES:
            ids, span, _ = R.render(role, body, D.Q_REPLY)
            assert (ids, span) == rr[rn], f'role {rn} renders differently from dsv41_natural3.Renderer'
    com = D.common_body(rr)
    seqs, rows = [], {}
    idx = {}
    for rn, _ in ROLES:
        idx[rn] = len(seqs)
        seqs.append((rr[rn][0], com[rn]))
    for i1, i2 in RPAIRS:
        a, c = ROLES[i1][0], ROLES[i2][0]
        rows[f'{a}-{c}'] = (idx[a], idx[c])
    unequal = []
    for p in pairs:
        r2 = {f'w{i + 1}': render(R, 'user', p['frame'].format(w=w) + body, body) for i, w in enumerate(p['w'])}
        c2 = D.common_body(r2)
        rows[p['id']] = (len(seqs), len(seqs) + 1)
        seqs.append((r2['w1'][0], c2['w1']))
        seqs.append((r2['w2'][0], c2['w2']))
        if len(r2['w1'][0]) != len(r2['w2'][0]):
            unequal.append(p['id'])
    return seqs, rows, unequal


@torch.inference_mode()
def forward_capture(M, model, seqs, keep, dev, stage=True):
    """the grouped forward of dsv41_modeswitch_nat3_20261108.forward_all, capturing per sequence the last position at every layer
    [L, hc * dim] and the body positions at the kept layers [K, T, hc * dim] (float32, on the host)"""
    sa = M.shared_attn
    kset = set(keep)
    st = []
    for ids_list, pos in seqs:
        ids = torch.tensor([ids_list], device=dev)
        h = model.embed(ids).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        st.append({'hashes': model.engram_hash(ids, 0, None) if model.engram_hash is not None else None, 'h': h,
                   'pre': M.make_identity_pre_mix(h, model.hc_mult), 'shared': (None, None, None, None),
                   'pos': torch.tensor(pos, device=dev), 'last': [], 'body': []})
    for li, layer in enumerate(model.layers):
        if stage:
            D.stage_layer(layer, dev)
        for x in st:
            sa.compress_kv, sa.index_k, sa.topk_idxs, sa.candidates = x['shared']
            h = x['h']
            if layer.engram is not None:
                h = layer.engram(h, x['hashes'][:, :, layer.engram.layer_hash_index, :], None)
            x['h'], x['pre'] = layer(h, 0, x['pre'], None)
            x['shared'] = (D._snap(sa.compress_kv), D._snap(sa.index_k), sa.topk_idxs, sa.candidates)
            x['last'].append(x['h'][0, -1].float().reshape(-1).cpu())
            if li in kset:
                b = x['h'][0, x['pos']].float()
                x['body'].append(b.reshape(b.shape[0], -1).cpu())
        if stage:
            D.unstage_layer(layer)
            if torch.device(dev).type == 'cuda':
                torch.cuda.empty_cache()
    return [(torch.stack(x['last']), torch.stack(x['body'])) for x in st]


def decomp(x1, x2):
    D_ = x1 - x2                                       # (K, T, d), float32
    T = D_.shape[1]
    p = torch.arange(T, dtype=torch.float32, device=D_.device) / T  # explicit device: run() sets the default device to cuda
    f, la = p < 1 / 3, p >= 2 / 3
    return D_[:, f].mean(1), D_[:, la].mean(1), (D_[:, f] ** 2).sum(-1).mean(1), (D_[:, la] ** 2).sum(-1).mean(1)


def cmp(ours, ref):
    ref = torch.as_tensor(np.asarray(ref, np.float32), device=ours.device).reshape(ours.shape[0], -1)  # explicit device (11-10 fix)
    cosv = torch.nn.functional.cosine_similarity(ours, ref, dim=-1)
    rel = (ours - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)
    return float(cosv.min()), float(rel.max())


def body_arrays(seqs_out, rows, row_names, store):
    vdt = np.float32 if store == 'float32' else np.float16
    d = seqs_out[0][1].shape[-1]
    K = seqs_out[0][1].shape[0]
    R_ = len(row_names)
    arrs = {'dec_mF': np.full((R_, K, d), np.nan, vdt), 'dec_mL': np.full((R_, K, d), np.nan, vdt),
            'dec_eF': np.full((R_, K), np.nan, np.float32), 'dec_eL': np.full((R_, K), np.nan, np.float32),
            'role_state': np.full((3, K, d), np.nan, vdt)}
    ncom = {}
    for qi, nm in enumerate(row_names):
        i1, i2 = rows[nm]
        mF, mL, eF, eL = decomp(seqs_out[i1][1], seqs_out[i2][1])
        arrs['dec_mF'][qi], arrs['dec_mL'][qi] = mF.numpy().astype(vdt), mL.numpy().astype(vdt)
        arrs['dec_eF'][qi], arrs['dec_eL'][qi] = eF.numpy(), eL.numpy()
        ncom[nm] = int(seqs_out[i1][1].shape[1])
    return arrs, ncom


def dry_run(a):
    import importlib.util
    spec = importlib.util.spec_from_file_location('ds_encoding', Path(a.model_dir) / 'encoding' / 'encoding.py')
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    R = D.Renderer(E, tok, False)
    W, pairs = load_words(a.words)
    bodies = load_bodies(a)[:a.limit] if a.limit else load_bodies(a)
    nseq, uneq, ncom, lmax = 0, {}, [], 0
    for b in bodies:
        seqs, rows, u = plan(R, b['body'], pairs, check_runner=True)
        nseq += len(seqs)
        lmax = max(lmax, max(len(s[0]) for s in seqs))
        for p in u:
            uneq[p] = uneq.get(p, 0) + 1
        ncom += [len(s[1]) for s in seqs]
    print(json.dumps({'bodies': len(bodies), 'order': 'natural', 'lang': 'zh', 'natural_system_ok': True,
                      'rows': ['A-U', 'A-S', 'U-S'] + [p['id'] for p in pairs], 'forwards': nseq, 'max_tokens': lmax,
                      'pairs_unequal_rendering_length': uneq, 'common_body_tokens_min_max': [min(ncom), max(ncom)],
                      'thinking_mode': R.kw}, ensure_ascii=False))
    print('DRY RUN OK')


def run(a):
    out = Path(a.out)
    (out / 'meta').mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: this script does not fall back to the CPU')
    dev = torch.device('cuda')
    W, pairs = load_words(a.words)
    row_names = ['A-U', 'A-S', 'U-S'] + [p['id'] for p in pairs]
    me = sha(__file__)
    if a.gate_from:
        g = json.loads(Path(a.gate_from).read_text(encoding='utf-8'))
        assert g.get('ok') and g.get('script_sha256') == me, f'--gate-from does not hold a passed gate of this script: {a.gate_from}'
    M, E = D.load_official(a.model_dir)
    t_load = time.time()
    model, tok, args, ck, loaded, n_host = D.build(M, a.model_dir, 1, a.max_seq, dev)
    torch.set_default_device(dev)  # as dsv41_natural3.run: only after the model is built on the host
    stats = {'expert_s': 0.0}
    D.patch_runtime(M, dev, stats)
    R = D.Renderer(E, tok, False)
    L = len(model.layers)
    keep = sorted({round(f * (L - 1)) for f in KEEP_FRACS})
    bodies_all = load_bodies(a)
    inf = Path(a.model_dir) / 'inference'
    ident = {'experiment': 'P2_words_along_dsv41_20261110', 'script_sha256': me, 'runner_sha256': sha(D.__file__),
             'kernel_sha256': sha(D.K.__file__), 'official_model_py_sha256': sha(inf / 'model.py'),
             'official_engram_py_sha256': sha(inf / 'engram.py'), 'encoding_py_sha256': sha(Path(a.model_dir) / 'encoding' / 'encoding.py'),
             'tokenizer_json_sha256': sha(Path(a.model_dir) / 'tokenizer.json'), 'words_sha256': sha(a.words),
             'words_version': W['version'], 'model': a.model_dir, 'tmpl_kwargs': R.kw, 'lang': 'zh', 'order': 'natural',
             'fixed_date': None, 'request': D.REQ, 'reply_question': D.Q_REPLY, 'natural_system_ok': True, 'rows': row_names,
             'pairs': 'all', 'pair_ids': [p['id'] for p in pairs], 'split': 'confirmation', 'n_layers': L, 'keep_layers': keep,
             'hc_mult': int(model.hc_mult), 'state': 'block output, hyper-connection streams concatenated [hc * dim]',
             'bodies_file': a.bodies, 'bodies_sha256': sha(a.bodies), 'torch': torch.__version__, 'cuda': torch.version.cuda,
             'store': a.store, 'group_bodies': a.group_bodies}
    shard_meta = {'shard': a.shard, 'n_shards': a.n_shards, 'gpu': torch.cuda.get_device_name(0), 'host': socket.gethostname(),
                  'load_s': time.time() - t_load, 'n_host_memory_tensors': n_host}

    # ---------------- gate (the smoke) ----------------
    if a.gate:
        gate = {'criterion': {'cos_min': GATE_COS, 'rel_max': GATE_REL}, 'paper_n': a.paper_n, 'bodies': [], 'ok': True}
        gb = bodies_all[:2]
        seqs = []
        for b in gb:
            s, _, _ = plan(R, b['body'], [], check_runner=True)
            seqs += s
        res = forward_capture(M, model, seqs, keep, dev)
        for j, b in enumerate(gb):
            rec = {'id': b['id'], 'checks': {}}
            with np.load(Path(a.paper_n) / f'{b["id"]}.npz') as z:
                for k, (rn, role) in enumerate(ROLES):
                    rec['checks'][f'onset_{rn}'] = cmp(res[3 * j + k][0], z[f'onset_{role}'])
            rec['ok'] = all(c >= GATE_COS and r <= GATE_REL for c, r in rec['checks'].values())
            gate['bodies'].append(rec)
            gate['ok'] &= rec['ok']
        gate['script_sha256'] = me
        D.atomic(out / 'gate.json', gate)
        print('gate:', json.dumps(gate), flush=True)
        if not gate['ok']:
            raise SystemExit(f'GATE FAILED: {json.dumps(gate)}')
        ident['gate'] = gate
    else:
        ident['gate'] = json.loads(Path(a.gate_from).read_text(encoding='utf-8'))
    D.atomic(out / 'meta' / f'identity_shard{a.shard}.json', {**ident, **shard_meta})

    # ---------------- measurement ----------------
    bodies = bodies_all[:a.limit] if a.limit else bodies_all
    if a.n_shards > 1:
        bodies = bodies[a.shard::a.n_shards]
    todo = [b for b in bodies if not (out / f'{b["id"]}.json').exists()]
    print(f'{len(bodies)} bodies in this shard, {len(todo)} to run', flush=True)
    t0, done = time.time(), 0
    for i in range(0, len(todo), a.group_bodies):
        if a.stop_file and Path(a.stop_file).exists():
            print(f'[stop] {a.stop_file} exists: stopping (rerun resumes)', flush=True)
            break
        grp = todo[i:i + a.group_bodies]
        tb = time.time()
        plans = [plan(R, b['body'], pairs) for b in grp]
        seqs = [s for p_ in plans for s in p_[0]]
        assert max(len(s[0]) for s in seqs) + 2 <= a.max_seq, 'raise --max-seq'
        res = forward_capture(M, model, seqs, keep, dev)
        off = 0
        for b, (s_, rows, uneq) in zip(grp, plans):
            mine = res[off:off + len(s_)]
            off += len(s_)
            arrs, ncom = body_arrays(mine, rows, row_names, a.store)
            for k in range(3):
                arrs['role_state'][k] = mine[k][0][keep].numpy().astype(arrs['role_state'].dtype)
            for kk, v in arrs.items():
                assert not np.isinf(v).any(), f'{kk} has inf in {b["id"]} (float16 overflow? rerun with --store float32)'
            part = out / f'{b["id"]}.npz.part'
            with open(part, 'wb') as fh:
                np.savez(fh, **arrs)
            part.replace(out / f'{b["id"]}.npz')
            D.atomic(out / f'{b["id"]}.json', {'id': b['id'], 'split': b.get('split'), 'category': b.get('category'),
                                                'token_counts': {nm: [len(s_[i1][0]), len(s_[i2][0])] for nm, (i1, i2) in rows.items()},
                                                'n_common': ncom, 'unequal_pairs': uneq,
                                                'seconds': round((time.time() - tb) / len(grp), 1), 'shard': a.shard})
            done += 1
        print(f'{done}/{len(todo)} bodies, {time.time() - t0:.0f}s, experts {stats["expert_s"]:.0f}s; last {grp[-1]["id"]}', flush=True)
    n_all = sum(1 for b in bodies if (out / f'{b["id"]}.json').exists())
    D.atomic(out / 'meta' / f'status_shard{a.shard}.json', {'bodies_on_disk': n_all, 'n_bodies': len(bodies), 'elapsed_s': time.time() - t0})
    print(f'shard {a.shard}: {n_all}/{len(bodies)} bodies on disk', flush=True)


def finalize(a):
    out = Path(a.out)
    bodies = load_bodies(a)
    missing = [b['id'] for b in bodies if not (out / f'{b["id"]}.json').exists()]
    assert not missing, f'{len(missing)} bodies missing, e.g. {missing[:3]}'
    idents = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'meta').glob('identity_shard*.json'))]
    keys = ('script_sha256', 'runner_sha256', 'kernel_sha256', 'official_model_py_sha256', 'encoding_py_sha256', 'tokenizer_json_sha256',
            'words_sha256', 'rows', 'keep_layers', 'tmpl_kwargs', 'store')
    for other in idents[1:]:
        for k in keys:
            assert other[k] == idents[0][k], f'shards differ in {k}'
    ident = {k: v for k, v in idents[0].items() if k not in ('shard', 'n_shards', 'gpu', 'host', 'load_s', 'n_host_memory_tensors')}
    ident.update(n_bodies=len(bodies), shards=[{k: d.get(k) for k in ('shard', 'gpu', 'host', 'load_s')} for d in idents],
                 finalize_script_sha256=sha(__file__))
    D.atomic(out / 'identity.json', ident)
    (out / 'DONE').write_text(time.strftime('%F %T') + '\n')
    print('finalized', len(bodies), 'bodies')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--bodies', required=True)
    p.add_argument('--words', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--gate', action='store_true', help='smoke: gate on the first 2 held-out bodies against --paper-n')
    p.add_argument('--paper-n', default=None, help="the paper's natural3 feature directory (N)")
    p.add_argument('--gate-from', default=None, help="gate.json of the passed smoke (shard runs)")
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--group-bodies', type=int, default=1, help='bodies per group (67 sequences each; the KL port used 63)')
    p.add_argument('--max-seq', type=int, default=1024)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=1)
    p.add_argument('--store', choices=('float16', 'float32'), default='float16')
    p.add_argument('--stop-file', default=None)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--finalize', action='store_true')
    a = p.parse_args()
    if a.dry_run:
        dry_run(a)
        return
    assert a.out, '--out is required'
    if a.finalize:
        finalize(a)
        return
    assert a.gate != bool(a.gate_from), 'give exactly one of --gate (with --paper-n) or --gate-from'
    try:
        run(a)
    except BaseException as exc:
        (Path(a.out) / 'meta').mkdir(parents=True, exist_ok=True)
        D.atomic(Path(a.out) / 'meta' / f'failed_shard{a.shard}.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
