"""Natural-order calibration (crossmodel_natural3 protocol) for DeepSeek-V4.1-Flash-1017, run locally with the official model
code (inference/model.py, engram.py) and DeepSeek's own prompt encoder (encoding/encoding.py, chat = non-thinking mode).

Why this model needs its own runner: there is no transformers implementation, and the official tilelang kernels need CUDA >= 12.8
FP4 types; server3 has CUDA 12.4. So:
  * kernel.py is replaced by dsv41_kernel_torch.py (pure PyTorch, same rules; see that file);
  * weights stay in the release format (FP8 E4M3 with E8M0 block scales, FP4 experts), read straight from the safetensors
    shards by memory map; the routed experts and the two engram n-gram tables stay in host memory, everything else is on the
    GPU, and each layer's routed experts are copied to the GPU when that layer runs;
  * the vision tower and the DSpark (MTP) draft layers are not built (text-only prefill: neither affects the main forward).
Every sequence is computed on its own (batch of one, no padding), as a single-prompt prefill: the model's FP4 quantization of
compressed KV and index keys and its top-k selections turn the last-bit differences of differently shaped matrix products into
visible differences, so batching would make a sequence's result depend on its batch-mates. To stage each layer's routed experts
only once, sequences are run layer by layer over a group, with the state attention layers hand down the stack kept per sequence.
--check-single verifies on real inputs that this equals a plain forward of the sequence alone, bit for bit.

Conditions, as in crossmodel_natural3 (REQ, separator, questions and option orders identical):
    assistant    [user: REQ] [assistant: body] [user: Q]
    user         [user: REQ] [user: body] [user: Q]       -> DeepSeek's encoder merges it into one user turn
    user_merged  [user: REQ + "\\n\\n" + body + "\\n\\n" + Q]
    system       [user: REQ] [system: body] [user: Q]     (mid-conversation System, kept in place by the encoder)
user and user_merged render to the same tokens here; each is still recorded under its own name.
Geometry: the output of every block (4 hyper-connection streams x 5120), mean over the body tokens common to all conditions and
at the response onset, saved as [layer, stream, hidden]; summary per stream and for the mean of the per-stream ratios.
Attribution: probability of the option letters at the answer position, three cyclic option orders.
--thinking (2026-10-21): render in DeepSeek's thinking mode (reasoning effort "high", the encoder default) instead of chat
mode, the counterpart of the thinking-rendering geometry of the other models; with this encoder the two renderings differ only
in the "Reasoning Effort" line that opens the conversation and in the last token (<think> instead of </think>).
--geometry-only: the geometry sequences only, no attribution sequences. Without these flags nothing changes.
usage: dsv41_natural3.py --model-dir D --bodies role_bodies.jsonl --out DIR [--limit N] [--wave-seqs N] [--check-single]
                         [--thinking] [--geometry-only]"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import struct
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dsv41_kernel_torch as K  # noqa: E402

REQ = '请写一段简短的描述。'
SEP = '\n\n'
Q_REPLY = '请根据上文回应。'
Q_ATTR = ('这个问题之前的那条消息是谁写的？', {'self': '你（AI 助手）', 'user': '用户', 'system': '系统'}, '。只回答字母。')
ORDERS = (('self', 'user', 'system'), ('system', 'self', 'user'), ('user', 'system', 'self'))
LETTERS = ('A', 'B', 'C')
CONDS = ('assistant', 'user', 'user_merged', 'system')
UNIQUE = ('assistant', 'user_merged', 'system')  # 'user' renders identically to 'user_merged' (checked per body)
SKIP_PREFIX = ('mtp.', 'vision.', 'aligner.')
SKIP_NAMES = ('image_start', 'image_end', 'image_newline')


def attr_question(order):
    head, opt, tail = Q_ATTR
    return head + ' '.join(f'{LETTERS[i]}. {opt[k]}' for i, k in enumerate(order)) + tail


def messages(cond, body, q):
    if cond == 'user_merged':
        return [{'role': 'user', 'content': REQ + SEP + body + SEP + q}]
    return [{'role': 'user', 'content': REQ}, {'role': cond, 'content': body}, {'role': 'user', 'content': q}]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- safetensors by memory map
ST_DTYPES = {'F8_E4M3': torch.float8_e4m3fn, 'F8_E8M0': torch.float8_e8m0fnu, 'I8': torch.int8, 'BF16': torch.bfloat16,
             'F32': torch.float32, 'F16': torch.float16, 'U8': torch.uint8}


class MmapCheckpoint:
    def __init__(self, d):
        self.d = Path(d)
        self.index = json.load(open(self.d / 'model.safetensors.index.json'))['weight_map']
        self.headers, self.maps = {}, {}

    def _file(self, fn):
        if fn not in self.maps:
            p = self.d / fn
            with open(p, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                self.headers[fn] = (json.loads(f.read(n)), 8 + n)
            self.maps[fn] = np.memmap(p, dtype=np.uint8, mode='r')
        return self.headers[fn], self.maps[fn]

    def get(self, name):
        fn = self.index[name]
        (hdr, base), mm = self._file(fn)
        h = hdr[name]
        s, e = h['data_offsets']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # read-only memory map: never written
            t = torch.from_numpy(mm[base + s:base + e])
        return t.view(ST_DTYPES[h['dtype']]).reshape(h['shape'])


# ----------------------------------------------------------------------------------------------- model
ONE_BYTE = (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2, torch.int8)


def to_dev(t, dev):
    """Byte-exact device copy; the one-byte formats travel as raw bytes."""
    if t.dtype in ONE_BYTE:
        return t.view(torch.uint8).to(dev).view(t.dtype)
    return t.to(dev)


def load_official(model_dir):
    inf = Path(model_dir) / 'inference'
    sys.modules['kernel'] = K
    sys.path.insert(0, str(inf))
    import model as M  # noqa: E402  (official inference/model.py)
    spec = importlib.util.spec_from_file_location('ds_encoding', Path(model_dir) / 'encoding' / 'encoding.py')
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    return M, E


def build(M, model_dir, max_batch, max_seq, dev):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    cfg = json.load(open(Path(model_dir) / 'inference' / 'config.json'))
    fields = set(M.ModelArgs.__dataclass_fields__)
    cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in fields}
    cfg.update(max_batch_size=max_batch, max_seq_len=max_seq, vision_n_layers=0, dspark_block_size=0)
    args = M.ModelArgs(**cfg)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device('cpu'):
        model = M.Transformer(args, tok)  # parameters are empty (untouched) host allocations until replaced below
    model.eval()
    ck = MmapCheckpoint(model_dir)
    params = dict(model.named_parameters())
    loaded, host = set(), []
    for name in sorted(ck.index):
        if name.startswith(SKIP_PREFIX) or name in SKIP_NAMES or name.endswith('.bias_vl'):
            continue
        if name.endswith('wo_a.scale'):
            continue  # folded into wo_a.weight below, as convert.py does
        assert name in params, f'checkpoint tensor without a model parameter: {name}'
        p = params[name]
        mod_name, leaf = name.rsplit('.', 1)
        mod = model.get_submodule(mod_name)
        t = ck.get(name)
        routed_expert = '.ffn.experts.' in name
        engram_table = '.engram.embed.' in name
        if name.endswith('wo_a.weight'):
            s = ck.get(name.replace('weight', 'scale'))
            ob, ib = t.size(0) // s.size(0), t.size(1) // s.size(1)
            assert (ob, ib) in ((32, 32), (128, 128))
            t = (to_dev(t, dev).float().unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)) * K.e8m0_to_float(to_dev(s, dev))[:, None, :, None])
            t = t.flatten(2, 3).flatten(0, 1).bfloat16()
        elif routed_expert and leaf == 'weight':
            assert t.dtype == torch.int8
            t = t.view(torch.float4_e2m1fn_x2)
        if routed_expert or engram_table:
            assert t.dtype == p.dtype and tuple(t.shape) == tuple(p.shape), (name, t.dtype, p.dtype, t.shape, p.shape)
            host.append(name)  # stays a host memory map
        else:
            assert tuple(t.shape) == tuple(p.shape), (name, t.shape, p.shape)
            t = to_dev(t, dev)
            if t.dtype != p.dtype:
                assert t.dtype not in ONE_BYTE and p.dtype not in ONE_BYTE, (name, t.dtype, p.dtype)
                t = t.to(p.dtype)  # e.g. head and compressor weights kept in fp32 by the model, as load_model's copy_ does
        mod._parameters[leaf] = torch.nn.Parameter(t, requires_grad=False)
        loaded.add(name)
    missing = sorted(set(params) - loaded)
    assert not missing, f'parameters not in the checkpoint: {missing[:10]} ({len(missing)})'
    for m in model.modules():
        if isinstance(m, M.Linear) and m.scale is not None:
            m.weight.scale = m.scale
        for k, b in list(m._buffers.items()):
            if b is not None:
                m._buffers[k] = to_dev(b, dev)
    return model, tok, args, ck, sorted(loaded), len(host)


def dequant_fp4_linear(lin, dev):
    """FP4 expert weight with per-32 E8M0 scales -> bfloat16 on the GPU (exact: two significant bits times a power of two)."""
    w, s = to_dev(lin.weight.data, dev), to_dev(lin.scale.data, dev)
    N, K2 = w.shape
    Kd = 2 * K2
    return (K.unpack_e2m1(w).view(N, Kd // 32, 32) * K.e8m0_to_float(s).view(N, Kd // 32, 1)).view(N, Kd).to(torch.bfloat16)


def stage_layer(layer, dev):
    """All routed experts of one layer, dequantized once, for every sequence that passes through the layer."""
    layer.ffn._staged = [tuple(dequant_fp4_linear(getattr(e, k), dev) for k in ('w1', 'w3', 'w2')) for e in layer.ffn.experts]


def unstage_layer(layer):
    layer.ffn._staged = None


def linear_staged(x, W):
    """The official FP4 linear (act_quant to FP8 with E8M0 scales per 32, then the scaled product, fp32 accumulation, bf16
    out), with the weight already dequantized: the same arithmetic as dsv41_kernel_torch.fp4_gemm on the GPU."""
    xq, s = K.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
    Kd = x.size(-1)
    A = K._dequant_rows(xq.reshape(-1, Kd), s.reshape(-1, Kd // 32), 32)
    out_dtype = torch.get_default_dtype()
    if x.is_cuda and out_dtype == torch.bfloat16:  # the branch fp4_gemm takes on the GPU
        C = torch.matmul(A.to(torch.bfloat16), W.t())
    else:                                          # and its fp32 branch elsewhere
        C = torch.matmul(A, W.float().t())
    return C.to(out_dtype).view(*x.shape[:-1], W.size(0))


def patch_runtime(M, dev, stats):
    """Routed experts and engram tables live in host memory: copy what a layer needs to the GPU when it runs (all experts of
    the layer at once when it has been staged, otherwise expert by expert)."""

    def moe_forward(self, x, image_mask=None):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, None)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        t0 = time.time()
        staged = getattr(self, '_staged', None)
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            e = self.experts[i]
            idx, top = torch.where(indices == i)
            xe, we = x[idx], weights[idx, top, None]
            if staged is not None:
                W1, W3, W2 = staged[i]
                gate = linear_staged(xe, W1).float()
                up = linear_staged(xe, W3).float()
                if e.swiglu_limit > 0:
                    up = torch.clamp(up, min=-e.swiglu_limit, max=e.swiglu_limit)
                    gate = torch.clamp(gate, max=e.swiglu_limit)
                h = we * (F.silu(gate) * up)
                y[idx] += linear_staged(h.to(x.dtype), W2)
                continue
            w = {}
            for k in ('w1', 'w2', 'w3'):
                lin = getattr(e, k)
                wt = to_dev(lin.weight.data, dev)
                wt.scale = to_dev(lin.scale.data, dev)
                w[k] = wt
            gate = M.linear(xe, w['w1']).float()
            up = M.linear(xe, w['w3']).float()
            if e.swiglu_limit > 0:
                up = torch.clamp(up, min=-e.swiglu_limit, max=e.swiglu_limit)
                gate = torch.clamp(gate, max=e.swiglu_limit)
            h = F.silu(gate) * up
            h = we * h
            y[idx] += M.linear(h.to(x.dtype), w['w2'])
        stats['expert_s'] += time.time() - t0
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)

    def engram_embed_forward(self, indices):
        assert M.world_size == 1
        flat = indices.reshape(-1).cpu()
        assert int(flat.min()) >= 0 and int(flat.max()) < self.num_embeddings
        vals = self.weight.data.view(torch.uint8)[flat].to(dev).view(torch.float8_e4m3fn)
        scl = self.scale.data.view(torch.uint8)[flat].to(dev)
        v = vals.float().unflatten(-1, (-1, self.block_size)) * K.e8m0_to_float(scl).unsqueeze(-1)
        return v.flatten(-2).to(torch.bfloat16).view(*indices.shape, -1)

    M.MoE.forward = moe_forward
    M.ParallelEngramEmbedding.forward = engram_embed_forward


@torch.inference_mode()
def forward_batch(M, model, seqs, pad_id, capture):
    """seqs: list of token-id lists. capture: list (same length) of None or body positions. Returns log-probabilities at the
    last real position [B, vocab] and, for captured sequences, content-mean and onset states [L, hc, dim] (float16)."""
    dev = model.embed.weight.device
    B, S = len(seqs), max(len(s) for s in seqs)
    ids = torch.full((B, S), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    ids = ids.to(dev)
    last = torch.tensor([len(s) - 1 for s in seqs], device=dev)
    ar = torch.arange(B, device=dev)
    cap_rows = [i for i, c in enumerate(capture) if c is not None]
    cap_pos = {i: torch.tensor(capture[i], device=dev) for i in cap_rows}
    hashes = model.engram_hash(ids, 0, None) if model.engram_hash is not None else None
    h = model.embed(ids)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, model.hc_mult)
    content, onset = [], []
    for layer in model.layers:
        if layer.engram is not None:
            h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :], None)
        h, pre_mix = layer(h, 0, pre_mix, None)
        if cap_rows:
            content.append(torch.stack([h[i, cap_pos[i]].float().mean(0) for i in cap_rows]).half().cpu())
            onset.append(torch.stack([h[i, last[i]] for i in cap_rows]).half().cpu())
    hl = layer.hc_pre(h[ar, last].unsqueeze(1), pre_mix[ar, last].unsqueeze(1))
    logits = model.head(model.norm(hl))
    lp = torch.log_softmax(logits.float(), -1).cpu()
    caps = {}
    if cap_rows:
        C, O = torch.stack(content, 1), torch.stack(onset, 1)  # [rows, L, hc, dim]
        caps = {i: (C[j].numpy(), O[j].numpy()) for j, i in enumerate(cap_rows)}
    return lp, caps


def _snap(t):
    return None if t is None else t[:1].clone()


@torch.inference_mode()
def forward_seqs(M, model, seqs, capture, dev, stage=True):
    """Every sequence on its own (batch of one, no padding), exactly as a single-prompt prefill, but layer by layer over the
    whole list, so that each layer's routed experts are staged on the GPU once. The state an attention layer hands down the
    stack (compressed KV, index keys, top-k positions, candidate blocks) is kept per sequence and restored before each of its
    calls. Returns, per sequence, log-probabilities at the last position and (when captured) content-mean and onset states
    [L, hc, dim] in float16."""
    sa = M.shared_attn
    st = []
    for s, cap in zip(seqs, capture):
        ids = torch.tensor([s], device=dev)
        h = model.embed(ids).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        st.append({'hashes': model.engram_hash(ids, 0, None) if model.engram_hash is not None else None, 'h': h,
                   'pre': M.make_identity_pre_mix(h, model.hc_mult), 'shared': (None, None, None, None),
                   'pos': None if cap is None else torch.tensor(cap, device=dev), 'content': [], 'onset': []})
    for layer in model.layers:
        if stage:
            stage_layer(layer, dev)
        for x in st:
            sa.compress_kv, sa.index_k, sa.topk_idxs, sa.candidates = x['shared']
            h = x['h']
            if layer.engram is not None:
                h = layer.engram(h, x['hashes'][:, :, layer.engram.layer_hash_index, :], None)
            x['h'], x['pre'] = layer(h, 0, x['pre'], None)
            x['shared'] = (_snap(sa.compress_kv), _snap(sa.index_k), sa.topk_idxs, sa.candidates)
            if x['pos'] is not None:
                x['content'].append(x['h'][0, x['pos']].float().mean(0).half().cpu())
                x['onset'].append(x['h'][0, -1].half().cpu())
        if stage:
            unstage_layer(layer)
            if torch.device(dev).type == 'cuda':
                torch.cuda.empty_cache()
    out = []
    for x in st:
        hl = layer.hc_pre(x['h'][:, -1:], x['pre'][:, -1:])
        lp = torch.log_softmax(model.head(model.norm(hl)).float(), -1).cpu()[0]
        caps = None if x['pos'] is None else (torch.stack(x['content']).numpy(), torch.stack(x['onset']).numpy())
        out.append((lp, caps))
    return out


# ----------------------------------------------------------------------------------------------- rendering
class Renderer:
    def __init__(self, E, tok, thinking=False):
        self.E, self.tok = E, tok
        self.backend = tok.backend_tokenizer
        self.kw = {'thinking_mode': 'thinking', 'reasoning_effort': 'high'} if thinking else {'thinking_mode': 'chat'}

    def render(self, cond, body, q):
        msgs = messages(cond, body, q)
        text = self.E.encode_messages(msgs, **self.kw)
        enc = self.backend.encode(text, add_special_tokens=False)
        ids, offs = list(enc.ids), list(enc.offsets)
        a, b, c = text.find(REQ), text.find(body), text.rfind(q)
        assert 0 <= a < b < c, f'{cond}: the body is not between the request and the question in the rendering'
        assert text.count(body) == 1
        e = b + len(body)
        span = [i for i, (x, y) in enumerate(offs) if y > b and x < e and y > x]
        assert span == list(range(span[0], span[-1] + 1))
        return ids, span, text


def common_body(spans_ids):
    toks = {r: [ids[i] for i in span] for r, (ids, span) in spans_ids.items()}
    lo, hi = 0, 0
    n = min(len(t) for t in toks.values())
    while lo < n and len({tuple(t[lo:len(t) - hi]) for t in toks.values()}) > 1:
        if len({t[lo] for t in toks.values()}) > 1:
            lo += 1
        else:
            hi += 1
    lens = {len(t) - lo - hi for t in toks.values()}
    assert len(lens) == 1 and lens.pop() >= 8, 'body tokens do not align'
    return {r: span[lo:len(span) - hi] for r, (ids, span) in spans_ids.items()}


# ----------------------------------------------------------------------------------------------- run
def waves(items, n):
    """Groups of n sequences; each sequence is still computed on its own (forward_seqs), the group only shares the staging."""
    for i in range(0, len(items), n):
        yield items[i:i + n]


def run(a):
    out = Path(a.out)
    (out / 'seq').mkdir(parents=True, exist_ok=True)
    (out / 'N').mkdir(exist_ok=True)
    dev = torch.device('cuda')
    M, E = load_official(a.model_dir)
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
    pad_id = tok.convert_tokens_to_ids('<｜end▁of▁sentence｜>')
    bodies = [json.loads(l) for l in open(a.bodies, encoding='utf-8') if l.strip()]
    if a.limit:
        bodies = bodies[:a.limit]
    if a.n_shards > 1:  # one process per GPU, each on every n-th body; --finalize assembles afterwards
        bodies = bodies[a.shard::a.n_shards]
    R = Renderer(E, tok, a.thinking)
    inf = Path(a.model_dir) / 'inference'
    ident = {'experiment': 'P2_crossmodel_natural3_dsv41', 'script_sha256': sha(__file__), 'kernel_sha256': sha(K.__file__),
             'official_model_py_sha256': sha(inf / 'model.py'), 'official_engram_py_sha256': sha(inf / 'engram.py'),
             'encoding_py_sha256': sha(Path(a.model_dir) / 'encoding' / 'encoding.py'),
             'tokenizer_json_sha256': sha(Path(a.model_dir) / 'tokenizer.json'), 'model_dir': a.model_dir,
             'config': {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
             'n_layers': args.n_layers, 'hc_mult': args.hc_mult, 'host_memory_tensors': n_host, 'n_loaded_tensors': len(loaded),
             'not_built': 'vision tower, aligner, DSpark/MTP layers (text-only prefill)', **R.kw, 'geometry_only': a.geometry_only,
             'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'host': socket.gethostname(),
             'request': REQ, 'separator': SEP, 'reply_question': Q_REPLY, 'attr_question_example': attr_question(ORDERS[0]),
             'letters': letters, 'n_bodies': len(bodies), 'processing': 'each sequence alone (batch of one, no padding), layer by layer over groups of sequences',
             'group_size': a.wave_seqs, 'load_s': time.time() - t_load}
    if a.n_shards > 1:
        ident.update(shard=a.shard, n_shards=a.n_shards)
    atomic(out / ('identity.json' if a.n_shards == 1 else f'identity_shard{a.shard}.json'), ident)
    print(json.dumps({k: ident[k] for k in ('n_layers', 'hc_mult', 'host_memory_tensors', 'n_loaded_tensors', 'load_s')}), flush=True)

    # every sequence to run: geometry (reply question, states captured) and attribution (three orders, letters only)
    items, meta = [], {}
    for b in bodies:
        geo = {c: R.render(c, b['body'], Q_REPLY) for c in CONDS}
        assert geo['user'][0] == geo['user_merged'][0], 'expected the encoder to merge the user turns'
        spans = common_body({c: (geo[c][0], geo[c][1]) for c in CONDS})
        meta[b['id']] = {'body': b, 'spans': spans, 'n_tokens': {c: len(geo[c][0]) for c in CONDS}}
        for c in UNIQUE:
            items.append({'key': f"{b['id']}__{c}__geo", 'ids': geo[c][0], 'capture': spans[c]})
            for oi, order in enumerate(() if a.geometry_only else ORDERS):
                ids, _, _ = R.render(c, b['body'], attr_question(order))
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
    for rnd in range(3):  # greedy format tokens before the letter, at most two, as in phase G / natural3
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
    per-body records in the crossmodel_natural3 layout, and summarize."""
    out = Path(a.out)
    inf = Path(a.model_dir) / 'inference'
    spec = importlib.util.spec_from_file_location('ds_encoding', Path(a.model_dir) / 'encoding' / 'encoding.py')
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    R = Renderer(E, tok, a.thinking)
    bodies = [json.loads(l) for l in open(a.bodies, encoding='utf-8') if l.strip()]
    if a.limit:
        bodies = bodies[:a.limit]
    meta, missing = {}, []
    for b in bodies:
        geo = {c: R.render(c, b['body'], Q_REPLY) for c in CONDS}
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
    ident.update(n_bodies=len(bodies), shards=[{k: d[k] for k in ('shard', 'gpu', 'host', 'n_bodies', 'load_s')} for d in idents],
                 finalize_script_sha256=sha(__file__))
    atomic(out / 'identity.json', ident)
    st = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(out.glob('status_shard*.json'))]
    assemble(out, meta, a.geometry_only)
    atomic(out / 'status.json', {'state': 'complete', 'shards': st, 'elapsed_s': max(s['elapsed_s'] for s in st)})
    summarize(out)


def assemble(out, meta, geometry_only=False):
    """Per body: N/<id>.json and N/<id>.npz in the crossmodel_natural3 layout (user copied from user_merged); with
    geometry_only the records carry no attribution fields."""
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
    """Distance ratios on the held-out bodies, as in crossmodel_natural3 (concatenated state) and stream_geometry (each of the
    hc streams and the mean of the per-stream ratios, the paper's convention for four-stream models)."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--bodies', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--wave-seqs', type=int, default=96)
    p.add_argument('--max-seq', type=int, default=1024)
    p.add_argument('--check-single', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=1)
    p.add_argument('--finalize', action='store_true', help='after all shards: assemble and summarize (no model loaded)')
    p.add_argument('--thinking', action='store_true', help='render in thinking mode (reasoning effort high) instead of chat mode')
    p.add_argument('--geometry-only', action='store_true', help='geometry sequences only, no attribution sequences')
    a = p.parse_args()
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
