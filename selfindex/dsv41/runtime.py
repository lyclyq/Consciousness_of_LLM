"""Running DeepSeek-V4.1-Flash-1017 with the official model code and the PyTorch kernels.

  * The official modules (inference/model.py and its imports, encoding/encoding.py) are loaded from official/ (or another copy of
    the release code) under private module names, with kernel_torch in place of the tilelang kernel module.
  * The weights stay in the release format and are read straight from the safetensors shards by memory map. The routed experts and
    the two engram n-gram tables stay in host memory, everything else is moved to the device; each layer's routed experts are
    dequantized onto the device when the layer runs (exact: FP4 values times power-of-two scales are exact in bfloat16).
  * The vision tower and the DSpark (MTP) draft layers are not built (text-only prefill: neither affects the main forward).
  * Every sequence is computed on its own (batch of one, no padding), as a single-prompt prefill: the model's FP4 quantization of
    compressed KV and index keys and its top-k selections turn the last-bit differences of differently shaped matrix products into
    visible differences, so batching would make a sequence's result depend on its batch-mates. To stage each layer's routed
    experts only once, forward_seqs runs the sequences layer by layer over a group and keeps, per sequence, the state that the
    attention layers hand down the stack; forward_batch is the plain forward of one sequence, against which it is checked.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import items as I
from . import OFFICIAL
from . import kernel_torch as K

LANG = 'zh'  # the runs use the Chinese items
LETTERS = ('A', 'B', 'C')
CONDS = tuple(I.templates()['natural_order']['conditions'])  # assistant, user, user_merged, system
UNIQUE = ('assistant', 'user_merged', 'system')  # 'user' renders identically to 'user_merged' (checked per body)
SKIP_PREFIX = ('mtp.', 'vision.', 'aligner.')
SKIP_NAMES = ('image_start', 'image_end', 'image_newline')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(path, obj):
    """JSON (indented) through a .tmp file: a result file on disk is always complete."""
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, path)


def messages(cond, body, q):
    """The natural conversation order: [user: request] [cond: body] [user: q]; user_merged joins the three in one user message."""
    req, sep = I.natural_order_request(LANG), I.templates()['natural_order']['merged_separator']
    if cond == 'user_merged':
        return [{'role': 'user', 'content': req + sep + body + sep + q}]
    return [{'role': 'user', 'content': req}, {'role': cond, 'content': body}, {'role': 'user', 'content': q}]


# ----------------------------------------------------------------------------------------------- the official code
def _load_file(name, path, register=True):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    if register:  # under its private name (dataclasses look their module up)
        sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_official(code_dir=None):
    """(model module, encoding module) of the release code in code_dir (default: official/), kernel module replaced by kernel_torch.
    model.py imports its siblings by top-level name (engram, image_processor, kernel, vision); those names are bound only while it
    loads, so no generic module names stay behind in sys.modules."""
    d = Path(code_dir or OFFICIAL)
    inf = d / 'inference'
    names = ('kernel', 'engram', 'image_processor', 'vision')
    saved = {n: sys.modules.get(n) for n in names}
    try:
        sys.modules['kernel'] = K
        for n in names[1:]:
            sys.modules[n] = _load_file(f'selfindex_dsv41_official_{n}', inf / f'{n}.py')
        M = _load_file('selfindex_dsv41_official_model', inf / 'model.py')
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    E = _load_file('ds_encoding', d / 'encoding' / 'encoding.py', register=False)
    M._p2_code_dir = d
    return M, E


def code_hashes(M):
    d = M._p2_code_dir
    return {'official_model_py_sha256': sha(d / 'inference' / 'model.py'), 'official_engram_py_sha256': sha(d / 'inference' / 'engram.py'),
            'encoding_py_sha256': sha(d / 'encoding' / 'encoding.py')}


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


def build(M, model_dir, max_batch, max_seq, dev):
    """The model of model_dir (inference/config.json, tokenizer, safetensors shards) with the weights of the checkpoint."""
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
    """FP4 expert weight with per-32 E8M0 scales -> bfloat16 on the device (exact: two significant bits times a power of two)."""
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
    out), with the weight already dequantized: the same arithmetic as kernel_torch.fp4_gemm."""
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
    """Routed experts and engram tables live in host memory: copy what a layer needs to the device when it runs (all experts of
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
    whole list, so that each layer's routed experts are staged once. The state an attention layer hands down the stack
    (compressed KV, index keys, top-k positions, candidate blocks) is kept per sequence and restored before each of its calls.
    Returns, per sequence, log-probabilities at the last position and (when captured) content-mean and onset states
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


def waves(items, n):
    """Groups of n sequences; each sequence is still computed on its own (forward_seqs), the group only shares the staging."""
    for i in range(0, len(items), n):
        yield items[i:i + n]


def device_or_exit(allow_cpu=False):
    """The GPU; a run refuses to fall back to the CPU (allow_cpu is for the tests on small random models)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if allow_cpu:
        return torch.device('cpu')
    sys.exit('CUDA unavailable: refuse to run on CPU')


def device_name(dev):
    return torch.cuda.get_device_name(0) if torch.device(dev).type == 'cuda' else None


# ----------------------------------------------------------------------------------------------- rendering
class Renderer:
    """DeepSeek's own encoder (chat mode, or thinking mode with reasoning effort high) and the tokenizer's offsets: token ids of a
    natural-order conversation and the positions of the body."""

    def __init__(self, E, tok, thinking=False):
        self.E, self.tok = E, tok
        self.backend = tok.backend_tokenizer
        self.kw = {'thinking_mode': 'thinking', 'reasoning_effort': 'high'} if thinking else {'thinking_mode': 'chat'}

    def render(self, cond, body, q):
        msgs = messages(cond, body, q)
        text = self.E.encode_messages(msgs, **self.kw)
        enc = self.backend.encode(text, add_special_tokens=False)
        ids, offs = list(enc.ids), list(enc.offsets)
        a, b, c = text.find(I.natural_order_request(LANG)), text.find(body), text.rfind(q)
        assert 0 <= a < b < c, f'{cond}: the body is not between the request and the question in the rendering'
        assert text.count(body) == 1
        e = b + len(body)
        span = [i for i, (x, y) in enumerate(offs) if y > b and x < e and y > x]
        assert span == list(range(span[0], span[-1] + 1))
        return ids, span, text
