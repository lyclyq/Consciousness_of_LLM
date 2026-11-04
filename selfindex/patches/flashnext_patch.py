"""Qwen3.8-Flash-Next (model_type qwen4_exp) without loading its hashed n-gram table into CPU memory.

The checkpoint's 51.2B-parameter hashed n-gram table (128 FP8 shards, one per-tensor BF16 scale) is ~48 GiB; transformers
concatenates it in CPU memory at load time (with a second temporary copy).
Here the table is never loaded: its converter is dropped, its module is replaced by a parameter-free placeholder before the
weights are read, and after loading each n-gram embedding reads only the rows a lookup needs from memory-mapped shards.
The lookup is the same computation as transformers' FP8Embedding.forward (FP8 rows cast to the scale dtype, times the scale)
on the same checkpoint bytes; rows are concatenated in shard order, as the checkpoint's Concatenate(dim=0) converter does.
_int64_scatter_indices is needed under torch < 2.7 only.
"""
from __future__ import annotations
import json
import os
import struct

import numpy as np
import torch
import torch.nn as nn

NGRAM_KEY = 'ngram_embedding'


def _header(path):
    with open(path, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        return 8 + n, json.loads(fh.read(n))


class _Placeholder(nn.Module):
    """Stands in for the n-gram nn.Embedding while the checkpoint loads (no parameters, so nothing is allocated)."""


class MMapNgramEmbedding(nn.Module):
    def __init__(self, model_dir: str, prefix: str):
        super().__init__()
        wm = json.load(open(os.path.join(model_dir, 'model.safetensors.index.json'), encoding='utf-8'))['weight_map']
        shard_keys = [k for k in wm if k.startswith(prefix + '.shard_') and k.endswith('.weight')]
        idx = sorted(int(k[len(prefix) + len('.shard_'):-len('.weight')]) for k in shard_keys)
        assert idx == list(range(len(idx))) and idx, (prefix, idx[:5])
        self.maps = []
        for i in idx:
            k = f'{prefix}.shard_{i}.weight'
            f = os.path.join(model_dir, wm[k])
            base, h = _header(f)
            e = h[k]
            assert e['dtype'] == 'F8_E4M3', (k, e['dtype'])
            rows, cols = e['shape']
            assert e['data_offsets'][1] - e['data_offsets'][0] == rows * cols
            self.maps.append(np.memmap(f, dtype=np.uint8, mode='r', offset=base + e['data_offsets'][0], shape=(rows, cols)))
        self.rows, self.dim = self.maps[0].shape
        assert all(m.shape == (self.rows, self.dim) for m in self.maps)
        k = f'{prefix}.weight_scale'
        f = os.path.join(model_dir, wm[k])
        base, h = _header(f)
        e = h[k]
        assert e['dtype'] == 'BF16' and e['shape'] == [1], e
        raw = np.fromfile(f, dtype=np.uint8, count=2, offset=base + e['data_offsets'][0])
        self.scale = torch.from_numpy(raw.copy()).view(torch.bfloat16)
        self.num_embeddings = self.rows * len(self.maps)
        self.embedding_dim = self.dim
        # the model reads `ngram_embedding.weight.device` to pick the lookup device: an empty CPU buffer makes it the CPU
        self.register_buffer('weight', torch.empty(0, dtype=torch.bfloat16), persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).to('cpu').numpy()
        assert flat.min() >= 0 and flat.max() < self.num_embeddings
        shard, row = np.divmod(flat, self.rows)
        out = np.empty((flat.size, self.dim), dtype=np.uint8)
        for s in np.unique(shard):
            m = shard == s
            out[m] = self.maps[int(s)][row[m]]
        rows = torch.from_numpy(out).view(torch.float8_e4m3fn)
        return (rows.to(self.scale.dtype) * self.scale).reshape(*ids.shape, self.dim)


def _int64_scatter_indices(M):
    """torch before 2.7 accepts only int64 scatter indices, and the QSA indexer of qwen4_exp builds int32 ones (it casts the
    selected token indices to int32; torch 2.11 accepts them). Its forward is
    rebuilt from its own source with the scatter indices cast to int64: the same indices, so the same mask."""
    import inspect
    import textwrap
    cls = M.Qwen4ExpTextQSAIndexer
    if getattr(cls.forward, '_p2_int64', False) or tuple(int(x) for x in torch.__version__.split('.')[:2]) >= (2, 7):
        return
    src = textwrap.dedent(inspect.getsource(cls.forward))
    old = 'scatter_indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)'
    assert src.count(old) == 1 and 'super()' not in src, 'qwen4_exp indexer source changed'
    ns = {}
    exec(compile(src.replace(old, old + '.long()'), inspect.getsourcefile(cls.forward), 'exec'), M.__dict__, ns)
    ns['forward']._p2_int64 = True
    cls.forward = ns['forward']
    print('[load] qwen4_exp: QSA indexer scatter indices cast to int64 (torch < 2.7)', flush=True)


def before_load(cfg):
    """Call before from_pretrained(config=cfg, ...)."""
    import transformers.conversion_mapping as CM
    import transformers.models.qwen4_exp.modeling_qwen4_exp as M
    _int64_scatter_indices(M)
    CM.get_checkpoint_conversion_mapping('qwen4_exp_text')
    cache = CM._checkpoint_conversion_mapping_cache
    for key in list(cache):
        if key.startswith('qwen4_exp'):
            cache[key] = [c for c in cache[key] if NGRAM_KEY not in str(getattr(c, 'source_patterns', ''))]
    orig = M.Qwen4ExpTextNGramEmbedding.__init__
    if not getattr(orig, '_mmap_patched', False):
        def init(self, *a, **k):
            orig(self, *a, **k)
            self.ngram_embedding = _Placeholder()
        init._mmap_patched = True
        M.Qwen4ExpTextNGramEmbedding.__init__ = init
    for cls_name in ('Qwen4ExpTextModel', 'Qwen4ExpModel', 'Qwen4ExpForConditionalGeneration', 'Qwen4ExpTextForCausalLM'):
        cls = getattr(M, cls_name, None)
        if cls is None:
            continue
        if getattr(cls, '_no_placement_params', None):
            cls._no_placement_params = None
        ign = set(getattr(cls, '_keys_to_ignore_on_load_unexpected', None) or [])
        cls._keys_to_ignore_on_load_unexpected = list(ign | {r'.*ngram_embedding\.shard_\d+\.weight', r'.*ngram_embedding\.weight_scale'})
    qc = getattr(cfg, 'quantization_config', None)
    if isinstance(qc, dict):
        qc['modules_to_convert'] = []
    elif qc is not None and hasattr(qc, 'modules_to_convert'):
        qc.modules_to_convert = []
    return cfg


def after_load(model, model_dir):
    """Attach the memory-mapped tables; returns the module names that got one."""
    done = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == 'Qwen4ExpTextNGramEmbedding':
            assert isinstance(mod.ngram_embedding, _Placeholder), type(mod.ngram_embedding)
            mod.ngram_embedding = MMapNgramEmbedding(model_dir, f'{name}.{NGRAM_KEY}')
            done.append(name)
    assert done, 'no n-gram embedding found'
    return done


def check_rows(model_dir, prefix, ids):
    """Independent check of a few rows against safetensors' own reader (bytes and the scale)."""
    from safetensors import safe_open
    wm = json.load(open(os.path.join(model_dir, 'model.safetensors.index.json'), encoding='utf-8'))['weight_map']
    emb = MMapNgramEmbedding(model_dir, prefix)
    got = emb(torch.tensor(ids))
    with safe_open(os.path.join(model_dir, wm[f'{prefix}.weight_scale']), 'pt') as fh:
        scale = fh.get_tensor(f'{prefix}.weight_scale')
    ref = []
    for i in ids:
        s, r = divmod(i, emb.rows)
        k = f'{prefix}.shard_{s}.weight'
        with safe_open(os.path.join(model_dir, wm[k]), 'pt') as fh:
            ref.append(fh.get_slice(k)[r:r + 1])
    ref = torch.cat(ref).to(scale.dtype) * scale
    return float((got.float() - ref.float()).abs().max()), emb.num_embeddings
