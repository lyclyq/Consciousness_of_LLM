"""Loading a checkpoint for the measurements: BF16 (or the checkpoint's own FP8/MXFP4 format), eval mode, the decoder layers found
by name, and a load that refuses to run when checkpoint tensors went unused or parameters stayed uninitialised.

Placement is a run-time choice: layers are spread over the visible GPUs by accelerate (device_map='auto'); --max-memory-gib caps
the memory used per GPU, --cpu-offload-gib allows weights to stay in CPU memory (moved to the GPU when used) and
--disk-offload-dir allows disk offload. Placement does not change the computation.
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
from dataclasses import dataclass

from .render import ids_of

TOKENIZER_PROBE = '账页青梧1列出三笔记录。甲签记着1，乙签记着2。 Ledger page Qingwu 1 lists three entries; tag Jia records 4.'


def find_layers(model):
    """The decoder layers: the longest ModuleList of '...Layer' or '...Block' modules outside vision, audio and MTP towers."""
    import torch
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8 and not any(k in name.lower() for k in ('vision', 'visual', 'audio', 'mtp')):
            cls = type(mod[0]).__name__
            if ('Layer' in cls or 'Block' in cls) and (best is None or len(mod) > len(best[1])):
                best = (name, mod)
    return best


def checked_tokenizer(tok, model_dir):
    """The tokenizer as loaded, unless it disagrees with the checkpoint's tokenizer.json on a mixed Chinese/English probe; then a
    fast tokenizer rebuilt from tokenizer.json (with the chat template of the checkpoint)."""
    tj = os.path.join(model_dir, 'tokenizer.json')
    if not os.path.exists(tj):
        return tok
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(tj)
    want = raw.encode(TOKENIZER_PROBE, add_special_tokens=False).ids
    if ids_of(tok(TOKENIZER_PROBE, add_special_tokens=False)) == want:
        return tok
    from transformers import PreTrainedTokenizerFast
    cfgp = os.path.join(model_dir, 'tokenizer_config.json')
    tc = json.load(open(cfgp, encoding='utf-8')) if os.path.exists(cfgp) else {}
    special = {}
    for k in ('bos_token', 'eos_token', 'pad_token', 'unk_token'):
        v = tc.get(k)
        if isinstance(v, dict):
            v = v.get('content')
        if v:
            special[k] = v
    fixed = PreTrainedTokenizerFast(tokenizer_file=tj, **special)
    tmpl = tc.get('chat_template')
    jp = os.path.join(model_dir, 'chat_template.jinja')
    if os.path.exists(jp):
        tmpl = open(jp, encoding='utf-8').read()
    fixed.chat_template = tmpl if tmpl is not None else tok.chat_template
    assert ids_of(fixed(TOKENIZER_PROBE, add_special_tokens=False)) == want, 'tokenizer rebuild failed'
    print(f'[tokenizer] {type(tok).__name__} disagreed with tokenizer.json; rebuilt from tokenizer.json', flush=True)
    fixed._p2_rebuilt_from = type(tok).__name__
    return fixed


def check_loading_info(info, cfg, model_dir):
    """Refuse a load in which checkpoint tensors went unused or parameters stayed uninitialised (e.g. transformers merging
    gate/up projections of an FP8 checkpoint and dropping gate_proj.weight_scale_inv). Allowed: multi-token-prediction weights
    (mtp.*, or decoder layers at index >= num_hidden_layers), weights deliberately skipped by a model patch (n-gram table),
    and, as missing only, the DSA indexer of a layer whose indexer type is 'shared': transformers builds an indexer in every
    layer, but a shared layer reuses the previous layer's top-k indices and never calls its own, and the checkpoint has none."""
    tc = getattr(cfg, 'text_config', cfg)
    nl = getattr(tc, 'num_hidden_layers', None)
    itypes = getattr(tc, 'indexer_types', None) or []

    def allowed(k):
        if k.startswith('mtp.') or '.mtp.' in k or 'ngram_embedding' in k:
            return True
        m = re.search(r'layers\.(\d+)\.', k)
        return bool(nl is not None and m and int(m.group(1)) >= nl)

    def unused_shared_indexer(k):
        m = re.search(r'layers\.(\d+)\.self_attn\.indexer\.', k)
        return bool(m and int(m.group(1)) < len(itypes) and itypes[int(m.group(1))] == 'shared')
    unexpected = [k for k in info.get('unexpected_keys', []) if not allowed(k)]
    missing = [k for k in info.get('missing_keys', []) if not allowed(k) and not unused_shared_indexer(k)]
    mism = list(info.get('mismatched_keys', []))
    print(f'[load] unexpected {len(unexpected)} missing {len(missing)} mismatched {len(mism)}', flush=True)
    bad = unexpected + missing + [str(x) for x in mism]
    assert not bad, f'checkpoint/model mismatch, refusing to run: {bad[:6]}'


@dataclass
class Loaded:
    torch: object
    tok: object
    model: object
    cfg: object
    placement: dict
    model_class: str
    fp8_backend: object = None
    sim_fp8: object = None


def load_model(model_dir, *, max_memory_gib=None, cpu_offload_gib=0.0, disk_offload_dir=None, sim_fp8=None, allow_cpu=False):
    """Load tokenizer and model. allow_cpu is for the tests on small random models; measurements run on GPUs."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    if not torch.cuda.is_available() and not allow_cpu:
        sys.exit('CUDA unavailable: refuse to run on CPU')
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tok = checked_tokenizer(tok, model_dir)
    kw = dict(dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    if max_memory_gib or cpu_offload_gib:
        from accelerate.utils import get_max_memory
        mm = {i: f'{max_memory_gib}GiB' for i in range(torch.cuda.device_count())} if max_memory_gib else \
            {k: v for k, v in get_max_memory().items() if k != 'cpu'}
        if cpu_offload_gib:
            mm['cpu'] = f'{cpu_offload_gib}GiB'
        kw['max_memory'] = mm
    if disk_offload_dir:
        kw['offload_folder'] = disk_offload_dir
    qc = getattr(cfg, 'quantization_config', None)
    qm = (qc.get('quant_method') if isinstance(qc, dict) else getattr(qc, 'quant_method', None)) if qc is not None else None
    fp8_backend = None
    if str(qm) in ('fp8', 'QuantizationMethod.FP8'):
        from .patches import fp8_fallback
        fp8_backend = fp8_fallback.install()
        print(f'[fp8] backend {fp8_backend}, disk patch {getattr(fp8_fallback.Q_CLS(), "_p2_disk_patch", False)}', flush=True)
    flashnext = getattr(cfg, 'model_type', '') == 'qwen4_exp'
    if flashnext:
        from .patches import flashnext_patch
        kw['config'] = flashnext_patch.before_load(cfg)
    if getattr(cfg, 'model_type', '') == 'glm5_next':
        # GLM-5.3-Flash keeps the linear-attention forget-gate projections in BF16 and lists them in modules_to_not_convert
        # as model.layers.N.self_attn.f_{a,b}_proj, but transformers 5.19 builds them as ...self_attn.forget_gate.f_{a,b}_proj,
        # so the names do not match and the layers were made FP8 with missing scales (load report: MISSING weight_scale_inv).
        # transformers matches a skip pattern only as a prefix (re.match) or a suffix (endswith), so name the suffixes.
        extra = ['forget_gate.f_a_proj', 'forget_gate.f_b_proj']
        q = getattr(cfg, 'quantization_config', None)
        if isinstance(q, dict):
            q['modules_to_not_convert'] = list(q.get('modules_to_not_convert') or []) + extra
        elif q is not None:
            q.modules_to_not_convert = list(getattr(q, 'modules_to_not_convert', None) or []) + extra
        kw['config'] = cfg
        print('[load] glm5_next: forget_gate kept in BF16 (modules_to_not_convert)', flush=True)
        # Its sparse-attention indexer returns int32 top-k key indices, which the attention scatters into a mask; torch 2.6
        # accepts only int64 scatter indices. Cast them before the mask is built (same values, same mask).
        from transformers.models.glm5_next import modeling_glm5_next as G
        A = G.Glm5NextTextAttention
        if not getattr(A, '_p2_long_topk', False):
            orig = A.build_attention_mask_from_topk

            def build_attention_mask_from_topk(self, topk_indices, *a, _orig=orig, **k):
                return _orig(self, topk_indices.long(), *a, **k)
            A.build_attention_mask_from_topk = build_attention_mask_from_topk
            A._p2_long_topk = True
        print('[load] glm5_next: top-k indices cast to int64 before the attention mask scatter', flush=True)
    multimodal = hasattr(cfg, 'vision_config') or any('ConditionalGeneration' in a for a in (cfg.architectures or []))
    cls = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    kw['output_loading_info'] = True
    first_error = None
    try:
        model, info = cls.from_pretrained(model_dir, attn_implementation='sdpa', **kw)
    except Exception as e:
        first_error = repr(e)[:300]
    if first_error is not None:                    # outside the except block, so the failed attempt's tensors can be freed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f'[load] sdpa attempt failed ({first_error}); retrying with the default attention', flush=True)
        model, info = cls.from_pretrained(model_dir, **kw)
    check_loading_info(info, cfg, model_dir)
    model.eval()
    if flashnext:
        flashnext_patch.after_load(model, model_dir)
    sim = None
    if sim_fp8:
        from .patches import fp8_fakequant
        sim = (sim_fp8, fp8_fakequant.apply(model, sim_fp8))
        print(f'[sim-fp8] {sim[1]} linear layers quantized ({sim_fp8})', flush=True)
    placement = {k: str(v) for k, v in getattr(model, 'hf_device_map', {}).items()}
    if not disk_offload_dir:
        assert not any(v == 'disk' for v in placement.values()), 'disk offload refused (pass --disk-offload-dir to allow it)'
    if not cpu_offload_gib and not allow_cpu:
        assert not any(v == 'cpu' for v in placement.values()), 'CPU offload refused (pass --cpu-offload-gib to allow weight offload)'
    return Loaded(torch, tok, model, cfg, placement, cls.__name__, fp8_backend, sim)
