"""Pure-torch stand-in for the `kernels-community/finegrained-fp8` Triton kernel that transformers uses for FP8 checkpoints.

Why: under torch < 2.6 the hub kernel cannot load (it registers its ops with torch.library.triton_op, added in torch
2.6). install() keeps the hub kernel wherever it loads and otherwise points
transformers.integrations.finegrained_fp8._load_finegrained_fp8_kernel at this module.

Same contract and numerics as the kernel source (finegrained-fp8 version 1: act_quant.py, matmul.py, batched.py, grouped.py):
  fp8_act_quant(x, block): per row, blocks of `block` channels; s = max|x_block| / 448 (float32); q = (x / s) cast to float8_e4m3fn
  fp8_matmul(A, B, As, Bs, block_size, output_dtype): block mode C = sum_k (A_k . B_k) * As[:, k] * Bs[n_blk, k], accumulated in
      float32 (here: both operands dequantized in float32, then one float32 matmul, which is the same sum); tensor mode
      (block_size None or [N, K]) C = (A . B^T) * As[:, None] * Bs
  batched_fp8_matmul(A, B, Bs, expert_ids, block_size): row s uses expert expert_ids[s]; A in high precision, quantized inside
      (fp8_act_quant with block_size[1], or the full K in tensor mode); output dtype of A
  grouped_fp8_matmul(A, B, Bs, offsets, tokens_per_expert, block_size): rows sorted by expert; expert e owns rows
      [offsets[e] - tokens_per_expert[e], offsets[e]); rows past offsets[-1] are left at zero (transformers masks them)

transformers 5.19 changed the interface (10-19): it keeps the kernel in the module global _FINEGRAINED_FP8 (a frozen
FineGrainedFP8(matmul, batched_matmul, grouped_matmul)) and every entry point takes raw high-precision activations with the
quantization fused in. Checked against the kernel it pins (kernels-community/finegrained-fp8 revision 8c178950, metadata
version 4, kernel commit 89d4054f; matmul.py, batched.py, grouped.py, utils.py): the activation scale is still per row and
per K tile of block_size[1] (full K in tensor mode), s = max|x| / 448 in float32, the cast divides by max(s, 1e-12); weight
scales are float32 per block and the dot is accumulated in float32; with a static activation scale, a = (x / s_static) cast
to FP8 and the output is multiplied by s_static once; batched rows whose expert id is >= the number of experts are skipped.
matmul_2d / matmul_batched / matmul_grouped below implement that contract. MX formats (UE8M0 group-32 scales, FP4-packed
int8 weights) are not implemented and raise.
"""
from __future__ import annotations
from types import SimpleNamespace

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def fp8_act_quant(x: torch.Tensor, block_size: int = 128):
    assert x.shape[-1] % block_size == 0
    xs = x.float().reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    s = xs.abs().amax(-1) / FP8_MAX
    y = (xs / s.clamp_min(1e-12).unsqueeze(-1)).to(FP8).reshape(x.shape)
    return y, s


def _expand(scale: torch.Tensor, rows: int, cols: int, bn: int, bk: int) -> torch.Tensor:
    return scale.repeat_interleave(bn, 0)[:rows].repeat_interleave(bk, 1)[:, :cols]


def _tensor_mode(block_size, B):
    return block_size is None or (block_size[0] == B.size(0) and block_size[1] == B.size(1))


def fp8_matmul(A, B, As, Bs, block_size, output_dtype=torch.float32):
    N, K = B.shape
    A2 = A.reshape(-1, K).float()
    if _tensor_mode(block_size, B):
        C = (A2 @ B.float().t()) * As.reshape(-1, 1).float() * Bs.float().reshape(())
    else:
        bn, bk = block_size
        A2 = A2 * As.reshape(-1, As.shape[-1]).float().repeat_interleave(bk, 1)[:, :K]
        C = A2 @ (B.float() * _expand(Bs.float(), N, K, bn, bk)).t()
    return C.to(output_dtype).reshape(*A.shape[:-1], N)


def _quant_matmul(A, W, Ws, block_size, output_dtype=None):
    blk = A.shape[-1] if _tensor_mode(block_size, W) else block_size[1]
    qa, sa = fp8_act_quant(A.contiguous(), blk)
    return fp8_matmul(qa, W, sa, Ws, block_size, output_dtype or A.dtype)


def batched_fp8_matmul(A, B, Bs, expert_ids=None, block_size=None):
    out = torch.zeros(A.shape[0], B.shape[1], dtype=A.dtype, device=A.device)
    for e in torch.unique(expert_ids).tolist():
        rows = (expert_ids == e).nonzero(as_tuple=True)[0]
        out[rows] = _quant_matmul(A[rows], B[e], Bs[e], block_size)
    return out


def grouped_fp8_matmul(A, B, Bs, offsets=None, tokens_per_expert=None, block_size=None):
    out = torch.zeros(A.shape[0], B.shape[1], dtype=A.dtype, device=A.device)
    ends = offsets.to(torch.long).tolist()
    counts = tokens_per_expert.to(torch.long).tolist()
    for e, (end, n) in enumerate(zip(ends, counts)):
        if n > 0:
            out[end - n:end] = _quant_matmul(A[end - n:end], B[e], Bs[e], block_size)
    return out


FALLBACK = SimpleNamespace(fp8_matmul=fp8_matmul, fp8_act_quant=fp8_act_quant,
                           batched_fp8_matmul=batched_fp8_matmul, grouped_fp8_matmul=grouped_fp8_matmul)


# ---- transformers >= 5.19 entry points (raw activations, quantization fused) ----

def _check_block_fp8(B, Bs):
    # Block scales may be stored as float32 (GLM-5.3-Flash) or bfloat16 (Qwen3.5-122B-A10B-FP8, 10-21); every use below converts
    # them with .float(), which is exact for bfloat16, so both give the same arithmetic as the kernel's float32 scale multiply.
    if B.dtype != FP8 or Bs.dtype not in (torch.float32, torch.bfloat16):
        raise NotImplementedError(f'torch fallback supports float8_e4m3fn weights with float32 or bfloat16 block scales only, got {B.dtype} / {Bs.dtype}')


def _static_quant(x, s_static):
    s = s_static.float().reshape(())
    return (x.float() / s).to(FP8), s


def matmul_2d(A, B, Bs, block_size, output_dtype=None, activation_scale=None):
    _check_block_fp8(B, Bs)
    out = output_dtype or A.dtype
    if activation_scale is None:
        return _quant_matmul(A, B, Bs, block_size, out)
    N, K = B.shape
    qa, s = _static_quant(A.reshape(-1, K), activation_scale)
    if _tensor_mode(block_size, B):
        C = (qa.float() @ B.float().t()) * Bs.float().reshape(())
    else:
        bn, bk = block_size
        C = qa.float() @ (B.float() * _expand(Bs.float(), N, K, bn, bk)).t()
    return (C * s).to(out).reshape(*A.shape[:-1], N)


def matmul_batched(A, B, Bs, expert_ids, block_size, output_dtype=None):
    _check_block_fp8(B, Bs)
    out = torch.zeros(A.shape[0], B.shape[1], dtype=output_dtype or A.dtype, device=A.device)
    for e in torch.unique(expert_ids).tolist():
        if e >= B.shape[0]:            # expert-parallel sentinel: the kernel skips these rows
            continue
        rows = (expert_ids == e).nonzero(as_tuple=True)[0]
        out[rows] = _quant_matmul(A[rows], B[e], Bs[e], block_size, out.dtype)
    return out


def matmul_grouped(A, B, Bs, offsets, tokens_per_expert, block_size, output_dtype=None):
    _check_block_fp8(B, Bs)
    out = torch.zeros(A.shape[0], B.shape[1], dtype=output_dtype or A.dtype, device=A.device)
    ends = offsets.to(torch.long).tolist()
    counts = tokens_per_expert.to(torch.long).tolist()
    for e, (end, n) in enumerate(zip(ends, counts)):
        if n > 0:
            out[end - n:end] = _quant_matmul(A[end - n:end], B[e], Bs[e], block_size, out.dtype)
    return out


def install() -> str:
    """Use the hub kernel if it loads, else this module. Returns the backend name. Safe to call repeatedly in one process.
    P2_FP8_FORCE_FALLBACK=1 skips the hub kernel, so that every run of a model takes the same path."""
    import os
    import transformers.integrations.finegrained_fp8 as F
    force = os.environ.get('P2_FP8_FORCE_FALLBACK', '0') == '1'
    if not getattr(F, '_p2_fp8_backend', None):
        if hasattr(F, '_FINEGRAINED_FP8'):          # transformers >= 5.19: the kernel lives in a module global
            ok = False
            if not force:
                try:
                    F._load_finegrained_fp8_kernel()
                    ok = F._FINEGRAINED_FP8 is not None
                except Exception:
                    ok = False
            if ok:
                F._p2_fp8_backend = 'hub-kernel'
            else:
                F._FINEGRAINED_FP8 = F.FineGrainedFP8(matmul=matmul_2d, batched_matmul=matmul_batched, grouped_matmul=matmul_grouped)
                F._p2_fp8_backend = 'torch-fallback-v519'
        else:
            try:
                if force:
                    raise ImportError('P2_FP8_FORCE_FALLBACK')
                F._load_finegrained_fp8_kernel()
                F._p2_fp8_backend = 'hub-kernel'
            except Exception:
                F._load_finegrained_fp8_kernel = lambda: FALLBACK
                F._p2_fp8_backend = 'torch-fallback'
    allow_disk_for_prequantized()      # always: a process that installed an older version of this module lacks the patch
    return F._p2_fp8_backend


def Q_CLS():
    from transformers.quantizers import quantizer_finegrained_fp8 as Q
    return Q.FineGrainedFP8HfQuantizer


def allow_disk_for_prequantized():
    """transformers 5.8 FineGrainedFP8HfQuantizer.validate_environment refuses any 'disk' entry in the device map, even for an
    already-quantized checkpoint (its condition `not pre_quantized and ... and 'cpu' in values or 'disk' in values` binds the
    `or` last). For pre-quantized checkpoints validate a copy of the map with disk entries shown as cpu, which the same check
    already accepts; everything else in the check is unchanged."""
    from transformers.quantizers import quantizer_finegrained_fp8 as Q
    cls = Q.FineGrainedFP8HfQuantizer
    if getattr(cls, '_p2_disk_patch', False):
        return
    orig = cls.validate_environment

    def validate_environment(self, *args, **kwargs):
        dm = kwargs.get('device_map')
        if self.pre_quantized and isinstance(dm, dict) and 'disk' in dm.values():
            kwargs = dict(kwargs, device_map={k: ('cpu' if v == 'disk' else v) for k, v in dm.items()})
        return orig(self, *args, **kwargs)
    cls.validate_environment = validate_environment
    cls._p2_disk_patch = True


def self_check(device='cpu', seed=0):
    """Fallback against an independent float64 dequantize-then-matmul reference (block 128 x 128)."""
    g = torch.Generator().manual_seed(seed)
    M, N, K, E = 37, 384, 512, 4
    A = torch.randn(M, K, generator=g).to(device, torch.bfloat16)
    W = (torch.randn(E, N, K, generator=g) * 0.05)
    Ws = W.abs().reshape(E, N // 128, 128, K // 128, 128).amax((2, 4)) / FP8_MAX
    Wq = (W / Ws.repeat_interleave(128, 1).repeat_interleave(128, 2)).to(FP8)
    Wq, Ws = Wq.to(device), Ws.to(device).float()
    qa, sa = fp8_act_quant(A, 128)
    ref = (qa.double() * sa.double().repeat_interleave(128, 1)) @ (Wq[0].double() * Ws[0].double().repeat_interleave(128, 0).repeat_interleave(128, 1)).t()
    got = fp8_matmul(qa, Wq[0], sa, Ws[0], [128, 128], torch.float32)
    err = float((got.double() - ref).abs().max() / ref.abs().max())
    ids = torch.randint(0, E, (M,), generator=g).to(device)
    b = batched_fp8_matmul(A, Wq, Ws, expert_ids=ids, block_size=[128, 128])
    order = torch.argsort(ids, stable=True)
    tpe = torch.bincount(ids, minlength=E)
    gr = grouped_fp8_matmul(A[order], Wq, Ws, offsets=torch.cumsum(tpe, 0), tokens_per_expert=tpe, block_size=[128, 128])
    bg = float((b[order].float() - gr.float()).abs().max())
    # 5.19 entry points: same numbers as the 5.8 ones on the same inputs; a sentinel expert id leaves its row at zero
    m2 = matmul_2d(A, Wq[0], Ws[0], [128, 128], torch.float32)
    m2_err = float((m2.double() - ref).abs().max() / ref.abs().max())
    ids_s = ids.clone()
    ids_s[0] = E
    b2 = matmul_batched(A, Wq, Ws, ids_s, [128, 128])
    b2_diff = float((b2[1:].float() - b[1:].float()).abs().max())
    g2 = matmul_grouped(A[order], Wq, Ws, torch.cumsum(tpe, 0), tpe, [128, 128])
    g2_diff = float((g2.float() - gr.float()).abs().max())
    s_static = A.float().abs().max() / FP8_MAX
    st = matmul_2d(A, Wq[0], Ws[0], [128, 128], torch.float32, activation_scale=s_static)
    ref_st = ((A.double() / s_static.double()).float().to(FP8).double() * s_static.double()) @ (Wq[0].double() * Ws[0].double().repeat_interleave(128, 0).repeat_interleave(128, 1)).t()
    st_err = float((st.double() - ref_st).abs().max() / ref_st.abs().max())
    z = matmul_2d(torch.zeros(3, K, dtype=torch.bfloat16, device=device), Wq[0], Ws[0], [128, 128])
    # bfloat16 block scales: identical results to the same scale values held in float32
    Ws_bf = Ws.to(torch.bfloat16)
    g3 = matmul_grouped(A[order], Wq, Ws_bf, torch.cumsum(tpe, 0), tpe, [128, 128])
    g3_ref = matmul_grouped(A[order], Wq, Ws_bf.float(), torch.cumsum(tpe, 0), tpe, [128, 128])
    m3 = matmul_2d(A, Wq[0], Ws_bf[0], [128, 128], torch.float32)
    m3_ref = matmul_2d(A, Wq[0], Ws_bf[0].float(), [128, 128], torch.float32)
    return {'rel_err_vs_float64': err, 'batched_vs_grouped_max_abs': bg, 'v519_matmul_rel_err': m2_err,
            'v519_batched_vs_v58_max_abs': b2_diff, 'v519_sentinel_row_zero': bool((b2[0] == 0).all()),
            'v519_grouped_vs_v58_max_abs': g2_diff, 'v519_static_rel_err': st_err, 'zero_rows_finite_zero': bool((z == 0).all()),
            'bf16_scales_equal_to_fp32_grouped': bool(torch.equal(g3, g3_ref)), 'bf16_scales_equal_to_fp32_2d': bool(torch.equal(m3, m3_ref))}
