"""Pure-torch stand-in for the `kernels-community/finegrained-fp8` Triton kernel that transformers uses for FP8 checkpoints.

Why: on server3 (stable torch 2.5.1, not to be upgraded) the hub kernel cannot load (it registers its ops with
torch.library.triton_op, added in torch 2.6). install() keeps the hub kernel wherever it loads and otherwise points
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
    y = (xs / s.unsqueeze(-1)).to(FP8).reshape(x.shape)
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


def _quant_matmul(A, W, Ws, block_size):
    blk = A.shape[-1] if _tensor_mode(block_size, W) else block_size[1]
    qa, sa = fp8_act_quant(A.contiguous(), blk)
    return fp8_matmul(qa, W, sa, Ws, block_size, A.dtype)


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


def install() -> str:
    """Use the hub kernel if it loads, else this module. Returns the backend name. Safe to call repeatedly in one process."""
    import transformers.integrations.finegrained_fp8 as F
    if not getattr(F, '_p2_fp8_backend', None):
        try:
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
    return {'rel_err_vs_float64': err, 'batched_vs_grouped_max_abs': bg}
