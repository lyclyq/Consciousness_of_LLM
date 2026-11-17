"""Pure-PyTorch replacements for the tilelang kernels of DeepSeek-V4.1-Flash-1017 (inference/kernel.py, same signatures).

The official kernels need tilelang JIT with CUDA >= 12.8 FP4 types; server3 has CUDA 12.4, so every kernel is re-implemented
here from the kernel source, rule by rule:
  act_quant      per row and K block: amax = max(|x|, 1e-4); scale = amax * (1/448), or with scale_fmt the power of two
                 2^ceil(log2(amax * (1/448))) computed from the float32 bits exactly as fast_round_scale does; y = e4m3(clamp(x/s))
  fp4_act_quant  per row and block: E8M0 scale 2^ceil(log2(max(amax, 6*2^-126) * (1/6))) or E4M3 scale e4m3(max(amax, 6*2^-9) / 6);
                 y = e2m1(clamp(x/s, -6, 6)) with round-to-nearest-even on the e2m1 grid
  fp8_gemm       C = sum over K blocks of (A_blk @ B_blk^T) * s_a[m, kb] * s_b[n // blk, kb], fp32 accumulation, output in the
                 default dtype. With power-of-two scales (E8M0, the only kind this model uses) every dequantized operand
                 a * s_a and b * s_b is exact in bfloat16, so the product is computed as one bf16 GEMM with fp32 accumulation
                 (reduced-precision reductions disabled); only the fp32 summation order differs from the kernel.
  fp4_gemm       the same with e2m1 weights (packed two per byte, low nibble first, as in convert.py) and per-32 E8M0 scales
  sparse_attn    the kernel's online softmax over blocks of 64 gathered positions in index order, including its two
                 roundings: q.k accumulated in fp32, probabilities cast to bf16 before the P.V product, running sum in fp32,
                 attention sink added to the denominator, rows with no valid index give zeros
  hc_split_sinkhorn  sigmoid pre/post mixes and the Sinkhorn-normalized comb matrix, fp32, same iteration order
"""
import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # codes 0..7; the low bit of the code is the mantissa bit
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                         dtype=torch.float32)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cudnn.allow_tf32 = False


def _f32(c):
    return torch.tensor(c, dtype=torch.float32)


def _pow2_ceil_log2(v):
    """2^ceil(log2(v)) for positive normal float32 v, from the IEEE bits (fast_log2_ceil + fast_pow2 in kernel.py)."""
    bits = v.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    e = exp - 127 + (man != 0).to(torch.int32)
    return ((e + 127) << 23).view(torch.float32)


def e8m0_to_float(s):
    """float8_e8m0fnu (or its uint8 bytes) -> float32 power of two."""
    u = s.view(torch.uint8) if s.dtype != torch.uint8 else s
    return torch.ldexp(torch.ones_like(u, dtype=torch.float32), u.to(torch.int32) - 127)


def float_to_e8m0(s):
    """Exact for powers of two in the normal range."""
    bits = s.contiguous().float().view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8).view(torch.float8_e8m0fnu)


def scale_to_float(s):
    if s.dtype in (torch.float8_e8m0fnu, torch.uint8):
        return e8m0_to_float(s)
    return s.float()


def e2m1_round(y):
    """Round float32 values in [-6, 6] to the nearest e2m1 value, ties to the even code."""
    a = y.abs()
    grid = torch.tensor(_E2M1_GRID, dtype=torch.float32, device=y.device)
    lo = (torch.bucketize(a, grid, right=True) - 1).clamp(0, 6)
    hi = lo + 1
    d_lo = a - grid[lo]
    d_hi = grid[hi] - a
    take_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    code = torch.where(take_hi, hi, lo)
    q = grid[code]
    q = torch.where(a >= 6.0, torch.full_like(q, 6.0), q)
    return torch.copysign(q, y)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(-1).clamp_min(1e-4)
    s = amax * _f32(1.0 / FP8_MAX)
    if scale_fmt is not None:
        s = _pow2_ceil_log2(s)
    yq = (xf / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        y = (yq.float() * s.unsqueeze(-1)).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    s_out = float_to_e8m0(s) if scale_dtype == torch.float8_e8m0fnu else s.to(scale_dtype)
    return yq.flatten(-2), s_out


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(FP4_MAX * 2.0 ** -9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(FP4_MAX * 2.0 ** -126)
        s = _pow2_ceil_log2(amax * _f32(1.0 / FP4_MAX))
    yq = e2m1_round((xf / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((yq * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    raise NotImplementedError('fp4_act_quant is only used in place by the model')


def _dequant_rows(q, s, block):
    """[R, K] fp8 values with one scale per (row, K block) -> bf16 (exact for power-of-two scales)."""
    R, K = q.shape
    v = q.float().view(R, K // block, block) * scale_to_float(s).view(R, K // block, 1)
    return v.view(R, K)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert K % block_size == 0 and a_s.numel() == M * (K // block_size)
    assert b_s.shape == ((N + block_size - 1) // block_size, K // block_size)
    out_dtype = torch.get_default_dtype()
    A = _dequant_rows(a.reshape(M, K), a_s.reshape(M, -1), block_size)
    sb = scale_to_float(b_s).repeat_interleave(block_size, 0)[:N]  # [N, K // block]
    B = (b.float().view(N, K // block_size, block_size) * sb.unsqueeze(-1)).view(N, K)
    exact = scale_dtype == torch.float8_e8m0fnu or a_s.dtype == torch.float8_e8m0fnu
    if exact and a.is_cuda:
        C = torch.matmul(A.to(torch.bfloat16), B.to(torch.bfloat16).t()) if out_dtype == torch.bfloat16 else \
            torch.matmul(A, B.t())
    else:
        C = torch.matmul(A, B.t())
    return C.to(out_dtype).view(*a.shape[:-1], N)


def unpack_e2m1(b):
    """[N, K//2] packed e2m1 (float4_e2m1fn_x2, int8 or uint8) -> [N, K] float32; low nibble is the first element."""
    u = b.view(torch.uint8)
    table = FP4_TABLE.to(u.device)
    lo = table[(u & 0x0F).long()]
    hi = table[((u >> 4) & 0x0F).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert K % act_block_size == 0 and a_s.numel() == M * (K // act_block_size)
    assert tuple(b_s.shape) == (N, K // 32)
    out_dtype = torch.get_default_dtype()
    A = _dequant_rows(a.reshape(M, K), a_s.reshape(M, -1), act_block_size)
    B = (unpack_e2m1(b).view(N, K // 32, 32) * scale_to_float(b_s).view(N, K // 32, 1)).view(N, K)
    if a.is_cuda and out_dtype == torch.bfloat16:
        C = torch.matmul(A.to(torch.bfloat16), B.to(torch.bfloat16).t())
    else:
        C = torch.matmul(A, B.t())
    return C.to(out_dtype).view(*a.shape[:-1], N)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, block=64, chunk_queries=4096):
    """o[b, m, h, :] = sum_j p_j kv[b, idx_j] / (sum_j p_j + exp(sink_h - max)), following the kernel's online softmax."""
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    out = torch.empty_like(q)
    qf = q.reshape(b * m, h, d)
    idx_all = topk_idxs.reshape(b * m, topk).long()
    batch_of = torch.arange(b, device=q.device).repeat_interleave(m)
    sink = attn_sink.float()
    for c0 in range(0, b * m, chunk_queries):
        c1 = min(b * m, c0 + chunk_queries)
        Q = qf[c0:c1]
        n = c1 - c0
        acc_o = torch.zeros(n, h, d, dtype=torch.float32, device=q.device)
        sum_exp = torch.zeros(n, h, dtype=torch.float32, device=q.device)
        smax = torch.full((n, h), -1e30, dtype=torch.float32, device=q.device)
        for t0 in range(0, topk, block):
            idx = idx_all[c0:c1, t0:t0 + block]
            valid = idx != -1
            rows = kv[batch_of[c0:c1].unsqueeze(1), idx.clamp_min(0)]  # [n, blk, d] bf16
            rows = torch.where(valid.unsqueeze(-1), rows, torch.zeros((), dtype=rows.dtype, device=rows.device))
            s = torch.einsum('nhd,nkd->nhk', Q.float(), rows.float())
            s = torch.where(valid.unsqueeze(1), s, torch.full((), float('-inf'), device=s.device))
            s = s * softmax_scale
            prev = smax
            smax = torch.maximum(prev, s.amax(-1))
            scale_prev = torch.exp(prev - smax)
            p = torch.exp(s - smax.unsqueeze(-1))
            sum_exp = sum_exp * scale_prev + p.sum(-1)
            acc_o = acc_o * scale_prev.unsqueeze(-1) + torch.einsum('nhk,nkd->nhd', p.to(torch.bfloat16).float(), rows.float())
        sum_exp = sum_exp + torch.exp(sink.unsqueeze(0) - smax)
        out.view(b * m, h, d)[c0:c1] = (acc_o / sum_exp.unsqueeze(-1)).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    sc = hc_scale.float()
    base = hc_base.float()
    pre = torch.sigmoid(m[..., :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * sc[1] + base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * sc[2] + base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
