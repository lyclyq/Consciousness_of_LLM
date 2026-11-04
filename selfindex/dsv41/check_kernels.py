"""Checks of selfindex.dsv41.kernel_torch against float64 references written from the tilelang kernel source (official/inference/
kernel.py): the power-of-two scale rule, e2m1 rounding, act_quant (E8M0 scales, also in place), fp8_gemm, fp4_gemm, fp4_act_quant
(indexer and compressed-KV variants), sparse_attn and the hyper-connection Sinkhorn mixes.
usage: python -m selfindex.dsv41.check_kernels [cpu|cuda]   -> one line per check and a JSON summary; exit status 1 on any failure."""
import json
import math
import sys

import torch

from . import kernel_torch as K

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else ('cuda' if torch.cuda.is_available() else 'cpu'))
torch.manual_seed(0)
results = {}


def check(name, ok, detail):
    results[name] = {'ok': bool(ok), **detail}
    print(('PASS ' if ok else 'FAIL ') + name, detail, flush=True)


def ref_pow2_ceil(v):
    m, e = math.frexp(v)  # v = m * 2^e, 0.5 <= m < 1
    return 2.0 ** (e - 1) if m == 0.5 else 2.0 ** e


# 1. power-of-two scale rule, including exact powers of two and values just above them
vals = [2.0 ** k for k in range(-20, 10)] + [2.0 ** k * (1 + 2 ** -23) for k in range(-20, 10)] + [0.3, 1.7, 448 / 3, 1e-4 / 448]
v = torch.tensor(vals, dtype=torch.float32)
got = K._pow2_ceil_log2(v).tolist()
want = [ref_pow2_ceil(float(x)) for x in v.tolist()]
check('pow2_ceil_log2', got == want, {'n': len(vals), 'mismatch': sum(a != b for a, b in zip(got, want))})

# 2. e2m1 rounding: nearest grid value, ties to the even code
grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def ref_e2m1(x):
    a = min(abs(x), 6.0)
    best = min(range(8), key=lambda i: (abs(grid[i] - a), i % 2))
    return math.copysign(grid[best], x)


xs = torch.cat([torch.linspace(-6, 6, 4801), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -5.0, 5.999, 6.0])])
got = K.e2m1_round(xs).tolist()
want = [ref_e2m1(float(x)) for x in xs.tolist()]
check('e2m1_round', got == want, {'n': len(want), 'mismatch': sum(a != b for a, b in zip(got, want))})

# 3. act_quant with E8M0 scales (the model's setting: block 32, scale_fmt ue8m0)
x = (torch.randn(64, 512) * torch.logspace(-3, 2, 64).unsqueeze(1)).to(torch.bfloat16).to(dev)
x[3, :32] = 0  # an all-zero block takes the 1e-4 floor
y, s = K.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
xd = x.double().cpu().view(64, 16, 32)
amax = xd.abs().amax(-1).clamp_min(1e-4)
inv448 = torch.tensor(1 / 448, dtype=torch.float32)
s_ref = torch.tensor([[ref_pow2_ceil(float(torch.tensor(float(a), dtype=torch.float32) * inv448)) for a in row] for row in amax])
s_got = K.e8m0_to_float(s.cpu()).double()
y_ref = (xd / s_ref.unsqueeze(-1)).clamp(-448, 448).float().to(torch.float8_e4m3fn).view(64, 512)
check('act_quant_e8m0', torch.equal(s_got, s_ref) and torch.equal(y.cpu().view(torch.uint8), y_ref.view(torch.uint8)),
      {'scale_mismatch': int((s_got != s_ref).sum()), 'value_mismatch': int((y.cpu().view(torch.uint8) != y_ref.view(torch.uint8)).sum())})
xi = x.clone()
K.act_quant(xi, 32, 'ue8m0', torch.float8_e8m0fnu, True)
check('act_quant_inplace', torch.equal(xi.cpu(), (y_ref.float().view(64, 16, 32) * s_ref.float().unsqueeze(-1)).view(64, 512).bfloat16()), {})


# 4. fp8_gemm against float64 sum over K blocks of the scaled block products
def rand_fp8(r, c):
    return (torch.randn(r, c) * 20).clamp(-448, 448).to(torch.float8_e4m3fn)


M_, N_, K_ = 48, 160, 512
a = rand_fp8(M_, K_).to(dev)
b = rand_fp8(N_, K_).to(dev)
a_s = K.float_to_e8m0(torch.exp2(torch.randint(-12, -4, (M_, K_ // 32)).float())).to(dev)
b_s = K.float_to_e8m0(torch.exp2(torch.randint(-12, -4, ((N_ + 31) // 32, K_ // 32)).float())).to(dev)
c = K.fp8_gemm(a, a_s, b, b_s, torch.float8_e8m0fnu, 32)
A = a.cpu().double().view(M_, K_ // 32, 32)
B = b.cpu().double().view(N_, K_ // 32, 32)
sa = K.e8m0_to_float(a_s.cpu()).double()
sb = K.e8m0_to_float(b_s.cpu()).double().repeat_interleave(32, 0)[:N_]
c_ref = torch.einsum('mkb,nkb,mk,nk->mn', A, B, sa, sb)
err = ((c.cpu().double() - c_ref).abs() / c_ref.abs().clamp_min(1e-30)).max().item()
check('fp8_gemm', err <= 2 ** -8, {'max_rel_err': err, 'out_dtype': str(c.dtype)})


# 5. fp4_gemm: weights packed two per byte (low nibble first, as convert.py decodes them), per-32 E8M0 scales
codes = torch.randint(0, 256, (N_, K_ // 2), dtype=torch.int32).to(torch.uint8)
table = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
Wd = torch.tensor([[table[int(v) & 15] for v in row for _ in (0,)] for row in codes], dtype=torch.float64)  # low nibbles
Wh = torch.tensor([[table[(int(v) >> 4) & 15] for v in row] for row in codes], dtype=torch.float64)
W = torch.stack([Wd, Wh], -1).view(N_, K_)
w_s = K.float_to_e8m0(torch.exp2(torch.randint(-6, 2, (N_, K_ // 32)).float()))
c4 = K.fp4_gemm(a, a_s, codes.view(torch.float4_e2m1fn_x2).to(dev) if dev.type == 'cpu' else codes.to(dev).view(torch.float4_e2m1fn_x2),
                w_s.to(dev), torch.float8_e8m0fnu, 32)
Wsc = W.view(N_, K_ // 32, 32) * K.e8m0_to_float(w_s).double().unsqueeze(-1)
c4_ref = torch.einsum('mkb,nkb,mk->mn', A, Wsc, sa)
err4 = ((c4.cpu().double() - c4_ref).abs() / c4_ref.abs().clamp_min(1e-30)).max().item()
check('fp4_gemm', err4 <= 2 ** -8, {'max_rel_err': err4})

# 6. fp4_act_quant (indexer: block 32, E8M0) and the compressed-KV variant (block 16, E4M3 scales), in place
z = (torch.randn(8, 4, 128) * 3).to(torch.bfloat16).to(dev)
for blk, sdt in ((32, torch.float8_e8m0fnu), (16, torch.float8_e4m3fn)):
    zi = z.clone()
    K.fp4_act_quant(zi, blk, True, sdt)
    zd = z.double().cpu().view(8, 4, 128 // blk, blk)
    amax = zd.abs().amax(-1)
    if sdt == torch.float8_e4m3fn:
        s_ref = (amax.clamp_min(6 * 2 ** -9).float() / 6).to(torch.float8_e4m3fn).double()
    else:
        s_ref = amax.clamp_min(6 * 2 ** -126).apply_(lambda q: ref_pow2_ceil(float(torch.tensor(q, dtype=torch.float32) * torch.tensor(1 / 6, dtype=torch.float32))))
    q_ref = torch.tensor([ref_e2m1(float(t)) for t in (zd.float() / s_ref.float().unsqueeze(-1)).clamp(-6, 6).flatten().tolist()],
                         dtype=torch.float64).view_as(zd)
    want = (q_ref * s_ref.unsqueeze(-1)).view(8, 4, 128).to(torch.bfloat16)
    check(f'fp4_act_quant_b{blk}', torch.equal(zi.cpu(), want), {'mismatch': int((zi.cpu() != want).sum())})


# 7. sparse_attn against a plain float64 softmax over the valid indices with the sink in the denominator
b_, m_, h_, d_, n_, topk = 2, 37, 8, 64, 90, 150
q = (torch.randn(b_, m_, h_, d_) / 4).to(torch.bfloat16).to(dev)
kv = torch.randn(b_, n_, d_).to(torch.bfloat16).to(dev)
sink = torch.randn(h_).to(dev)
idx = torch.randint(-1, n_, (b_, m_, topk), dtype=torch.int32)
idx[0, 0] = -1  # a row with no valid position must give zeros
idx = idx.to(dev)
o = K.sparse_attn(q, kv, sink, idx, d_ ** -0.5)
qd, kvd, sd, ix = q.double().cpu(), kv.double().cpu(), sink.double().cpu(), idx.long().cpu()
o_ref = torch.zeros(b_, m_, h_, d_, dtype=torch.float64)
for bb in range(b_):
    for mm in range(m_):
        sel = ix[bb, mm][ix[bb, mm] >= 0]
        if sel.numel() == 0:
            continue
        rows = kvd[bb, sel]
        sc = qd[bb, mm] @ rows.t() * d_ ** -0.5
        mx = torch.maximum(sc.amax(-1), torch.full((h_,), -1e30, dtype=torch.float64))
        p = torch.exp(sc - mx.unsqueeze(-1))
        o_ref[bb, mm] = (p @ rows) / (p.sum(-1) + torch.exp(sd - mx)).unsqueeze(-1)
erra = (o.cpu().double() - o_ref).abs().max().item()
check('sparse_attn', erra < 2e-2 and o[0, 0].abs().max().item() == 0, {'max_abs_err': erra, 'scale': o_ref.abs().max().item()})

# 8. hyper-connection mixes and Sinkhorn
mixes = torch.randn(3, 5, 24).to(dev)
hs, hb = torch.randn(3).to(dev), torch.randn(24).to(dev)
pre, post, comb = K.hc_split_sinkhorn(mixes, hs, hb, 4, 20, 1e-6)
md, hsd, hbd = mixes.double().cpu(), hs.double().cpu(), hb.double().cpu()
pre_r = torch.sigmoid(md[..., :4] * hsd[0] + hbd[:4]) + 1e-6
post_r = 2 * torch.sigmoid(md[..., 4:8] * hsd[1] + hbd[4:8])
cm = (md[..., 8:] * hsd[2] + hbd[8:]).view(3, 5, 4, 4)
cm = torch.softmax(cm, -1) + 1e-6
cm = cm / (cm.sum(-2, keepdim=True) + 1e-6)
for _ in range(19):
    cm = cm / (cm.sum(-1, keepdim=True) + 1e-6)
    cm = cm / (cm.sum(-2, keepdim=True) + 1e-6)
errh = max((pre.cpu().double() - pre_r).abs().max().item(), (post.cpu().double() - post_r).abs().max().item(), (comb.cpu().double() - cm).abs().max().item())
check('hc_split_sinkhorn', errh < 1e-5, {'max_abs_err': errh})

print(json.dumps({'device': str(dev), 'torch': torch.__version__, 'all_ok': all(r['ok'] for r in results.values()), 'results': results}))
sys.exit(0 if all(r['ok'] for r in results.values()) else 1)
