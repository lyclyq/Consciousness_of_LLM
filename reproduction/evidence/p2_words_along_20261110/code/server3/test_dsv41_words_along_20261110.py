"""CPU check of dsv41_words_along_20261110.py on a small random-weight model built from the official model.py (as
test_dsv41_forward.py: ModelArgs defaults, 5 layers, compressed and windowed attention, indexer, hyper-connections, FP8 linears,
FP4 experts, dsv41_kernel_torch). Checks:
  1. forward_capture's last-position states equal the onset states of the paper's runner (dsv41_natural3.forward_seqs) bit for bit
     at every layer, after the runner's float16 rounding;
  2. the mean of forward_capture's body states at the kept layers equals the runner's content means bit for bit (same rounding);
  3. decomp equals a float64 numpy computation of the first- and last-third mean differences and mean squared differences;
  4. the grouped capture of one sequence equals that sequence captured alone (no dependence on the group).
usage: CUDA_VISIBLE_DEVICES= test_dsv41_words_along_20261110.py OFFICIAL_INFERENCE_DIR"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dsv41_kernel_torch as K  # noqa: E402

sys.modules['kernel'] = K
sys.path.insert(0, sys.argv[1])
import model as M  # noqa: E402
import dsv41_natural3 as R  # noqa: E402
import dsv41_words_along_20261110 as W  # noqa: E402

dev = torch.device('cpu')
torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
args = M.ModelArgs(max_batch_size=4, max_seq_len=320)
with torch.device(dev):
    model = M.Transformer(args)
g = torch.Generator(device='cpu').manual_seed(1)
for name, p in list(model.named_parameters()):
    mod_name, leaf = name.rsplit('.', 1) if '.' in name else ('', name)
    mod = model.get_submodule(mod_name) if mod_name else model
    if p.dtype == torch.float8_e4m3fn:
        t = (torch.randn(p.shape, generator=g) * 0.6).clamp(-448, 448).to(torch.float8_e4m3fn)
    elif p.dtype == torch.float8_e8m0fnu:
        t = K.float_to_e8m0(torch.exp2(torch.randint(-7, -4, p.shape, generator=g).float()))
    elif p.dtype == torch.float4_e2m1fn_x2:
        t = torch.randint(0, 256, p.shape, generator=g, dtype=torch.int32).to(torch.uint8).view(torch.float4_e2m1fn_x2)
    elif 'norm' in name:
        t = torch.ones(p.shape, dtype=p.dtype)
    elif leaf == 'attn_sink' or name.endswith('gate.bias'):
        t = torch.zeros(p.shape, dtype=p.dtype)
    else:
        t = (torch.randn(p.shape, generator=g) * 0.02).to(p.dtype)
    mod._parameters[leaf] = torch.nn.Parameter(R.to_dev(t, dev), requires_grad=False)
for m in model.modules():
    if isinstance(m, M.Linear) and m.scale is not None:
        m.weight.scale = m.scale
model.eval()
torch.set_default_device(dev)
stats = {'expert_s': 0.0}
for layer in model.layers:  # routed experts in host memory, as in the real run
    for e in layer.ffn.experts:
        for k in ('w1', 'w2', 'w3'):
            lin = getattr(e, k)
            lin._parameters['weight'] = torch.nn.Parameter(R.to_dev(lin.weight.data, 'cpu'), requires_grad=False)
            lin._parameters['scale'] = torch.nn.Parameter(R.to_dev(lin.scale.data, 'cpu'), requires_grad=False)
            lin.weight.scale = lin.scale
R.patch_runtime(M, dev, stats)

L = len(model.layers)
keep = sorted({round(f * (L - 1)) for f in W.KEEP_FRACS})
seqs = [torch.randint(0, 129280, (n,), generator=g, device='cpu').tolist() for n in (150, 211, 97)]
caps = [list(range(10, 60)), list(range(30, 140)), list(range(5, 90))]
res = {'n_layers': L, 'keep': keep}

ref = R.forward_seqs(M, model, seqs, caps, dev, stage=True)            # the paper's runner: (lp, (content, onset)) [L, hc, dim] f16
mine = W.forward_capture(M, model, list(zip(seqs, caps)), keep, dev)  # (last [L, hc*dim] f32, body [K, T, hc*dim] f32)
d_onset, d_content = 0.0, 0.0
for (lp, (content, onset)), (last, body) in zip(ref, mine):
    d_onset = max(d_onset, float(np.abs(last.half().numpy().astype(np.float32) - onset.reshape(L, -1).astype(np.float32)).max()))
    cm = body.mean(1).half().numpy().astype(np.float32)               # [K, hc*dim]
    d_content = max(d_content, float(np.abs(cm - content[keep].reshape(len(keep), -1).astype(np.float32)).max()))
res['onset_max_abs_diff'] = d_onset
res['content_max_abs_diff'] = d_content

# 4. group independence: the second sequence alone
alone = W.forward_capture(M, model, [(seqs[1], caps[1])], keep, dev)[0]
res['group_vs_alone'] = max(float((alone[0] - mine[1][0]).abs().max()), float((alone[1] - mine[1][1]).abs().max()))

# 3. decomposition against float64 numpy, on the captured body states of sequences 0 and 2 (common length: the shorter one)
T = min(mine[0][1].shape[1], mine[2][1].shape[1])
x1, x2 = mine[0][1][:, :T], mine[2][1][:, :T]
mF, mL, eF, eL = W.decomp(x1, x2)
Dn = x1.double().numpy() - x2.double().numpy()
pos = np.arange(T) / T
f, la = pos < 1 / 3, pos >= 2 / 3
ref_d = (Dn[:, f].mean(1), Dn[:, la].mean(1), (Dn[:, f] ** 2).sum(-1).mean(1), (Dn[:, la] ** 2).sum(-1).mean(1))
res['decomp_max_rel'] = max(float(np.abs(a.double().numpy() - b).max() / (np.abs(b).max() + 1e-12)) for a, b in zip((mF, mL, eF, eL), ref_d))
res['ok'] = res['onset_max_abs_diff'] == 0.0 and res['content_max_abs_diff'] == 0.0 and res['group_vs_alone'] == 0.0 and res['decomp_max_rel'] < 1e-5
print(json.dumps(res))
sys.exit(0 if res['ok'] else 1)
