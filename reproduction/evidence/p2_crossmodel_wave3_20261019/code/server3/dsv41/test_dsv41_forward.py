"""Small random-weight model built from the official model.py (ModelArgs defaults: 5 layers, compressed and windowed attention,
indexer, hyper-connections, FP8 linears, FP4 experts), run with dsv41_kernel_torch. Checks that
  1. dsv41_natural3.forward_batch reproduces the official Transformer.forward on one sequence (log-probabilities at the last position);
  2. the host-memory expert path (patch_runtime) gives the same result as the official MoE.forward;
  3. the runner's grouped path (forward_seqs: one sequence at a time, layer by layer, staged experts) equals each sequence run
     alone, bit for bit (log-probabilities and captured states).
usage: test_dsv41_forward.py OFFICIAL_INFERENCE_DIR [cpu|cuda]"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dsv41_kernel_torch as K  # noqa: E402

sys.modules['kernel'] = K
sys.path.insert(0, sys.argv[1])
import model as M  # noqa: E402
import dsv41_natural3 as R  # noqa: E402

dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else 'cpu')
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
torch.set_default_device(dev)  # as generate.py: get_window_topk_idxs creates its tensors on the default device

seqs = [torch.randint(0, 129280, (n,), generator=g, device='cpu').tolist() for n in (150, 211, 97)]
pad = 1
res = {}

# 1. official forward vs forward_batch on single sequences (original MoE path)
lp_off, lp_mine = [], []
for s in seqs:
    _, logits, _ = model(torch.tensor([s], device=dev))
    lp_off.append(torch.log_softmax(logits.float(), -1).cpu()[0])
    lp, _ = R.forward_batch(M, model, [s], pad, [None])
    lp_mine.append(lp[0])
d1 = max(float((a - b).abs().max()) for a, b in zip(lp_off, lp_mine))
res['forward_batch_vs_official'] = d1

# 2. host-memory expert path
stats = {'expert_s': 0.0}
for layer in model.layers:  # move the routed experts to host memory as in the real run
    for e in layer.ffn.experts:
        for k in ('w1', 'w2', 'w3'):
            lin = getattr(e, k)
            lin._parameters['weight'] = torch.nn.Parameter(R.to_dev(lin.weight.data, 'cpu'), requires_grad=False)
            lin._parameters['scale'] = torch.nn.Parameter(R.to_dev(lin.scale.data, 'cpu'), requires_grad=False)
            lin.weight.scale = lin.scale
R.patch_runtime(M, dev, stats)
lp_host = [R.forward_batch(M, model, [s], pad, [None])[0][0] for s in seqs]
d2 = max(float((a - b).abs().max()) for a, b in zip(lp_off, lp_host))
res['host_experts_vs_official'] = d2

# 3. the runner's path: every sequence alone, layer by layer over the group, routed experts staged (dequantized once per layer),
#    attention state handed down per sequence; it must equal each sequence run on its own, bit for bit
caps = [list(range(10, 60)), None, list(range(5, 90))]
grouped = R.forward_seqs(M, model, seqs, caps, dev, stage=True)
d3 = max(float((grouped[i][0] - lp_host[i]).abs().max()) for i in range(3))
st = []
for i in (0, 2):
    _, c1 = R.forward_batch(M, model, [seqs[i]], pad, [caps[i]])
    st.append(max(float(abs(c1[0][k].astype('float32') - grouped[i][1][k].astype('float32')).max()) for k in (0, 1)))
res['grouped_vs_single_logprob'] = d3
res['grouped_vs_single_states'] = max(st)
res['logprob_scale'] = float(lp_off[0].abs().max())
res['ok'] = d1 == 0.0 and d2 == 0.0 and d3 == 0.0 and res['grouped_vs_single_states'] == 0.0
print(json.dumps(res))
sys.exit(0 if res['ok'] else 1)
