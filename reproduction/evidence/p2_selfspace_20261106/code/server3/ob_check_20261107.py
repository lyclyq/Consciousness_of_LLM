"""Diagnostic for output_breadth: per condition, median over held-out body tokens of p(<|im_end|>), top-1 prob, entropy, and
log-prob of the actual next body token. Reads stored L63 states only."""
import json, sys
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
run, M = Path(sys.argv[1]), Path(sys.argv[2])
ident = json.loads((run / 'identity.json').read_text())
names = [c['name'] for c in ident['conditions']]
li = list(ident['keep_layers']).index(63)
wm = json.loads((M / 'model.safetensors.index.json').read_text())['weight_map']
def get(k):
    with safe_open(str(M / wm[k]), framework='pt', device='cpu') as f:
        return f.get_tensor(k)
dev = torch.device('cuda:0')
wn = get('model.language_model.norm.weight').float().to(dev); W = get('lm_head.weight').to(dev).float()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(M))
iend = tok.convert_tokens_to_ids('<|im_end|>')
eps = 1e-6
want = ['A', 'U', 'S', 'A_switch', 'U_hospital', 'U_school']
acc = {n: {'p_end': [], 'top1': [], 'ent': [], 'lp_next': []} for n in want}
for p in sorted(run.glob('*.json')):
    if p.name == 'identity.json': continue
    m = json.loads(p.read_text())
    if m['split'] != 'confirmation': continue
    h = torch.from_numpy(np.ascontiguousarray(np.load(run / f"{m['id']}.npz")['tokens'][:, li])).to(dev).float()
    hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + wn)
    nxt = torch.tensor(m['body_ids'][1:], device=dev)
    for n in want:
        lp = torch.log_softmax(hn[names.index(n)] @ W.T, -1)
        pr = lp.exp()
        acc[n]['p_end'].append(float(pr[:-1, iend].median()))
        acc[n]['top1'].append(float(pr[:-1].max(-1).values.median()))
        acc[n]['ent'].append(float((-(pr * lp).sum(-1))[:-1].median()))
        acc[n]['lp_next'].append(float(lp[torch.arange(len(nxt)), nxt].mean()))
print(json.dumps({n: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for n, d in acc.items()}, indent=1))
