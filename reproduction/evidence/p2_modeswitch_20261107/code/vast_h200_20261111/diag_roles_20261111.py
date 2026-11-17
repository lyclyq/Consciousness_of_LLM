"""Diagnostic (11-11): role renderings of the first held-out body only, against server3. Last-token states at the kept layers vs
stage 1 (role_state) and role KLs vs the paper's ctrl run, under the given placement (--max-memory-gib, --cpu-offload-gib).
usage: diag_roles_20261111.py --model DIR --max-memory-gib G --cpu-offload-gib G --tag NAME"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402
from crossmodel_suite import find_layers, load  # noqa: E402
from persistence_readout_20261106 import KEEP_FRACS, ROLES, natural_system_ok  # noqa: E402
from mode_switch_20261107 import PAIRS, REQ, renderings  # noqa: E402
from words_along_20261110 import plan, load_words  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--max-memory-gib', type=float, required=True)
ap.add_argument('--cpu-offload-gib', type=float, required=True)
ap.add_argument('--tag', required=True)
a = ap.parse_args()
R0 = Path('/root/glm53/ref')
allb = [json.loads(l) for l in open(R0 / 'role_bodies.jsonl', encoding='utf-8') if l.strip()]
b = [x for x in allb if x['split'] == 'confirmation'][0]
import torch  # noqa: E402
t0 = time.time()
torch, tok, model, cfg, placement, cls = load(argparse.Namespace(model=a.model, max_memory_gib=a.max_memory_gib, cpu_offload_gib=a.cpu_offload_gib))
import collections  # noqa: E402
print('load', round(time.time() - t0), 's placement', dict(collections.Counter(str(v) for v in placement.values())), flush=True)
_, layers = find_layers(model)
L = len(layers)
keep = sorted({round(f * (L - 1)) for f in KEEP_FRACS})
cap = {}


def hook(i):
    def h(mod, inp, outp):
        x = outp[0] if isinstance(outp, tuple) else outp
        if x.dim() == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        cap[i] = x[0, -1].detach().float()
    return h


for l in keep:
    layers[l].register_forward_hook(hook(l))
dev0 = model.get_input_embeddings().weight.device
kw = {}
q = cs.Q_REPLY
_, R, _, spans = plan(tok, kw, b['body'], 'main', q, None, True, [], True)
Z = np.load(R0 / 'words_along/b150_main/glm53_flash' / f'{b["id"]}.npz')
out = {'tag': a.tag, 'placement': dict(collections.Counter(str(v) for v in placement.values())), 'states': {}, 'kl': {}}
for ri, (rn, _) in enumerate(ROLES):
    ids = R[rn][0]
    x = torch.tensor([ids], device=dev0)
    t = time.time()
    with torch.no_grad():
        model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
    ours = torch.stack([cap[l].to(dev0) for l in keep])
    ref = torch.tensor(Z['role_state'][ri].astype(np.float32), device=dev0)
    cosv = torch.nn.functional.cosine_similarity(ours, ref, dim=-1)
    rel = (ours - ref).norm(dim=-1) / ref.norm(dim=-1)
    out['states'][rn] = {'cos_per_layer': [round(float(v), 5) for v in cosv], 'rel_per_layer': [round(float(v), 4) for v in rel],
                         'seconds': round(time.time() - t, 1)}
# role KLs on the paper's KL renderings
sys_ok = natural_system_ok(tok, kw, q, REQ['zh'], allb[0]['body'])
Rk = renderings(tok, kw, b['body'], 'zh', sys_ok)['main']
ctrl_ids = sorted(set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}).keys()))
lps = {}
for rn, v in Rk.items():
    x = torch.tensor([v['ids']], device=dev0)
    with torch.no_grad():
        lg = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=0).logits[0].float()
    mask = torch.zeros(lg.shape[-1], dtype=torch.bool, device=lg.device)
    mask[[i for i in ctrl_ids if i < lg.shape[-1]]] = True
    mask[len(tok):] = True
    lp = torch.log_softmax(lg[torch.tensor(v['common'][:-1], device=lg.device)], -1)
    lc = lp[:, ~mask]
    lps[rn] = (lp, lc - torch.logsumexp(lc, -1, keepdim=True))
ref = json.loads((R0 / 'modeswitch/ctrl/glm53_flash/bodies' / f'{b["id"]}.json').read_text(encoding='utf-8'))['pairs']
for X, Y in PAIRS:
    kb = float((lps[X][0].exp() * (lps[X][0] - lps[Y][0])).sum(-1).median())
    kc = float((lps[X][1].exp() * (lps[X][1] - lps[Y][1])).sum(-1).median())
    out['kl'][f'{X}-{Y}'] = {'ours': [round(kb, 5), round(kc, 5)], 'ctrl': [round(ref[f'{X}-{Y}']['kl_body'], 5), round(ref[f'{X}-{Y}']['kl_content'], 5)]}
print('DIAG', json.dumps(out), flush=True)
Path(f'/root/glm53/out/diag_{a.tag}.json').write_text(json.dumps(out, indent=1))
