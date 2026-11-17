"""Diagnostic for the smoke gate (11-06): does batching with right padding change the late-layer states of Qwen3.8-27B?
For the given bodies, the A / U / S states (mean over the body) are computed three ways and compared with the archived phase-G
features (single sequence per forward, attention mask of ones):
  single   one sequence per forward, attention mask of ones (as phase G)
  padded   all 18 conditions in one forward, right padding (as the smoke run)
  grouped  conditions of equal length in one forward, no padding
Prints, per way and role, the largest relative L2 error over layers and the layers above 3%.
usage: diag_batching_20261106.py --model DIR --bodies role_bodies.jsonl --ids ID [ID ...] --gdir suite/.../G"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossmodel_suite import find_layers, load  # noqa: E402
from selfspace_tokens_20261106 import COND, render_body  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--bodies', required=True)
ap.add_argument('--ids', nargs='+', required=True)
ap.add_argument('--gdir', required=True)
ap.add_argument('--max-memory-gib', type=int, default=44)
args = ap.parse_args()
bodies = {b['id']: b for b in (json.loads(l) for l in open(args.bodies, encoding='utf-8') if l.strip())}
torch, tok, model, cfg, placement, cls = load(argparse.Namespace(model=args.model, max_memory_gib=args.max_memory_gib))
_, layers = find_layers(model)
L = len(layers)
cap = {}
for i in range(L):
    layers[i].register_forward_hook(lambda m, inp, out, i=i: cap.__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach()))
dev0 = model.get_input_embeddings().weight.device
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
names = [c[0] for c in COND]


def run(seqs, padded):
    T = max(len(s) for s in seqs)
    x = torch.tensor([s + [pad] * (T - len(s)) for s in seqs], dtype=torch.long, device=dev0)
    am = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs], dtype=torch.long, device=dev0)
    cap.clear()
    with torch.no_grad():
        model(input_ids=x, attention_mask=am, use_cache=False, return_dict=True, logits_to_keep=1)
    return {i: cap[i].float() for i in range(L)}


for bid in args.ids:
    ids, span, span3 = render_body(tok, bodies[bid])
    g = np.load(Path(args.gdir) / f'{bid}.npz')
    ref = {r: g[f'content_{full}'].astype(np.float32) for r, full in (('A', 'assistant'), ('U', 'user'), ('S', 'system'))}
    ways = {}
    ways['single'] = {}
    for r in ('A', 'U', 'S'):
        h = run([ids[r]], False)
        ways['single'][r] = np.stack([h[i][0, span3[r]].mean(0).cpu().numpy() for i in range(L)])
    h = run([ids[n] for n in names], True)
    ways['padded'] = {r: np.stack([h[i][names.index(r), span3[r]].mean(0).cpu().numpy() for i in range(L)]) for r in ('A', 'U', 'S')}
    ways['grouped'] = {}
    for r in ('A', 'U', 'S'):
        same = [n for n in names if len(ids[n]) == len(ids[r])]
        h = run([ids[n] for n in same], False)
        ways['grouped'][r] = np.stack([h[i][same.index(r), span3[r]].mean(0).cpu().numpy() for i in range(L)])
    for w, d in ways.items():
        msg = []
        for r in ('A', 'U', 'S'):
            rel = np.linalg.norm(d[r] - ref[r], axis=-1) / np.linalg.norm(ref[r], axis=-1)
            msg.append(f'{r} max {rel.max():.4f} @L{int(rel.argmax())} >3%: {np.where(rel > 0.03)[0].tolist()}')
        print(bid, f'{w:8s}', ' | '.join(msg), flush=True)
    sp = ways['single']['U']
    rel = np.linalg.norm(ways['padded']['U'] - sp, axis=-1) / np.linalg.norm(sp, axis=-1)
    print(bid, 'padded vs single (U): max', round(float(rel.max()), 4), '@L', int(rel.argmax()), flush=True)
print('DIAG DONE', flush=True)
