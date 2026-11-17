"""Overlap of the Chinese and the English single-pass subspaces of Qwen3.8-27B (fits of crossmodel_singlepass_ablate.py e848ae5c on
PRO, 11-02: singlepass_ablate/qwen38_27b_{zh,en}/fit.pt): per layer, ||V_zh^T V_en||_F^2 / k for the first k columns (k 8, the
edit of the causal tests, and 64), against the expectation k / d for two independent random k-dim subspaces; the same for the
stored random subspaces R (same seed per layer in both fits, so their overlap is 1 by construction and is printed as a check).
Runs where torch is installed (PRO venv); writes OUT_JSON.
usage: python xlang_overlap_20261102.py ZH_FIT EN_FIT OUT_JSON"""
import json
import statistics
import sys

import torch

zh, en = (torch.load(p, map_location='cpu', weights_only=False) for p in sys.argv[1:3])
out = {'zh_fit': sys.argv[1], 'en_fit': sys.argv[2]}
def keys(V):
    return sorted(V) if isinstance(V, dict) else list(range(len(V)))


layers = keys(zh['V'])
assert layers == keys(en['V'])
d = zh['V'][layers[0]].shape[0]
for k in (8, 64):
    ov = []
    for l in layers:
        a, b = zh['V'][l][:, :k].double(), en['V'][l][:, :k].double()
        assert torch.allclose(a.T @ a, torch.eye(k, dtype=a.dtype), atol=1e-4), 'basis not orthonormal'
        ov.append(float((a.T @ b).pow(2).sum() / k))
    rr = [float((zh['R'][l][:, :k].double().T @ en['R'][l][:, :k].double()).pow(2).sum() / k) for l in layers]
    third = len(layers) // 3
    out[f'k{k}'] = {'by_layer': [round(x, 4) for x in ov], 'median': round(statistics.median(ov), 4), 'min': round(min(ov), 4),
                    'max': round(max(ov), 4), 'min_layer': layers[ov.index(min(ov))], 'max_layer': layers[ov.index(max(ov))],
                    'median_by_third': [round(statistics.median(ov[i * third:(i + 1) * third if i < 2 else None]), 4) for i in range(3)],
                    'random_expected': round(k / d, 5), 'stored_random_overlap_median': round(statistics.median(rr), 4)}
out['hidden'] = d
out['n_layers'] = len(layers)
open(sys.argv[3], 'w', encoding='utf-8').write(json.dumps(out, indent=1))
print(json.dumps({k: {kk: v for kk, v in x.items() if kk != 'by_layer'} for k, x in out.items() if k.startswith('k')}, indent=1))
