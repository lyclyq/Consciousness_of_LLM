"""Rewrite fit files written by the first version of crossmodel_role_swap_r05.py --make-fits (10-30), whose metadata held a
TorchVersion object that torch >= 2.6 refuses under weights_only=True: the versions become plain strings; tensors are unchanged
(checked). Our own files, so they are loaded once with weights_only=False.
usage: r05r06_fix_fit_meta.py FIT_DIR"""
import sys
from pathlib import Path

import torch

for f in sorted(Path(sys.argv[1]).glob('fit_*.pt')):
    F = torch.load(f, weights_only=False)
    before = {k: (v[0].clone() if isinstance(v, dict) and 0 in v else None) for k, v in F.items() if k in ('V', 'R')}
    F['torch'], F['transformers'] = str(F['torch']), str(F['transformers'])
    torch.save(F, f)
    G = torch.load(f)  # the default (weights_only=True) must now succeed
    assert all(torch.equal(G[k][0], v) for k, v in before.items() if v is not None)
    print(f.name, 'rewritten; loads with weights_only=True')
