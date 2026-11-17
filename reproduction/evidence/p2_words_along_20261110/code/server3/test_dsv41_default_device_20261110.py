"""CPU check (11-10) for the smoke's device error. run() sets the default device to cuda after building the model, so a factory
call without a device lands there while forward_capture's outputs are on the host. decomp and cmp must give the same result
whatever the default device is. The 'meta' default device stands in for cuda on a CPU-only run: before the fix both functions
raise a device error here, after it they return exactly the values computed with the default device on the host.
usage: CUDA_VISIBLE_DEVICES= test_dsv41_default_device_20261110.py OFFICIAL_INFERENCE_DIR"""
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
import model as M  # noqa: E402,F401
import dsv41_words_along_20261110 as W  # noqa: E402

g = torch.Generator().manual_seed(0)
x1, x2 = torch.randn(3, 31, 16, generator=g), torch.randn(3, 31, 16, generator=g)
ours = torch.randn(5, 16, generator=g)
ref = np.random.default_rng(0).standard_normal((5, 16)).astype(np.float32)
base_d, base_c = W.decomp(x1, x2), W.cmp(ours, ref)
res = {}
torch.set_default_device('meta')
try:
    d, c = W.decomp(x1, x2), W.cmp(ours, ref)
    res['decomp_equal'] = all(a.device.type == 'cpu' and torch.equal(a, b) for a, b in zip(d, base_d))
    res['cmp'], res['cmp_host_default'] = c, base_c
    res['ok'] = res['decomp_equal'] and c == base_c
except RuntimeError as e:
    res['error'], res['ok'] = str(e)[:200], False
finally:
    torch.set_default_device('cpu')
print(json.dumps(res))
sys.exit(0 if res['ok'] else 1)
