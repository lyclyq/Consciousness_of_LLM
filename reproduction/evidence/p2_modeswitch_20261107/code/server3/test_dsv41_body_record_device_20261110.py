"""CPU test of body_record in dsv41_modeswitch_nat3_20261108.py (11-10 fix, 46e977c4): with the default device set to a non-CPU
device (cuda; meta does not reproduce the failure), body_record and the control-token mask must run on CPU log-probabilities and give the same
record as with the default CPU device; the old version 6917870c must fail the same way the smoke did. No model is loaded.
usage: CUDA_VISIBLE_DEVICES=0 python test_dsv41_body_record_device_20261110.py NEW.py OLD.py"""
import importlib.machinery
import importlib.util
import sys
import types

import torch

DEV = 'cuda'


def load(path, name):
    sys.modules.setdefault('dsv41_natural3', types.ModuleType('dsv41_natural3'))  # the runner is not needed for body_record
    loader = importlib.machinery.SourceFileLoader(name, path)  # also for the kept old copy *.py.v6917870c
    spec = importlib.util.spec_from_loader(name, loader)
    m = importlib.util.module_from_spec(spec)
    m.__file__ = path
    loader.exec_module(m)
    return m


def make_inputs(V=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    R, lps = {}, {}
    for k, rn in enumerate(('A', 'U', 'S')):
        ids = list(range(5 + k, 25 + k))
        common = list(range(3, 15))
        R[rn] = {'ids': ids, 'common': common, 'close': 7}
        lps[rn] = torch.log_softmax(torch.randn(len(common), V, generator=g), -1)  # [:-1] in body_record -> len(common)-1 rows
    return R, lps


def mask_of(V, ctrl):
    mask = torch.zeros(V, dtype=torch.bool, device='cpu')
    mask[torch.tensor([i for i in ctrl if i < V], dtype=torch.long, device='cpu')] = True
    return mask


def main(new, old):
    V, ctrl = 50, [0, 1, 2, 7, 49, 60]
    b = {'id': 'x', 'split': 'held', 'category': 'c'}
    N = load(new, 'new_mod')
    O = load(old, 'old_mod')
    R, lps = make_inputs(V)
    ref = N.body_record(b, R, lps, mask_of(V, ctrl))
    torch.set_default_device(DEV)
    try:
        got = N.body_record(b, R, lps, mask_of(V, ctrl))
    finally:
        torch.set_default_device('cpu')
    assert got == ref, (got, ref)
    print('new: same record with default device cuda and cpu:', ref['pairs'])
    torch.set_default_device(DEV)
    try:
        O.body_record(b, R, lps, mask_of(V, ctrl))
        print('old: no error (unexpected)')
        bad = True
    except RuntimeError as e:
        print('old: fails as the smoke did:', str(e)[:120])
        bad = False
    finally:
        torch.set_default_device('cpu')
    assert not bad
    print('TEST PASS')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
