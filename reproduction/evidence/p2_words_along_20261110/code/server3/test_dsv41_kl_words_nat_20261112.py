"""CPU check of dsv41_kl_words_nat_20261112.py (11-12) on a small random-weight model built from the official model.py, as
test_dsv41_words_along_20261110.py (ModelArgs defaults, 5 layers, compressed and windowed attention, indexer, hyper-connections,
FP8 linears, FP4 experts, dsv41_kernel_torch, routed experts in host memory), with the real tokenizer and encoder for the
renderings (the first held-out body, the roles and the first 3 word pairs). Checks:
  1. the word renderings equal those of the words-along DeepSeek run (dsv41_words_along_20261110.plan), token for token
  2. the last row of forward_all's log-probabilities equals the paper's runner (dsv41_natural3.forward_seqs) for a role and a word
     sequence (the KL port's own --check and criterion of its smoke), maximum absolute difference <= 1e-3 and the same argmax
  3. a sequence run alone equals the same sequence inside the group, bit for bit
  4. body_record: the role KLs equal the KL port's body_record; the word KLs (full and content, both directions) equal a float64
     numpy computation from the same log-probabilities (relative difference < 1e-4); the check against a reference holding the
     KL port's role KLs gives max_rel 0 and ok
  5. summarize on 75 synthetic records with the held-out families writes to_US, to_AU and the word medians
usage: CUDA_VISIBLE_DEVICES= test_dsv41_kl_words_nat_20261112.py OFFICIAL_INFERENCE_DIR MODEL_DIR TMPDIR"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import dsv41_kernel_torch as K  # noqa: E402

sys.modules['kernel'] = K
sys.path.insert(0, sys.argv[1])
import model as M  # noqa: E402
import dsv41_natural3 as R  # noqa: E402
import dsv41_modeswitch_nat3_20261108 as KN  # noqa: E402
import dsv41_words_along_20261110 as WA  # noqa: E402
import dsv41_kl_words_nat_20261112 as X  # noqa: E402

MD, TMP = sys.argv[2], Path(sys.argv[3])
dev = torch.device('cpu')
torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
args = M.ModelArgs(max_batch_size=4, max_seq_len=512)
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

E, tok = KN.encoder_and_tok(MD)
Rd = R.Renderer(E, tok, False)
W, pairs = WA.load_words(str(HERE.parent / 'words_v2.json'))
pairs3 = pairs[:3]
a = type('A', (), {'bodies': '/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl'})()
b = X.held_out(a)[0]
res = {}

# 1. renderings
seqs, poss, meta = X.plan(Rd, b, pairs3)
wa_seqs, _, _ = WA.plan(Rd, b['body'], pairs3, check_runner=True)
res['words_equal_words_along'] = seqs[3:] == [s[0] for s in wa_seqs[3:]]
res['n_seqs'] = len(seqs)
res['max_tokens'] = max(len(s) for s in seqs)
assert res['max_tokens'] + 2 <= args.max_seq_len

# 2. forward_all against the paper's runner (last position)
lps = KN.forward_all(M, model, seqs, poss, dev)
d2, am = 0.0, True
for k in (0, 3):
    ref = R.forward_seqs(M, model, [seqs[k]], [None], dev)[0][0]
    d2 = max(d2, float((ref.float() - lps[k][-1].float()).abs().max()))
    am &= int(ref.argmax()) == int(lps[k][-1].argmax())
res['last_logprob_max_abs_diff'] = d2
res['same_argmax'] = am

# 3. alone vs in the group
alone = KN.forward_all(M, model, [seqs[4]], [poss[4]], dev)[0]
res['alone_vs_group'] = float((alone - lps[4]).abs().max())

# 4. body_record
ctrl = KN.control_ids(tok)
V = args.vocab_size
mask = torch.zeros(V, dtype=torch.bool, device='cpu')
mask[torch.tensor([i for i in ctrl if i < V], dtype=torch.long, device='cpu')] = True
mask[len(tok):] = True
kr = KN.body_record(b, meta['roles'], {rn: lps[k] for k, (rn, _) in enumerate(KN.ROLES)}, mask)
refd = TMP / 'ref'
(refd / 'bodies').mkdir(parents=True, exist_ok=True)
R.atomic(refd / 'bodies' / f'{b["id"]}.json', {'id': b['id'], 'pairs': kr['pairs']})
rec = X.body_record(b, meta, lps, mask, refd)
res['roles_equal_kl_port'] = rec['roles'] == kr['pairs']
res['check'] = rec['check']


def kl64(x, y):
    x, y = x.double().numpy(), y.double().numpy()
    return float(np.median((np.exp(x) * (x - y)).sum(-1)))


def cont64(lp):
    lc = lp[:, ~mask].double()
    return lc - torch.logsumexp(lc, -1, keepdim=True)


rmax = 0.0
for pid, m in meta['pairs'].items():
    l1, l2 = lps[m['i']][:-1], lps[m['i'] + 1][:-1]
    c1, c2 = cont64(l1), cont64(l2)
    want = {'kl_body_12': kl64(l1, l2), 'kl_body_21': kl64(l2, l1), 'kl_content_12': kl64(c1, c2), 'kl_content_21': kl64(c2, c1)}
    for k, v in want.items():
        rmax = max(rmax, abs(rec['words'][pid][k] - v) / max(abs(v), 1e-12))
    assert rec['words'][pid]['n_positions'] == len(poss[m['i']])
res['word_kl_max_rel_vs_float64'] = rmax

# 5. summarize on synthetic records
rng = np.random.default_rng(0)
out = TMP / 'summ'
(out / 'bodies').mkdir(parents=True, exist_ok=True)
ids = [p['id'] for p in pairs]
for bb in X.held_out(a):
    r_ = {'id': bb['id'], 'category': bb['category'], 'roles': {pn: {'kl_body': float(rng.uniform(.1, 1)), 'kl_content': float(rng.uniform(.1, 1))}
                                                               for pn in ('A-U', 'A-S', 'U-S')},
          'words': {pid: {f'{k}_{d}': float(rng.uniform(0, .1)) for k in ('kl_body', 'kl_content') for d in ('12', '21')} for pid in ids},
          'check': {'max_rel': 0.0}}
    R.atomic(out / 'bodies' / f'{bb["id"]}.json', r_)
X.summarize(out, ids)
s = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
c = s['kinds']['kl_content']
res['summary_ok'] = s['n_bodies'] == 75 and s['families'] == 15 and 'to_US' in c and 'to_AU' in c and 'word_median_to_AU_12' in c

res['ok'] = (res['words_equal_words_along'] and res['last_logprob_max_abs_diff'] <= 1e-3 and res['same_argmax']
             and res['alone_vs_group'] == 0.0 and res['roles_equal_kl_port'] and res['check']['ok'] and res['check']['max_rel'] == 0.0
             and res['word_kl_max_rel_vs_float64'] < 1e-4 and res['summary_ok'])
print(json.dumps(res))
sys.exit(0 if res['ok'] else 1)
