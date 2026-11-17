"""Role words in the input embedding and output unembedding of gpt-oss-20b (weights only, CPU), as the token-embedding check
of the paper (section 'Token Embeddings'): R_A, R_U, R_S of the rows of 'assistant', 'user', 'system' in raw and cosine
distance. Run on PRO on 2026-10-21 from the crossmodel folder (inline, this file is the same code); output
runs/analysis/gptoss20b_embedding_roles.json."""
import json
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

M = '/root/hf_models/openai/gpt-oss-20b'
tok = AutoTokenizer.from_pretrained(M)
ids = {r: tok.encode(r, add_special_tokens=False) for r in ('assistant', 'user', 'system')}
print('token ids', ids)
idx = json.load(open(M + '/model.safetensors.index.json'))['weight_map']
names = [n for n in idx if n.endswith('embed_tokens.weight') or n == 'lm_head.weight']
out = {}
for n in names:
    with safe_open(M + '/' + idx[n], 'pt') as f:
        W = f.get_tensor(n).float()
    v = {r: W[i[0]] for r, i in ids.items()}
    a, u, s = v['assistant'], v['user'], v['system']

    def R(dist):
        au, as_, us = dist(a, u), dist(a, s), dist(u, s)
        return {'R_A': float(us / ((au + as_) / 2)), 'R_U': float(as_ / ((au + us) / 2)), 'R_S': float(au / ((as_ + us) / 2))}
    out[n] = {'raw': R(lambda x, y: (x - y).norm()), 'cosine': R(lambda x, y: 1 - torch.nn.functional.cosine_similarity(x, y, 0))}
json.dump({'token_ids': ids, 'R': out}, open('embedding_roles_gptoss20b.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
