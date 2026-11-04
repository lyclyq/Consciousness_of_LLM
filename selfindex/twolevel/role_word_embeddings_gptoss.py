"""Role words in the input embedding and output unembedding of gpt-oss-20b (weights only, CPU): R_A, R_U and R_S of the rows of
'assistant', 'user' and 'system', in raw (Euclidean) and cosine distance (the token-embedding check of Section 5.4).

The token ids come from the checkpoint's tokenizer (the first token of each word); the tensors are every tensor whose name ends
in embed_tokens.weight, and lm_head.weight. R_X = D(other two) / mean(D(X, other)).
usage: python -m selfindex.twolevel.role_word_embeddings_gptoss --model DIR --out JSON
"""
from __future__ import annotations

import argparse
import json
import os


def run(args):
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = {r: tok.encode(r, add_special_tokens=False) for r in ('assistant', 'user', 'system')}
    print('token ids', ids)
    idx = json.load(open(os.path.join(args.model, 'model.safetensors.index.json')))['weight_map']
    names = [n for n in idx if n.endswith('embed_tokens.weight') or n == 'lm_head.weight']
    out = {}
    for n in names:
        with safe_open(os.path.join(args.model, idx[n]), 'pt') as f:
            W = f.get_tensor(n).float()
        v = {r: W[i[0]] for r, i in ids.items()}
        a, u, s = v['assistant'], v['user'], v['system']

        def R(dist):
            au, as_, us = dist(a, u), dist(a, s), dist(u, s)
            return {'R_A': float(us / ((au + as_) / 2)), 'R_U': float(as_ / ((au + us) / 2)), 'R_S': float(au / ((as_ + us) / 2))}
        out[n] = {'raw': R(lambda x, y: (x - y).norm()), 'cosine': R(lambda x, y: 1 - torch.nn.functional.cosine_similarity(x, y, 0))}
    with open(args.out, 'w') as f:
        json.dump({'token_ids': ids, 'R': out}, f, indent=1)
    print(json.dumps(out, indent=1))
    return out


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', required=True, help='checkpoint folder with the tokenizer and model.safetensors.index.json')
    p.add_argument('--out', required=True, help='output JSON')
    return p


def main(argv=None):
    run(parser().parse_args(argv))


if __name__ == '__main__':
    main()
