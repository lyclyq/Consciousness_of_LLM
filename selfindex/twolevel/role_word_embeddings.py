"""Distance ratio of the three role words in the input embedding and the output unembedding of Qwen3.8-27B (weights only; no input).

Role-word token ids of the Qwen3.8 tokenizer: assistant 74455, user 846, system 8678. R_A = D(U,S) / mean(D(A,U), D(A,S)), with D
the raw Euclidean and the cosine distance; also every pairwise distance. The rows are read from the checkpoint's safetensors
shards (the shortest tensor name ending in embed_tokens.weight or lm_head.weight).
usage: python -m selfindex.twolevel.role_word_embeddings --model DIR --out JSON
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IDS = {'A': 74455, 'U': 846, 'S': 8678}


def rows(model_dir, wanted):
    from safetensors import safe_open
    idx = json.load(open(Path(model_dir) / 'model.safetensors.index.json'))['weight_map']
    out = {}
    for key in wanted:
        names = [k for k in idx if k.endswith(key)]
        assert names, key
        name = sorted(names, key=len)[0]
        with safe_open(str(Path(model_dir) / idx[name]), framework='pt') as f:
            t = f.get_slice(name)
            out[key] = (name, {r: t[i:i + 1].float().numpy()[0] for r, i in IDS.items()})
    return out


def dist(v, a, b, cosine):
    if cosine:
        return 1 - float(v[a] @ v[b] / (np.linalg.norm(v[a]) * np.linalg.norm(v[b])))
    return float(np.linalg.norm(v[a] - v[b]))


def ratio(v, cosine):
    return dist(v, 'U', 'S', cosine) / ((dist(v, 'A', 'U', cosine) + dist(v, 'A', 'S', cosine)) / 2)


def run(args):
    res = {}
    for key, (name, v) in rows(args.model, ['embed_tokens.weight', 'lm_head.weight']).items():
        res[key] = {'tensor': name, 'R_A_raw': ratio(v, False), 'R_A_cos': ratio(v, True),
                    'D_raw': {p: dist(v, p[0], p[1], False) for p in ('AU', 'AS', 'US')},
                    'D_cos': {p: dist(v, p[0], p[1], True) for p in ('AU', 'AS', 'US')}}
        print(key, res[key])
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    return res


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', required=True, help='checkpoint folder with model.safetensors.index.json')
    p.add_argument('--out', required=True, help='output JSON')
    return p


def main(argv=None):
    run(parser().parse_args(argv))


if __name__ == '__main__':
    main()
