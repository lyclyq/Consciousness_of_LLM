"""Distance ratio of the three role words in the model's input embedding and output unembedding (weights only; no input).
Role-word token IDs: assistant 74455, user 846, system 8678. R_A = D(U,S) / mean(D(A,U), D(A,S)), raw Euclidean and cosine.
usage: role_word_embeddings.py MODEL_DIR OUT_JSON"""
import json
import sys
from pathlib import Path
import numpy as np
from safetensors import safe_open

IDS = {'A': 74455, 'U': 846, 'S': 8678}


def rows(model_dir, wanted):
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
    def d(a, b):
        if cosine:
            return 1 - float(v[a] @ v[b] / (np.linalg.norm(v[a]) * np.linalg.norm(v[b])))
        return float(np.linalg.norm(v[a] - v[b]))
    return d('U', 'S') / ((d('A', 'U') + d('A', 'S')) / 2)


def main(model_dir, out_json):
    res = {}
    for key, (name, v) in rows(model_dir, ['embed_tokens.weight', 'lm_head.weight']).items():
        res[key] = {'tensor': name, 'R_A_raw': ratio(v, False), 'R_A_cos': ratio(v, True),
                    'D_raw': {p: dist(v, p[0], p[1], False) for p in ('AU', 'AS', 'US')},
                    'D_cos': {p: dist(v, p[0], p[1], True) for p in ('AU', 'AS', 'US')}}
        print(key, res[key])
    json.dump(res, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
