"""Gate for the self-space run: the smoke bodies' A / U / S states without prefix (pooled3) must reproduce the archived phase-G
message-content features of the same model (suite/qwen38_27b_bf16_s3/G, the paper's Qwen3.8-27B geometry on server3).
Pass: at every layer, cosine >= 0.999 and relative L2 error <= 0.03 for each role, and cosine >= 0.99 for the own offset
A - (U + S) / 2; every condition and stored layer finite; spans of the same-length pairs identical (checked at render time).
Writes GATE_OK or GATE_FAIL (with the numbers) into the smoke directory.
usage: validate_selfspace_smoke_20261106.py SMOKE_DIR G_DIR"""
import json
import sys
from pathlib import Path
import numpy as np

smoke, gdir = Path(sys.argv[1]), Path(sys.argv[2])
rows, ok = [], True
for j in sorted(smoke.glob('*.json')):
    if j.name == 'identity.json':
        continue
    meta = json.loads(j.read_text(encoding='utf-8'))
    z = np.load(smoke / f'{meta["id"]}.npz')
    g = np.load(gdir / f'{meta["id"]}.npz')
    p3 = z['pooled3'].astype(np.float32)
    ref = np.stack([g[f'content_{r}'].astype(np.float32) for r in ('assistant', 'user', 'system')])
    assert p3.shape == ref.shape, (p3.shape, ref.shape)
    cos = (p3 * ref).sum(-1) / (np.linalg.norm(p3, axis=-1) * np.linalg.norm(ref, axis=-1))
    rel = np.linalg.norm(p3 - ref, axis=-1) / np.linalg.norm(ref, axis=-1)
    off, offr = p3[0] - (p3[1] + p3[2]) / 2, ref[0] - (ref[1] + ref[2]) / 2
    ocos = (off * offr).sum(-1) / (np.linalg.norm(off, axis=-1) * np.linalg.norm(offr, axis=-1))
    fin = bool(np.isfinite(z['pooled']).all() and np.isfinite(z['tokens']).all())
    good = bool(fin and cos.min() >= 0.999 and rel.max() <= 0.03 and ocos.min() >= 0.99)  # plain bool: json cannot write np.bool_
    ok &= good
    rows.append({'id': meta['id'], 'n_body_tokens': meta['n_body_tokens'], 'cos_min': float(cos.min()), 'rel_max': float(rel.max()),
                 'offset_cos_min': float(ocos.min()), 'finite': fin, 'pass': good})
ok &= len(rows) > 0
res = {'pass': bool(ok), 'bodies': rows}
(smoke / ('GATE_OK' if ok else 'GATE_FAIL')).write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8')
print(json.dumps(res, indent=1))
sys.exit(0 if ok else 1)
