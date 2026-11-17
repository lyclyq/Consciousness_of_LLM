"""Compact, exact summary of one run directory for the local analysis (11-10, ../../PROTOCOL_zh.md), numpy only, no GPU.
Works on the 11-06 pilot runs (persistence_readout_20261106.py: dec rows A-U, A-S, U-S, then the pilot words in the order of
its WORDS when the run is not lean and in the main order) and on the runs of words_along_20261110.py (rows from identity.json).
For the held-out bodies (split == confirmation) and every row r:
  family means MF, ML of the first- and last-third mean difference vectors (K, d), float64, family = the body's category
  (15 wording families for the 150-body set; every phase-3 body is its own family);
  Gram matrices G_FF[r] = <MF_f, MF_g>, G_LL, G_FL per kept layer (R, K, F, F): the family U-statistic of any resample of
  families is a function of these (analyze_persistence_20261106.ustats);
  per-body energies eF, eL (N, R, K).
Writes OUT.npz and OUT.json (rows, ids, families, counts, kept layers, identity excerpt, sha256 of every body npz read).
usage: extract_gram_20261110.py RUN_DIR OUT [--split confirmation]"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PILOT_WORD_IDS = ['place1', 'time1', 'name1', 'feeling1']   # persistence_readout_20261106.WORDS order: place, time, name, feeling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('out')
    ap.add_argument('--split', default='confirmation')
    a = ap.parse_args()
    run = Path(a.run)
    ident = json.loads((run / 'identity.json').read_text(encoding='utf-8'))
    metas = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(run.glob('*.json')) if p.name not in ('identity.json', 'gate.json')]
    held = [m for m in metas if m.get('split') == a.split]
    assert held, 'no held-out body'
    first = np.load(run / f'{held[0]["id"]}.npz')
    Q = first['dec_mF'].shape[0]
    if 'rows' in ident:
        rows = ident['rows']
        source = 'words_along'
    else:
        rows = ['A-U', 'A-S', 'U-S'] + (PILOT_WORD_IDS if Q == 7 else [])
        source = 'pilot'
    assert len(rows) == Q, (rows, Q)
    fams = sorted({m['category'] for m in held})
    fi = {f: i for i, f in enumerate(fams)}
    F, K = len(fams), first['dec_mF'].shape[1]
    d = first['dec_mF'].shape[2]
    SF = np.zeros((Q, F, K, d))
    SL = np.zeros((Q, F, K, d))
    cnt = np.zeros(F, int)
    eF = np.zeros((len(held), Q, K))
    eL = np.zeros((len(held), Q, K))
    sha = {}
    for j, m in enumerate(held):
        p = run / f'{m["id"]}.npz'
        sha[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p) as z:
            f = fi[m['category']]
            SF[:, f] += z['dec_mF'].astype(np.float64)
            SL[:, f] += z['dec_mL'].astype(np.float64)
            eF[j], eL[j] = z['dec_eF'], z['dec_eL']
        cnt[f] += 1
    MF = SF / cnt[None, :, None, None]
    ML = SL / cnt[None, :, None, None]
    gdt = np.float64 if F <= 50 else np.float32
    G = {k: np.einsum('rfkd,rgkd->rkfg', A, B).astype(gdt) for k, (A, B) in {'FF': (MF, MF), 'LL': (ML, ML), 'FL': (MF, ML)}.items()}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out) + '.npz', G_FF=G['FF'], G_LL=G['LL'], G_FL=G['FL'], eF=eF, eL=eL, counts=cnt,
             body_family=np.array([fi[m['category']] for m in held]))
    meta = {'run': str(run), 'source': source, 'rows': rows, 'ids': [m['id'] for m in held], 'families': fams,
            'counts': cnt.tolist(), 'n_layers': ident['n_layers'], 'keep_layers': ident['keep_layers'], 'hidden': int(d),
            'identity': {k: ident.get(k) for k in ('model', 'tmpl_kwargs', 'lang', 'order', 'fixed_date', 'script_sha256', 'lean',
                                                   'words_sha256', 'gate', 'natural_system_ok', 'torch', 'transformers')},
            'rows_finite': [bool(np.isfinite(G['FF'][r]).all() and np.isfinite(G['LL'][r]).all()) for r in range(Q)],
            # pairs whose two renderings differ in length (PROTOCOL rule: run as usual, flag); words_along runs only
            'unequal_pairs': {k: sum(k in m.get('unequal_pairs', []) for m in held) for k in sorted({u for m in held for u in m.get('unequal_pairs', [])})},
            'body_npz_sha256': sha}
    Path(str(out) + '.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print('extracted', run.name, source, 'rows', Q, 'families', F, 'bodies', len(held))


if __name__ == '__main__':
    main()
