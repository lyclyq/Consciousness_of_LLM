"""Synthetic check of extract_gram_20261110.py + analyze_words_along_20261110.py (11-10). Builds run directories in the pilot
format (rows A-U, A-S, U-S, place1, time1, name1, feeling1; natural: three role rows with S missing) and in the words_along
format (rows from identity.json), with known shares kept and initial sizes, runs the extraction and the analysis, and checks:
  - every share and initial size is recovered (b150: 15 families x 5 bodies; p3: each body its own family);
  - the pilot's own computation (family means, U-statistic at full counts, sqrt(C_last / C_first), median over layers) gives
    the same share as the analysis for the pilot rows;
  - rows merge across sources, a missing System row drops A-S and U-S, a row given twice is refused;
  - the magnitude check sees a planted correlation between initial size and share over the words.
usage: test_analyze_words_along_20261110.py WORKDIR"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXTRACT = HERE.parent / 'server3' / 'extract_gram_20261110.py'
ANALYZE = HERE / 'analyze_words_along_20261110.py'
WORDS = HERE.parent.parent / 'items' / 'words_v2.json'
COVER = HERE.parent.parent / 'items' / 'content_index_coverage.json'
K, D = 6, 64
W = json.loads(WORDS.read_text(encoding='utf-8'))
WIDS = [p['id'] for p in W['pairs']]
PILOT_WIDS = ['place1', 'time1', 'name1', 'feeling1']


def truth(seed, words_corr):
    rng = np.random.default_rng(seed)
    t = {'A-U': (1.00, 3.0), 'A-S': (0.95, 2.5), 'U-S': (0.50, 1.5)}
    keeps = rng.uniform(0.3, 0.7, len(WIDS))
    inits = 1.0 + (keeps - 0.3) * 2.5 if words_corr else rng.uniform(1.0, 2.0, len(WIDS))
    for w, k, i in zip(WIDS, keeps, inits):
        t[w] = (float(k), float(i))
    return t


def bodies(set_name):
    if set_name == 'b150':
        return [{'id': f'b{f:02d}_{j}', 'category': f'fam{f:02d}', 'split': 'confirmation'} for f in range(15) for j in range(5)] + \
               [{'id': f'd{f:02d}_{j}', 'category': f'fam{f:02d}', 'split': 'discovery'} for f in range(15) for j in range(5)]
    return [{'id': f'p3_{i:03d}', 'category': f'p3_{i:03d}', 'split': 'confirmation'} for i in range(60)]


def write_run(dirp, rows, T, bl, seed, ident_extra, drop=()):
    rng = np.random.default_rng(seed)
    dirp.mkdir(parents=True, exist_ok=True)
    U = {r: rng.normal(size=(K, D)) for r in rows}
    for r in rows:
        U[r] /= np.linalg.norm(U[r], axis=-1, keepdims=True)
    ident = {'n_layers': 24, 'keep_layers': [9, 12, 14, 17, 20, 23], 'model': 'synthetic', 'tmpl_kwargs': {}, 'lang': 'zh'}
    ident.update(ident_extra)
    (dirp / 'identity.json').write_text(json.dumps(ident), encoding='utf-8')
    fam_eff = {}
    for b in bl:
        mF = np.full((len(rows), K, D), np.nan, np.float16)
        mL = np.full((len(rows), K, D), np.nan, np.float16)
        eF = np.full((len(rows), K), np.nan, np.float32)
        eL = np.full((len(rows), K), np.nan, np.float32)
        for qi, r in enumerate(rows):
            if r in drop:
                continue
            keep, init = T[r]
            fe = fam_eff.setdefault((b['category'], r), rng.normal(scale=0.3, size=(K, D)) / np.sqrt(D))
            a = init * U[r] + fe + rng.normal(scale=0.5, size=(K, D)) / np.sqrt(D)
            c = keep * init * U[r] + keep * fe + rng.normal(scale=0.5, size=(K, D)) / np.sqrt(D)
            mF[qi], mL[qi] = a, c
            eF[qi], eL[qi] = (a ** 2).sum(-1) + 1.0, (c ** 2).sum(-1) + 1.0
        np.savez(dirp / f'{b["id"]}.npz', dec_mF=mF, dec_mL=mL, dec_eF=eF, dec_eL=eL)
        (dirp / f'{b["id"]}.json').write_text(json.dumps(b), encoding='utf-8')


def pilot_share(dirp, qi):
    """the pilot's computation (analyze_persistence_20261106: fam_means, ustats with full counts, median over layers)"""
    metas = [json.loads(p.read_text()) for p in sorted(dirp.glob('*.json')) if p.name != 'identity.json']
    held = [m for m in metas if m['split'] == 'confirmation']
    fams = sorted({m['category'] for m in held})
    Z = {m['id']: dict(np.load(dirp / f'{m["id"]}.npz')) for m in held}
    MF = np.stack([np.mean([Z[m['id']]['dec_mF'][qi].astype(np.float64) for m in held if m['category'] == f], 0) for f in fams])
    ML = np.stack([np.mean([Z[m['id']]['dec_mL'][qi].astype(np.float64) for m in held if m['category'] == f], 0) for f in fams])
    F = len(fams)

    def C(M):
        G = np.einsum('fkd,gkd->kfg', M, M)
        return (G.sum((1, 2)) - np.einsum('kff->k', G)) / (F * F - F)
    return float(np.nanmedian(np.sqrt(C(ML) / C(MF))))


def run(cmd):
    r = subprocess.run([sys.executable, '-I'] + [str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr)
        raise SystemExit(f'failed: {cmd}')
    return r.stdout


def main():
    work = Path(sys.argv[1])
    if work.exists():
        shutil.rmtree(work)
    runs, ex = work / 'runs', work / 'extracts'
    T = truth(1, words_corr=False)
    Tc = truth(2, words_corr=True)
    # b150 main, tested setting: pilot (roles + 4 pilot words) + new run (28 new words)
    b150 = bodies('b150')
    write_run(runs / 'pilot_main_tagA', ['A-U', 'A-S', 'U-S'] + PILOT_WIDS, T, b150, 11, {'order': 'main'})
    new28 = [w for w in WIDS if w not in PILOT_WIDS]
    write_run(runs / 'new_main_tagA', new28, T, b150, 12, {'order': 'main', 'rows': new28})
    # b150 natural, tested setting without System: pilot roles (S missing) + 32 words; correlated words
    write_run(runs / 'pilot_nat_tagA', ['A-U', 'A-S', 'U-S'], Tc, b150, 13, {'order': 'natural'}, drop=('A-S', 'U-S'))
    write_run(runs / 'new_nat_tagA', WIDS, Tc, b150, 14, {'order': 'natural', 'rows': WIDS})
    # p3 main, untested setting: roles + 32 words in one run
    write_run(runs / 'new_p3_tagB', ['A-U', 'A-S', 'U-S'] + WIDS, T, bodies('p3'), 15, {'order': 'main', 'rows': ['A-U', 'A-S', 'U-S'] + WIDS})
    # GLM-5.3-like: stage 1 = roles + the first pair of each category; stage 2 = the other 24 pairs (a second source)
    first8 = [w for w in WIDS if w.endswith('1')]
    rest24 = [w for w in WIDS if not w.endswith('1')]
    write_run(runs / 'g53_stage1', ['A-U', 'A-S', 'U-S'] + first8, T, b150, 16, {'order': 'main', 'rows': ['A-U', 'A-S', 'U-S'] + first8})
    write_run(runs / 'g53_stage2', rest24, T, b150, 17, {'order': 'main', 'rows': rest24})
    run([EXTRACT, runs / 'g53_stage1', ex / 'b150/main/glm53_flash/words'])
    run([EXTRACT, runs / 'g53_stage1', ex / 'b150/main/glm53_both/words'])
    run([EXTRACT, runs / 'g53_stage2', ex / 'b150/main/glm53_both/words_stage2'])
    for src, dst in [('pilot_main_tagA', 'b150/main/qwen38_27b/pilot'), ('new_main_tagA', 'b150/main/qwen38_27b/words'),
                     ('pilot_nat_tagA', 'b150/natural/qwen38_27b/pilot'), ('new_nat_tagA', 'b150/natural/qwen38_27b/words'),
                     ('new_p3_tagB', 'p3/main/qwen36_27b/words')]:
        run([EXTRACT, runs / src, ex / dst])
    run([ANALYZE, ex, work / 'summary', '--words', WORDS, '--coverage', COVER])
    S = json.loads((work / 'summary.json').read_text(encoding='utf-8'))['units']
    bad = []
    for (st, od, tag, TT, tol) in [('b150', 'main', 'qwen38_27b', T, 0.06), ('b150', 'natural', 'qwen38_27b', Tc, 0.06), ('p3', 'main', 'qwen36_27b', T, 0.06)]:
        u = S[st][od][tag]
        for r, v in u['rows'].items():
            if abs(v['share'] - TT[r][0]) > tol or abs(v['init_size'] - TT[r][1]) / TT[r][1] > 0.1:
                bad.append((st, od, r, v['share'], TT[r]))
        assert u['n_words'] == 32, u['n_words']
    g1, g2 = S['b150']['main']['glm53_flash'], S['b150']['main']['glm53_both']
    assert g1['n_words'] == 8 and g1['advantage_vs_first8_median']['n_words'] == 8, g1['n_words']
    assert g2['n_words'] == 32 and abs(g2['rows']['A-U']['share'] - g1['rows']['A-U']['share']) < 1e-12
    assert abs(g1['advantage_vs_first8_median']['value'] - g2['advantage_vs_first8_median']['value']) < 1e-12
    un = S['b150']['natural']['qwen38_27b']
    assert un['self_pairs'] == ['A-U'] and 'A-S' not in un['rows'] and 'U-S' not in un['rows'], un['self_pairs']
    assert un['magnitude_words_spearman']['value'] > 0.8, un['magnitude_words_spearman']
    assert abs(S['b150']['main']['qwen38_27b']['magnitude_words_spearman']['value']) < 0.5
    for qi, r in enumerate(['A-U', 'A-S', 'U-S'] + PILOT_WIDS):
        a, b = pilot_share(runs / 'pilot_main_tagA', qi), S['b150']['main']['qwen38_27b']['rows'][r]['share']
        assert abs(a - b) < 1e-9, ('pilot computation differs', r, a, b)
    # a row given twice is refused
    run([EXTRACT, runs / 'new_main_tagA', ex / 'b150/main/dup/words'])
    run([EXTRACT, runs / 'new_main_tagA', ex / 'b150/main/dup/words2'])
    r = subprocess.run([sys.executable, '-I', str(ANALYZE), str(ex), str(work / 'summary2'), '--words', str(WORDS), '--coverage', str(COVER)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and 'row from two sources' in r.stderr, 'duplicate rows were not refused'
    if bad:
        raise SystemExit(f'recovery failed: {bad}')
    print('SYNTHETIC CHECK OK')
    for k in ('b150/main/qwen38_27b', 'b150/natural/qwen38_27b', 'p3/main/qwen36_27b'):
        s, o, t = k.split('/')
        u = S[s][o][t]
        print(k, 'self', round(u['self']['share'], 3), 'adv vs median', round(u['advantage_vs_word_median']['value'], 3),
              u['advantage_vs_word_median']['ci'], 'rho words', round(u['magnitude_words_spearman']['value'], 2),
              'self - fit', {p: round(v['value'], 3) for p, v in u['self_minus_word_fit'].items()})


if __name__ == '__main__':
    main()
