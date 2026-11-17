"""Does the next-token divergence follow from the geometry of the last layer? (11-14, user: "可以加统计"; theory check for the NMI
framework: geometry, share kept and next-token KL are three readings of one difference h^A - h^U.)
For small differences KL(A || U) is a quadratic form of the last-layer difference, KL ~ (1/2) d^T F d with F the Fisher matrix of
the output distribution. If the output metric does not single out or hide the role directions, the KL ratio of the paper,
KL(self) / KL(U || S), should be about the squared ratio of the last-layer distances, 1 / R_A^2, with R_A the distance ratio of the
geometry (User-System distance over the mean distance of Assistant to the two). Tested across the 25 models of the main probe:
  y = log10 of the content-KL ratio self / external (as Fig. 2a; smaller of A-U and A-S over U-S, ratio of sums over bodies),
  x = -log10 of the mean R_A over the 75 held-out bodies at a given layer, message content (the paper's cross-model geometry:
      mean residual state over the body tokens; summary files of the geometry runs, rendering as in Table tab:self_crossmodel;
      the two four-stream models, Qwen3.8-Flash-Next and GLM-5.3-Flash, as everywhere in the paper: the mean of the four streams,
      from their stream geometry files).
Reported: Spearman rho of y with x at the last layer (one-sided permutation test, 20,000), the least-squares slope of y on x
(prediction 2; bootstrap over models, 10,000), the same rho at earlier depths, and rho of y with the share of layers that carry the
self-index (the statistic of the paper, rho = 0.82) for comparison. Nothing is fitted to choose a layer. Read-only.
usage (from projects/consciousness): kl_vs_lastlayer_geometry_20261114.py"""
import json
import sys
from pathlib import Path

import numpy as np

EV = Path('evidence')
WA = EV / 'p2_words_along_20261110'
sys.path.insert(0, str(Path(__file__).resolve().parent))
FT = json.loads((WA / 'results/final_20261113/final_tables_20261113.json').read_text(encoding='utf-8'))
A8, A3 = EV / 'p2_crossmodel_8models_20261018/models', EV / 'p2_crossmodel_wave3_20261019/runs'
G8 = lambda m, mode='non_thinking', d='geo_attr': A8 / m / mode / d / '_run_summary.json'  # noqa: E731
GEO = {'qwen38_27b': G8('qwen38_27b'), 'qwen38_flashnext': A3 / 'pro/flashnext_stream_20261020/flashnext_stream_geometry_non_20261020.json', 'glm46v_flash': G8('glm46v_flash'),
       'glm47_flash': G8('glm47_flash'), 'gemma4_31b_en': G8('gemma4_31b'), 'qwen36_27b': G8('qwen36_27b'),
       'qwen36_35b_a3b': G8('qwen36_35b'), 'r1_distill_llama70b': G8('r1_llama70b', 'thinking', 'geo'),
       'gpt_oss_20b_en': A3 / 'pro/suite_think/gpt_oss_20b_mxfp4_en/summary.json', 'gpt_oss_120b_en': A3 / 'pro/suite_think/gpt_oss_120b_en/summary.json',
       'glm53_flash': A3 / 'server3/suite_think/glm53_flash_stream_geometry.json', 'exaone4_32b_en': A3 / 'server3/suite/exaone4_32b_en/summary.json',
       'mimo_v26_distill_qwen9b': A3 / 'pro/suite/mimo_v26_distill_qwen9b/summary.json', 'phi4_en': A3 / 'server3/suite/phi4_en/summary.json',
       'granite42_30b_en': A3 / 'server3/suite/granite42_30b_en/summary.json', 'granite41_30b_en': A3 / 'server3/suite/granite41_30b_en/summary.json',
       'olmo2_13b_en': A3 / 'pro/suite/olmo2_13b_en/summary.json', 'olmo2_7b_en': A3 / 'pro/suite/olmo2_7b_en/summary.json',
       'apertus8b_en': A3 / 'pro/suite/apertus8b_instruct_2509_en/summary.json',
       'moonlight16b_a3b_instruct': A3 / 'server3/suite/moonlight16b_a3b_instruct/summary.json',
       'mimo7b_rl_0530': A3 / 'server3/suite/mimo7b_rl_0530/summary.json', 'mimo_vl7b_rl_2508': A3 / 'pro/suite/mimo_vl7b_rl_2508/summary.json',
       'ernie45_21b_a3b_thinking': A3 / 'server3/suite/ernie45_21b_a3b_thinking/summary.json',
       'seed_oss_36b': A3 / 'server3/suite/seed_oss_36b/summary.json', 'qwen35_27b': A3 / 'server3/suite/qwen35_27b/summary.json'}
DEPTHS = (0.25, 0.5, 0.75, 1.0)


def rank(a):
    a = np.asarray(a, float)
    o = a.argsort()
    r = np.empty(len(a))
    r[o] = np.arange(len(a))
    for v in np.unique(a):
        m = a == v
        r[m] = r[m].mean()
    return r


def rho(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])

STREAMS = {'qwen38_flashnext', 'glm53_flash'}


def kl_self_ext(src):
    s = json.loads((EV / src / 'summary.json').read_text(encoding='utf-8'))['kinds']['kl_content']['to_US']
    return min(s['A-U']['ratio'], s['A-S']['ratio'])


def main():
    rows = [r for r in FT['rows']['main'] if r['group'] != '-']
    assert len(rows) == 25 and set(GEO) == {r['tag'] for r in rows}, set(GEO) ^ {r['tag'] for r in rows}
    rng = np.random.default_rng(20261114)
    data = []
    for r in rows:
        g = json.loads(GEO[r['tag']].read_text(encoding='utf-8'))['content']
        if r['tag'] in STREAMS:   # four-stream models: the mean of the streams (R_A averaged over the streams, then over bodies)
            g = {'mean_R': [g['mean_of_streams']['R_A_by_layer']], 'share_A_lowest': g['mean_of_streams']['share_A_lowest_by_layer']}
        ra = np.array(g['mean_R'][0], float)   # R_A by layer
        L = len(ra)
        data.append({'tag': r['tag'], 'name': r['name'], 'group': r['group'], 'kl_ratio': kl_self_ext(r['kl']['src']), 'layers': L,
                     'share_layers_content': float((np.array(g['share_A_lowest']) >= 0.8).mean()),
                     'R_A': {str(d): float(ra[round(d * (L - 1))]) for d in DEPTHS}})
    y = np.log10([d['kl_ratio'] for d in data])
    res = {}
    for dp in DEPTHS:
        x = -np.log10([d['R_A'][str(dp)] for d in data])
        r0 = rho(x, y)
        perm = np.array([rho(x, rng.permutation(y)) for _ in range(20000)])
        b = np.polyfit(x, y, 1)
        bs = []
        for _ in range(10000):
            i = rng.integers(0, len(x), len(x))
            if np.ptp(x[i]) > 0:
                bs.append(np.polyfit(x[i], y[i], 1)[0])
        res[str(dp)] = {'spearman': r0, 'p_one_sided': float((perm >= r0).mean()), 'slope': float(b[0]), 'intercept': float(b[1]),
                        'slope_ci95': [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                        'pearson_log': float(np.corrcoef(x, y)[0, 1])}
    sh = [d['share_layers_content'] for d in data]
    res['share_of_layers_content'] = {'spearman': rho(sh, y)}
    out = WA / 'results/final_20261113/kl_vs_lastlayer_geometry_20261114.json'
    out.write_text(json.dumps({'method': __doc__.split('usage')[0].strip(), 'models': data, 'results': res}, ensure_ascii=False, indent=1), encoding='utf-8')
    L = ['| depth | Spearman rho | p (one-sided) | slope of log KL ratio on -log R_A | 95% interval | Pearson (log-log) |', '|---|---|---|---|---|---|']
    for dp in DEPTHS:
        v = res[str(dp)]
        L.append(f"| {dp:.2f} | {v['spearman']:+.2f} | {v['p_one_sided']:.4f} | {v['slope']:.2f} | [{v['slope_ci95'][0]:.2f}, {v['slope_ci95'][1]:.2f}] | {v['pearson_log']:+.2f} |")
    L += ['', f"share of layers with the self-index on message content: rho = {res['share_of_layers_content']['spearman']:+.2f}", '',
          '| model | group | KL ratio | R_A last layer | 1 / R_A^2 |', '|---|---|---|---|---|']
    for d in sorted(data, key=lambda d: -d['kl_ratio']):
        L.append(f"| {d['name']} | {d['group']} | {d['kl_ratio']:.2f} | {d['R_A']['1.0']:.3f} | {1 / d['R_A']['1.0'] ** 2:.2f} |")
    out.with_suffix('.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
