"""Published benchmark scores against the two levels (Appendix app:benchmarks, Tables tab:self_benchmarks and
tab:self_benchmark_corr).

gpqa (stats/gpqa_vs_levels_20261029.json): rank correlations over the 16 models whose developers publish a GPQA score
  (stats/v6_benchmarks_20261104.csv of the results package; v5 plus the models added on 11-04, compiled from the model cards; EXAONE-4.0.1-32B: reasoning mode,
  as in the paper), against
    group   : ordinal group of Section sec:res_crossmodel (neither 0 < self-index only 1 < self-reference after thinking 2 < both 3)
    selfref : recognizes its own messages in some mode (0/1)
    index   : self-index share of layers (Figure self_crossmodel; non-thinking rendering where the model has one, else thinking)
    claim   : self-reference in its strongest mode (Figure self_crossmodel points: own minus the larger external, either mode)
  Spearman rho with a two-sided permutation p (100,000 permutations, seed 0) and a 95% bootstrap interval over models (10,000,
  seed 0); for selfref also the Mann-Whitney comparison of GPQA between models with and without self-reference.
pooled (stats/gpqa_vs_levels_pooled_20261029.json): the same correlation over both conversation orders, each model weighted
  equally and, where a model was measured both under the main probe and in natural order, each measurement with half its weight
  (realized by averaging the two values of such a model, with the ordinary Spearman rho; the value of the one order otherwise).
    main probe : the gpqa output (14 models; as tab:self_benchmarks)
    natural    : self-index share, three-role models (stats/self_natural_points_v1.json of natural_order.py) and
                 Ministral-3-14B-Reasoning (0 of 40 layers at content and onset; natural order only, GPQA 71.2 from its model card,
                 self-reference not measured); self-reference strength, three-role models (own minus the larger external, three
                 option orders) and the two-role models of tab:self_natural2 (own minus User, in the mode of tab:self_crossmodel)
    group      : unchanged except Gemma-4-31B (neither level under the main probe, both levels in natural order: mean 1.5)
  Spearman rho, two-sided permutation p (100,000, seed 0), 95% bootstrap interval over models (10,000, seed 0).
tables (stats/appendix_c_20261029/benchmarks_{en,zh}.tex): both tables in one float, from the two JSON files and the CSV; label
  and setting as in the model cards; every GPQA value is checked against the CSV. The English block is the body between
  %<<tab_benchmarks and %>>tab_benchmarks of the appendix source.

usage: benchmarks.py --results RES --out OUT   (needs self_crossmodel_points_v6.json from fig_crossmodel.py and
       self_natural_points_v1.json from natural_order.py in OUT)
"""
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import W3, Roots, arguments  # noqa: E402


def gpqa(S, IN):
    """S: output stats folder; IN: stats folder of the results package (the benchmark CSV)."""
    KEY = {'Qwen3.8-27B': 'qwen38_27b', 'Qwen3.8-Flash-Next': 'qwen38_flashnext', 'gpt-oss-120b': 'gpt_oss_120b', 'gpt-oss-20b': 'gpt_oss_20b',
           'EXAONE-4.0.1-32B': 'exaone4_32b', 'GLM-4.7-Flash': 'glm47_flash', 'Granite-4.2-30B': 'granite42_30b',
           'DeepSeek-R1-Distill-Llama-70B': 'r1_llama70b', 'Granite-4.1-30B': 'granite41_30b', 'Qwen3.6-27B': 'qwen36_27b',
           'Qwen3.6-35B-A3B': 'qwen36_35b', 'Qwen3.5-27B': 'qwen35_27b', 'Gemma-4-31B': 'gemma4_31b', 'Seed-OSS-36B': 'seed_oss_36b',
           'MiMo-7B-RL-0530': 'mimo7b_rl', 'phi-4': 'phi4'}  # 11-04: the two new models whose cards publish GPQA
    GROUP = {'neither level': 0, 'self-index without self-reference': 1, 'self-reference after reasoning': 2, 'both levels': 3}
    P = json.load(open(S / 'self_crossmodel_points_v6.json', encoding='utf-8'))
    rows = {}
    for r in csv.DictReader(open(IN / 'v6_benchmarks_20261104.csv', encoding='utf-8')):
        if r['model'] in KEY and r['value'] and r['model'] not in rows:  # first row per model (EXAONE: reasoning mode)
            rows[r['model']] = r
    names = sorted(rows)
    g = np.array([float(rows[m]['value']) for m in names])
    k = [KEY[m] for m in names]
    grp = np.array([GROUP[rows[m]['group']] for m in names], float)
    sr = (grp >= 2).astype(float)
    idx = np.array([(P.get(f'{x}/non_thinking') or P[f'{x}/thinking'])[0] for x in k])
    clm = np.array([max(P[f'{x}/{md}'][1] for md in ('non_thinking', 'thinking') if f'{x}/{md}' in P) for x in k])
    rng = np.random.default_rng(0)

    def rho(a, b):
        return stats.spearmanr(a, b)[0]

    print(f'n = {len(names)} models with a published GPQA score')
    OUT = {'n': len(names), 'models': {m: {'gpqa': float(x), 'group': int(y), 'index': float(z), 'claim': float(w)}
                                       for m, x, y, z, w in zip(names, g, grp, idx, clm)}, 'rank': {}}
    for nm, v in (('group', grp), ('selfref', sr), ('index', idx), ('claim', clm)):
        r0 = rho(g, v)
        perm = np.array([rho(g, rng.permutation(v)) for _ in range(100_000)])
        p = float((np.abs(perm) >= abs(r0) - 1e-12).mean())
        bs = []
        for _ in range(10_000):
            i = rng.integers(0, len(g), len(g))
            if len(set(v[i])) > 1 and len(set(g[i])) > 1:
                bs.append(rho(g[i], v[i]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f'{nm:8s} Spearman rho = {r0:+.2f}  permutation p = {p:.2f}  95% bootstrap [{lo:+.2f}, {hi:+.2f}]')
        OUT['rank'][nm] = {'rho': float(r0), 'p': p, 'ci95': [float(lo), float(hi)]}
    u = stats.mannwhitneyu(g[sr == 1], g[sr == 0], alternative='two-sided')
    OUT['mannwhitney'] = {'median_with': float(np.median(g[sr == 1])), 'n_with': int(sr.sum()), 'median_without': float(np.median(g[sr == 0])),
                          'n_without': int((1 - sr).sum()), 'p': float(u.pvalue)}
    json.dump(OUT, open(S / 'gpqa_vs_levels_20261029.json', 'w'), indent=1)
    print(f'GPQA with self-reference: median {np.median(g[sr == 1]):.1f} (n={int(sr.sum())}); without: median {np.median(g[sr == 0]):.1f} '
          f'(n={int((1 - sr).sum())}); Mann-Whitney p = {u.pvalue:.2f}')
    for m, x, y, z, w in sorted(zip(names, g, grp, idx, clm), key=lambda t: -t[1]):
        print(f'  {m:32s} GPQA {x:5.1f}  group {int(y)}  index {z:.2f}  claim {w:+.2f}')


def pooled(S):
    M = json.loads((S / 'gpqa_vs_levels_20261029.json').read_text(encoding='utf-8'))['models']
    NP = json.loads((S / 'self_natural_points_v1.json').read_text(encoding='utf-8'))
    nat_index = {'Seed-OSS-36B': NP['seed_oss_36b/natural'][0], 'Gemma-4-31B': NP['gemma4_31b/natural'][0],
                 'EXAONE-4.0.1-32B': NP['exaone4_32b/natural'][0], 'Granite-4.1-30B': NP['granite41_30b/natural'][0],
                 'MiMo-7B-RL-0530': NP['mimo7b_rl/natural'][0], 'phi-4': NP['phi4/natural'][0]}
    nat_claim = {'Seed-OSS-36B': NP['seed_oss_36b/natural'][1], 'Gemma-4-31B': NP['gemma4_31b/natural'][1],
                 'EXAONE-4.0.1-32B': NP['exaone4_32b/natural'][1], 'Granite-4.1-30B': NP['granite41_30b/natural'][1],
                 'MiMo-7B-RL-0530': NP['mimo7b_rl/natural'][1], 'phi-4': NP['phi4/natural'][1],
                 'Qwen3.8-27B': 0.76 - 0.02, 'gpt-oss-20b': 0.93 - 0.03, 'DeepSeek-R1-Distill-Llama-70B': 0.34 - 0.16}
    EXTRA = {'Ministral-3-14B-Reasoning': {'gpqa': 71.2, 'index': 0.0}}  # natural order only
    rng = np.random.default_rng(0)

    def series(key, nat):
        rows = []
        for m, d in M.items():
            if key == 'selfref':  # presence of self-reference (group >= 2); computed after the other three, which keep their random draws
                v = 0.5 if m == 'Gemma-4-31B' else float(d['group'] >= 2)
            else:
                v = d[key] if m not in nat else (d[key] + nat[m]) / 2
            if key == 'group' and m == 'Gemma-4-31B':
                v = (0 + 3) / 2
            rows.append((m, d['gpqa'], v))
        if key == 'index':
            rows += [(m, e['gpqa'], e['index']) for m, e in EXTRA.items()]
        return rows

    OUT = {}
    for key, nat in (('group', {}), ('index', nat_index), ('claim', nat_claim), ('selfref', {})):
        rows = series(key, nat)
        g = np.array([r[1] for r in rows])
        v = np.array([r[2] for r in rows], float)
        r0 = stats.spearmanr(g, v)[0]
        perm = np.array([stats.spearmanr(g, rng.permutation(v))[0] for _ in range(100_000)])
        p = float((np.abs(perm) >= abs(r0) - 1e-12).mean())
        bs = []
        for _ in range(10_000):
            i = rng.integers(0, len(g), len(g))
            if len(set(v[i])) > 1 and len(set(g[i])) > 1:
                bs.append(stats.spearmanr(g[i], v[i])[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        OUT[key] = {'n': len(rows), 'rho': float(r0), 'p': p, 'ci95': [float(lo), float(hi)]}
        print(f'{key:6s} n={len(rows)} rho={r0:+.2f} p={p:.2f} 95% [{lo:+.2f}, {hi:+.2f}]', flush=True)
    json.dump(OUT, open(S / 'gpqa_vs_levels_pooled_20261029.json', 'w'), indent=1)


def tables_tex(S, IN):
    J = json.loads((S / 'gpqa_vs_levels_20261029.json').read_text(encoding='utf-8'))
    # both conversation orders, each model weighted equally (pooled)
    PJ = json.loads((S / 'gpqa_vs_levels_pooled_20261029.json').read_text(encoding='utf-8'))
    CSV = {}
    for r in csv.DictReader(open(IN / 'v6_benchmarks_20261104.csv', encoding='utf-8')):
        if r['value'] and r['model'] not in CSV:
            CSV[r['model']] = float(r['value'])
    # (group, model, label, setting), in table order; label and setting as in the model cards
    ROWS = [(3, 'Qwen3.8-27B', 'GPQA Diamond', ''),
            (2, 'Qwen3.8-Flash-Next', 'GPQA Diamond', ''), (2, 'gpt-oss-120b', 'GPQA Diamond', 'no tools, reasoning high'),
            (2, 'EXAONE-4.0.1-32B', 'GPQA-Diamond', 'reasoning mode (61.6 without)'), (2, 'gpt-oss-20b', 'GPQA Diamond', 'no tools, reasoning high'),
            (1, 'GLM-4.7-Flash', 'GPQA', ''), (1, 'Granite-4.2-30B', 'GPQA', ''), (1, 'DeepSeek-R1-Distill-Llama-70B', 'GPQA Diamond', 'pass@1'),
            (1, 'Granite-4.1-30B', 'GPQA', '0-shot, CoT'), (1, 'phi-4', 'GPQA', ''),
            (0, 'Qwen3.6-27B', 'GPQA Diamond', ''), (0, 'Qwen3.6-35B-A3B', 'GPQA', ''), (0, 'Qwen3.5-27B', 'GPQA Diamond', ''),
            (0, 'Gemma-4-31B', 'GPQA Diamond', ''), (0, 'Seed-OSS-36B', 'GPQA-D', ''), (0, 'MiMo-7B-RL-0530', 'GPQA-Diamond', 'Pass@1')]
    SHOWN = {'DeepSeek-R1-Distill-Llama-70B': 'R1-Distill-Llama-70B', 'MiMo-7B-RL-0530': 'MiMo-7B-RL'}
    GNAME = {'en': {3: ('Both levels', ''), 2: ('Self-reference', 'after reasoning'), 1: ('Self-index without', 'self-reference'), 0: ('Neither level', '')},
             'zh': {3: ('两层都有', ''), 2: ('推理后自指', ''), 1: ('有索引无自指', ''), 0: ('两层都没有', '')}}
    for g, m, *_ in ROWS:
        d = J['models'][m]
        assert d['group'] == g and abs(d['gpqa'] - CSV[m]) < 1e-9, m
    assert len(ROWS) == J['n']

    def num(x, sign=False):
        s = f'{x:+.2f}' if sign else f'{x:.2f}'
        return s.replace('-', '$-$')

    def gpqa_(x):
        return f'{x:g}' if abs(x * 10 - round(x * 10)) > 1e-9 else f'{x:.1f}'

    def tables(L):
        R = J['rank']
        if L == 'en':
            cap1 = ('GPQA scores as published by the developers, with the group of each model and its two levels under the main probe '
                    '(Section~\\ref{sec:res_crossmodel}). Label: the name of the benchmark in the source; setting: as stated in the source (empty where none is stated). '
                    'Index: share of layers at which the model\'s own message is singled out in at least 80\\% of the held-out bodies at message '
                    'content or at the response onset, under the non-thinking rendering (thinking rendering for models with only a thinking '
                    'mode), as in Figure~\\ref{fig:self_crossmodel}. Self-reference: the model\'s own messages minus the larger of User and '
                    'System messages, in the mode in which it is larger (probability of ``you\'\' without thinking, share of finished items '
                    'answered ``you\'\' with thinking), as in Figure~\\ref{fig:self_crossmodel}.')
            head = 'Group & Model & GPQA & Label & Setting & Index & Self-ref. \\\\'
            cap2 = ('Rank correlation between the published GPQA score and the two levels. Main probe: the '
                    f'{J["n"]} models of Table~\\ref{{tab:self_benchmarks}}. With natural order: the measurements in natural order '
                    '(Appendix~\\ref{app:natural}) included, each model weighted equally: a model measured in both orders contributes the mean '
                    'of its two values, and Ministral-3-14B-Reasoning, measured in natural order only, is added to the self-index (GPQA Diamond '
                    '71.2 in its model card); in natural order, self-reference is averaged over the option orders of Appendix~\\ref{app:natural} '
                    '(three where three roles are compared, two otherwise) and, for the models compared '
                    'on two roles, taken against User messages. Spearman $\\rho$, two-sided permutation $p$ (100,000 permutations), and 95\\% '
                    'bootstrap interval over models (10,000 resamples). Group: neither level, self-index without self-reference, self-reference '
                    'after reasoning, both levels, in this order; presence of self-reference: one of the last two groups.')
            head2 = ('& \\multicolumn{3}{c}{Main probe ($n = ' + str(J['n']) + '$)} & \\multicolumn{4}{c}{With natural order} \\\\\n'
                     '\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\nLevel & $\\rho$ & $p$ & 95\\% interval & $n$ & $\\rho$ & $p$ & 95\\% interval \\\\')
            names = [('group', 'Group'), ('index', 'Self-index'), ('selfref', 'Self-reference, presence'),
                     ('claim', 'Self-reference, strength')]
        else:
            cap1 = ('开发方公布的 GPQA 分数，以及每个模型在主探针下的分组与两个层级（第~\\ref{sec:res_crossmodel} 节）。名称：来源中的基准名称；设置：来源中写明的设置'
                    '（未写明则空）。索引：在至少 80\\% 的留出正文中，模型自己的消息在消息内容或回应起点被单独分出的层所占的比例，取非思考渲染'
                    '（只有思考模式的模型取思考渲染），同图~\\ref{fig:self_crossmodel}。自指：模型自己的消息减去 User 与 System 消息中较大者，'
                    '取两种模式中较大的一种（不思考时为“你”的概率，思考时为答“你”的已完成题目比例），同图~\\ref{fig:self_crossmodel}。')
            head = '组别 & 模型 & GPQA & 名称 & 设置 & 索引 & 自指 \\\\'
            cap2 = ('公布的 GPQA 分数与两个层级的秩相关。主探针：表~\\ref{tab:self_benchmarks} 中的 '
                    f'{J["n"]} 个模型。含自然顺序：纳入自然顺序的测量（附录~\\ref{{app:natural}}），每个模型权重相同：两种顺序都测过的模型取两个值的平均，'
                    '只在自然顺序下测过的 Ministral-3-14B-Reasoning 加入自我索引一项（其模型卡中 GPQA Diamond 为 71.2）；自然顺序下的自指按附录~\\ref{app:natural} 的选项顺序平均（比较三种角色时为三种，否则为两种），'
                    '只比较两种角色的模型取相对 User 消息的差。Spearman $\\rho$、双侧置换检验 $p$（100,000 次置换）与按模型的 95\\% bootstrap 区间（10,000 次重抽样）。'
                    '组别依次为：两层都没有、有索引无自指、推理后自指、两层都有；有无自指：属于后两组之一。')
            head2 = ('& \\multicolumn{3}{c}{主探针（$n = ' + str(J['n']) + '$）} & \\multicolumn{4}{c}{含自然顺序} \\\\\n'
                     '\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\n层级 & $\\rho$ & $p$ & 95\\% 区间 & $n$ & $\\rho$ & $p$ & 95\\% 区间 \\\\')
            names = [('group', '组别'), ('index', '自我索引'), ('selfref', '自指，有无'), ('claim', '自指，强度')]
        body, last, k = [], None, 0
        for g, m, lab, sett in ROWS:
            d = J['models'][m]
            if last is not None and g != last:
                body.append('\\midrule')
            k = 0 if g != last else k + 1
            gtxt = GNAME[L][g][k] if k < 2 else ''  # group name on the first row, its second line (if any) on the second
            body.append(f'{gtxt} & {SHOWN.get(m, m)} & {gpqa_(d["gpqa"])} & {lab} & {sett} & {num(d["index"])} & {num(d["claim"], True)} \\\\')
            last = g
        # both tables in one float, so that they stay on one page; each keeps its own caption and number
        t1 = ('\\begin{table}[htbp]\n\\centering\n\\caption{' + cap1 + '}\n\\label{tab:self_benchmarks}\n{\\footnotesize\n'
              '\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{llrl>{\\raggedright\\arraybackslash}p{2.7cm}rr}\n\\toprule\n' + head + '\n\\midrule\n' + '\n'.join(body)
              + '\n\\bottomrule\n\\end{tabular}}\n')

        def ci(d):
            return f'[{num(d["ci95"][0], True)}, {num(d["ci95"][1], True)}]'
        rows2 = [f'{lab} & {num(R[k]["rho"], True)} & {R[k]["p"]:.2f} & {ci(R[k])} & {PJ[k]["n"]} & {num(PJ[k]["rho"], True)} & '
                 f'{PJ[k]["p"]:.2f} & {ci(PJ[k])} \\\\' for k, lab in names]
        t2 = ('\n\\vspace{1.2em}\n\\caption{' + cap2 + '}\n\\label{tab:self_benchmark_corr}\n{\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n'
              '\\begin{tabular}{lrrcrrrc}\n\\toprule\n' + head2 + '\n\\midrule\n' + '\n'.join(rows2) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n')
        return t1 + t2

    out = S / 'appendix_c_20261029'
    out.mkdir(parents=True, exist_ok=True)
    for L in ('en', 'zh'):
        (out / f'benchmarks_{L}.tex').write_text(tables(L), encoding='utf-8')
    print(tables('en'))


def main(argv=None):
    args = arguments(__doc__).parse_args(argv)
    r = Roots(args)
    S, IN = r.out(W3 + '/stats'), r.res(W3 + '/stats')
    gpqa(S, IN)
    pooled(S)
    tables_tex(S, IN)


if __name__ == '__main__':
    main()
