"""Appendix C tables of the P2 paper (user 10-29: the capability claim needs its data in a table). From stats/gpqa_vs_levels_20261029.json
(gpqa_vs_levels_20261029.py) and stats/v5_benchmarks_20261029.csv:
  tab:self_benchmarks      GPQA as published, by group, with each model's self-index (share of layers singled out at content or onset,
                           non-thinking rendering, thinking rendering for models with only a thinking mode; as Table tab:self_crossmodel)
                           and self-reference (own minus the larger external, in the mode where it is larger; as Figure self_crossmodel)
  tab:self_benchmark_corr  Spearman rho between GPQA and the two levels, permutation p and bootstrap interval.
Label and setting are those of the table of 10-29 (checked against the model cards then); every GPQA value is checked against the CSV.
  tex   write stats/appendix_c_20261029/benchmarks_{en,zh}.tex
  fill  put the English block between %<<tab_benchmarks and %>>tab_benchmarks of paper/source/self_en_appendix.tex (first run: replace
        the table of 10-29), and write the sentences that cite the numbers (Appendix C text, Section 6.2, Conclusion) from the same JSON
usage: make_benchmark_tables.py tex | fill   (run from evidence/p2_crossmodel_wave3_20261019)"""
import csv
import json
import sys
from pathlib import Path

A = Path('.').resolve()
SRC = A.parents[1] / 'paper' / 'source'
J = json.loads((A / 'stats' / 'gpqa_vs_levels_20261029.json').read_text(encoding='utf-8'))
# both conversation orders, each model weighted equally (user 10-29; gpqa_vs_levels_pooled_20261029.py)
PJ = json.loads((A / 'stats' / 'gpqa_vs_levels_pooled_20261029.json').read_text(encoding='utf-8'))
CSV = {}
for r in csv.DictReader(open(A / 'stats' / 'v5_benchmarks_20261029.csv', encoding='utf-8')):
    if r['value'] and r['model'] not in CSV:
        CSV[r['model']] = float(r['value'])
# (group, model, label, setting) in the order of the table of 10-29
ROWS = [(3, 'Qwen3.8-27B', 'GPQA Diamond', ''),
        (2, 'Qwen3.8-Flash-Next', 'GPQA Diamond', ''), (2, 'gpt-oss-120b', 'GPQA Diamond', 'no tools, reasoning high'),
        (2, 'EXAONE-4.0.1-32B', 'GPQA-Diamond', 'reasoning mode (61.6 without)'), (2, 'gpt-oss-20b', 'GPQA Diamond', 'no tools, reasoning high'),
        (1, 'GLM-4.7-Flash', 'GPQA', ''), (1, 'Granite-4.2-30B', 'GPQA', ''), (1, 'DeepSeek-R1-Distill-Llama-70B', 'GPQA Diamond', 'pass@1'),
        (1, 'Granite-4.1-30B', 'GPQA', '0-shot, CoT'),
        (0, 'Qwen3.6-27B', 'GPQA Diamond', ''), (0, 'Qwen3.6-35B-A3B', 'GPQA', ''), (0, 'Qwen3.5-27B', 'GPQA Diamond', ''),
        (0, 'Gemma-4-31B', 'GPQA Diamond', ''), (0, 'Seed-OSS-36B', 'GPQA-D', '')]
SHOWN = {'DeepSeek-R1-Distill-Llama-70B': 'R1-Distill-Llama-70B'}
GNAME = {'en': {3: ('Both levels', ''), 2: ('Self-reference', 'after reasoning'), 1: ('Self-index without', 'self-reference'), 0: ('Neither level', '')},
         'zh': {3: ('两层都有', ''), 2: ('推理后自指', ''), 1: ('有索引无自指', ''), 0: ('两层都没有', '')}}
for g, m, *_ in ROWS:
    d = J['models'][m]
    assert d['group'] == g and abs(d['gpqa'] - CSV[m]) < 1e-9, m
assert len(ROWS) == J['n']


def num(x, sign=False):
    s = f'{x:+.2f}' if sign else f'{x:.2f}'
    return s.replace('-', '$-$')


def gpqa(x):
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
        body.append(f'{gtxt} & {SHOWN.get(m, m)} & {gpqa(d["gpqa"])} & {lab} & {sett} & {num(d["index"])} & {num(d["claim"], True)} \\\\')
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


def fill():
    R = J['rank']
    p = SRC / 'self_en_appendix.tex'
    s = p.read_text(encoding='utf-8')
    block = tables('en')
    if '%<<tab_benchmarks\n' in s:
        i, j = s.index('%<<tab_benchmarks\n') + len('%<<tab_benchmarks\n'), s.index('%>>tab_benchmarks\n')
        s = s[:i] + block + s[j:]
    else:
        i = s.index('\\begin{table}[htbp]\n\\centering\n\\caption{GPQA scores as published by the developers')
        j = s.index('\\end{table}\n', i) + len('\\end{table}\n')
        s = s[:i] + '%<<tab_benchmarks\n' + block + '%>>tab_benchmarks\n' + s[j:]
    def swap(t, olds, new):
        """write new in place of the first old wording found once (the text of 10-29 or an earlier version); nothing if new is there"""
        if new in t:
            return t
        for o in olds:
            if t.count(o) == 1:
                return t.replace(o, new)
        raise AssertionError('no known wording for: ' + new[:70])

    P_ = PJ['claim']
    a0 = ('Within each group of more than one model the scores span a wide range, and the four highest scores of the models without '
          'self-reference belong to models with neither level.')
    a1 = (a0 + f' Over the {J["n"]} models with a score, neither the group of a model nor the extent of its self-index follows the score, '
          'and the strength of self-reference tends to be higher in higher-scoring models, but not reliably (Table~\\ref{tab:self_benchmark_corr}).')
    a2 = ('Within each group of more than one model the scores span a wide range, and the four highest scores of the models without '
          f'self-reference belong to models with neither level under the main probe. Over the {J["n"]} models with a score, neither the group '
          'of a model nor the extent of its self-index follows the score, and the strength of self-reference tends to be higher in '
          'higher-scoring models, but not reliably; with the measurements in natural order included, none of the three follows the score '
          '(Table~\\ref{tab:self_benchmark_corr}).')
    # 10-30 (audit: a capability "tendency" asserted and withdrawn): the statistic only
    a3 = ('Within each group of more than one model the scores span a wide range, and the four highest scores of the models without '
          f'self-reference belong to models with neither level under the main probe. Over the {J["n"]} models with a score, neither the group '
          f'of a model nor the extent of its self-index follows the score; for the strength of self-reference, $\\rho = {R["claim"]["rho"]:.2f}$ '
          f'($p = {R["claim"]["p"]:.2f}$), and with the measurements in natural order included none of the three follows the score '
          '(Table~\\ref{tab:self_benchmark_corr}).')
    s = swap(s, [a2, a1, a0], a3)
    p.write_text(s, encoding='utf-8')
    q = SRC / 'self_en_body.tex'
    b = q.read_text(encoding='utf-8')
    b = swap(b, ['The levels and problem-solving ability are not unrelated, since both levels are strongest in the Qwen3.8 models, but they '
                 'are not proportional: within the Qwen3.8 template, the 27B model recognizes its own messages in a single pass and the larger '
                 'Qwen3.8-Flash-Next does not.'],
             'Within the Qwen3.8 template, the 27B model recognizes its own messages in a single pass, and the larger Qwen3.8-Flash-Next, '
             'which scores higher, does not.')
    # the counterexamples rest on the three Qwen models, which do not depend on the order of the conversation (user 10-29)
    b = swap(b, ['On GPQA, the four models without self-reference that score highest, Qwen3.6-27B (87.8), Qwen3.6-35B-A3B (86.0), '
                 'Qwen3.5-27B (85.5), and Gemma-4-31B (84.3), have neither level,'],
             'On GPQA, the three models without self-reference that score highest, Qwen3.6-27B (87.8), Qwen3.6-35B-A3B (86.0), and '
             'Qwen3.5-27B (85.5), have neither level,')
    anchor = 'and EXAONE-4.0.1-32B (74.3) and gpt-oss-20b (71.5) recognize their own messages after reasoning.'
    v1 = (f' Across the {J["n"]} models with a published score, the group of a model does not follow its score (Spearman '
          f'$\\rho = {R["group"]["rho"]:.2f}$), nor does the extent of its self-index ($\\rho = {R["index"]["rho"]:.2f}$); the strength of '
          f'self-reference tends to be higher in higher-scoring models, but not reliably ($\\rho = {R["claim"]["rho"]:.2f}$, permutation '
          f'$p = {R["claim"]["p"]:.2f}$; Table~\\ref{{tab:self_benchmark_corr}}).')
    v2 = (f' Under the main probe, across the {J["n"]} models with a published score, the group of a model does not follow its score '
          f'(Spearman $\\rho = {R["group"]["rho"]:.2f}$), nor does the extent of its self-index ($\\rho = {R["index"]["rho"]:.2f}$); the '
          f'strength of self-reference tends to be higher in higher-scoring models, but not reliably ($\\rho = {R["claim"]["rho"]:.2f}$, '
          f'permutation $p = {R["claim"]["p"]:.2f}$), and with the measurements in natural order included, this tendency falls to '
          f'$\\rho = {P_["rho"]:.2f}$ (Table~\\ref{{tab:self_benchmark_corr}}).')
    v3 = (f' Under the main probe, across the {J["n"]} models with a published score, the group of a model does not follow its score '
          f'(Spearman $\\rho = {R["group"]["rho"]:.2f}$), nor does the extent of its self-index ($\\rho = {R["index"]["rho"]:.2f}$); for the '
          f'strength of self-reference, $\\rho = {R["claim"]["rho"]:.2f}$ (permutation $p = {R["claim"]["p"]:.2f}$), and '
          f'$\\rho = {P_["rho"]:.2f}$ with the measurements in natural order included (Table~\\ref{{tab:self_benchmark_corr}}).')
    b = swap(b, [anchor + v2, anchor + v1, anchor], anchor + v3)
    # 10-30 (supervisor's review R11): no claim of independence; the counterexample stays (fix_three_claims_20261030.py)
    b = swap(b, ['The levels do not follow model size or reasoning ability: the functional self is separate from problem-solving ability, and a high benchmark score does not bring it.'],
             'The levels do not follow model size or reasoning ability, and a high benchmark score does not bring them: the three models without self-reference that score highest on GPQA have neither level.')
    q.write_text(b, encoding='utf-8')
    print('filled')


if sys.argv[1] == 'tex':
    out = A / 'stats' / 'appendix_c_20261029'
    out.mkdir(parents=True, exist_ok=True)
    for L in ('en', 'zh'):
        (out / f'benchmarks_{L}.tex').write_text(tables(L), encoding='utf-8')
    print(tables('en'))
else:
    fill()
