"""Appendix B (Configurations) of the P2 paper (user 10-29: reproducible settings, no hardware). The table of checkpoints is built
from the archived renderings of Appendix A (items/appendix_a_20261029/*.json: download commit of the template and tokenizer files,
which is the commit of every file of the checkpoint, and the template arguments of each mode); DeepSeek-V4.1-Flash-1017 comes from
ModelScope (revision 9f9818127fdf, README_zh.md section 3). The text states the software, decoding, API and statistics settings
recorded in the run identities and analysis code (README_zh.md sections 2 and 6; bootstrap: p2_geometry_analysis.py 1,000
family resamples, make_v4_mechanism.py 4,000, crossmodel_role_swap.py 5,000, analyze_thinking.py / analyze_reasoning.py 10,000,
all seed 0; random control subspace: crossmodel_thinking_ablate2.py, seed 20261022 + layer).
usage: make_appendix_b_tex.py ITEMS_DIR OUT_DIR   (writes appendix_b_{en,zh}.tex)"""
import json
import sys
from pathlib import Path

ITEMS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
D = ITEMS / 'appendix_a_20261029'
M = {}
for f in ('appendix_a_render_s3.json', 'appendix_a_render_pro.json', 'appendix_a_render_pro_tokonly.json'):
    for k, v in json.loads((D / f).read_text(encoding='utf-8'))['models'].items():
        M.setdefault(k, v)
ORDER = ['Qwen/Qwen3.8-27B', 'Qwen/Qwen3.8-Flash-Next-FP8', 'Qwen/Qwen3.6-27B', 'Qwen/Qwen3.6-35B-A3B', 'Qwen/Qwen3.5-27B',
         'zai-org/GLM-4.6V-Flash', 'zai-org/GLM-4.7-Flash', 'zai-org/GLM-5.3-Flash', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
         'openai/gpt-oss-20b', 'openai/gpt-oss-120b', 'google/gemma-4-31B-it', 'ByteDance-Seed/Seed-OSS-36B-Instruct',
         'LGAI-EXAONE/EXAONE-4.0.1-32B', 'ibm-granite/granite-4.2-30b', 'ibm-granite/granite-4.1-30b',
         'allenai/OLMo-2-1124-13B-Instruct', 'allenai/OLMo-2-1124-7B-Instruct',
         'mistralai/Ministral-3-14B-Instruct-2512-BF16', 'mistralai/Ministral-3-14B-Reasoning-2512']
LANG = {'en': {'zh': 'Chinese', 'en': 'English', 'both': 'Chinese, English', 'default': 'default'},
        'zh': {'zh': '中文', 'en': '英文', 'both': '中文、英文', 'default': '默认'}}


def tt(s):
    return '\\texttt{' + s.replace('_', '\\_') + '}'


def args(kw, L):
    if kw is None:
        return '--'
    if not kw:
        return LANG[L]['default']
    return tt(', '.join(f'{k}={v}' for k, v in kw.items()))


def commit(rec):
    c = {v['commit'] for k, v in rec['files'].items() if k in ('chat_template.jinja', 'tokenizer_config.json') and v['commit']}
    assert len(c) == 1, c
    return c.pop()[:12]


def rows(L):
    out = []
    for repo in ORDER:
        rec = M[repo]
        modes = {**rec.get('main', {}), **rec.get('natural', {})}
        nt = modes.get('non-thinking', {}).get('kwargs') if 'non-thinking' in modes else None
        th = modes.get('thinking', {}).get('kwargs') if 'thinking' in modes else None
        out.append(f'{tt(repo)} & {tt(commit(rec))} & {args(nt, L)} & {args(th, L)} \\\\')
        if repo == 'Qwen/Qwen3.8-27B':  # format ablation (Section 5.2): the arguments recorded in the identity files of its runs (10-30)
            tag = ' (format ablation)' if L == 'en' else '（格式消融）'
            nt_abl = '\\texttt{enable\\_thinking=\\allowbreak False, preserve\\_thinking=\\allowbreak False}'  # breakable in the narrow column
            out.append(f'{tt(repo)}{tag} & {tt(commit(rec))} & {nt_abl} & {tt("preserve_thinking=False")} \\\\')
        if repo == 'allenai/OLMo-2-1124-7B-Instruct':  # DeepSeek-V4.1-Flash before the two Ministral models, as in Appendix C
            enc = ('\\texttt{thinking\\_mode=chat}', '\\texttt{thinking\\_mode=thinking, reasoning\\_effort=high}')
            out.append(f'{tt("deepseek-ai/DeepSeek-V4.1-Flash-1017")} & {tt("9f9818127fdf")}$^{{a}}$ & {enc[0]} & {enc[1]} \\\\')
    return '\n'.join(out)


def build(L):
    if L == 'en':
        head = ('Table~\\ref{tab:self_config} identifies every checkpoint by its repository and the revision of its files; all files of '
                'a checkpoint come from that one revision, and the weights are used in the format of the release '
                '(Table~\\ref{tab:self_models}). The table also gives the template arguments of each mode, with which the items of '
                'Appendix~\\ref{app:templates} are rendered (``default\'\': no argument).\n\n')
        cap = ('Checkpoints and template arguments. Revision: the first 12 characters of the commit. '
               '$^{a}$~ModelScope revision; the files are identical to those on Hugging Face. '
               'DeepSeek-V4.1-Flash has no chat template, and its arguments are those of the prompt encoder of the release.')
        hdr = 'Repository & Revision & Without thinking & With thinking \\\\'
        text = ('\\paragraph{Software.} Local runs use PyTorch (2.5.1 to 2.11.0) and Hugging Face Transformers (5.8.0 or 5.19.0); the '
                'identity file of each run in the data package records the versions it used. DeepSeek-V4.1-Flash has no Transformers '
                'implementation and runs on the model code of its release, with the custom kernels of the release replaced by PyTorch '
                'implementations of the same arithmetic and the weights kept in the release format (FP8, with FP4 expert weights). '
                'Both Ministral-3-14B models run in BF16.\n\n'
                '\\paragraph{Decoding and APIs.} In thinking mode, local runs generate greedily, up to 4,096 tokens for Qwen3.8-27B in '
                'Section~\\ref{sec:res_thinking} and up to 3,072 tokens elsewhere. The thinking-mode '
                'self-reference of two models comes from the developer\'s API with the same inputs: GLM-5.3-Flash through the Z.ai API '
                '(model \\texttt{glm-5.3-flash}, temperature 0, at most 3,072 tokens), and DeepSeek-V4.1-Flash through the DeepSeek API '
                '(model \\texttt{deepseek-flash}, which the provider documents as served by DeepSeek-V4.1-Flash-1017; reasoning effort '
                'high, temperature 1.0 and top-$p$ 1.0, the settings the developer recommends, since the thinking mode offers no greedy '
                'decoding; at most 3,072 tokens). For every request, the number of prompt tokens reported by the API equals that of the '
                'local rendering.\n\n'
                '\\paragraph{Statistics.} All intervals are 95\\% intervals with a fixed seed: percentile bootstrap intervals over wording '
                'families for the geometry (1,000 resamples), over bodies or items elsewhere (4,000 to 10,000 resamples), and over models '
                'for the rank correlations of Appendix~\\ref{app:benchmarks} (10,000 resamples); the shares of finished items in '
                'Figure~\\ref{fig:self_mechanism} carry Wilson intervals. Each subspace of the replacement is fitted on the discovery bodies '
                'and stored with the random subspace of its control (the orthonormalized columns of a Gaussian matrix, drawn per layer with '
                'a fixed seed); every test that uses it reads the stored coordinates.\n')
    else:
        head = ('表~\\ref{tab:self_config} 用仓库和修订版本标识每个检查点：一个检查点的全部文件都来自同一个修订版本，权重按发布格式使用'
                '（表~\\ref{tab:self_models}）。表中还给出每种模式的模板参数，附录~\\ref{app:templates} 的题目就按这些参数渲染（“默认”：不传参数）。\n\n')
        cap = ('检查点与模板参数。修订版本：提交号的前 12 位。$^{a}$~ModelScope 修订版本，文件与 Hugging Face 上的相同。'
               'DeepSeek-V4.1-Flash 没有对话模板，其参数是发布附带的提示编码器的参数。')
        hdr = '仓库 & 修订版本 & 不思考 & 思考 \\\\'
        text = ('\\paragraph{软件。} 本地运行使用 PyTorch（2.5.1 至 2.11.0）与 Hugging Face Transformers（5.8.0 或 5.19.0），'
                '每次运行所用的版本记录在数据包中该运行的身份文件里。DeepSeek-V4.1-Flash 没有 Transformers 实现，运行于其发布附带的模型代码，'
                '其中的自定义内核换成按相同算术规则实现的 PyTorch 版本，权重保持发布格式（FP8，专家权重为 FP4）。两个 Ministral-3-14B 模型以 BF16 运行。\n\n'
                '\\paragraph{解码与 API。} 本地运行在思考模式下贪心生成：第~\\ref{sec:res_thinking} 节的 Qwen3.8-27B 最多 4,096 个 token，'
                '其余最多 3,072 个。两个模型的思考模式自指来自开发方 API，输入相同：'
                'GLM-5.3-Flash 用 Z.ai API（模型 \\texttt{glm-5.3-flash}，temperature 0，最多 3,072 个 token）；DeepSeek-V4.1-Flash 用 DeepSeek API'
                '（模型 \\texttt{deepseek-flash}，服务方文档写明由 DeepSeek-V4.1-Flash-1017 提供；推理强度 high，temperature 1.0、top-$p$ 1.0，'
                '即开发方推荐的设置，因为思考模式不提供贪心解码；最多 3,072 个 token）。每个请求中 API 报告的提示 token 数都与本地渲染相同。\n\n'
                '\\paragraph{统计。} 所有区间都是固定随机种子的 95\\% 区间：几何为按措辞族重抽样的百分位 bootstrap 区间（1,000 次），其余按正文或题目重抽样'
                '（4,000 至 10,000 次），附录~\\ref{app:benchmarks} 的秩相关按模型重抽样（10,000 次）；图~\\ref{fig:self_mechanism} 中已完成题目的比例用 Wilson 区间。'
                '替换所用的每个子空间都在发现组正文上拟合，并与其对照用的随机子空间（高斯矩阵各列正交化，每层用固定种子抽取）一起保存；'
                '用到它的每项检验都读取保存的坐标。\n')
    table = ('\\begin{table}[htbp]\n\\centering\n\\caption{' + cap + '}\n\\label{tab:self_config}\n\\scriptsize\n'
             '\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{llp{3.2cm}p{3.5cm}}\n\\toprule\n' + hdr + '\n\\midrule\n'
             + rows(L) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n')
    return head + table + text


OUT.mkdir(parents=True, exist_ok=True)
for L in ('en', 'zh'):
    (OUT / f'appendix_b_{L}.tex').write_text(build(L), encoding='utf-8')
print(rows('en'))
