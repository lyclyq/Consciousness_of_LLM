"""Next-token prediction (Section "The Self-Index in Next-Token Prediction", Appendix "Next-Token Prediction"): the index of the
next-token runs and the relation between the next-token separation and the self-index, from the run summaries of
selfindex.next_token and selfindex.next_token_swap in the next-token archive (evidence/p2_modeswitch_20261107,
results/<group>/modeswitch/...). Three analyses, run in this order (--only NAME[,NAME] runs some of them):

index (KL_INDEX_zh.md, in the archive's language): every next-token run by kind, with its status (DONE or the bodies on disk) and
  its main numbers: A. the full distributions, main and natural order (<setting>/); B. control and content tokens apart, main
  order (ctrl/<setting>/); C. the same in natural order (nat/<setting>/); D. the templates that accept only the natural order
  (nat3/<setting>/, Table tab:self_natural3); E. the replacement of the eight coordinates on the body (swap/<readout>/, content
  divergences: the share left after removal, the share closed by installation, the random controls and the content ratio after
  the edit); F. the format ablation of Qwen3.8-27B (qwen38_{nt,think}_switch/); G. the replacement runs of the fifth causal
  readout, MiMo-V2.6-Distill-Qwen-9B (thinking_swap/mimo_v26_*: the changes of the "you" share); H. a list of the other files
  of the archive. Each setting is tagged with the groups it belongs to: the mechanism group (the five causal readouts), the main
  table (the 25 models of the main comparison, one setting per model, non-thinking rendering where the template has one) and the
  natural-order list (the smallest model per developer in the natural-order appendix).
ratio_distribution (results/kl_ratio_distribution_20261110.md): per model of the main comparison, the two content ratios
  KL(A||U)/KL(U||S) and KL(A||S)/KL(U||S) of the main probe, the smaller of the two with its own 95% family interval, the paper's
  self-index group (13 with a self-index, 12 without) and the share of message-content layers that carry the index (Table
  tab:self_crossmodel; four-stream models: mean of the streams); the ranges by group and Spearman's rho between the share and the
  smaller ratio; then the natural order (Table tab:self_natural: full ratios; Table tab:self_natural3: content ratios).
boundary_stats (results/kl_boundary_stats_20261110.md): without a threshold set in advance, (1) Spearman's rho between the
  content-index share and the smaller content ratio with a one-sided permutation p (100,000 shuffles, seed 1); (2) one- against
  two-component Gaussian mixtures on log10(smaller ratio) (EM, 50 random starts, seed 1), the BIC of each and the point between
  the two means where the posterior is 0.5; (3) the models on each side and their content-index shares. Main probe (25 models)
  and natural order (13 models).
The markdown outputs keep the wording of the archived files (Chinese).

usage: next_token.py --results RES --out OUT [--only index,ratio_distribution,boundary_stats]
"""
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MS, Roots, arguments, result_path, write_text  # noqa: E402


def load(d):
    f = Path(d) / 'summary.json'
    return json.loads(f.read_text(encoding='utf-8')) if f.exists() else None


def subdirs(M, sub):
    """setting folders under results/<group>/modeswitch/<sub> of every group, by name (a folder name may repeat across groups only
    for folders that are not runs, such as ctrl/ or nat3/ themselves; a run is found under one group)"""
    out = {}
    for g in sorted(glob.glob(str(Path(M) / 'results' / '*' / 'modeswitch'))):
        b = Path(g) / sub if sub else Path(g)
        if b.exists():
            for x in b.iterdir():
                if x.is_dir():
                    if x.name in out:
                        assert not ((x / 'identity.json').exists() and (out[x.name] / 'identity.json').exists()), \
                            f'{sub}/{x.name} under more than one group'
                        if not (x / 'identity.json').exists():
                            continue
                    out[x.name] = x
    return [out[k] for k in sorted(out)]


# ---------------------------------------------------------------------------------------------------------------------------
# index

MECH = {'qwen38_nt', 'qwen38_think', 'gpt_oss_20b_en', 'gpt_oss_120b_en', 'mimo_v26_default'}
MAIN = {'qwen38_nt': 'Qwen3.8-27B', 'flashnext_nt': 'Qwen3.8-Flash-Next', 'qwen36_27b': 'Qwen3.6-27B', 'qwen36_35b_a3b': 'Qwen3.6-35B-A3B',
        'qwen35_27b': 'Qwen3.5-27B', 'glm46v_flash': 'GLM-4.6V-Flash', 'glm47_flash': 'GLM-4.7-Flash', 'glm53_flash': 'GLM-5.3-Flash',
        'r1_distill_llama70b': 'DeepSeek-R1-Distill-Llama-70B', 'gpt_oss_20b_en': 'gpt-oss-20b', 'gpt_oss_120b_en': 'gpt-oss-120b',
        'gemma4_31b_en': 'Gemma-4-31B', 'seed_oss_36b': 'Seed-OSS-36B', 'exaone4_32b_en': 'EXAONE-4.0.1-32B', 'granite42_30b_en': 'Granite-4.2-30B',
        'granite41_30b_en': 'Granite-4.1-30B', 'olmo2_13b_en': 'OLMo-2-13B', 'olmo2_7b_en': 'OLMo-2-7B', 'ernie45_21b_a3b_thinking': 'ERNIE-4.5-21B-A3B',
        'mimo7b_rl_0530': 'MiMo-7B-RL', 'mimo_vl7b': 'MiMo-VL-7B', 'mimo_v26_nt': 'MiMo-V2.6-Distill-9B', 'moonlight16b_a3b_instruct': 'Moonlight-16B-A3B',
        'apertus8b_en': 'Apertus-8B', 'phi4_en': 'phi-4'}
NATLIST = {'qwen38_nt', 'qwen38_think', 'gpt_oss_20b_en', 'apertus8b_en', 'olmo2_7b_en', 'mimo7b_rl_0530', 'glm46v_flash', 'phi4_en',
           'moonlight16b_a3b_instruct', 'ernie45_21b_a3b_thinking', 'granite41_30b_en', 'gemma4_31b_en', 'exaone4_32b_en', 'seed_oss_36b',
           'r1_distill_llama70b'}
NAT3 = {'ministral3_14b_instruct_en': 'Ministral-3-14B-Instruct', 'ministral3_14b_reasoning_en': 'Ministral-3-14B-Reasoning',
        'dsv41_flash': 'DeepSeek-V4.1-Flash'}
FORMAT = {'qwen38_nt_switch', 'qwen38_think_switch'}


def status(d, n=150):
    d = Path(d)
    if (d / 'DONE').exists():
        return 'DONE'
    b = d / 'bodies'
    return f'未完成 {len(list(b.glob("*.json")))}/{n}' if b.exists() else '未开始'


def rc(v):
    if v is None:
        return '—'
    if isinstance(v, dict) and 'ratio' in v:
        return f"{v['ratio']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]"
    if isinstance(v, dict) and 'value' in v:
        return f"{v['value']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]"
    return f'{v:.3f}'


def group(tag):
    g = []
    if tag in MECH:
        g.append('机制组')
    if tag in MAIN:
        g.append('主表')
    if tag in NATLIST:
        g.append('自然顺序名单')
    if tag in FORMAT:
        g.append('格式消融（机制部分）')
    return '、'.join(g) or '附带'


def index(M, O):
    ms = lambda rel: result_path(M, f'modeswitch/{rel}')  # noqa: E731
    L = ['# 模式切换（KL）存档：分类索引', '',
         '由 `code/local/make_kl_index_20261108.py` 从本地镜像 `results/pro/` 和 `results/server3/` 生成。每次同步镜像后重跑。镜像核对清单：`SHA256SUMS_pro_20261108.txt`、`SHA256SUMS_server3_20261109.txt`（路径相对本存档）。', '',
         'GLM-5.3-Flash 和 DeepSeek-V4.1-Flash 在 server3 上跑，结果在 `results/server3/modeswitch/`，目录结构与 PRO 相同；下表的路径写的是 `results/pro/`，这两个设置对应换成 `results/server3/`。', '',
         '归属规则（用户 11-07/11-08）：',
         '- 机制组：Qwen3.8 非思考、思考，gpt-oss-20b、120b，以及 MiMo-V2.6 默认调用。',
         '- 主表：论文主比较中的 25 个模型，每个模型一个设置，模板有非思考模式时用非思考渲染。',
         '- 自然顺序名单：附录里每家最小的模型，加上 Table tab:self_natural3 中的 3 个。',
         '- 两类都不属于的是附带结果。', '']
    L += ['## A. 全量 KL，主探针 + 自然顺序（`mode_switch_20261107.py`，`results/pro/modeswitch/<设置>/`）', '',
          '| 设置 | 归属 | 状态 | 主探针 A–U / U–S | 主探针 A–S / U–S | 自然顺序 A–U | 自然顺序 A–U / U–S | 自然顺序 A–S / U–S |', '|---|---|---|---|---|---|---|---|']
    for d in (x for x in subdirs(M, '') if (x / 'identity.json').exists() and not x.name.startswith('_') and x.name not in FORMAT):
        s = load(d) or {}
        m, n = s.get('orders', {}).get('main', {}), s.get('orders', {}).get('natural', {})
        na = n.get('pairs', {}).get('A-U', {}).get('kl_body', {}).get('mean')
        L.append(f"| {d.name} | {group(d.name)} | {status(d)} | {rc(m.get('ratios', {}).get('A-U/U-S_kl_body'))} | {rc(m.get('ratios', {}).get('A-S/U-S_kl_body'))} | "
                 f"{rc(na)} | {rc(n.get('ratios', {}).get('A-U/U-S_kl_body'))} | {rc(n.get('ratios', {}).get('A-S/U-S_kl_body'))} |")
    L += ['', '自然顺序只有主表名单之外的模型也算了出来，这部分记录在案，但不进协议表。Qwen、gpt-oss、Apertus、R1 的模板不让 System 留在对话中间，自然顺序下没有 U–S，只报 A–U 的绝对值。', '']
    L += ['## B. 控制 token 拆分，主探针（`mode_switch_ctrl_20261107.py`，`results/pro/modeswitch/ctrl/<设置>/`）', '',
          '| 设置 | 归属 | 状态 | 控制 token 概率 A / U / S | 全量 A–U / U–S | 内容 A–U / U–S | 内容 A–S / U–S |', '|---|---|---|---|---|---|---|']
    for d in subdirs(M, 'ctrl'):
        s = load(d) or {}
        pc = ' / '.join(f"{s['roles'][r]['p_ctrl']['mean']:.3f}" for r in ('A', 'U', 'S') if r in s.get('roles', {})) if s else '—'
        r = s.get('ratios', {})
        L.append(f"| {d.name} | {group(d.name)} | {status(d)} | {pc} | {rc(r.get('A-U/U-S_kl_body'))} | {rc(r.get('A-U/U-S_kl_content'))} | {rc(r.get('A-S/U-S_kl_content'))} |")
    L += ['']
    L += ['## C. 控制 token 拆分，自然顺序（`mode_switch_nat_20261108.py --order natural`，`results/pro/modeswitch/nat/<设置>/`）', '',
          '| 设置 | 状态 | 控制 token 概率 A / U | 全量 A–U | 内容 A–U |', '|---|---|---|---|---|']
    for d in subdirs(M, 'nat'):
        s = load(d) or {}
        pc = ' / '.join(f"{s['roles'][r]['p_ctrl']['mean']:.4f}" for r in ('A', 'U') if r in s.get('roles', {})) if s else '—'
        p = s.get('pairs', {}).get('A-U', {})
        L.append(f"| {d.name} | {status(d)} | {pc} | {rc(p.get('kl_body', {}).get('mean'))} | {rc(p.get('kl_content', {}).get('mean'))} |")
    L += ['', '其余模型在主探针上的控制 token 概率约为 0（见 B），自然顺序下的内容 KL 与全量 KL 相同，未单独拆分。', '']
    L += ['## D. 只能用自然顺序的模板（Table tab:self_natural3，`mode_switch_nat_20261108.py --order natural3`，`results/pro/modeswitch/nat3/<设置>/`）', '',
          '| 设置 | 模型 | 状态 | 全量 A–U / U–S | 内容 A–U / U–S | 内容 A–S / U–S |', '|---|---|---|---|---|---|']
    for tag, name in NAT3.items():
        d = ms(f'nat3/{tag}')
        s = load(d) or {}
        r = s.get('ratios', {})
        st = status(d) if d.exists() else '排队中'
        L.append(f"| {tag} | {name} | {st} | {rc(r.get('A-U/U-S_kl_body'))} | {rc(r.get('A-U/U-S_kl_content'))} | {rc(r.get('A-S/U-S_kl_content'))} |")
    L += ['']
    L += ['## E. KL 替换消融（`mode_switch_swap_20261108.py`，`results/pro/modeswitch/swap/<读出>/`，留出 75 段，内容 KL）', '',
          '| 读出 | 状态 | 剩余 own←User | 剩余 own←System | 拉近 User←own | 拉近 System←own | 随机：剩余 / 拉近 | 编辑后 A–U / U–S |', '|---|---|---|---|---|---|---|---|']
    for t in ('qsp', 'qth', 'g20', 'g120', 'm26'):
        d = ms(f'swap/{t}')
        s = load(d)
        if not s:
            L.append(f'| {t} | {status(d) if d.exists() else "未开始"} | — | — | — | — | — | — |')
            continue
        lf, mv = s['left']['content'], s['moved']['content']
        L.append(f"| {t} | {status(d, 75)} | {rc(lf['A_U'])} | {rc(lf['A_S'])} | {rc(mv['U_A'])} | {rc(mv['S_A'])} | {lf['A_Ur']['value']:.2f} / {mv['U_Ar']['value']:.2f} | "
                 f"{rc(s['asym']['content']['A_U||U/U||S'])} |")
    L += ['', '读出：qsp = Qwen3.8 非思考，qth = Qwen3.8 思考，g20 / g120 = gpt-oss-20b / 120b，m26 = MiMo-V2.6 默认调用。拟合在 `r05r06_fits/<读出>/fit_m.pt`；m26 的拟合已镜像到本存档 `results/pro/r05r06_fits/m26/`（若已同步），其余 4 个读出的拟合在 `evidence/paper_data_v4`。烟雾测试 `_smoke_g20` 不进结果。', '']
    L += ['## F. 格式消融（机制部分，不进主表）：`results/pro/modeswitch/qwen38_{nt,think}_switch/`', '']
    for t in sorted(FORMAT):
        s = load(ms(t)) or {}
        m = s.get('orders', {}).get('main', {}).get('ratios', {})
        L.append(f"- {t}（{status(ms(t))}）：主探针 A–U / U–S {rc(m.get('A-U/U-S_kl_body'))}，A–S / U–S {rc(m.get('A-S/U-S_kl_body'))}")
    L += ['']
    L += ['## G. MiMo-V2.6 机制读出（归因与对照，不是 KL；`results/pro/thinking_swap/mimo_v26_*`、`results/pro/games_swap/mimo_v26_*`）', '',
          '| 运行 | 条件：回答"你"比例的变化 [95%] |', '|---|---|']
    for d in sorted(Path(x) for x in glob.glob(str(Path(M) / 'results' / '*' / 'thinking_swap' / 'mimo_v26_*'))):
        if not d.is_dir():
            continue
        s = load(d)
        if not s:
            continue
        if any('you_minus_base' in v for v in s.values()):
            cells = [f"{k} {v['you_minus_base']['mean']:+.3f} [{v['you_minus_base']['ci95'][0]:+.2f}, {v['you_minus_base']['ci95'][1]:+.2f}]" for k, v in s.items() if 'you_minus_base' in v]
        else:
            cells = [f"{k} 正确率 {v.get('content_accuracy', float('nan')):.2f}" for k, v in s.items()]
        L.append(f"| {d.name} | {'；'.join(cells)} |")
    L += ['', '秘密数字游戏 v5：烟雾测试没过事先定的放行规则，没有跑全量（`games_swap/mimo_v26_s5_smoke_*`）。论文措辞见 README。', '']
    L += ['## H. 诊断与日志', '',
          '- gpt-oss-120b 的正文预测诊断：`results/pro/modeswitch/g120_diag.log`、`g_gen_120b.log`、`g_gen_20b.log`（脚本 `code/pro/g120_logit_diag_20261108.py`、`g_gen_diag_20261108.py`）。',
          '- 每个模型跑之前的渲染存档和只渲染检查：`results/pro/modeswitch/nat10/<设置>.render.txt`、`.dry.log`，以及 `results/pro/modeswitch/main6/`。',
          '- 下载记录：`results/pro/modeswitch/{nat10,main6}/dl_*.log`。',
          '- 队列日志：`results/pro/modeswitch/queue.log`（总），`nat10/queue_nat10.log`，`main6/queue_main6.log`，`swap/queue_swap.log`，`mech_m26/status.log`。',
          '- 输入：`inputs/`（正文、英文译文、游戏题目）。服务器依赖脚本：`code/pro_deps/`。本存档写的脚本：`code/pro/`。',
          '- 11-08 之前零散拉回的 summary 已核对与镜像一致（39 份），退役到 `tmp/retired_results_partial_20261108/`。', '']
    out = O / 'KL_INDEX_zh.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')
    print('written', out, len(L), 'lines')


# ---------------------------------------------------------------------------------------------------------------------------
# ratio_distribution

# setting, name, paper self-index group, content coverage (Table tab:self_crossmodel, content column), four-stream
MODELS = [('qwen38_nt', 'Qwen3.8-27B', 1, (64, 64), 0), ('gpt_oss_120b_en', 'gpt-oss-120b', 1, (25, 36), 0), ('gpt_oss_20b_en', 'gpt-oss-20b', 1, (20, 24), 0),
          ('flashnext_nt', 'Qwen3.8-Flash-Next', 1, (38, 48), 1), ('glm53_flash', 'GLM-5.3-Flash', 1, (18, 45), 1), ('mimo_v26_nt', 'MiMo-V2.6-Distill-9B', 1, (32, 32), 0),
          ('exaone4_32b_en', 'EXAONE-4.0.1-32B', 1, (64, 64), 0), ('glm46v_flash', 'GLM-4.6V-Flash', 1, (40, 40), 0), ('glm47_flash', 'GLM-4.7-Flash', 1, (34, 47), 0),
          ('r1_distill_llama70b', 'R1-Distill-Llama-70B', 1, (0, 80), 0), ('phi4_en', 'phi-4', 1, (35, 40), 0), ('granite42_30b_en', 'Granite-4.2-30B', 1, (30, 64), 0),
          ('granite41_30b_en', 'Granite-4.1-30B', 1, (17, 64), 0), ('olmo2_13b_en', 'OLMo-2-13B', 0, (4, 40), 0), ('olmo2_7b_en', 'OLMo-2-7B', 0, (5, 32), 0),
          ('apertus8b_en', 'Apertus-8B', 0, (0, 32), 0), ('moonlight16b_a3b_instruct', 'Moonlight-16B-A3B', 0, (0, 27), 0), ('mimo7b_rl_0530', 'MiMo-7B-RL', 0, (0, 36), 0),
          ('mimo_vl7b', 'MiMo-VL-7B', 0, (0, 36), 0), ('ernie45_21b_a3b_thinking', 'ERNIE-4.5-21B-A3B', 0, (0, 28), 0), ('seed_oss_36b', 'Seed-OSS-36B', 0, (0, 64), 0),
          ('gemma4_31b_en', 'Gemma-4-31B', 0, (0, 60), 0), ('qwen36_27b', 'Qwen3.6-27B', 0, (6, 64), 0), ('qwen36_35b_a3b', 'Qwen3.6-35B-A3B', 0, (1, 40), 0),
          ('qwen35_27b', 'Qwen3.5-27B', 0, (13, 64), 0)]
NAT = [('exaone4_32b_en', 'EXAONE-4.0.1-32B', '64/64'), ('glm46v_flash', 'GLM-4.6V-Flash', '36/40'), ('granite41_30b_en', 'Granite-4.1-30B', '61/64'),
       ('moonlight16b_a3b_instruct', 'Moonlight-16B-A3B', '27/27'), ('mimo7b_rl_0530', 'MiMo-7B-RL', '34/36'), ('phi4_en', 'phi-4', '25/40'),
       ('ernie45_21b_a3b_thinking', 'ERNIE-4.5-21B-A3B', '16/28'), ('olmo2_7b_en', 'OLMo-2-7B', '5/32'), ('seed_oss_36b', 'Seed-OSS-36B', '3/64'),
       ('gemma4_31b_en', 'Gemma-4-31B', '0/60')]
NAT_ONLY = (('dsv41_flash', 'DeepSeek-V4.1-Flash（只接受自然顺序）', '38/40'), ('ministral3_14b_instruct_en', 'Ministral-3-14B-Instruct', '0/40'),
            ('ministral3_14b_reasoning_en', 'Ministral-3-14B-Reasoning', '0/40'))


def summary(M, rel):
    f = result_path(M, f'modeswitch/{rel}') / 'summary.json'
    if not f.exists():
        raise FileNotFoundError(rel)
    return json.loads(f.read_text(encoding='utf-8'))


def distribution(M):
    """rows (main probe, sorted by the smaller content ratio) and nat (natural order: (smaller ratio, name, content index, interval))"""
    rows = []
    for tag, name, idx, (c, n), hc in MODELS:
        r = summary(M, f'ctrl/{tag}')['ratios']
        a, b = r['A-U/U-S_kl_content'], r['A-S/U-S_kl_content']
        lo = a if a['ratio'] <= b['ratio'] else b
        rows.append(dict(name=name, idx=idx, cov=c / n, cn=f'{c}/{n}', hc=hc, au=a['ratio'], as_=b['ratio'], mn=lo['ratio'], ci=lo['ci95'],
                         both=a['ci95'][0] > 1 and b['ci95'][0] > 1, one=(a['ci95'][0] > 1) + (b['ci95'][0] > 1)))
    rows.sort(key=lambda x: x['mn'])
    nat = []
    for tag, name, ci in NAT:
        r = summary(M, tag)['orders']['natural']['ratios']
        a, b = r['A-U/U-S_kl_body'], r['A-S/U-S_kl_body']
        lo = a if a['ratio'] <= b['ratio'] else b
        nat.append((lo['ratio'], name, ci, lo['ci95']))
    for tag, name, ci in NAT_ONLY:
        r = summary(M, f'nat3/{tag}')['ratios']
        a, b = r['A-U/U-S_kl_content'], r['A-S/U-S_kl_content']
        lo = a if a['ratio'] <= b['ratio'] else b
        nat.append((lo['ratio'], name, ci, lo['ci95']))
    return rows, nat


def rank_avg(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return rk


def spearman_py(x, y):
    rx, ry = rank_avg(x), rank_avg(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    den = (sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry)) ** 0.5
    return num / den


def ratio_distribution(M, O):
    rows, nat = distribution(M)
    L = ['# KL 比值在 25 个模型上的分布（11-10）', '', '主探针，内容。较小比值 = min(A–U/U–S, A–S/U–S)，区间是该比值自己的家族重抽样 95% 区间。索引分组按论文（13 / 12）。'
         '正文位置索引覆盖 = 主表 content 列（四流模型为四流平均）。', '',
         '| 模型 | 论文分组 | 正文位置索引 | A–U/U–S | A–S/U–S | 较小比值 [95%] | 过几个 |', '|---|---|---|---|---|---|---|']
    for x in rows:
        L.append(f"| {x['name']}{' (四流)' if x['hc'] else ''} | {'有索引' if x['idx'] else '无索引'} | {x['cn']} ({x['cov']:.0%}) | {x['au']:.2f} | {x['as_']:.2f} | "
                 f"{x['mn']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}] | {x['one']} |")
    noidx = [x['mn'] for x in rows if not x['idx']]
    both = [x['mn'] for x in rows if x['both']]
    rest_idx = [x['mn'] for x in rows if x['idx'] and not x['both']]
    L += ['', f"- 无索引 12 个的较小比值：{min(noidx):.2f}–{max(noidx):.2f}；两个比值都过的 8 个：{min(both):.2f}–{max(both):.2f}；其余 5 个有索引的：{min(rest_idx):.2f}–{max(rest_idx):.2f}。",
          f"- 25 个模型中，较小比值落在 {max(x['mn'] for x in rows if x['mn'] < 1.5):.2f} 与 {min(x['mn'] for x in rows if x['mn'] >= 1.5):.2f} 之间的没有。",
          f"- Spearman(正文位置索引覆盖, log 较小比值)，25 个模型：{spearman_py([x['cov'] for x in rows], [x['mn'] for x in rows]):+.2f}；"
          f"只算 23 个单流模型：{spearman_py([x['cov'] for x in rows if not x['hc']], [x['mn'] for x in rows if not x['hc']]):+.2f}。"]
    single = [x for x in rows if not x['hc']]
    L.append('- 单流模型按正文位置索引覆盖：' + '；'.join(f"{x['name']} {x['cov']:.0%}→{x['mn']:.2f}" for x in sorted(single, key=lambda x: -x['cov']) if x['cov'] > 0))
    L += ['', '## 自然顺序（完整分布；这些模型控制 token 概率约 0）', '', '| 模型 | 正文位置索引（自然顺序） | 较小比值 [95%] |', '|---|---|---|']
    for mn, name, ci, c in sorted(nat):
        L.append(f'| {name} | {ci} | {mn:.2f} [{c[0]:.2f}, {c[1]:.2f}] |')
    out = O / 'results' / 'kl_ratio_distribution_20261110.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')
    print('\n'.join(L))
    print('written', out)


# ---------------------------------------------------------------------------------------------------------------------------
# boundary_stats

def rank_np(v):
    o = np.argsort(v, kind='mergesort')
    r = np.empty(len(v))
    r[o] = np.arange(1, len(v) + 1)
    for x in set(v):  # average ties
        idx = [i for i, y in enumerate(v) if y == x]
        if len(idx) > 1:
            r[idx] = np.mean(r[idx])
    return r


def spearman_np(x, y):
    a, b = rank_np(list(x)), rank_np(list(y))
    return float(np.corrcoef(a, b)[0, 1])


def perm_p(x, y, n=100000, seed=1):
    rng = np.random.default_rng(seed)
    r0 = spearman_np(x, y)
    a, b = rank_np(list(x)), rank_np(list(y))
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    cnt = 0
    for _ in range(n):
        if np.mean(a * rng.permutation(b)) >= r0 - 1e-12:
            cnt += 1
    return r0, (cnt + 1) / (n + 1)


def gmm(z, k, starts=50, seed=1):
    rng = np.random.default_rng(seed)
    best = None
    n = len(z)
    for _ in range(starts):
        mu = rng.choice(z, k, replace=False).astype(float)
        sd = np.full(k, z.std() + 1e-3)
        w = np.full(k, 1 / k)
        for _ in range(500):
            dens = np.array([w[j] * np.exp(-0.5 * ((z - mu[j]) / sd[j]) ** 2) / (sd[j] * math.sqrt(2 * math.pi)) for j in range(k)])
            tot = dens.sum(0) + 1e-300
            g = dens / tot
            nk = g.sum(1)
            if np.any(nk < 1e-6):
                break
            w = nk / n
            mu = (g * z).sum(1) / nk
            sd = np.sqrt((g * (z - mu[:, None]) ** 2).sum(1) / nk) + 1e-3
        ll = float(np.log(tot).sum())
        if best is None or ll > best[0]:
            best = (ll, mu.copy(), sd.copy(), w.copy())
    ll, mu, sd, w = best
    bic = -2 * ll + (3 * k - 1) * math.log(n)
    return ll, bic, mu, sd, w


def boundary(mu, sd, w):
    lo, hi = sorted(range(2), key=lambda j: mu[j])
    xs = np.linspace(mu[lo], mu[hi], 20001)
    p = lambda j: w[j] * np.exp(-0.5 * ((xs - mu[j]) / sd[j]) ** 2) / sd[j]  # noqa: E731
    d = p(hi) / (p(lo) + p(hi))
    return float(xs[np.argmin(abs(d - 0.5))])


def boundary_stats(M, O):
    rows, nat = distribution(M)
    L = ['# KL 比值的统计边界（11-10）', '', '没有事先设阈值。较小比值 = min(A–U/U–S, A–S/U–S)，内容；z = log10(较小比值)。', '']
    for lab, its in (('主探针，25 个模型', [(x['name'] + (' (四流)' if x['hc'] else ''), x['cov'], x['mn']) for x in rows]),
                     ('自然顺序，13 个模型', [(n, int(c.split('/')[0]) / int(c.split('/')[1]), mn) for mn, n, c, _ in nat])):
        names = [a for a, _, _ in its]
        cov = np.array([b for _, b, _ in its])
        z = np.log10(np.array([c for _, _, c in its]))
        r, p = perm_p(cov, z)
        l1, b1, *_ = gmm(z, 1)
        l2, b2, mu, sd, w = gmm(z, 2)
        bd = boundary(mu, sd, w)
        hi = [(n, c, 10 ** v) for n, c, v in zip(names, cov, z) if v > bd]
        lo = [(n, c, 10 ** v) for n, c, v in zip(names, cov, z) if v <= bd]
        L += [f'## {lab}', '',
              f'- Spearman(正文位置索引覆盖, 较小比值) = {r:+.2f}，置换 p = {p:.1e}（单侧，100,000 次）。',
              f'- 单峰 BIC {b1:.1f}，双峰 BIC {b2:.1f}（ΔBIC = {b1 - b2:+.1f}，正数表示双峰更好）；双峰均值 {10 ** min(mu):.2f} 与 {10 ** max(mu):.2f}，'
              f'边界（后验 0.5）在比值 {10 ** bd:.2f}。',
              f'- 边界以上 {len(hi)} 个：' + '；'.join(f'{n} {c:.0%}→{v:.2f}' for n, c, v in sorted(hi, key=lambda x: x[2])),
              f'- 边界以下 {len(lo)} 个：' + '；'.join(f'{n} {c:.0%}→{v:.2f}' for n, c, v in sorted(lo, key=lambda x: x[2])),
              f'- 覆盖：边界以上 {min(c for _, c, _ in hi):.0%}–{max(c for _, c, _ in hi):.0%}，边界以下 {min(c for _, c, _ in lo):.0%}–{max(c for _, c, _ in lo):.0%}。', '']
    out = O / 'results' / 'kl_boundary_stats_20261110.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')
    print('\n'.join(L))
    print('written', out)


STEPS = [('index', index), ('ratio_distribution', ratio_distribution), ('boundary_stats', boundary_stats)]


def main(argv=None):
    ap = arguments(__doc__)
    ap.add_argument('--only', help='comma-separated: ' + ','.join(n for n, _ in STEPS))
    args = ap.parse_args(argv)
    r = Roots(args)
    M, O = r.res(MS), r.out(MS)
    only = set(args.only.split(',')) if args.only else None
    for name, fn in STEPS:
        if only is None or name in only:
            print(f'== {name}', flush=True)
            fn(M, O)


if __name__ == '__main__':
    main()
