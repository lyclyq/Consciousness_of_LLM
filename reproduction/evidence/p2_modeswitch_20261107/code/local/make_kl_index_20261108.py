"""Build KL_INDEX_zh.md, the category index of the mode-switch (KL) archive, from the local mirror results/pro (read only).
Rerun after every sync of the mirror.  usage: python -I make_kl_index_20261108.py [ARCHIVE_DIR]"""
import json
import sys
from pathlib import Path

A = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
P = A / 'results' / 'pro'
MS = P / 'modeswitch'
MS3 = A / 'results' / 'server3' / 'modeswitch'  # GLM-5.3-Flash and DeepSeek-V4.1-Flash (11-09), same layout as MS


def subdirs(sub):
    """setting dirs under modeswitch/<sub> on both machines (PRO first)"""
    out = {}
    for root in (MS3, MS):
        b = root / sub if sub else root
        if b.exists():
            out.update({x.name: x for x in b.iterdir() if x.is_dir()})
    return [out[k] for k in sorted(out)]

# protocol membership (user 11-07/11-08): mechanism group; the paper's main comparison (one setting per model, non-thinking
# rendering where the template has one); the natural-order appendix (smallest model per developer, + Table tab:self_natural3)
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
SERVER3 = {'glm53_flash', 'dsv41_flash'}


def load(d):
    f = d / 'summary.json'
    return json.loads(f.read_text(encoding='utf-8')) if f.exists() else None


def status(d, n=150):
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


L = ['# 模式切换（KL）存档：分类索引', '',
     f'由 `code/local/make_kl_index_20261108.py` 从本地镜像 `results/pro/` 和 `results/server3/` 生成。每次同步镜像后重跑。镜像核对清单：`SHA256SUMS_pro_20261108.txt`、`SHA256SUMS_server3_20261109.txt`（路径相对本存档）。', '',
     'GLM-5.3-Flash 和 DeepSeek-V4.1-Flash 在 server3 上跑，结果在 `results/server3/modeswitch/`，目录结构与 PRO 相同；下表的路径写的是 `results/pro/`，这两个设置对应换成 `results/server3/`。', '',
     '归属规则（用户 11-07/11-08）：',
     '- 机制组：Qwen3.8 非思考、思考，gpt-oss-20b、120b，以及 MiMo-V2.6 默认调用。',
     '- 主表：论文主比较中的 25 个模型，每个模型一个设置，模板有非思考模式时用非思考渲染。',
     '- 自然顺序名单：附录里每家最小的模型，加上 Table tab:self_natural3 中的 3 个。',
     '- 两类都不属于的是附带结果。', '']

# A. full KL, main + natural (mode_switch_20261107.py)
L += ['## A. 全量 KL，主探针 + 自然顺序（`mode_switch_20261107.py`，`results/pro/modeswitch/<设置>/`）', '',
      '| 设置 | 归属 | 状态 | 主探针 A–U / U–S | 主探针 A–S / U–S | 自然顺序 A–U | 自然顺序 A–U / U–S | 自然顺序 A–S / U–S |', '|---|---|---|---|---|---|---|---|']
for d in (x for x in subdirs('') if (x / 'identity.json').exists() and not x.name.startswith('_') and x.name not in FORMAT):
    s = load(d) or {}
    m, n = s.get('orders', {}).get('main', {}), s.get('orders', {}).get('natural', {})
    na = n.get('pairs', {}).get('A-U', {}).get('kl_body', {}).get('mean')
    L.append(f"| {d.name} | {group(d.name)} | {status(d)} | {rc(m.get('ratios', {}).get('A-U/U-S_kl_body'))} | {rc(m.get('ratios', {}).get('A-S/U-S_kl_body'))} | "
             f"{rc(na)} | {rc(n.get('ratios', {}).get('A-U/U-S_kl_body'))} | {rc(n.get('ratios', {}).get('A-S/U-S_kl_body'))} |")
L += ['', '自然顺序只有主表名单之外的模型也算了出来，这部分记录在案，但不进协议表。Qwen、gpt-oss、Apertus、R1 的模板不让 System 留在对话中间，自然顺序下没有 U–S，只报 A–U 的绝对值。', '']

# B. control-token split, main order (mode_switch_ctrl_20261107.py)
L += ['## B. 控制 token 拆分，主探针（`mode_switch_ctrl_20261107.py`，`results/pro/modeswitch/ctrl/<设置>/`）', '',
      '| 设置 | 归属 | 状态 | 控制 token 概率 A / U / S | 全量 A–U / U–S | 内容 A–U / U–S | 内容 A–S / U–S |', '|---|---|---|---|---|---|---|']
for d in subdirs('ctrl'):
    s = load(d) or {}
    pc = ' / '.join(f"{s['roles'][r]['p_ctrl']['mean']:.3f}" for r in ('A', 'U', 'S') if r in s.get('roles', {})) if s else '—'
    r = s.get('ratios', {})
    tag = d.name.replace('flashnext_think', 'flashnext_think')
    L.append(f"| {d.name} | {group(tag)} | {status(d)} | {pc} | {rc(r.get('A-U/U-S_kl_body'))} | {rc(r.get('A-U/U-S_kl_content'))} | {rc(r.get('A-S/U-S_kl_content'))} |")
L += ['']

# C. control-token split, natural order (mode_switch_nat_20261108.py --order natural)
L += ['## C. 控制 token 拆分，自然顺序（`mode_switch_nat_20261108.py --order natural`，`results/pro/modeswitch/nat/<设置>/`）', '',
      '| 设置 | 状态 | 控制 token 概率 A / U | 全量 A–U | 内容 A–U |', '|---|---|---|---|---|']
for d in sorted(x for x in (MS / 'nat').iterdir() if x.is_dir()) if (MS / 'nat').exists() else []:
    s = load(d) or {}
    pc = ' / '.join(f"{s['roles'][r]['p_ctrl']['mean']:.4f}" for r in ('A', 'U') if r in s.get('roles', {})) if s else '—'
    p = s.get('pairs', {}).get('A-U', {})
    L.append(f"| {d.name} | {status(d)} | {pc} | {rc(p.get('kl_body', {}).get('mean'))} | {rc(p.get('kl_content', {}).get('mean'))} |")
L += ['', '其余模型在主探针上的控制 token 概率约为 0（见 B），自然顺序下的内容 KL 与全量 KL 相同，未单独拆分。', '']

# D. natural3 (Table tab:self_natural3)
L += ['## D. 只能用自然顺序的模板（Table tab:self_natural3，`mode_switch_nat_20261108.py --order natural3`，`results/pro/modeswitch/nat3/<设置>/`）', '',
      '| 设置 | 模型 | 状态 | 全量 A–U / U–S | 内容 A–U / U–S | 内容 A–S / U–S |', '|---|---|---|---|---|---|']
for tag, name in NAT3.items():
    d = MS / 'nat3' / tag if (MS / 'nat3' / tag).exists() else MS3 / 'nat3' / tag
    s = load(d) or {}
    r = s.get('ratios', {})
    st = status(d) if d.exists() else ('等 server3' if tag in SERVER3 else '排队中')
    L.append(f"| {tag} | {name} | {st} | {rc(r.get('A-U/U-S_kl_body'))} | {rc(r.get('A-U/U-S_kl_content'))} | {rc(r.get('A-S/U-S_kl_content'))} |")
L += ['']

# E. KL replacement (mode_switch_swap_20261108.py)
L += ['## E. KL 替换消融（`mode_switch_swap_20261108.py`，`results/pro/modeswitch/swap/<读出>/`，留出 75 段，内容 KL）', '',
      '| 读出 | 状态 | 剩余 own←User | 剩余 own←System | 拉近 User←own | 拉近 System←own | 随机：剩余 / 拉近 | 编辑后 A–U / U–S |', '|---|---|---|---|---|---|---|---|']
for t in ('qsp', 'qth', 'g20', 'g120', 'm26'):
    d = MS / 'swap' / t
    s = load(d)
    if not s:
        L.append(f'| {t} | {status(d) if d.exists() else "未开始"} | — | — | — | — | — | — |')
        continue
    lf, mv = s['left']['content'], s['moved']['content']
    L.append(f"| {t} | {status(d, 75)} | {rc(lf['A_U'])} | {rc(lf['A_S'])} | {rc(mv['U_A'])} | {rc(mv['S_A'])} | {lf['A_Ur']['value']:.2f} / {mv['U_Ar']['value']:.2f} | "
             f"{rc(s['asym']['content']['A_U||U/U||S'])} |")
L += ['', '读出：qsp = Qwen3.8 非思考，qth = Qwen3.8 思考，g20 / g120 = gpt-oss-20b / 120b，m26 = MiMo-V2.6 默认调用。拟合在 `r05r06_fits/<读出>/fit_m.pt`；m26 的拟合已镜像到本存档 `results/pro/r05r06_fits/m26/`（若已同步），其余 4 个读出的拟合在 `evidence/paper_data_v4`。烟雾测试 `_smoke_g20` 不进结果。', '']

# F. format ablation (mechanism section)
L += ['## F. 格式消融（机制部分，不进主表）：`results/pro/modeswitch/qwen38_{nt,think}_switch/`', '']
for t in sorted(FORMAT):
    s = load(MS / t) or {}
    m = s.get('orders', {}).get('main', {}).get('ratios', {})
    L.append(f"- {t}（{status(MS / t)}）：主探针 A–U / U–S {rc(m.get('A-U/U-S_kl_body'))}，A–S / U–S {rc(m.get('A-S/U-S_kl_body'))}")
L += ['']

# G. MiMo-V2.6 mechanism readout (not KL)
TS, GS = P / 'thinking_swap', P / 'games_swap'
L += ['## G. MiMo-V2.6 机制读出（归因与对照，不是 KL；`results/pro/thinking_swap/mimo_v26_*`、`results/pro/games_swap/mimo_v26_*`）', '',
      '| 运行 | 条件：回答"你"比例的变化 [95%] |', '|---|---|']
for d in sorted(TS.glob('mimo_v26_*')) if TS.exists() else []:
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

# H. diagnostics and logs
L += ['## H. 诊断与日志', '',
      '- gpt-oss-120b 的正文预测诊断：`results/pro/modeswitch/g120_diag.log`、`g_gen_120b.log`、`g_gen_20b.log`（脚本 `code/pro/g120_logit_diag_20261108.py`、`g_gen_diag_20261108.py`）。',
      '- 每个模型跑之前的渲染存档和只渲染检查：`results/pro/modeswitch/nat10/<设置>.render.txt`、`.dry.log`，以及 `results/pro/modeswitch/main6/`。',
      '- 下载记录：`results/pro/modeswitch/{nat10,main6}/dl_*.log`。',
      '- 队列日志：`results/pro/modeswitch/queue.log`（总），`nat10/queue_nat10.log`，`main6/queue_main6.log`，`swap/queue_swap.log`，`mech_m26/status.log`。',
      '- 输入：`inputs/`（正文、英文译文、游戏题目）。服务器依赖脚本：`code/pro_deps/`。本存档写的脚本：`code/pro/`。',
      '- 11-08 之前零散拉回的 summary 已核对与镜像一致（39 份），退役到 `tmp/retired_results_partial_20261108/`。', '']

(A / 'KL_INDEX_zh.md').write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')
print('written', A / 'KL_INDEX_zh.md', len(L), 'lines')
