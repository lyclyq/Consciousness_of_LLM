"""Distribution of the next-token (KL) separation over the 25 models of the main comparison (user 11-10: "比值具体是多少，从分布上
统计上看是否有一个大致的边界"). Per model: content ratios A-U/U-S and A-S/U-S under the main probe (ctrl summaries), the smaller
of the two with its own 95% family interval, the paper's self-index group (13 / 12), and the content-position index coverage
(layers singled out on message content / all layers, Table tab:self_crossmodel; four-stream models: mean of streams, strongest
stream in brackets). Also the natural order (Table tab:self_natural, natural3). Reads only; writes results/kl_ratio_distribution_20261110.md.
usage: kl_ratio_distribution_20261110.py [ARCHIVE]"""
import json
import sys
from pathlib import Path

A = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
ROOTS = [A / 'results' / 'server3' / 'modeswitch', A / 'results' / 'pro' / 'modeswitch']


def load(rel):
    for r in ROOTS:
        f = r / rel / 'summary.json'
        if f.exists():
            return json.loads(f.read_text(encoding='utf-8'))
    raise FileNotFoundError(rel)


# tag, name, paper self-index group, content coverage (Table tab:self_crossmodel, content column), four-stream
M = [('qwen38_nt', 'Qwen3.8-27B', 1, (64, 64), 0), ('gpt_oss_120b_en', 'gpt-oss-120b', 1, (25, 36), 0), ('gpt_oss_20b_en', 'gpt-oss-20b', 1, (20, 24), 0),
     ('flashnext_nt', 'Qwen3.8-Flash-Next', 1, (38, 48), 1), ('glm53_flash', 'GLM-5.3-Flash', 1, (18, 45), 1), ('mimo_v26_nt', 'MiMo-V2.6-Distill-9B', 1, (32, 32), 0),
     ('exaone4_32b_en', 'EXAONE-4.0.1-32B', 1, (64, 64), 0), ('glm46v_flash', 'GLM-4.6V-Flash', 1, (40, 40), 0), ('glm47_flash', 'GLM-4.7-Flash', 1, (34, 47), 0),
     ('r1_distill_llama70b', 'R1-Distill-Llama-70B', 1, (0, 80), 0), ('phi4_en', 'phi-4', 1, (35, 40), 0), ('granite42_30b_en', 'Granite-4.2-30B', 1, (30, 64), 0),
     ('granite41_30b_en', 'Granite-4.1-30B', 1, (17, 64), 0), ('olmo2_13b_en', 'OLMo-2-13B', 0, (4, 40), 0), ('olmo2_7b_en', 'OLMo-2-7B', 0, (5, 32), 0),
     ('apertus8b_en', 'Apertus-8B', 0, (0, 32), 0), ('moonlight16b_a3b_instruct', 'Moonlight-16B-A3B', 0, (0, 27), 0), ('mimo7b_rl_0530', 'MiMo-7B-RL', 0, (0, 36), 0),
     ('mimo_vl7b', 'MiMo-VL-7B', 0, (0, 36), 0), ('ernie45_21b_a3b_thinking', 'ERNIE-4.5-21B-A3B', 0, (0, 28), 0), ('seed_oss_36b', 'Seed-OSS-36B', 0, (0, 64), 0),
     ('gemma4_31b_en', 'Gemma-4-31B', 0, (0, 60), 0), ('qwen36_27b', 'Qwen3.6-27B', 0, (6, 64), 0), ('qwen36_35b_a3b', 'Qwen3.6-35B-A3B', 0, (1, 40), 0),
     ('qwen35_27b', 'Qwen3.5-27B', 0, (13, 64), 0)]
rows = []
for tag, name, idx, (c, n), hc in M:
    r = load(f'ctrl/{tag}')['ratios']
    a, b = r['A-U/U-S_kl_content'], r['A-S/U-S_kl_content']
    lo = a if a['ratio'] <= b['ratio'] else b
    rows.append(dict(name=name, idx=idx, cov=c / n, cn=f'{c}/{n}', hc=hc, au=a['ratio'], as_=b['ratio'], mn=lo['ratio'], ci=lo['ci95'],
                     both=a['ci95'][0] > 1 and b['ci95'][0] > 1, one=(a['ci95'][0] > 1) + (b['ci95'][0] > 1)))
rows.sort(key=lambda x: x['mn'])


def rank(v):
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


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    den = (sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry)) ** 0.5
    return num / den


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
      f"- Spearman(正文位置索引覆盖, log 较小比值)，25 个模型：{spearman([x['cov'] for x in rows], [x['mn'] for x in rows]):+.2f}；"
      f"只算 23 个单流模型：{spearman([x['cov'] for x in rows if not x['hc']], [x['mn'] for x in rows if not x['hc']]):+.2f}。"]
single = [x for x in rows if not x['hc']]
L.append('- 单流模型按正文位置索引覆盖：' + '；'.join(f"{x['name']} {x['cov']:.0%}→{x['mn']:.2f}" for x in sorted(single, key=lambda x: -x['cov']) if x['cov'] > 0))
# natural order
NAT = [('exaone4_32b_en', 'EXAONE-4.0.1-32B', '64/64'), ('glm46v_flash', 'GLM-4.6V-Flash', '36/40'), ('granite41_30b_en', 'Granite-4.1-30B', '61/64'),
       ('moonlight16b_a3b_instruct', 'Moonlight-16B-A3B', '27/27'), ('mimo7b_rl_0530', 'MiMo-7B-RL', '34/36'), ('phi4_en', 'phi-4', '25/40'),
       ('ernie45_21b_a3b_thinking', 'ERNIE-4.5-21B-A3B', '16/28'), ('olmo2_7b_en', 'OLMo-2-7B', '5/32'), ('seed_oss_36b', 'Seed-OSS-36B', '3/64'),
       ('gemma4_31b_en', 'Gemma-4-31B', '0/60')]
L += ['', '## 自然顺序（完整分布；这些模型控制 token 概率约 0）', '', '| 模型 | 正文位置索引（自然顺序） | 较小比值 [95%] |', '|---|---|---|']
nat = []
for tag, name, ci in NAT:
    r = load(tag)['orders']['natural']['ratios']
    a, b = r['A-U/U-S_kl_body'], r['A-S/U-S_kl_body']
    lo = a if a['ratio'] <= b['ratio'] else b
    nat.append((lo['ratio'], name, ci, lo['ci95']))
for tag, name, ci in (('dsv41_flash', 'DeepSeek-V4.1-Flash（只接受自然顺序）', '38/40'), ('ministral3_14b_instruct_en', 'Ministral-3-14B-Instruct', '0/40'),
                      ('ministral3_14b_reasoning_en', 'Ministral-3-14B-Reasoning', '0/40')):
    r = load(f'nat3/{tag}')['ratios']
    a, b = r['A-U/U-S_kl_content'], r['A-S/U-S_kl_content']
    lo = a if a['ratio'] <= b['ratio'] else b
    nat.append((lo['ratio'], name, ci, lo['ci95']))
for mn, name, ci, c in sorted(nat):
    L.append(f'| {name} | {ci} | {mn:.2f} [{c[0]:.2f}, {c[1]:.2f}] |')
out = A / 'results' / 'kl_ratio_distribution_20261110.md'
out.write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')
print('written', out)
