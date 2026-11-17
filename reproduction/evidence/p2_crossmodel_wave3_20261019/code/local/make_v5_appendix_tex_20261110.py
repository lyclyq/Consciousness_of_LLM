"""Appendix table bodies for the 11-10 revision (user: KL, MiMo-V2.6-Distill-Qwen-9B as the fifth readout, the wording test), all
read from data files, none typed by hand. Output stats/v5_appendix_tex.json: {block: [cells or rows]}.
  mimo_seeds     MiMo cells of tab:self_seeds (10 data rows: own <- User random for the seeds 20261022..26, then User <- own random);
                 seed 20261022 = mimo_v26_r05q_all, the others mimo_v26_r06_s1..s4; changes as make_v4_mechanism.summarize
  mimo_cluster   MiMo cells of tab:self_cluster (11 data rows), from stats/r09_cluster_mimo_20261110.json (family intervals)
  mimo_question  MiMo cells of tab:self_question (10 data rows), runs r05q_all, r05q_msg, r05m_all, r05m_msg
  kl_main_{en,zh}, kl_nat_{en,zh}, kl_swap_{en,zh}, wording_{en,zh}: full LaTeX rows of the new tables tab:self_kl,
                 tab:self_kl_natural, tab:self_kl_swap, tab:self_wording
Sources: the KL archive ../p2_modeswitch_20261107/results/{pro,server3}/modeswitch (summary.json of each setting) and the wording
archive ../p2_wording_swap_20261109/results/wording_summary_all.json. Self-index groups and the natural-order content index are those
of the paper's Tables tab:self_crossmodel and tab:self_natural (listed below with their source).
usage: make_v5_appendix_tex_20261110.py ARCHIVE"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_v4_mechanism import items, summarize  # noqa: E402

A = Path(sys.argv[1]).resolve()
MIMO = (A / '..' / 'p2_modeswitch_20261107' / 'results' / 'pro' / 'thinking_swap').resolve()
KL = [(A / '..' / 'p2_modeswitch_20261107' / 'results' / m / 'modeswitch').resolve() for m in ('server3', 'pro')]
WORD = (A / '..' / 'p2_wording_swap_20261109' / 'results' / 'wording_summary_all.json').resolve()
out = {}


def sg(x, d=2):
    s = f'{x:+.{d}f}'
    return f'\\({abs(x):.{d}f}\\)' if s in ('+0.00', '-0.00', '+0.000', '-0.000') else f'\\({s}\\)'


def ch(s):
    if s is None or 'diff' not in s:
        return '--'
    d = 3 if abs(s['diff']) < 0.005 and s['diff'] != 0 else 2
    return f"{sg(s['diff'], d)} [{sg(s['diff_ci'][0], d)}, {sg(s['diff_ci'][1], d)}]"


def val(s):
    return '--' if s is None else f"{s['stat']:.2f}" + (f" ({s['finished']}/{s['n']})" if s.get('finished', s['n']) != s['n'] else '')


run = {k: summarize(items(str(MIMO / f'mimo_v26_{k}')), k) for k in ('r05q_all', 'r05q_msg', 'r05m_all', 'r05m_msg', 'r06_s1', 'r06_s2', 'r06_s3', 'r06_s4')}
seeds = ['r05q_all', 'r06_s1', 'r06_s2', 'r06_s3', 'r06_s4']
out['mimo_seeds'] = [ch(run[r].get('rswU8_A')) for r in seeds] + [ch(run[r].get('rswA8_U')) for r in seeds]
C = json.load(open(A / 'stats' / 'r09_cluster_mimo_20261110.json', encoding='utf-8'))['results']


def cl(s):
    return '--' if not s or 'change' not in s else f"{sg(s['change'])} [{sg(s['cluster_ci'][0])}, {sg(s['cluster_ci'][1])}]"


at, co = C['attribution'], C['content']
out['mimo_cluster'] = [cl(at.get('own <- User')), '--', cl(at.get('own <- User, random')), '--', cl(at.get('User <- own')),
                       cl(at.get('User <- own, random')), '--', cl(co.get('swU8_A')), '--', '--', cl(co.get('swA8_U'))]
Q = ['r05q_all', 'r05q_msg', 'r05m_all', 'r05m_msg']
out['mimo_question'] = ([val(run['r05q_all'].get('base_A'))] + [ch(run[r].get('swU8_A')) for r in Q] + [val(run['r05q_all'].get('base_U'))]
                        + [ch(run[r].get('swA8_U')) for r in Q])


def load(rel):
    for root in KL:
        f = root / rel / 'summary.json'
        if f.exists():
            return json.loads(f.read_text(encoding='utf-8'))
    raise FileNotFoundError(rel)


def rc(v, d=2):
    if v is None:
        return '--'
    x, lo, hi = (v['ratio'] if 'ratio' in v else v['value']), v['ci95'][0], v['ci95'][1]
    f = (lambda y: f'{y:.0f}') if x >= 100 else (lambda y: f'{y:.1f}') if x >= 10 else (lambda y: f'{y:.{d}f}')
    return f'{f(x)} [{f(lo)}, {f(hi)}]'


def ok(v):
    return v is not None and v['ci95'][0] > 1


# Table tab:self_crossmodel order and self-index group (13 with a self-index, 12 without; paper Section 4.4)
MAIN = [('qwen38_nt', 'Qwen3.8-27B', 1), ('gpt_oss_120b_en', 'gpt-oss-120b', 1), ('gpt_oss_20b_en', 'gpt-oss-20b', 1),
        ('flashnext_nt', 'Qwen3.8-Flash-Next', 1), ('glm53_flash', 'GLM-5.3-Flash', 1), ('mimo_v26_nt', 'MiMo-V2.6-Distill-9B', 1),
        ('exaone4_32b_en', 'EXAONE-4.0.1-32B', 1), ('glm46v_flash', 'GLM-4.6V-Flash', 1), ('glm47_flash', 'GLM-4.7-Flash', 1),
        ('r1_distill_llama70b', 'R1-Distill-Llama-70B', 1), ('phi4_en', 'phi-4', 1), ('granite42_30b_en', 'Granite-4.2-30B', 1),
        ('granite41_30b_en', 'Granite-4.1-30B', 1), ('olmo2_13b_en', 'OLMo-2-13B', 0), ('olmo2_7b_en', 'OLMo-2-7B', 0),
        ('apertus8b_en', 'Apertus-8B', 0), ('moonlight16b_a3b_instruct', 'Moonlight-16B-A3B', 0), ('mimo7b_rl_0530', 'MiMo-7B-RL', 0),
        ('mimo_vl7b', 'MiMo-VL-7B', 0), ('ernie45_21b_a3b_thinking', 'ERNIE-4.5-21B-A3B', 0), ('seed_oss_36b', 'Seed-OSS-36B', 0),
        ('gemma4_31b_en', 'Gemma-4-31B', 0), ('qwen36_27b', 'Qwen3.6-27B', 0), ('qwen36_35b_a3b', 'Qwen3.6-35B-A3B', 0),
        ('qwen35_27b', 'Qwen3.5-27B', 0)]
# 11-10 (user: a relation, not a threshold): the second column is the content index of Table tab:self_crossmodel (layers singled out
# on message content; four-stream models: mean of the streams, strongest stream in parentheses), not the yes/no self-index group
CONTENT = {'Qwen3.8-27B': '64/64', 'gpt-oss-120b': '25/36', 'gpt-oss-20b': '20/24', 'Qwen3.8-Flash-Next': '38 (44)/48', 'GLM-5.3-Flash': '18 (28)/45',
           'MiMo-V2.6-Distill-9B': '32/32', 'EXAONE-4.0.1-32B': '64/64', 'GLM-4.6V-Flash': '40/40', 'GLM-4.7-Flash': '34/47',
           'R1-Distill-Llama-70B': '0/80', 'phi-4': '35/40', 'Granite-4.2-30B': '30/64', 'Granite-4.1-30B': '17/64', 'OLMo-2-13B': '4/40',
           'OLMo-2-7B': '5/32', 'Apertus-8B': '0/32', 'Moonlight-16B-A3B': '0/27', 'MiMo-7B-RL': '0/36', 'MiMo-VL-7B': '0/36',
           'ERNIE-4.5-21B-A3B': '0/28', 'Seed-OSS-36B': '0/64', 'Gemma-4-31B': '0/60', 'Qwen3.6-27B': '6/64', 'Qwen3.6-35B-A3B': '1/40',
           'Qwen3.5-27B': '13/64'}
for lang in ('en', 'zh'):
    rows, tally = [], {0: [0, 0, 0], 1: [0, 0, 0]}
    for tag, name, idx in MAIN:
        s = load(f'ctrl/{tag}')
        r = s['ratios']
        pc = ' / '.join(f"{s['roles'][x]['p_ctrl']['mean']:.2f}" for x in ('A', 'U', 'S'))
        a, b = r.get('A-U/U-S_kl_content'), r.get('A-S/U-S_kl_content')
        tally[idx][ok(a) + ok(b)] += 1
        rows.append(f"{name} & {CONTENT[name]} & {pc} & {rc(r.get('A-U/U-S_kl_body'))} & {rc(a)} & {rc(b)} \\\\")
    out[f'kl_main_{lang}'] = rows
out['kl_main_tally'] = {'with index: none/one/both': tally[1], 'without: none/one/both': tally[0]}
# natural order: Table tab:self_natural (content index), the four templates without a System message mid-conversation (A-U only),
# and Table tab:self_natural3
NAT = [('exaone4_32b_en', 'EXAONE-4.0.1-32B', '64/64'), ('glm46v_flash', 'GLM-4.6V-Flash', '36/40'), ('granite41_30b_en', 'Granite-4.1-30B', '61/64'),
       ('moonlight16b_a3b_instruct', 'Moonlight-16B-A3B', '27/27'), ('mimo7b_rl_0530', 'MiMo-7B-RL', '34/36'), ('phi4_en', 'phi-4', '25/40'),
       ('ernie45_21b_a3b_thinking', 'ERNIE-4.5-21B-A3B', '16/28'), ('olmo2_7b_en', 'OLMo-2-7B', '5/32'), ('seed_oss_36b', 'Seed-OSS-36B', '3/64'),
       ('gemma4_31b_en', 'Gemma-4-31B', '0/60')]
NAT3 = [('dsv41_flash', 'DeepSeek-V4.1-Flash', '38/40'), ('ministral3_14b_instruct_en', 'Ministral-3-14B-Instruct', '0/40'),
        ('ministral3_14b_reasoning_en', 'Ministral-3-14B-Reasoning', '0/40')]
for lang in ('en', 'zh'):
    rows = []
    for tag, name, ci in NAT:
        n = load(tag)['orders']['natural']['ratios']
        rows.append(f"{name} & {ci} & {rc(n.get('A-U/U-S_kl_body'))} & {rc(n.get('A-S/U-S_kl_body'))} \\\\")
    rows.append('\\midrule')
    for tag, name, ci in NAT3:
        r = load(f'nat3/{tag}')['ratios']
        rows.append(f"{name} & {ci} & {rc(r.get('A-U/U-S_kl_content'))} & {rc(r.get('A-S/U-S_kl_content'))} \\\\")
    out[f'kl_nat_{lang}'] = rows
RO = [('qsp', 'Qwen3.8-27B, no thinking', 'Qwen3.8-27B，不思考'), ('qth', 'Qwen3.8-27B, thinking', 'Qwen3.8-27B，思考'),
      ('g20', 'gpt-oss-20b, thinking', 'gpt-oss-20b，思考'), ('g120', 'gpt-oss-120b, thinking', 'gpt-oss-120b，思考'),
      ('m26', 'MiMo-V2.6-Distill-9B, default call', 'MiMo-V2.6-Distill-9B，默认调用')]
UNED = {'qsp': 'ctrl/qwen38_nt', 'qth': 'ctrl/qwen38_think', 'g20': 'ctrl/gpt_oss_20b_en', 'g120': 'ctrl/gpt_oss_120b_en', 'm26': 'ctrl/mimo_v26_nt'}
for lang in ('en', 'zh'):
    rows = []
    for t, en, zh in RO:
        s = load(f'swap/{t}')
        lf, mv = s['left']['content'], s['moved']['content']
        rows.append(f"{en if lang == 'en' else zh} & {rc(lf['A_U'])} & {rc(lf['A_S'])} & {rc(mv['U_A'])} & {rc(mv['S_A'])} & "
                    f"{lf['A_Ur']['value']:.2f} / {abs(mv['U_Ar']['value']) if round(mv['U_Ar']['value'], 2) == 0 else mv['U_Ar']['value']:.2f} & {rc(s['asym']['content']['A_U||U/U||S'])} \\\\")
    out[f'kl_swap_{lang}'] = rows
# wording test (p2_wording_swap_20261109): unedited own / User on the option naming the model, removal and installation changes
W = json.load(open(WORD, encoding='utf-8'))['by_readout']
WR = [('qsp', 'Qwen3.8-27B, no thinking, Chinese', 'Qwen3.8-27B，不思考，中文'), ('qth', 'Qwen3.8-27B, thinking, Chinese', 'Qwen3.8-27B，思考，中文'),
      ('qsp_en', 'Qwen3.8-27B, no thinking, English', 'Qwen3.8-27B，不思考，英文'), ('qth_en', 'Qwen3.8-27B, thinking, English', 'Qwen3.8-27B，思考，英文'),
      ('g20', 'gpt-oss-20b, thinking', 'gpt-oss-20b，思考'), ('g120', 'gpt-oss-120b, thinking', 'gpt-oss-120b，思考'),
      ('m26', 'MiMo-V2.6-Distill-9B, default call', 'MiMo-V2.6-Distill-9B，默认调用')]
WL = {'en': {'paper': 'main', 'pron': 'pronoun', 'role': 'role name'}, 'zh': {'paper': '原版', 'pron': '代词', 'role': '角色名'}}
for lang in ('en', 'zh'):
    rows = []
    for k, en, zh in WR:
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{en if lang == 'en' else zh}}}}} \\\\")
        for w in ('paper', 'pron', 'role'):
            x = W[k]['wordings'][w]
            a, u = x['unedited']['base_A']['self'], x['unedited']['base_U']['self']
            rm, ins = x['去除']['self'], x['植入']['self']
            rr = max(abs(x[n]['self']['mean']) for n in ('随机，自己的消息', '随机，User 消息') if n in x) if any(n in x for n in ('随机，自己的消息', '随机，User 消息')) else None
            gap = '' if k == 'm26' else f" ({rm['share']:.2f})"
            gi = '' if k == 'm26' else f" ({ins['share']:.2f})"
            rows.append(f"\\quad {WL[lang][w]} & {a:.2f} & {u:.2f} & {sg(rm['mean'])} [{sg(rm['ci_families'][0])}, {sg(rm['ci_families'][1])}]{gap} & "
                        f"{sg(ins['mean'])} [{sg(ins['ci_families'][0])}, {sg(ins['ci_families'][1])}]{gi} & {'--' if rr is None else f'{rr:.2f}'} \\\\")
    out[f'wording_{lang}'] = rows
(A / 'stats' / 'v5_appendix_tex.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k.startswith('mimo') or k == 'kl_main_tally'}, ensure_ascii=False, indent=1))
for k in ('kl_main_en', 'kl_nat_en', 'kl_swap_en', 'wording_en'):
    print('\n'.join(out[k][:30]))
