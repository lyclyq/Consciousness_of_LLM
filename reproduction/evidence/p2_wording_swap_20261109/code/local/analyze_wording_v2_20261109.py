"""Wording variant of R05 across the mechanism group (user 11-09), v2 of analyze_wording_20261109.py: every readout whose runs are on
the local mirror, single pass and thinking. Per readout, three wordings of the attribution question with the same edit (fit_m, message
region, 8 dims at every layer, 75 held-out bodies x 2 orders):
  paper  the anchor <dir>_r05m_msg (no rerun)            你（AI 助手）/ 用户 / 系统  |  You (the AI assistant) / The user / The system
  pron   wording_swap(_v2)_20261109.py --wording pron     你 / 我 / 系统              |  You / Me / The system
  role   wording_swap(_v2)_20261109.py --wording role     AI 助手 / 用户 / 系统       |  The AI assistant / The user / The system
Value of an item for option o (self = the option naming the model, user = the option naming the user): single pass, its probability;
thinking, 1 if the finished answer is o, else 0 (unfinished items are left out; pairs need both items finished, gap shares need the
item finished in base_A, base_U and the edited condition). Per wording: unedited values, item-paired changes (item and 15-family
bootstrap, 4,000, seed 1), the share of the unedited A-U gap the edit closes (family bootstrap), and the wording contrasts of that share.
Reads only; usage: python -I analyze_wording_v2_20261109.py [ARCHIVE_DIR] -> results/wording_summary_all.md / .json"""
import json
import random
import sys
from pathlib import Path

A = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
W3 = A.parent / 'p2_crossmodel_wave3_20261019' / 'runs' / 'pro'
MS = A.parent / 'p2_modeswitch_20261107' / 'results' / 'pro'
NEW_S = A / 'results' / 'pro' / 'singlepass_swap' / 'wording_20261109'
NEW_T = A / 'results' / 'pro' / 'thinking_swap' / 'wording_20261109'
READOUTS = {  # name: (label, thinking, paper anchor, pron, role)
    'qsp': ('Qwen3.8-27B，非思考，中文', False, W3 / 'singlepass_swap' / 'qwen38_27b_r05m_msg', NEW_S / 'pron', NEW_S / 'role'),
    'qth': ('Qwen3.8-27B，思考，中文', True, W3 / 'thinking_swap' / 'qwen38_27b_r05m_msg', NEW_T / 'qth_pron', NEW_T / 'qth_role'),
    # English Qwen (user 11-09): no English R05 cell existed, so the paper wording was run here too (wording_swap_qwen_en_pro_20261109.sh)
    'qsp_en': ('Qwen3.8-27B，非思考，英文', False, NEW_S / 'qsp_en_paper', NEW_S / 'qsp_en_pron', NEW_S / 'qsp_en_role'),
    'qth_en': ('Qwen3.8-27B，思考，英文', True, NEW_T / 'qth_en_paper', NEW_T / 'qth_en_pron', NEW_T / 'qth_en_role'),
    'g20': ('gpt-oss-20b，思考', True, W3 / 'thinking_swap' / 'gpt_oss_20b_r05m_msg', NEW_T / 'g20_pron', NEW_T / 'g20_role'),
    'g120': ('gpt-oss-120b，思考', True, W3 / 'thinking_swap' / 'gpt_oss_120b_r05m_msg', NEW_T / 'g120_pron', NEW_T / 'g120_role'),
    'm26': ('MiMo-V2.6，默认调用', True, MS / 'thinking_swap' / 'mimo_v26_r05m_msg', NEW_T / 'm26_pron', NEW_T / 'm26_role')}
EDITS = [('去除', 'swU8_A', 'base_A'), ('植入', 'swA8_U', 'base_U'), ('随机，自己的消息', 'rswU8_A', 'base_A'), ('随机，User 消息', 'rswA8_U', 'base_U')]
REPS = 4000


def fam(i):
    return i.rsplit('_c', 1)[0]


def load(d):
    by = {}
    for f in sorted((d / 'items').glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        by.setdefault(r['cond'], {})[(r['id'], tuple(r['order']))] = r
    return by


def value(r, o, thinking):
    if not thinking:
        return r['p'][o]
    return float(r.get('on_answer') == o) if r.get('finished') else None


def pct(xs):
    xs = sorted(xs)
    return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float('nan')


def fam_draws(keys, seed=1):
    fams = sorted({fam(k[0]) for k in keys})
    byf = {f: [k for k in keys if fam(k[0]) == f] for f in fams}
    rng = random.Random(seed)
    return [[k for f in (rng.choice(fams) for _ in fams) for k in byf[f]] for _ in range(REPS)]


def vals(by, c, o, th):
    return {k: v for k, r in by.get(c, {}).items() if (v := value(r, o, th)) is not None}


def share(by, c, base, o, th, kk):
    """share of the unedited A-U gap on option o closed by edit c (1 = answers as the other role), over items kk"""
    a, u, e, b = vals(by, 'base_A', o, th), vals(by, 'base_U', o, th), vals(by, c, o, th), vals(by, base, o, th)
    kk = [k for k in kk if k in a and k in u and k in e]
    if not kk:
        return float('nan')
    d = mean(e[k] - b[k] for k in kk)
    g = mean(a[k] - u[k] for k in kk)
    return (-d if base == 'base_A' else d) / g if g else float('nan')


S, C = {}, {}
for ro, (label, th, *dirs) in READOUTS.items():
    data = {w: load(d) for w, d in zip(('paper', 'pron', 'role'), dirs) if (d / 'items').exists()}
    if not data:
        continue
    S[ro] = {'label': label, 'thinking': th, 'wordings': {}}
    for w, by in data.items():
        if 'base_A' not in by or 'base_U' not in by:
            S[ro]['wordings'][w] = {'note': 'no unedited items'}
            continue
        keys = sorted(set(by['base_A']) & set(by['base_U']))
        draws = fam_draws(keys)
        s = {'n_items': len(keys), 'unedited': {}, 'unparsed': {}}
        for c in ('base_A', 'base_U'):
            s['unedited'][c] = {o: round(mean(vals(by, c, o, th).values()), 4) for o in ('self', 'user', 'system')}
            if th:
                s['unedited'][c]['finished'] = sum(bool(r.get('finished')) for r in by[c].values())
        for c in by:
            if th:
                s['unparsed'][c] = sum(1 for r in by[c].values() if r.get('finished') and r.get('on_answer') is None)
        for name, c, base in EDITS:
            if c not in by:
                continue
            e = {'n': len(by[c])}
            for o in ('self', 'user'):
                ev, bv = vals(by, c, o, th), vals(by, base, o, th)
                dd = {k: ev[k] - bv[k] for k in ev if k in bv}
                rng = random.Random(1)
                ds = list(dd.values())
                e[o] = {'n_pairs': len(dd), 'mean': round(mean(ds), 4),
                        'ci_items': pct(sum(rng.choice(ds) for _ in ds) / len(ds) for _ in range(REPS)),
                        'ci_families': pct(mean(dd[k] for k in dr if k in dd) for dr in draws),
                        'share': round(share(by, c, base, o, th, keys), 4),
                        'share_ci_families': pct(share(by, c, base, o, th, dr) for dr in draws)}
            s[name] = e
        S[ro]['wordings'][w] = s
    for a_, b_ in (('pron', 'role'), ('pron', 'paper'), ('role', 'paper')):
        if a_ not in data or b_ not in data:
            continue
        for name, c, base in EDITS[:2]:
            if c not in data[a_] or c not in data[b_]:
                continue
            for o in ('self', 'user'):
                keys = sorted(set(data[a_]['base_A']) & set(data[a_]['base_U']) & set(data[b_]['base_A']) & set(data[b_]['base_U']))
                draws = fam_draws(keys, seed=2)
                v = share(data[a_], c, base, o, th, keys) - share(data[b_], c, base, o, th, keys)
                C[f'{ro} | {a_} - {b_} | {name} | {o}'] = {
                    'diff_share': round(v, 4), 'n': len(keys),
                    'ci_families': pct(share(data[a_], c, base, o, th, dr) - share(data[b_], c, base, o, th, dr) for dr in draws)}

out = A / 'results'
out.mkdir(exist_ok=True)
(out / 'wording_summary_all.json').write_text(json.dumps({'by_readout': S, 'contrasts': C}, ensure_ascii=False, indent=1), encoding='utf-8')
WL = {'paper': '论文原版', 'pron': '代词版', 'role': '角色名版'}
L = ['# 归属题措辞变体：机制组各读出汇总', '',
     '由 `code/local/analyze_wording_v2_20261109.py` 从本地镜像生成。fit_m，只改消息，每层 8 维，留出 75 段 × 2 种顺序。'
     '非思考读概率，思考读完成条目中各答案的份额。"指模型"：你（AI 助手）/ 你 / AI 助手；"指用户"：用户 / 我 / 用户（英文同理）。'
     '"关掉的差距"：改动关掉不改时 A–U 差距的比例（1 = 与另一角色相同），家族重抽样 95% 区间。', '']
for ro, s in S.items():
    L += [f"## {s['label']}", '', '| 措辞 | 不改：自己的消息 指模型 / 指用户 | 不改：User 消息 指模型 / 指用户 | 去除：指模型的变化 | 去除关掉的差距 | 植入：指模型的变化 | 植入关掉的差距 | 随机（自己 / User） |',
          '|---|---|---|---|---|---|---|---|']
    for w, x in s['wordings'].items():
        if 'note' in x:
            L.append(f'| {WL[w]} | {x["note"]} | | | | | | |')
            continue
        a, u = x['unedited']['base_A'], x['unedited']['base_U']
        cell = lambda nm, k: (f"{x[nm][k]['mean']:+.3f} [{x[nm][k]['ci_families'][0]:+.2f}, {x[nm][k]['ci_families'][1]:+.2f}]" if nm in x else '—')  # noqa: E731
        shr = lambda nm: (f"{x[nm]['self']['share']:.2f} [{x[nm]['self']['share_ci_families'][0]:.2f}, {x[nm]['self']['share_ci_families'][1]:.2f}]" if nm in x else '—')  # noqa: E731
        rnd = ' / '.join(f"{x[nm]['self']['mean']:+.3f}" if nm in x else '—' for nm in ('随机，自己的消息', '随机，User 消息'))
        L.append(f"| {WL[w]} | {a['self']:.3f} / {a['user']:.3f} | {u['self']:.3f} / {u['user']:.3f} | {cell('去除', 'self')} | {shr('去除')} | {cell('植入', 'self')} | {shr('植入')} | {rnd} |")
    if s['thinking']:
        L += ['', '完成情况与未解析答案：' + '；'.join(f"{WL[w]} 不改完成 {x['unedited']['base_A'].get('finished')}/{x['unedited']['base_U'].get('finished')}，未解析 {sum(x['unparsed'].values())}"
                                     for w, x in s['wordings'].items() if 'note' not in x)]
    L.append('')
L += ['## 措辞之间的比较（关掉的差距比例之差，家族重抽样 95% 区间）', '', '| 比较 | 差 | 区间 | 条数 |', '|---|---|---|---|']
for k, v in C.items():
    L.append(f"| {k} | {v['diff_share']:+.3f} | [{v['ci_families'][0]:+.3f}, {v['ci_families'][1]:+.3f}] | {v['n']} |")
(out / 'wording_summary_all.md').write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')
print('written', out / 'wording_summary_all.md')
