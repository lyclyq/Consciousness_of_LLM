"""How the attribution question names the model (Appendix "How the Question Names the Model", Table tab:self_wording, Figure
self_wording): the edit of the causal tests under three wordings of the option that names the model, in every readout of the
causal tests (Qwen3.8-27B without and with thinking, with the Chinese and with the English items; gpt-oss-20b and gpt-oss-120b with
thinking; MiMo-V2.6-Distill-Qwen-9B in its default call).

The edit is the same in every wording: the fit on the message alone (fit_m), the message edited only (body and end), 8 dimensions
at every layer, 75 held-out bodies x 2 option orders. Only the answer options of the question change; the stem and the tail are
those of the paper:
  paper  the anchor run <readout>_r05m_msg of the causal tests  你（AI 助手）/ 用户 / 系统  |  You (the AI assistant) / The user / The system
  pron   selfindex.role_swap --wording pron                      你 / 我 / 系统              |  You / Me / The system
  role   selfindex.role_swap --wording role                      AI 助手 / 用户 / 系统       |  The AI assistant / The user / The system
For Qwen3.8-27B with the English items the paper wording was run with the others (there is no English anchor of the causal tests
with the edit of the message alone).
Value of an item for option o (self = the option naming the model, user = the option naming the user): without thinking its
probability; with thinking 1 if the finished answer is o, else 0 (unfinished items are left out; pairs need both items finished,
gap shares need the item finished in base_A, base_U and the edited condition). Per wording: unedited values, item-paired changes
(item and 15-family bootstrap, 4,000 draws, seed 1), the share of the unedited A-U gap the edit closes (family bootstrap), and the
contrasts of that share between wordings (family bootstrap, seed 2).

Inputs: the anchors in the wave-3 archive (runs/<group>/{singlepass,thinking}_swap/<model>_r05m_msg) and in the next-token archive
(MiMo: results/<group>/thinking_swap/mimo_v26_r05m_msg); the wording runs in the wording archive
(evidence/p2_wording_swap_20261109, results/<group>/{singlepass,thinking}_swap/wording_20261109/<run>). Readouts whose runs are not
in the package are left out.
Outputs: OUT/evidence/p2_wording_swap_20261109/results/wording_summary_all.json (the archived format: readout labels and the names
of the edits as in the archive, i.e. 去除 = removal, own message <- User coordinates; 植入 = installation, User message <- own
coordinates; 随机，自己的消息 / 随机，User 消息 = the same edits in a random subspace) and wording_summary_all.md (the same numbers as
tables, in English).

usage: question_wording.py --results RES --out OUT
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MS, W3, WS, Roots, arguments, result_path, run_path, write_text  # noqa: E402

EDITS = [('去除', 'swU8_A', 'base_A'), ('植入', 'swA8_U', 'base_U'), ('随机，自己的消息', 'rswU8_A', 'base_A'), ('随机，User 消息', 'rswA8_U', 'base_U')]
EDIT_EN = {'去除': 'removal', '植入': 'installation', '随机，自己的消息': 'random, own message', '随机，User 消息': 'random, User message'}
REPS = 4000
NEW_S, NEW_T = 'singlepass_swap/wording_20261109', 'thinking_swap/wording_20261109'


def readouts(r):
    """name: (label as archived, English label, thinking, paper anchor, pron, role)"""
    A3, M, A = r.res(W3), r.res(MS), r.res(WS)
    S = lambda x: result_path(A, f'{NEW_S}/{x}')  # noqa: E731
    T = lambda x: result_path(A, f'{NEW_T}/{x}')  # noqa: E731
    return {
        'qsp': ('Qwen3.8-27B，非思考，中文', 'Qwen3.8-27B, no thinking, Chinese items', False,
                run_path(A3, 'singlepass_swap/qwen38_27b_r05m_msg'), S('pron'), S('role')),
        'qth': ('Qwen3.8-27B，思考，中文', 'Qwen3.8-27B, thinking, Chinese items', True,
                run_path(A3, 'thinking_swap/qwen38_27b_r05m_msg'), T('qth_pron'), T('qth_role')),
        'qsp_en': ('Qwen3.8-27B，非思考，英文', 'Qwen3.8-27B, no thinking, English items', False, S('qsp_en_paper'), S('qsp_en_pron'), S('qsp_en_role')),
        'qth_en': ('Qwen3.8-27B，思考，英文', 'Qwen3.8-27B, thinking, English items', True, T('qth_en_paper'), T('qth_en_pron'), T('qth_en_role')),
        'g20': ('gpt-oss-20b，思考', 'gpt-oss-20b, thinking', True, run_path(A3, 'thinking_swap/gpt_oss_20b_r05m_msg'), T('g20_pron'), T('g20_role')),
        'g120': ('gpt-oss-120b，思考', 'gpt-oss-120b, thinking', True, run_path(A3, 'thinking_swap/gpt_oss_120b_r05m_msg'), T('g120_pron'),
                 T('g120_role')),
        'm26': ('MiMo-V2.6，默认调用', 'MiMo-V2.6-Distill-Qwen-9B, default call', True, result_path(M, 'thinking_swap/mimo_v26_r05m_msg'),
                T('m26_pron'), T('m26_role'))}


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


def summary(RO):
    S, C = {}, {}
    for ro, (label, _, th, *dirs) in RO.items():
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
    return S, C


def markdown(S, C, RO):
    WL = {'paper': 'paper (main)', 'pron': 'pronoun', 'role': 'role name'}
    L = ['# Wording of the attribution question: every readout of the causal tests', '',
         'fit_m, the message edited only, 8 dimensions at every layer, 75 held-out bodies x 2 option orders. Without thinking: '
         'probabilities; with thinking: shares of the finished items. "Names the model": You (the AI assistant) / You / The AI assistant '
         '(Chinese items: 你（AI 助手）/ 你 / AI 助手); "names the user": The user / Me / The user (用户 / 我 / 用户). "Gap closed": the share '
         'of the unedited A-U gap that the edit closes (1 = as the other role), 95% interval over the 15 wording families.', '']
    for ro, s in S.items():
        L += [f"## {RO[ro][1]}", '', '| wording | unedited own message: model / user | unedited User message: model / user | removal: change '
              '(model) | removal: gap closed | installation: change (model) | installation: gap closed | random (own / User) |',
              '|---|---|---|---|---|---|---|---|']
        for w, x in s['wordings'].items():
            if 'note' in x:
                L.append(f'| {WL[w]} | {x["note"]} | | | | | | |')
                continue
            a, u = x['unedited']['base_A'], x['unedited']['base_U']
            cell = lambda nm, k: (f"{x[nm][k]['mean']:+.3f} [{x[nm][k]['ci_families'][0]:+.2f}, {x[nm][k]['ci_families'][1]:+.2f}]" if nm in x else '—')  # noqa: E731
            shr = lambda nm: (f"{x[nm]['self']['share']:.2f} [{x[nm]['self']['share_ci_families'][0]:.2f}, {x[nm]['self']['share_ci_families'][1]:.2f}]" if nm in x else '—')  # noqa: E731
            rnd = ' / '.join(f"{x[nm]['self']['mean']:+.3f}" if nm in x else '—' for nm in ('随机，自己的消息', '随机，User 消息'))
            L.append(f"| {WL[w]} | {a['self']:.3f} / {a['user']:.3f} | {u['self']:.3f} / {u['user']:.3f} | {cell('去除', 'self')} | {shr('去除')} | "
                     f"{cell('植入', 'self')} | {shr('植入')} | {rnd} |")
        if s['thinking']:
            L += ['', 'Finished and unparsed answers: ' + '; '.join(
                f"{WL[w]}: unedited finished {x['unedited']['base_A'].get('finished')}/{x['unedited']['base_U'].get('finished')}, "
                f"unparsed {sum(x['unparsed'].values())}" for w, x in s['wordings'].items() if 'note' not in x)]
        L.append('')
    L += ['## Contrasts between wordings (difference of the gap closed, 95% interval over the families)', '',
          '| readout | wordings | edit | option | difference | interval | items |', '|---|---|---|---|---|---|---|']
    for k, v in C.items():
        ro, pair, name, o = k.split(' | ')
        L.append(f"| {ro} | {pair} | {EDIT_EN[name]} | {o} | {v['diff_share']:+.3f} | [{v['ci_families'][0]:+.3f}, {v['ci_families'][1]:+.3f}] | {v['n']} |")
    return '\n'.join(L) + '\n'


def main(argv=None):
    args = arguments(__doc__).parse_args(argv)
    r = Roots(args)
    RO = readouts(r)
    S, C = summary(RO)
    out = r.out(WS + '/results')
    (out / 'wording_summary_all.json').write_text(json.dumps({'by_readout': S, 'contrasts': C}, ensure_ascii=False, indent=1), encoding='utf-8')
    write_text(out / 'wording_summary_all.md', markdown(S, C, RO))
    print('readouts:', ', '.join(S), '| written', out / 'wording_summary_all.json')


if __name__ == '__main__':
    main()
