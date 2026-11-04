"""Controls and variants of the causal tests, from the per-item results of the replacement runs. Statistics as in the paper
(common.items, common.summarize): the share answering "you" (thinking) or mean P("you") (no thinking), and the item-paired
difference from the unedited condition of the same recipient with a 95% bootstrap interval. Eight analyses, run in this order:

without_question (stats/r05r06_20261031.json, .md): the edited conditions of the paper, removal swU8_A and installation swA8_U,
  k = 8, per readout:
    r05q_all  the paper's fit (fit_q) and positions (all), with the random controls of base seed 20261022 (rerun of the paper's run)
    r05q_msg  fit_q, edits only in the message (body and end)
    r05m_msg  fit_m (fitted on the message only), edits only in the message: neither the fit nor the edit sees the question
    r05m_all  fit_m, all positions
    r06_s1..4 random subspaces of base seeds 20261023..20261026, all positions (with r05q_all: five consecutive seeds)
  Dose: per edited item, the relative size of the edit over all edited layers and positions, sqrt(sum ||delta||^2 / sum ||x||^2).
  The ratio random / V is also given without the final layer (the band ends at the model's last block, whose output at the
  edited prompt positions reaches no later layer and no read logit: the read position is not edited) and per quarter of the
  layers. Runs without summary.json (not finished) are left out and listed. The paper's run (v4_mechanism.json, 'all') is shown
  alongside.
random_floor (stats/r06f_20261031.json, .md): each random edit (same replacement in a random 8-dim subspace, base seeds
  20261022..26) scaled up per item and per layer to at least the realized relative size of the V edit on the same item
  (r05q_all; never scaled down); the unedited items are those of r05q_all, copied into every run.
    frswU8_A  random, own message (floor: the V removal swU8_A)
    frswA8_U  random, User message (floor: the V installation swA8_U)
  Dose check: per item, the realized relative size of the scaled random edit over all edited layers divided by that of the V
  edit of the same item, and per layer the smallest ratio of the two relative sizes (>= 1 if the floor held). For comparison:
  the V edit and the unscaled random edit of the same base seeds (stats/r05r06_20261031.json).
exchange (stats/r08_20261031.json, .md): do the self-index coordinates also tell one external source from the other? A User and a
  System message exchange their eight coordinates (User <- System: swS8_U; System <- User: swU8_S), with the same exchange in the
  random subspace of base seed 20261022 (rswS8_U, rswU8_S); fit_q, all positions, k = 8. The unedited own and User items are
  those of r05q_all (one baseline for these analyses). For comparison, the installation of the own coordinates into the User
  message (swA8_U) is read from r05q_all. Per answer (you / user / system): the share of finished items so answered (thinking)
  or the mean probability of that option (no thinking), and the item-paired difference from the unedited message of the same role.
exchange_traces (stats/r08_traces_20261101.json, .md): the gpt-oss template renders a System message as a developer message; the
  traces that call the edited System message "the developer message" are counted with one rule for the exchange runs and, for
  comparison, the paper's runs:
    rule: "developer message" in the generated text, case-insensitive, over all 150 items of a condition (finished or not)
    runs: thinking_swap/<model>_r08_us (base_S, swU8_S: the User coordinates written into a System message, rswU8_S: the same in
          the random subspace; also base_U, swS8_U, rswS8_U) and thinking_swap/<model>_all (base_S, swA8_S)
  Also given per condition: the answers of the finished items (you / user / system).
rest (stats/r03_20261101.json, .md): is the source read through the eight coordinates or through the rest of the state
  (epsilon of Equation 1)?
  Part A (<dir>_r03c_all): at every layer of the band and aligned position, all positions, fit_q, k = 8,
    V  the recipient takes the donor's 8 coordinates (the paper's edit; swU8_A / swA8_U of <dir>_r05q_all)
    C  the recipient takes the donor's state outside the 8 directions and keeps its own coordinates inside them (cswU8_A / cswA8_U)
    F  the recipient takes the donor's whole state (fswU8_A / fswA8_U)
    R  V in the random subspace of base seed 20261022 (rswU8_A / rswA8_U of <dir>_r05q_all)
  Removal: the own message given the User state; installation: the User message given the own state. The unedited items of
  r05q_all are copied into r03c_all. A thinking cell with unfinished items also gets the bounds of the change with the unfinished
  items counted as "you" and as not "you". Edit size: per item sqrt(sum of squared changes / sum of squared states) over all
  layers and edited positions, median over items.
  Part B (<dir>_r03fd): central differences D_v f of f = logit("you" letter) - logit("user" letter), eps = 0.25, along v = d
  (total), P_V d (V), d - P_V d (perp) and P_R d (R), d = h^A - h^U, added at one layer at a time; per item the sum over the
  computed layers (the first-order change of f when the role difference is applied at every such layer), mean over the items
  complete at both endpoints, with the share of D_V and D_perp in D_total (95% bootstrap interval of the ratio of means, seed 1).
  The logits are bfloat16, so every D is a multiple of 0.25; the share of zero entries is given.
fd_layers (stats/r03b_layers_20261101.json, .md): the first-order sensitivities of <dir>_r03fd by depth instead of summed over
  layers. Per readout and endpoint (A: the model's own message, U: the User message), per computed layer: mean over items of D
  along d (total), its part in the fitted subspace (V), the rest (perp), the random part (R) and the unperturbed rows (zero, the
  numerical noise), and the median share of the squared norm of d in the subspace. Bands: the thirds of the layers, and the
  window band in which installation exceeds +0.10 in the layer-window runs of the paper (Table tab:self_mechanism_windows:
  Qwen3.8-27B L24-L47 without and L28-L43 with thinking, gpt-oss-20b L0-L15, gpt-oss-120b L8-L23). Shares are ratios of summed
  means (as in rest), with a 95% bootstrap interval over items (2000 draws, seed 1).
families (stats/r09_cluster_20261101.json, .md): uncertainty unit, completion and robustness of the causal tests, from the items of
  the paper's main runs. The 75 held-out bodies are 15 wording families x 5 numerical variants (item id
  role150_<family>_c<variant>), each in two option orders. Per readout and condition, against the unedited message of the same role:
    - n, finished (edited / unedited), and the paired intersection (both finished), on which the paper's change is computed;
    - the change with the paper's item bootstrap (common.boot, seed 1, 4000 draws) and with a family-cluster bootstrap (15 families
      drawn with replacement, every item of a drawn family kept with its option orders and paired conditions; seed 1, 4000 draws);
    - with thinking, bounds over all items that treat an unfinished output as unknown (worst case for the edited minus unedited
      share);
    - control tasks (content, arithmetic; 75 items per condition): paired change in accuracy, item and family-cluster intervals.
strict (stats/p56full_20261101.json, .md): the stricter versions of the two self-related tasks, in all four readouts, with fit_m
  (fitted on message positions only) and the paper's k = 8. Of five eight-item pilots with rules fixed before any pilot ran (T1
  attribution in a held-out wording: signal; T2 numbered options: none; T3 game: signal; T4 polarity: none; T5 content
  direction: not valid), T1 and T3 were run in full:
    T1 <dir>_p56full_alt: the attribution question in a held-out wording (attr_alt), the message edited only, 75 held-out bodies x
       2 option orders. Reference: <dir>_r05m_msg, the same fit and region with the paper's wording.
    T3 games_swap/<model>_p56full_game_{main,control}: the secret game, the coordinates edited at X only (the setter's or the
       user's announcement), all games. Reference: the paper's game runs (*_s6_main / *_s6_control; the paper's fit, all
       positions).
  T1: share of finished items answered "you" (thinking) or mean probability of the "you" option (no thinking), item-paired
  change from the unedited message of the same role, 95% bootstrap interval (seed 1). T3: share of finished games in which the
  model acts as the setter, item-paired change on games finished in both conditions (causal_tests.game_summary); control: share
  answered correctly. Runs not in the package are left out.

usage: causal_controls.py --results RES --out OUT [--item-cache DIR] [--only NAME[,NAME...]]
       (needs OUT/evidence/p2_crossmodel_wave3_20261019/stats/v4_mechanism.json from causal_tests.py)
"""
import json
import math
import os
import random
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import W3, BASE, Roots, arguments, boot, items, run_path, summarize  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------------
# without_question: fit and edit without the question; five random seeds

READOUTS = [('qsp', 'qwen_sp', 'singlepass_swap/qwen38_27b', 'Qwen3.8-27B, no thinking'),
            ('qth', 'qwen_th', 'thinking_swap/qwen38_27b', 'Qwen3.8-27B, thinking'),
            ('g20', 'oss_th', 'thinking_swap/gpt_oss_20b', 'gpt-oss-20b, thinking'),
            ('g120', 'oss120_th', 'thinking_swap/gpt_oss_120b', 'gpt-oss-120b, thinking')]
Q_RUNS = ['r05q_all', 'r05q_msg', 'r05m_msg', 'r05m_all', 'r06_s1', 'r06_s2', 'r06_s3', 'r06_s4']
SEED = {'r05q_all': 20261022, 'r06_s1': 20261023, 'r06_s2': 20261024, 'r06_s3': 20261025, 'r06_s4': 20261026}
EDITED = ('swU8_A', 'swA8_U', 'rswU8_A', 'rswA8_U')


def layer_sets(n):
    last = n - 1
    sets = [('without_final', lambda l: l != last)]
    for j in range(4):
        sets.append((f'quarter{j + 1}', lambda l, j=j: j * n // 4 <= l < (j + 1) * n // 4 and l != last))
    return sets


def dose(recs, n_layers):
    out = {}
    for c in EDITED:
        ds = [r['dose'] for r in recs if r['cond'] == c and r.get('dose')]
        if ds:
            rel = [math.sqrt(sum(d['edit2']) / sum(d['state2'])) for d in ds]
            q = statistics.quantiles(rel, n=4)
            out[c] = {'n': len(rel), 'median': statistics.median(rel), 'q1': q[0], 'q3': q[2],
                      'n_pos_median': statistics.median(d['n_pos'] for d in ds),
                      'final_share': sum(d['edit2'][d['layers'].index(n_layers - 1)] for d in ds) / sum(sum(d['edit2']) for d in ds)}
            for name, keep in layer_sets(n_layers):
                ix = [[i for i, l in enumerate(d['layers']) if keep(l)] for d in ds]
                out[c]['median_' + name] = statistics.median(
                    math.sqrt(sum(d['edit2'][i] for i in k) / sum(d['state2'][i] for i in k)) for d, k in zip(ds, ix))
    return out


def f(s, key='diff'):
    if not s or key not in s:
        return '—'
    lo, hi = s[key + '_ci'] if key == 'diff' else s['ci']
    return f"{s[key]:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def without_question(A, S):
    paper = json.load(open(S / 'v4_mechanism.json', encoding='utf-8'))['results']
    res, missing = {}, []
    for rd, key, root, _ in READOUTS:
        res[rd] = {'paper_all': {c: paper[key]['all'].get(c) for c in ('base_A', 'base_U') + EDITED}}
        for run in Q_RUNS:
            d = run_path(A, f'{root}_{run}')
            if not (d / 'summary.json').exists():
                missing.append(f'{root}_{run}')
                continue
            recs = items(str(d))
            s = summarize(recs, None)
            n_layers = json.load(open(d / 'identity.json', encoding='utf-8'))['n_layers']
            res[rd][run] = {'stats': s, 'dose': dose(recs, n_layers), 'n_items': len(recs)}
    L = ['# R05 / R06 (10-30/31)', '',
         'Item-paired difference from the unedited condition (95% bootstrap interval); thinking: share answering "you", '
         'no thinking: mean P("you"). k = 8.', '']
    L += ['## R05: removal (← User, in the own message) and installation (← own, in the User message)', '',
          '| readout | edit | paper | fit_q, all | fit_q, message | fit_m, message | fit_m, all |', '|---|---|---|---|---|---|---|']
    for rd, _, _, name in READOUTS:
        for c, lab in (('swU8_A', 'removal'), ('swA8_U', 'installation')):
            row = [name, lab, f(res[rd]['paper_all'].get(c))]
            row += [f(res[rd][r]['stats'].get(c)) if r in res[rd] else '—' for r in ('r05q_all', 'r05q_msg', 'r05m_msg', 'r05m_all')]
            L.append('| ' + ' | '.join(row) + ' |')
    L += ['', '## R06: random subspaces, five consecutive base seeds (all positions)', '',
          '| readout | edit | paper (20261022) | ' + ' | '.join(str(v) for v in SEED.values()) + ' | max abs |', '|---' * 9 + '|']
    for rd, _, _, name in READOUTS:
        for c, lab in (('rswU8_A', 'random, own message'), ('rswA8_U', 'random, User message')):
            vals = [res[rd][r]['stats'].get(c) if r in res[rd] else None for r in SEED]
            ds = [v['diff'] for v in vals if v and 'diff' in v]
            row = [name, lab, f(res[rd]['paper_all'].get(c))] + [f(v) for v in vals]
            row.append(f'{max(abs(x) for x in ds):.3f} (n={len(ds)})' if ds else '—')
            L.append('| ' + ' | '.join(row) + ' |')
    L += ['', '## Dose: relative size of the edit, sqrt(sum ||delta||^2 / sum ||x||^2), median [quartiles] over items', '',
          '| readout | run | ← User (V) | ← own (V) | ← random, own message (R) | ← random, User message (R) | positions (median, ← User) |',
          '|---|---|---|---|---|---|---|']
    for rd, _, _, name in READOUTS:
        for r in ('r05q_all', 'r05q_msg', 'r05m_msg', 'r05m_all'):
            if r not in res[rd]:
                continue
            D = res[rd][r]['dose']
            cell = lambda c: f"{D[c]['median']:.4f} [{D[c]['q1']:.4f}, {D[c]['q3']:.4f}]" if c in D else '—'  # noqa: E731
            L.append(f"| {name} | {r} | {cell('swU8_A')} | {cell('swA8_U')} | {cell('rswU8_A')} | {cell('rswA8_U')} | "
                     f"{D['swU8_A']['n_pos_median'] if 'swU8_A' in D else '—'} |")
    keys = ['median', 'median_without_final'] + [f'median_quarter{j}' for j in range(1, 5)]
    L += ['', '## Dose ratio random / V by layers', '',
          'V: r05q_all; random: median over the five base seeds (r05q_all, r06_s1..4). The final layer\'s output at the edited prompt '
          'positions reaches no later layer and no read logit; the quarters split the layers 0..n-1, the fourth without the final one. '
          'Last column: the final layer\'s share of the summed squared V edit.', '',
          '| readout | edit | all layers | without final | quarter 1 | quarter 2 | quarter 3 | quarter 4 | final layer, share of V |',
          '|---|---|---|---|---|---|---|---|---|']
    for rd, _, _, name in READOUTS:
        if 'r05q_all' not in res[rd]:
            continue
        for v, rr, lab in (('swU8_A', 'rswU8_A', 'removal'), ('swA8_U', 'rswA8_U', 'installation')):
            Dv = res[rd]['r05q_all']['dose'].get(v)
            Dr = [res[rd][r]['dose'][rr] for r in SEED if r in res[rd] and rr in res[rd][r]['dose']]
            if Dv and Dr:
                L.append(f'| {name} | {lab} | ' + ' | '.join(f"{statistics.median(x[kk] for x in Dr) / Dv[kk]:.2f}" for kk in keys)
                         + f" | {Dv['final_share']:.2f} |")
    if missing:
        L += ['', 'Not finished (left out): ' + ', '.join(missing)]
    (S / 'r05r06_20261031.json').write_text(json.dumps({'results': res, 'missing': missing}, indent=1), encoding='utf-8')
    (S / 'r05r06_20261031.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------------------------------------
# random_floor: random edits scaled to at least the size of the fitted edit

F_RUNS = [f'r06f_s{j}' for j in range(5)]
PAIRS = (('frswU8_A', 'swU8_A', 'rswU8_A', 'removal (random, own message)'), ('frswA8_U', 'swA8_U', 'rswA8_U', 'installation (random, User message)'))


def rel(d, keep=None):
    ix = [i for i, l in enumerate(d['layers']) if keep is None or l in keep]
    return math.sqrt(sum(d['edit2'][i] for i in ix) / sum(d['state2'][i] for i in ix))


def dose_check(recs, vrecs, c, vc):
    V = {(r['id'], tuple(r['order'])): r['dose'] for r in vrecs if r['cond'] == vc and r.get('dose')}
    ratio, lmin, up, sc = [], [], 0, []
    for r in recs:
        if r['cond'] != c or not r.get('dose') or (r['id'], tuple(r['order'])) not in V:
            continue
        d, v = r['dose'], V[(r['id'], tuple(r['order']))]
        ratio.append(rel(d) / rel(v))
        per = [math.sqrt(d['edit2'][i] / d['state2'][i]) / math.sqrt(v['edit2'][j] / v['state2'][j])
               for i, l in enumerate(d['layers']) for j, m in enumerate(v['layers']) if l == m and v['edit2'][j] > 0]
        lmin.append(min(per))
        s = r['floor']['scale']
        up += sum(x > 1 for x in s)
        sc += s
    if not ratio:
        return None
    return {'n': len(ratio), 'ratio_median': statistics.median(ratio), 'ratio_min': min(ratio), 'layer_ratio_min': min(lmin),
            'scaled_up_share': up / len(sc), 'scale_median': statistics.median(sc)}


def random_floor(A, S):
    prev = json.load(open(S / 'r05r06_20261031.json', encoding='utf-8'))['results']
    res, missing = {}, []
    for rd, _, root, _ in READOUTS:
        vrecs = items(str(run_path(A, f'{root}_r05q_all')))
        res[rd] = {'V': {vc: prev[rd]['r05q_all']['stats'].get(vc) for _, vc, _, _ in PAIRS},
                   'unscaled': {rc: [prev[rd][r]['stats'].get(rc) for r in SEED if r in prev[rd]] for _, _, rc, _ in PAIRS}}
        for run in F_RUNS:
            d = run_path(A, f'{root}_{run}')
            if not (d / 'summary.json').exists():
                missing.append(f'{root}_{run}')
                continue
            recs = items(str(d))
            res[rd][run] = {'stats': summarize(recs, None), 'dose': {c: dose_check(recs, vrecs, c, vc) for c, vc, _, _ in PAIRS},
                            'n_items': len(recs)}
    L = ['# R06 follow-up: dose-floor random control (10-31)', '',
         'Item-paired difference from the unedited condition (95% bootstrap interval); thinking: share answering "you", '
         'no thinking: mean P("you"). k = 8, all positions. The random edit of each base seed is scaled up per item and layer to at '
         'least the realized relative size of the V edit on the same item (never down).', '',
         '## Effect', '',
         '| readout | edit | V (r05q_all) | random, unscaled: max abs over 5 seeds | ' + ' | '.join(str(v) for v in SEED.values())
         + ' | max abs | seeds with interval excluding 0 |', '|---' * 11 + '|']
    for rd, _, _, name in READOUTS:
        for c, vc, rc, lab in PAIRS:
            un = [x['diff'] for x in res[rd]['unscaled'][rc] if x and 'diff' in x]
            vals = [res[rd][r]['stats'].get(c) if r in res[rd] else None for r in F_RUNS]
            ds = [v for v in vals if v and 'diff' in v]
            excl = sum(v['diff_ci'][0] > 0 or v['diff_ci'][1] < 0 for v in ds)
            row = [name, lab, f(res[rd]['V'].get(vc)), f'{max(abs(x) for x in un):.3f} (n={len(un)})' if un else '—']
            row += [f(v) for v in vals]
            row += [f"{max(abs(v['diff']) for v in ds):.3f} (n={len(ds)})" if ds else '—', f'{excl} of {len(ds)}']
            L.append('| ' + ' | '.join(row) + ' |')
    L += ['', '## Dose check', '',
          'Per item: relative size of the scaled random edit over all layers / that of the V edit of the same item (median, minimum '
          'over items); the smallest per-layer ratio over all items and layers (>= 1 if the floor held); the share of item-layers '
          'scaled up and the median scale factor. Pooled over the five seeds.', '',
          '| readout | edit | items | ratio, median | ratio, min | per-layer ratio, min | layers scaled up | scale, median |',
          '|---|---|---|---|---|---|---|---|']
    for rd, _, _, name in READOUTS:
        for c, _, _, lab in PAIRS:
            D = [res[rd][r]['dose'][c] for r in F_RUNS if r in res[rd] and res[rd][r]['dose'].get(c)]
            if not D:
                continue
            n = sum(x['n'] for x in D)
            L.append(f"| {name} | {lab} | {n} | {statistics.median(x['ratio_median'] for x in D):.2f} | {min(x['ratio_min'] for x in D):.2f} | "
                     f"{min(x['layer_ratio_min'] for x in D):.3f} | {sum(x['scaled_up_share'] * x['n'] for x in D) / n:.2f} | "
                     f"{statistics.median(x['scale_median'] for x in D):.2f} |")
    if missing:
        L += ['', 'Not finished (left out): ' + ', '.join(missing)]
    (S / 'r06f_20261031.json').write_text(json.dumps({'results': res, 'missing': missing}, indent=1), encoding='utf-8')
    (S / 'r06f_20261031.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------------------------------------
# exchange: the coordinates of a User and a System message exchanged

ANSWERS = ('self', 'user', 'system')
X_EDITS = [('swS8_U', 'User ← System'), ('rswS8_U', 'User ← System, random'), ('swU8_S', 'System ← User'),
           ('rswU8_S', 'System ← User, random'), ('swA8_U', 'User ← own (R05)')]


def x_value(r, a):
    if 'p' in r:
        return r['p'][a]
    return float(r.get('on_answer') == a) if r.get('finished', True) else None


def x_level(rs):
    out = {'n': len(rs)}
    if rs and 'p' not in rs[0]:
        out['finished'] = sum(r.get('finished', True) for r in rs)
    for a in ANSWERS:
        v = [x for x in (x_value(r, a) for r in rs) if x is not None]
        out[a] = sum(v) / len(v) if v else None
    return out


def paired(rs, bs):
    b = {(r['id'], tuple(r['order'])): r for r in bs}
    out = {}
    for a in ANSWERS:
        d = [x_value(r, a) - x_value(b[k], a) for r in rs for k in [(r['id'], tuple(r['order']))]
             if k in b and x_value(r, a) is not None and x_value(b[k], a) is not None]
        if d:
            out[a] = {'diff': sum(d) / len(d), 'ci': list(boot(d, seed=1)), 'n_pairs': len(d)}
    return out


def exchange(A, S):
    res, missing = {}, []
    for rd, _, root, _ in READOUTS:
        d = run_path(A, f'{root}_r08_us')
        if not (d / 'summary.json').exists():
            missing.append(f'{root}_r08_us')
            continue
        recs = items(str(d))
        by = {}
        for r in recs:
            by.setdefault(r['cond'], []).append(r)
        by['swA8_U'] = [r for r in items(str(run_path(A, f'{root}_r05q_all'))) if r['cond'] == 'swA8_U']
        res[rd] = {'levels': {c: x_level(rs) for c, rs in sorted(by.items())}, 'n_items': len(recs)}
        for c, _ in X_EDITS:
            if c in by:
                res[rd][c] = paired(by[c], by[BASE[by[c][0]['recipient']]])
    fmt = lambda x: '—' if x is None else f'{x:.3f}'  # noqa: E731
    lv = lambda s: ' / '.join(fmt(s.get(a)) for a in ANSWERS)  # noqa: E731
    dd = lambda s, a: f"{s[a]['diff']:+.3f} [{s[a]['ci'][0]:+.3f}, {s[a]['ci'][1]:+.3f}]" if s and a in s else '—'  # noqa: E731
    L = ['# R08: exchange of the User and System coordinates (10-31)', '',
         'Thinking: share of finished items answered you / user / system; no thinking: mean probability of the three options. '
         'Changes: item-paired difference from the unedited message of the same role (95% bootstrap interval). k = 8, fit_q, '
         'all positions; random = the same exchange in the random subspace of base seed 20261022.', '',
         '| readout | message | edit | unedited you / user / system | edited you / user / system | change in "you" | change in "user" | '
         'change in "system" |', '|---|---|---|---|---|---|---|---|']
    for rd, _, _, name in READOUTS:
        if rd not in res:
            continue
        Lv = res[rd]['levels']
        for c, lab in X_EDITS:
            if c not in Lv:
                continue
            msg, base = ('User', 'base_U') if c.endswith('_U') else ('System', 'base_S')
            L.append(f'| {name} | {msg} | {lab} | {lv(Lv[base])} | {lv(Lv[c])} | {dd(res[rd].get(c), "self")} | '
                     f'{dd(res[rd].get(c), "user")} | {dd(res[rd].get(c), "system")} |')
    L += ['', 'Items: ' + ', '.join(f"{name} {res[rd]['n_items']}" for rd, _, _, name in READOUTS if rd in res)]
    if missing:
        L += ['', 'Not finished (left out): ' + ', '.join(missing)]
    (S / 'r08_20261031.json').write_text(json.dumps({'results': res, 'missing': missing}, indent=1), encoding='utf-8')
    (S / 'r08_20261031.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------------------------------------
# exchange_traces: "developer message" in the reasoning traces of the gpt-oss models

T_MODELS = [('g20', 'gpt_oss_20b', 'gpt-oss-20b'), ('g120', 'gpt_oss_120b', 'gpt-oss-120b')]
T_RUNS = [('r08_us', ['base_S', 'swU8_S', 'rswU8_S', 'base_U', 'swS8_U', 'rswS8_U']), ('all', ['base_S', 'swA8_S'])]
T_NAMES = {'base_S': 'System, unedited', 'swU8_S': 'System <- User', 'rswU8_S': 'System <- User, random',
           'base_U': 'User, unedited', 'swS8_U': 'User <- System', 'rswS8_U': 'User <- System, random', 'swA8_S': 'System <- own'}


def exchange_traces(A, S):
    res = {}
    for key, d, _ in T_MODELS:
        res[key] = {}
        for run, conds in T_RUNS:
            by = {}
            for r in items(str(run_path(A, f'thinking_swap/{d}_{run}')), cache=False):  # reads the generated text
                by.setdefault(r['cond'], []).append(r)
            res[key][run] = {}
            for c in conds:
                rs = by[c]
                fin = [r for r in rs if r.get('finished')]
                ans = Counter(r.get('on_answer') for r in fin)
                res[key][run][c] = {'n': len(rs), 'developer_message': sum('developer message' in r['gen'].lower() for r in rs),
                                    'finished': len(fin), 'answers': {k: ans.get(k, 0) for k in ('self', 'user', 'system')}}
    L = ['# R08: reasoning traces of the gpt-oss models for System and User messages (11-01)', '',
         'Traces that contain "developer message" (case-insensitive), over all items of a condition; answers of the finished items. '
         'Runs: thinking_swap/<model>_r08_us (exchange of the User and System coordinates, k = 8, fit_q, all positions) and '
         'thinking_swap/<model>_all (the paper\'s runs).', '',
         '| model | run | condition | items | "developer message" | finished | you / user / system |', '|---|---|---|---|---|---|---|']
    for key, _, name in T_MODELS:
        for run, conds in T_RUNS:
            for c in conds:
                s = res[key][run][c]
                a = s['answers']
                L.append(f"| {name} | {run} | {T_NAMES[c]} | {s['n']} | {s['developer_message']} | {s['finished']} | "
                         f"{a['self']} / {a['user']} / {a['system']} |")
    (S / 'r08_traces_20261101.json').write_text(json.dumps({'results': res}, indent=1), encoding='utf-8')
    (S / 'r08_traces_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------------------------------------
# rest: the eight coordinates against the rest of the state (part A) and the first-order sensitivity (part B)

R_EDITS = [('removal', 'base_A', [('V', 'swU8_A'), ('C', 'cswU8_A'), ('F', 'fswU8_A'), ('R', 'rswU8_A')]),
           ('installation', 'base_U', [('V', 'swA8_U'), ('C', 'cswA8_U'), ('F', 'fswA8_U'), ('R', 'rswA8_U')])]
FROM_R05 = ('swU8_A', 'swA8_U', 'rswU8_A', 'rswA8_U')
R_DIRS = ('total', 'V', 'perp', 'R')


def you(r, unfinished=None):
    if 'p' in r:
        return r['p']['self']
    if not r.get('finished', True):
        return unfinished
    return float(r.get('on_answer') == 'self')


def level(rs):
    v = [x for x in (you(r) for r in rs) if x is not None]
    out = {'n': len(rs), 'you': sum(v) / len(v) if v else None}
    if rs and 'p' not in rs[0]:
        out['finished'] = sum(r.get('finished', True) for r in rs)
    return out


def size(r):
    d = r.get('dose')
    if not d or not d.get('state2'):
        return None
    return math.sqrt(sum(d['edit2']) / sum(d['state2']))


def change(rs, bs):
    b = {(r['id'], tuple(r['order'])): r for r in bs}
    pairs = [(r, b[k]) for r in rs for k in [(r['id'], tuple(r['order']))] if k in b]
    d = [you(r) - you(q) for r, q in pairs if you(r) is not None and you(q) is not None]
    out = {'diff': sum(d) / len(d), 'ci': list(boot(d, seed=1)), 'n_pairs': len(d)}
    if any(not r.get('finished', True) for r, _ in pairs):
        bounds = []
        for u in (0.0, 1.0):
            e = [you(r, u) - you(q) for r, q in pairs if you(q) is not None]
            bounds.append(sum(e) / len(e))
        out['bounds'] = bounds
    s = [x for x in (size(r) for r in rs) if x is not None]
    out['size'] = statistics.median(s) if s else None
    return out


def part_b(d):
    recs = items(str(d))
    ok = [r for r in recs if all('D' in r['endpoints'].get(e, {}) for e in ('A', 'U'))]
    out = {'n_items': len(recs), 'n_complete': len(ok)}
    for e in ('A', 'U'):
        S = {k: [sum(r['endpoints'][e]['D'][k]) for r in ok] for k in R_DIRS}
        mean = {k: sum(v) / len(v) for k, v in S.items()}
        rng = random.Random(1)
        share = {}
        for k in ('V', 'perp'):
            bs = []
            for _ in range(4000):
                ix = [rng.randrange(len(ok)) for _ in ok]
                t = sum(S['total'][i] for i in ix)
                if t:
                    bs.append(sum(S[k][i] for i in ix) / t)
            bs.sort()
            share[k] = {'ratio': mean[k] / mean['total'] if mean['total'] else None,
                        'ci': [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]] if bs else None}
        zero = sum(x == 0 for r in ok for x in r['endpoints'][e]['D']['total']) / sum(len(r['endpoints'][e]['D']['total']) for r in ok)
        out[e] = {'mean_sum': mean, 'share': share, 'zero_total': zero,
                  'zero_dir_nonzero': sum(x != 0 for r in ok for x in r['endpoints'][e]['D']['zero'])}
    out['norm_share'] = {k: statistics.median(sum(r['norms'][k]) / sum(r['norms']['total']) for r in ok) for k in ('V', 'perp', 'R')}
    out['n_layers'] = len(ok[0]['layers']) if ok else 0
    return out


def rest(A, S):
    res, missing = {}, []
    for rd, _, root, _ in READOUTS:
        da, db = run_path(A, f'{root}_r03c_all'), run_path(A, f'{root}_r03fd')
        if not (da / 'summary.json').exists():
            missing.append(f'{root}_r03c_all')
            continue
        by = {}
        for r in items(str(da)):
            by.setdefault(r['cond'], []).append(r)
        for r in items(str(run_path(A, f'{root}_r05q_all'))):
            if r['cond'] in FROM_R05:
                by.setdefault(r['cond'], []).append(r)
        res[rd] = {'levels': {c: level(rs) for c, rs in sorted(by.items())}}
        for _, base, eds in R_EDITS:
            for _, c in eds:
                if c in by:
                    assert BASE[by[c][0]['recipient']] == base
                    res[rd][c] = change(by[c], by[base])
        if (db / 'summary.json').exists():
            res[rd]['B'] = part_b(db)
        else:
            missing.append(f'{root}_r03fd')
    f3 = lambda x: '—' if x is None else f'{x:.3f}'  # noqa: E731
    ch = lambda s: f"{s['diff']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]" if s else '—'  # noqa: E731
    L = ['# R03: is the source read through the eight coordinates or through the rest of the state? (10-31)', '',
         '## Part A: replacing the coordinates (V), everything else (C), the whole state (F)', '',
         'Thinking: share of finished items answered "you"; no thinking: mean probability of the "you" option. Change: item-paired '
         'difference from the unedited message of the same role (95% bootstrap interval). V = the paper\'s edit (8 coordinates of the '
         'donor); C = the donor\'s state outside the 8 directions, the recipient\'s coordinates inside them; F = the donor\'s whole '
         'state; R = V in a random 8-dim subspace (base seed 20261022). fit_q, all positions, every layer of the band. V and R are the '
         'items of r05q_all. Size: median over items of the edit\'s norm relative to the state\'s (all layers).', '',
         '| readout | test | unedited "you" | basis | edited "you" | change in "you" | size | finished |', '|---|---|---|---|---|---|---|---|']
    notes = []
    for rd, _, _, name in READOUTS:
        if rd not in res:
            continue
        Lv = res[rd]['levels']
        for test, base, eds in R_EDITS:
            for w, c in eds:
                if c not in Lv:
                    continue
                s = res[rd].get(c)
                fin = f"{Lv[c]['finished']}/{Lv[c]['n']}" if 'finished' in Lv[c] else '—'
                L.append(f"| {name} | {test} | {f3(Lv[base]['you'])} | {w} | {f3(Lv[c]['you'])} | {ch(s)} | "
                         f"{f3(s.get('size') if s else None)} | {fin} |")
                if s and 'bounds' in s:
                    notes.append(f"{name}, {test}, {w}: {Lv[c]['finished']} of {Lv[c]['n']} finished; change {s['bounds'][0]:+.3f} with the "
                                 f"unfinished items counted as not \"you\", {s['bounds'][1]:+.3f} counted as \"you\".")
    if notes:
        L += ['', 'Unfinished items:'] + [f'- {x}' for x in notes]
    L += ['', '## Part B: first-order sensitivity of the readout to the role difference, inside and outside the 8 coordinates', '',
          'f = logit("you" letter) - logit("user" letter) where the answer letter is read (onset without thinking; after the model\'s own '
          'reasoning, teacher-forced, with thinking). D_v f by central differences (eps = 0.25) along d = h^A - h^U (total), its part '
          'in the 8 fitted directions (V), the rest (perp) and its part in a random 8-dim subspace (R), at one layer at a time; per item '
          'summed over the layers, mean over items. Share: mean D_V / mean D_total and mean D_perp / mean D_total (95% bootstrap '
          'interval over items). bf16 logits: every D is a multiple of 0.25.', '',
          '| readout | endpoint | items | D_total | D_V | D_perp | D_R | V share | perp share | D_total = 0 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    sh = lambda s: f"{s['ratio']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]" if s and s['ratio'] is not None and s['ci'] else '—'  # noqa: E731
    for rd, _, _, name in READOUTS:
        B = res.get(rd, {}).get('B')
        if not B:
            continue
        for e, lab in (('A', 'own message'), ('U', 'User message')):
            m = B[e]['mean_sum']
            L.append(f"| {name} | {lab} | {B['n_complete']}/{B['n_items']} | {m['total']:+.2f} | {m['V']:+.2f} | {m['perp']:+.2f} | "
                     f"{m['R']:+.2f} | {sh(B[e]['share']['V'])} | {sh(B[e]['share']['perp'])} | {B[e]['zero_total']:.2f} |")
    L += ['', 'Norm of the parts relative to |d| (median over items, summed over layers): ' + '; '.join(
        f"{name} V {res[rd]['B']['norm_share']['V']:.2f}, perp {res[rd]['B']['norm_share']['perp']:.2f}, R {res[rd]['B']['norm_share']['R']:.3f} "
        f"({res[rd]['B']['n_layers']} layers)" for rd, _, _, name in READOUTS if 'B' in res.get(rd, {}))]
    if missing:
        L += ['', 'Not finished (left out): ' + ', '.join(missing)]
    (S / 'r03_20261101.json').write_text(json.dumps({'results': res, 'missing': missing}, indent=1), encoding='utf-8')
    (S / 'r03_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------------------------------------
# fd_layers: the first-order sensitivity by layer and by band

RD = [('qsp', 'singlepass_swap/qwen38_27b', 'Qwen3.8-27B, no thinking', 64, (24, 47)),
      ('qth', 'thinking_swap/qwen38_27b', 'Qwen3.8-27B, thinking', 64, (28, 43)),
      ('g20', 'thinking_swap/gpt_oss_20b', 'gpt-oss-20b, thinking', 24, (0, 15)),
      ('g120', 'thinking_swap/gpt_oss_120b', 'gpt-oss-120b, thinking', 36, (8, 23))]
FD_DIRS = ('total', 'V', 'perp', 'R', 'zero')


def share(items, layers, ep, num, den='total', draws=2000):
    def ratio(sample):
        n = sum(sum(it[ep][num][l] for l in layers) for it in sample)
        d = sum(sum(it[ep][den][l] for l in layers) for it in sample)
        return n / d if d else float('nan')
    est = ratio(items)
    rng = random.Random(1)
    bs = sorted(ratio([items[rng.randrange(len(items))] for _ in items]) for _ in range(draws))
    bs = [x for x in bs if x == x]
    return est, (bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]) if bs else (float('nan'), float('nan'))


def sum_ci(items, layers, ep, draws=2000):
    """95% bootstrap interval over items of the band's summed slope along d (mean over items)"""
    vals = [sum(it[ep]['total'][l] for l in layers) for it in items]
    rng = random.Random(1)
    bs = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals) for _ in range(draws))
    return bs[int(0.025 * draws)], bs[int(0.975 * draws) - 1]


def fd_layers(A, S):
    res, L = {}, ['# R03 Part B by layer (11-01)', '',
                  'Mean over items of the central-difference slope D of f = logit(you letter) - logit(user letter), by computed layer, '
                  'at the model\'s own message (A) and at the User message (U); shares are ratios of summed means over a band of '
                  'layers (95% bootstrap interval over items). Install band: the layers of the windows in which installation exceeds '
                  '+0.10 in the paper\'s layer-window runs. Thinking: the model\'s own reasoning is teacher-forced, so D measures the '
                  'letter read after a fixed reasoning text.', '']
    for key, d, name, nl, band in RD:
        items = []
        for p in sorted((run_path(A, f'{d}_r03fd') / 'items').glob('*.json')):
            j = json.load(open(p, encoding='utf-8'))
            lays = j['layers']
            it = {'layers': lays}
            for ep in ('A', 'U'):  # an endpoint without a finished reasoning has no slopes (gpt-oss-20b: one item at A)
                if ep in j['endpoints'] and 'D' in j['endpoints'][ep]:
                    it[ep] = {k: dict(zip(lays, j['endpoints'][ep]['D'][k])) for k in FD_DIRS}
                    it[ep]['f0'] = j['endpoints'][ep]['f0']
            if 'norms' in j:
                it['vshare'] = {l: (j['norms']['V'][i] / j['norms']['total'][i]) ** 2 if j['norms']['total'][i] else 0.0
                                for i, l in enumerate(lays)}
            if 'A' in it or 'U' in it:
                items.append(it)
        lays = items[0]['layers']
        thirds = {'first third': [l for l in lays if l < nl / 3], 'middle third': [l for l in lays if nl / 3 <= l < 2 * nl / 3],
                  'last third': [l for l in lays if l >= 2 * nl / 3], 'install band': [l for l in lays if band[0] <= l <= band[1]],
                  'all layers': lays}
        res[key] = {'n_items': len(items), 'layers': lays, 'by_layer': {}, 'bands': {}}
        for ep in ('A', 'U'):
            its = [it for it in items if ep in it]
            res[key]['by_layer'][ep] = {l: {k: sum(it[ep][k][l] for it in its) / len(its) for k in FD_DIRS} for l in lays}
            res[key]['bands'][ep] = {}
            for bn, bl in thirds.items():
                v, vci = share(its, bl, ep, 'V')
                q, qci = share(its, bl, ep, 'perp')
                tot = sum(res[key]['by_layer'][ep][l]['total'] for l in bl)
                vs = sorted(it['vshare'][l] for it in its if 'vshare' in it for l in bl)
                med_vs = vs[len(vs) // 2]
                res[key]['bands'][ep][bn] = {'layers': [bl[0], bl[-1]] if bl else [], 'sum_mean_D_total': tot,
                                             'sum_ci': sum_ci(its, bl, ep), 'V_share': v,
                                             'V_share_ci': vci, 'perp_share': q, 'perp_share_ci': qci, 'median_size_share_V': med_vs}
        L += [f'## {name} ({len(items)} items; computed layers {lays[0]}-{lays[-1]}, step {lays[1] - lays[0]})', '',
              '| endpoint | band | layers | summed mean D_total | V share | perp share | size share of V (median) |', '|---|---|---|---|---|---|---|']
        for ep in ('A', 'U'):
            for bn, b in res[key]['bands'][ep].items():
                L.append(f"| {ep} | {bn} | {b['layers'][0] if b['layers'] else '-'}-{b['layers'][1] if b['layers'] else '-'} | "
                         f"{b['sum_mean_D_total']:+.2f} | {b['V_share']:.2f} [{b['V_share_ci'][0]:.2f}, {b['V_share_ci'][1]:.2f}] | "
                         f"{b['perp_share']:.2f} [{b['perp_share_ci'][0]:.2f}, {b['perp_share_ci'][1]:.2f}] | {b['median_size_share_V']:.2f} |")
        L += ['', '| layer | A: total | A: V | A: perp | U: total | U: V | U: perp | U: R | U: zero |', '|---|---|---|---|---|---|---|---|---|']
        for l in lays:
            a, u = res[key]['by_layer']['A'][l], res[key]['by_layer']['U'][l]
            L.append(f"| L{l} | {a['total']:+.2f} | {a['V']:+.2f} | {a['perp']:+.2f} | {u['total']:+.2f} | {u['V']:+.2f} | {u['perp']:+.2f} | "
                     f"{u['R']:+.2f} | {u['zero']:+.2f} |")
        L.append('')
    (S / 'r03b_layers_20261101.json').write_text(json.dumps({'results': res}, indent=1), encoding='utf-8')
    (S / 'r03b_layers_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(x for x in L if not x.startswith('| L')))


# ---------------------------------------------------------------------------------------------------------------------------
# families: intervals over the 15 wording families, completion bounds, control tasks

FAM_READOUTS = [('qsp', 'Qwen3.8-27B, no thinking', 'singlepass_swap/qwen38_27b_all', False),
                ('qth', 'Qwen3.8-27B, thinking', 'thinking_swap/qwen38_27b_all', True),
                ('g20', 'gpt-oss-20b, thinking', 'thinking_swap/gpt_oss_20b_all', True),
                ('g120', 'gpt-oss-120b, thinking', 'thinking_swap/gpt_oss_120b_all', True)]
ATTR = [('own <- User', 'swU8_A', 'base_A'), ('own <- System', 'swS8_A', 'base_A'), ('own <- User, random', 'rswU8_A', 'base_A'),
        ('own <- own (copy)', 'idA8_A', 'base_A'), ('User <- own', 'swA8_U', 'base_U'), ('User <- own, random', 'rswA8_U', 'base_U'),
        ('System <- own', 'swA8_S', 'base_S')]


def fam(i):
    return i.rsplit('_c', 1)[0]


def cluster_ci(pairs, reps=4000, seed=1):
    """pairs: list of (family, diff); resample families with replacement"""
    fams = sorted({f for f, _ in pairs})
    byf = {f: [d for g, d in pairs if g == f] for f in fams}
    rng = random.Random(seed)
    ms = []
    for _ in range(reps):
        ds = [d for f in (rng.choice(fams) for _ in fams) for d in byf[f]]
        ms.append(sum(ds) / len(ds))
    ms.sort()
    return ms[int(0.025 * reps)], ms[int(0.975 * reps) - 1]


def fam_value(r, thinking):
    if not thinking:
        return r['p']['self']
    return float(r.get('on_answer') == 'self') if r.get('finished') else None


def correct(r):
    for k in ('content', 'arith', 'answer'):
        if isinstance(r.get(k), dict) and 'correct' in r[k]:
            return float(bool(r[k]['correct'])) if r.get('finished', True) else None
    raise KeyError(sorted(r.keys()))


def compare(recs, cond, base, val):
    by = {}
    for r in recs:
        if r['cond'] in (cond, base):
            by.setdefault((r['id'], str(r.get('order'))), {})[r['cond']] = r
    keys = [k for k, v in by.items() if cond in v and base in v]
    e = {k: val(by[k][cond]) for k in keys}
    b = {k: val(by[k][base]) for k in keys}
    both = [k for k in keys if e[k] is not None and b[k] is not None]
    d = [e[k] - b[k] for k in both]
    out = {'n': len(keys), 'finished': [sum(e[k] is not None for k in keys), sum(b[k] is not None for k in keys)], 'pairs': len(both)}
    if d:
        out.update(change=sum(d) / len(d), item_ci=list(boot(d, seed=1)), cluster_ci=list(cluster_ci([(fam(k[0]), x) for k, x in zip(both, d)])))
    lo = sum((0.0 if e[k] is None else e[k]) - (1.0 if b[k] is None else b[k]) for k in keys) / len(keys)
    hi = sum((1.0 if e[k] is None else e[k]) - (0.0 if b[k] is None else b[k]) for k in keys) / len(keys)
    out['bounds'] = [lo, hi]
    return out


def families(A, S):
    res = {}
    for key, name, d, thinking in FAM_READOUTS:
        recs = items(str(run_path(A, d)))
        res[key] = {'attribution': {lab: compare(recs, c, b, lambda r: fam_value(r, thinking)) for lab, c, b in ATTR}}
        for task in ('content', 'arith'):
            trecs = items(str(run_path(A, f'{d}_{task}')))
            conds = sorted({r['cond'] for r in trecs})
            res[key][task] = {}
            for c in conds:
                if c.startswith('base_'):
                    continue
                base = 'base_U' if c.endswith('_U') else ('base_S' if c.endswith('_S') else 'base_A')
                if base in conds:
                    res[key][task][c] = compare(trecs, c, base, correct)
    f3 = lambda x: f'{x:+.2f}'  # noqa: E731
    L = ['# R09: uncertainty unit, completion and bounds of the causal tests (11-01)', '',
         'Change: item-paired difference on the items finished in both conditions (the paper\'s estimand). Item CI: the paper\'s item '
         'bootstrap; cluster CI: 15 wording families resampled with all their items. Bounds: over all items, unfinished outputs '
         'treated as unknown (thinking). Finished: edited / unedited.', '']
    for key, name, _, thinking in FAM_READOUTS:
        L += [f'## {name}', '', '| condition | finished | pairs | change | item CI | cluster CI | bounds (unknown unfinished) |', '|---|---|---|---|---|---|---|']
        for lab, s in res[key]['attribution'].items():
            if 'change' not in s:
                continue
            L.append(f"| {lab} | {s['finished'][0]}/{s['finished'][1]} of {s['n']} | {s['pairs']} | {f3(s['change'])} | "
                     f"[{f3(s['item_ci'][0])}, {f3(s['item_ci'][1])}] | [{f3(s['cluster_ci'][0])}, {f3(s['cluster_ci'][1])}] | "
                     f"[{f3(s['bounds'][0])}, {f3(s['bounds'][1])}] |")
        for task in ('content', 'arith'):
            L += ['', f'{task}: paired change in accuracy', '', '| condition | finished | pairs | change | item CI | cluster CI |', '|---|---|---|---|---|---|']
            for c, s in res[key][task].items():
                if 'change' in s:
                    L.append(f"| {c} | {s['finished'][0]}/{s['finished'][1]} of {s['n']} | {s['pairs']} | {f3(s['change'])} | "
                             f"[{f3(s['item_ci'][0])}, {f3(s['item_ci'][1])}] | [{f3(s['cluster_ci'][0])}, {f3(s['cluster_ci'][1])}] |")
        L.append('')
    (S / 'r09_cluster_20261101.json').write_text(json.dumps({'results': res}, indent=1), encoding='utf-8')
    (S / 'r09_cluster_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------------------------------------
# strict: held-out wording of the attribution question, and the secret game with X alone edited

GN = {'qsp': 'qwen38_27b_sp', 'qth': 'qwen38_27b_th', 'g20': 'gpt_oss_20b', 'g120': 'gpt_oss_120b'}
REF_GAME = {'qsp': 'qwen_sp', 'qth': 'qwen_th', 'g20': 'oss_th', 'g120': 'oss120_th'}
T1 = [('removal', 'swU8_A', 'base_A'), ('installation', 'swA8_U', 'base_U'), ('random removal', 'rswU8_A', 'base_A')]


def attr(d):
    by = {}
    for r in items(str(d)):
        by.setdefault(r['cond'], []).append(r)
    out = {'levels': {c: level(rs) for c, rs in sorted(by.items())}}
    for _, c, base in T1:
        if c in by and base in by:
            out[c] = change(by[c], by[base])
    return out


def strict(A, S):
    from causal_tests import GAMES, game_summary
    res, missing = {'T1': {}, 'T3': {}}, []
    for rd, _, root, _ in READOUTS:
        d = run_path(A, f'{root}_p56full_alt')
        if (d / 'summary.json').exists():
            res['T1'][rd] = {'full': attr(d), 'ref': attr(run_path(A, f'{root}_r05m_msg'))}
        else:
            missing.append(f'{root}_p56full_alt')
        g = {q: run_path(A, f'games_swap/{GN[rd]}_p56full_game_{q}') for q in ('main', 'control')}
        if all((x / 'summary.json').exists() for x in g.values()):
            res['T3'][rd] = {'full': {q: game_summary(str(x)) for q, x in g.items()},
                             'ref': {q: game_summary(str(run_path(A, GAMES[REF_GAME[rd]].format(q)))) for q in ('main', 'control')}}
        else:
            missing += [f'games_swap/{GN[rd]}_p56full_game_{q}' for q, x in g.items() if not (x / 'summary.json').exists()]
    f3 = lambda x: '—' if x is None else f'{x:.3f}'  # noqa: E731
    ch = lambda s: f"{s['diff']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]" if s else '—'  # noqa: E731
    L = ['# Full runs of the signal pilots (R05-1: held-out wording, X-only game; from 10-31)', '',
         'Pilot verdicts (pilot_check_p56.py): T1 attr_alt signal, T2 attr_num no signal, T3 game signal, T4 polarity no signal, '
         'T5 content direction not valid. fit_m of r05r06_fits, k = 8.', '',
         '## T1: attribution question in a held-out wording, the message edited only', '',
         'Thinking: share of finished items answered "you"; no thinking: mean probability of the "you" option. Change: item-paired '
         'difference from the unedited message of the same role (95% bootstrap interval). Reference: r05m_msg, the same fit and region '
         'with the paper\'s wording.', '',
         '| readout | wording | unedited own / User | removal | installation | random removal | finished (fewest) |',
         '|---|---|---|---|---|---|---|']
    for rd, _, _, name in READOUTS:
        if rd not in res['T1']:
            continue
        for lab, k in (('held-out', 'full'), ('paper (r05m_msg)', 'ref')):
            s = res['T1'][rd][k]
            Lv = s['levels']
            fins = [(v['finished'], v['n']) for v in Lv.values() if 'finished' in v]
            fin = '{}/{}'.format(*min(fins)) if fins else '—'
            L.append(f"| {name} | {lab} | {f3(Lv['base_A']['you'])} / {f3(Lv['base_U']['you'])} | {ch(s.get('swU8_A'))} | "
                     f"{ch(s.get('swA8_U'))} | {ch(s.get('rswU8_A'))} | {fin} |")
    L += ['', '## T3: the secret game, the coordinates edited at X only', '',
          'Share of finished games in which the model acts as the setter; change: item-paired difference from the unedited '
          'announcement of the same role (95% bootstrap interval); control: share of the control question answered correctly. '
          'Reference: the paper\'s game runs (s6; the paper\'s fit, all positions).', '',
          '| readout | run | unedited own / User | removal | installation | random removal | control correct (own / User) |',
          '|---|---|---|---|---|---|---|']
    gd = lambda s: f"{s['diff']:+.3f} [{s['diff_ci'][0]:+.3f}, {s['diff_ci'][1]:+.3f}]" if s and 'diff' in s else '—'  # noqa: E731
    for rd, _, _, name in READOUTS:
        if rd not in res['T3']:
            continue
        for lab, k in (('X only (fit_m)', 'full'), ('paper (s6)', 'ref')):
            M, Ct = res['T3'][rd][k]['main'], res['T3'][rd][k]['control']
            L.append(f"| {name} | {lab} | {f3(M['base_A']['stat'])} / {f3(M['base_U']['stat'])} | {gd(M.get('swU8_A'))} | "
                     f"{gd(M.get('swA8_U'))} | {gd(M.get('rswU8_A'))} | {f3(Ct['base_A']['accuracy'])} / {f3(Ct['base_U']['accuracy'])} |")
    if missing:
        L += ['', 'Not finished or not yet archived (left out): ' + ', '.join(missing)]
    (S / 'p56full_20261101.json').write_text(json.dumps({'results': res, 'missing': missing}, indent=1), encoding='utf-8')
    (S / 'p56full_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


STEPS = [('without_question', without_question), ('random_floor', random_floor), ('exchange', exchange),
         ('exchange_traces', exchange_traces), ('rest', rest), ('fd_layers', fd_layers), ('families', families), ('strict', strict)]


def main(argv=None):
    ap = arguments(__doc__, item_cache=True)
    ap.add_argument('--only', help='comma-separated subset of: ' + ', '.join(n for n, _ in STEPS))
    args = ap.parse_args(argv)
    r = Roots(args)
    A, S = r.res(W3), r.out(W3 + '/stats')
    only = set(args.only.split(',')) if args.only else None
    for name, fn in STEPS:
        if only is None or name in only:
            print(f'== {name}', flush=True)
            fn(A, S)


if __name__ == '__main__':
    main()
