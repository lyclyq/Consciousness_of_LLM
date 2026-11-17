"""Check the hand-entered rows of the seven models added on 11-03/04 against the analysis outputs (release run of 11-04, OUT) and the
archive: Table tab:self_crossmodel (both languages) against v6_newdev_*.csv; Table tab:self_natural rows against the suite_nat3
summaries; the Apertus row of tab:self_natural2 against its records (two option orders); the own-step row against v4_ownstep.csv.
Prints paper / recomputed / OK and writes stats/check_newdev_tables_20261104.txt.
11-15: the own-step row is checked only when OUT holds it (the analysis run of 11-04); OUT_TXT (optional) is the output file.
usage: python -I check_newdev_tables_20261104.py ARCHIVE_W PAPER_SOURCE OUT [OUT_TXT]"""
import csv
import glob
import json
import math
import os
import sys

W, SRC, OUT = sys.argv[1:4]
S = os.path.join(OUT, 'evidence', 'p2_crossmodel_wave3_20261019', 'stats')
f2 = lambda x: f'{x:.2f}'  # noqa: E731
rows = lambda f: list(csv.DictReader(open(os.path.join(S, f), encoding='utf-8')))  # noqa: E731
geo = {(r['model'], r['mode'], r['position']): r for r in rows('v6_newdev_geometry_summary.csv')}
att = {r['model']: r for r in rows('v6_newdev_attribution_non_thinking.csv') if r['orders'] == '2_orders_as_paper'}
thk = {(r['model'], r['role']): r for r in rows('v6_newdev_thinking_counts.csv')}
NAME = {'mimo_distill9b': 'MiMo-V2.6-Distill-9B$^{a}$', 'phi4': 'phi-4', 'apertus8b': 'Apertus-8B', 'moonlight16b': 'Moonlight-16B-A3B',
        'mimo7b_rl': 'MiMo-7B-RL', 'mimo_vl7b': 'MiMo-VL-7B', 'ernie45_21b': 'ERNIE-4.5-21B-A3B'}
out, bad = [], 0


def expect_crossmodel(k):
    mode = 'non_thinking' if (k, 'non_thinking', 'content') in geo else 'thinking'
    c, o = geo[(k, mode, 'content')], geo[(k, mode, 'onset')]
    L = int(c['n_layers'])
    lc = set(map(int, c['layers_80pct'].split())) if c['layers_80pct'] else set()
    lo = set(map(int, o['layers_80pct'].split())) if o['layers_80pct'] else set()
    t0 = math.ceil(2 * L / 3)
    last = sum(1 for i in range(t0, L) if i in lc | lo)
    cells = [f"{c['layers_A_apart_80pct']}/{L}", f"{o['layers_A_apart_80pct']}/{L}", f'{last}/{L - t0}']
    if k in att:
        a = att[k]
        cells += [f2(float(a['p_you_own'])), f2(max(float(a['p_you_user']), float(a['p_you_system']))), a['bodies_positive']]
    else:
        cells += ['--'] * 3
    if (k, 'assistant') in thk:
        cells += [f2(float(thk[(k, r)]['you_share'])) for r in ('assistant', 'user', 'system')] + [thk[(k, 'assistant')]['n_items']]
    else:
        cells += ['--'] * 4
    return ' & '.join([NAME[k]] + cells) + ' \\\\'


for L in ('en', 'zh'):
    body = open(os.path.join(SRC, f'self_{L}_body.tex'), encoding='utf-8').read().split('\n')
    for k in NAME:
        e = expect_crossmodel(k)
        got = [x for x in body if x.startswith(NAME[k] + ' & ') and x.count('&') == 10]
        ok = len(got) == 1 and got[0] == e
        bad += not ok
        out.append(f"Table 5 [{L}] {NAME[k]}: {'OK' if ok else 'MISMATCH'}" + ('' if ok else f'\n   paper {got}\n   recomputed {e}'))
NAT = {'mimo7b_rl': ('MiMo-7B-RL', 'runs/server3/suite_nat3/mimo7b_rl_0530'), 'moonlight16b': ('Moonlight-16B-A3B', 'runs/server3/suite_nat3/moonlight16b_a3b_instruct'),
       'phi4': ('phi-4', 'runs/server3/suite_nat3/phi4_en'), 'ernie45_21b': ('ERNIE-4.5-21B-A3B$^{a}$', 'runs/server3/suite_nat3/ernie45_21b_a3b_thinking')}
for L in ('en', 'zh'):
    app = open(os.path.join(SRC, f'self_{L}_appendix.tex'), encoding='utf-8').read().split('\n')
    for k, (nm, d) in NAT.items():
        s = json.load(open(os.path.join(W, d, 'summary.json'), encoding='utf-8'))
        n = s['n_layers']
        cells = [f"{sum(x >= 0.8 for x in s['separate_content']['share_A_lowest'])}/{n}", f"{sum(x >= 0.8 for x in s['separate_onset']['share_A_lowest'])}/{n}"]
        cells += [f2(s['attr'][r]['self']) for r in ('assistant', 'user', 'system')] if 'attr' in s else ['--'] * 3
        e = ' & '.join([nm] + cells) + ' \\\\'
        got = [x for x in app if x.startswith(nm + ' & ') and x.count('&') == 5 and x.count('/') >= 2]   # not the option-order rows
        ok = len(got) == 1 and got[0] == e
        bad += not ok
        out.append(f"Natural [{L}] {nm}: {'OK' if ok else 'MISMATCH'}" + ('' if ok else f'\n   paper {got}\n   recomputed {e}'))
rec = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(os.path.join(W, 'runs/pro/natural/apertus8b_instruct_2509_en/role150_*.json')))]
nat2 = [f2(sum(sum(x['roles'][r]['by_order'][i]['self'] for i in (0, 1)) / 2 for x in rec) / len(rec)) for r in ('assistant', 'user')]
a = att['apertus8b']
main2 = [f2(float(a['p_you_own'])), f2(float(a['p_you_user']))]
for L, mode in (('en', 'single pass'), ('zh', '单次前向')):
    app = open(os.path.join(SRC, f'self_{L}_appendix.tex'), encoding='utf-8').read()
    e = ' & '.join(['Apertus-8B', mode] + nat2 + main2) + ' \\\\'
    ok = e in app
    bad += not ok
    out.append(f"Natural two roles [{L}] Apertus-8B: {'OK' if ok else 'MISMATCH ' + e}")
own = {r['model']: r for r in rows('v4_ownstep.csv')}.get('mimo_v26_distill_qwen9b')
if own is None:   # 11-15: the row comes from the analysis run of 11-04 (OUT); the archive's v4_ownstep.csv of 10-24 has none
    out.append('Own steps MiMo-V2.6-Distill-9B: not checked (no row of this model in v4_ownstep.csv of OUT; a long-version table)')
else:
    e = f"& {f2(float(own['own']))} & {f2(float(own['user']))} & \\(+{f2(float(own['own_minus_user']))}\\) [{f2(float(own['ci_lo']))}, {f2(float(own['ci_hi']))}] \\\\"
    for L in ('en', 'zh'):
        app = open(os.path.join(SRC, f'self_{L}_appendix.tex'), encoding='utf-8').read()
        ok = e in app
        bad += not ok
        out.append(f"Own steps [{L}] MiMo-V2.6-Distill-9B: {'OK' if ok else 'MISMATCH ' + e}")
out.append(f'{len(out) - (own is None)} checks, {bad} mismatches')   # 11-15: the note on the own-step row is not a check
txt = '\n'.join(out)
print(txt)
open(sys.argv[4] if len(sys.argv) > 4 else os.path.join(W, 'stats', 'check_newdev_tables_20261104.txt'), 'w', encoding='utf-8').write(txt + '\n')
