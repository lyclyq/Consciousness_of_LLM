"""11-03 copy of check_crossmodel_hand_numbers_20261026.py with the two OLMo-2 models of Table 5 (added to the paper 10-30) in
NAMES and GEO (runs/pro/suite/olmo2_{13b,7b}_en/summary.json, non-thinking only); otherwise unchanged.
Cross-model numbers of Section 4.5 that were counted by hand, recomputed from the archived per-layer shares and per-body records
(user 10-26: every number of the paper has a script). Prints "paper | recomputed | OK/MISMATCH" for each; never fits a rule to a value.
Paper values are parsed from paper/source/self_en_body.tex (Table tab:self_crossmodel) or written below with the sentence they come from.
Shared rule: a layer singles out the own message if share_A_lowest >= 0.8 (the own message has the smallest R_A in at least 80% of the
held-out bodies; >= is the convention of the stored layers_A_lowest_80 lists); the last third is layers >= ceil(2L/3), the last quarter
layers >= ceil(3L/4); "content or onset" = union of the two positions, a layer counted once; four-stream models (Qwen3.8-Flash-Next,
GLM-5.3-Flash): mean_of_streams share_A_lowest_by_layer, strongest stream = the stream with the most layers.
 1. Table 5, columns content / onset / last third, under the non-thinking rendering (thinking rendering for models with only a thinking
    mode), from the run summaries listed in GEO (the same files as make_fig_crossmodel_v5.py); Qwen3.5-27B and Granite-4.1/4.2 also
    against stats/v4_geometry_summary.csv (max over positions of last_third_80pct).
 2. "In every model and mode in which the model recognizes its own messages, the own message is singled out at every layer of the last
    third ... in the models without a self-index, it covers at most 3 of the layers of the last third (Gemma-4-31B under the thinking
    rendering)" and "The same holds for the last quarter of the depth" (README 8a: the full-coverage half of the criterion), over the
    renderings in SELFREF (geometry under the rendering of the mode with self-reference) and NOINDEX (every rendering measured).
 3. Body-level P("you") from geo_attr per-body records roles.<role>.by_order[o].<option>: per body the mean over the orders used, then
    the mean over bodies; "own" = P(you | assistant), "other" = max over User and System of the body-mean P(you) (Table 5 convention:
    Qwen3.8-27B 0.7712 vs max(0.0359, 0.0696)); "excess" = mean over bodies of P(you|A) - max(P(you|U), P(you|S)) (stats CSV
    own_minus_max_other). Orders: o0 you/user/system, o1 system/you/user, o2 user/system/you; the paper uses o0 and o1.
    GLM-4.6V-Flash (8-model archive models/glm46v_flash/non_thinking/geo_attr): first-option probability per order and role; excess.
    GLM-4.7-Flash (models/glm47_flash/non_thinking/geo_attr): excess over o0+o1 and its bodies > 0; excess in o2 alone ("you" last).
 4. Qwen3.8-27B simulated FP8 (models/qwen38_27b/non_thinking/geo_attr_simfp8): own, other, bodies with excess > 0, index layers.
    Qwen3.8-Flash-Next natural order (models/qwen38_flashnext/non_thinking/natural, roles assistant and user only): own and User P(you).
 5. Qwen3.8-27B English (models/qwen38_27b/non_thinking/geo_attr_en): own and other (o0+o1), content layers.
usage: check_crossmodel_hand_numbers_20261026.py [ARCHIVE8 ARCHIVE3 TEX]   (defaults: run from evidence/p2_crossmodel_wave3_20261019)"""
import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

A8, A3, TEX = (Path(p) for p in (sys.argv[1:4] if len(sys.argv) > 3 else
                                 ('../p2_crossmodel_8models_20261018', '.', '../../paper/source/self_en_body.tex')))
M8, R = A8 / 'models', A3 / 'runs'
NAMES = {'qwen38_27b': 'Qwen3.8-27B', 'qwen38_flashnext': 'Qwen3.8-Flash-Next', 'gpt_oss_120b': 'gpt-oss-120b', 'gpt_oss_20b': 'gpt-oss-20b',
         'glm53_flash': 'GLM-5.3-Flash', 'exaone4_32b': 'EXAONE-4.0.1-32B', 'glm46v_flash': 'GLM-4.6V-Flash', 'glm47_flash': 'GLM-4.7-Flash',
         'r1_llama70b': 'R1-Distill-Llama-70B', 'seed_oss_36b': 'Seed-OSS-36B', 'gemma4_31b': 'Gemma-4-31B', 'qwen36_27b': 'Qwen3.6-27B',
         'qwen36_35b': 'Qwen3.6-35B-A3B', 'qwen35_27b': 'Qwen3.5-27B', 'granite41_30b': 'Granite-4.1-30B', 'granite42_30b': 'Granite-4.2-30B', 'olmo2_13b': 'OLMo-2-13B', 'olmo2_7b': 'OLMo-2-7B'}
STREAM = 'stream'
GEO = {  # (model, rendering) -> run summary with per-layer share_A_lowest, or (STREAM, file) for the four-stream models
    ('qwen38_27b', 'non_thinking'): M8 / 'qwen38_27b/non_thinking/geo_attr/_run_summary.json',
    ('qwen38_27b', 'thinking'): M8 / 'qwen38_27b/thinking/geo/_run_summary.json',
    ('gpt_oss_120b', 'thinking'): R / 'pro/suite_think/gpt_oss_120b_en/summary.json',
    ('gpt_oss_20b', 'thinking'): R / 'pro/suite_think/gpt_oss_20b_mxfp4_en/summary.json',
    ('qwen38_flashnext', 'non_thinking'): (STREAM, R / 'pro/flashnext_stream_20261020/flashnext_stream_geometry_non_20261020.json'),
    ('qwen38_flashnext', 'thinking'): (STREAM, R / 'pro/flashnext_stream_20261020/flashnext_stream_geometry_think_20261020.json'),
    ('glm53_flash', 'thinking'): (STREAM, R / 'server3/suite_think/glm53_flash_stream_geometry.json'),
    ('exaone4_32b', 'non_thinking'): R / 'server3/suite/exaone4_32b_en/summary.json',
    ('exaone4_32b', 'thinking'): R / 'server3/suite_think/exaone4_32b_en/summary.json',
    ('r1_llama70b', 'thinking'): M8 / 'r1_llama70b/thinking/geo/_run_summary.json',
    ('granite42_30b', 'non_thinking'): R / 'server3/suite/granite42_30b_en/summary.json',
    ('granite41_30b', 'non_thinking'): R / 'server3/suite/granite41_30b_en/summary.json',
    ('seed_oss_36b', 'non_thinking'): R / 'server3/suite/seed_oss_36b/summary.json',
    ('seed_oss_36b', 'thinking'): R / 'server3/suite_think/seed_oss_36b/summary.json',
    ('qwen35_27b', 'non_thinking'): R / 'server3/suite/qwen35_27b/summary.json',
    ('olmo2_13b', 'non_thinking'): R / 'pro/suite/olmo2_13b_en/summary.json',
    ('olmo2_7b', 'non_thinking'): R / 'pro/suite/olmo2_7b_en/summary.json'}
for m in ('glm46v_flash', 'glm47_flash', 'gemma4_31b', 'qwen36_27b', 'qwen36_35b'):
    GEO[(m, 'non_thinking')] = M8 / m / 'non_thinking/geo_attr/_run_summary.json'
    GEO[(m, 'thinking')] = M8 / m / 'thinking/geo/_run_summary.json'
TABLE_RENDERING = {m: 'thinking' for m in ('gpt_oss_120b', 'gpt_oss_20b', 'glm53_flash', 'r1_llama70b')}
SELFREF = [('qwen38_27b', 'non_thinking'), ('qwen38_27b', 'thinking'), ('gpt_oss_120b', 'thinking'), ('gpt_oss_20b', 'thinking'),
           ('qwen38_flashnext', 'thinking'), ('glm53_flash', 'thinking'), ('exaone4_32b', 'thinking')]
NOINDEX = [k for k in GEO if k[0] in ('qwen36_27b', 'qwen36_35b', 'qwen35_27b', 'gemma4_31b', 'seed_oss_36b')]
results = []


def check(name, paper, got):
    results.append((name, paper, got, 'OK' if str(paper) == str(got) else 'MISMATCH'))


def geo(key):
    """content and onset per-layer shares (mean of streams for four-stream models) and the strongest single stream per position"""
    src = GEO[key]
    if isinstance(src, tuple):
        s = json.load(open(src[1], encoding='utf-8'))
        sh = {p: s[p]['mean_of_streams']['share_A_lowest_by_layer'] for p in ('content', 'onset')}
        best = {p: max(sum(x >= 0.8 for x in s[p][f'stream{i}']['share_A_lowest_by_layer']) for i in range(4)) for p in ('content', 'onset')}
        return sh, best
    s = json.load(open(src, encoding='utf-8'))
    return {p: s[p]['share_A_lowest'] for p in ('content', 'onset')}, None


def tail_cover(sh, frac):
    """(layers singled out at content or onset among layers >= ceil(frac * L), size of that tail)"""
    L = len(sh['content'])
    t0 = math.ceil(frac * L)
    return sum(sh['content'][l] >= 0.8 or sh['onset'][l] >= 0.8 for l in range(t0, L)), L - t0


# 1. Table 5
tex = TEX.read_text(encoding='utf-8')
tab = tex[tex.index('\\label{tab:self_crossmodel}'):]
tab = tab[tab.index('\\midrule'):tab.index('\\bottomrule')]
paper_rows = {}
for line in tab.splitlines()[1:]:
    cells = [c.strip() for c in line.rstrip('\\ ').split('&')]
    if len(cells) > 3:
        paper_rows[cells[0]] = cells[1:4]
key_of = {v: k for k, v in NAMES.items()}
# 11-15: the rows of the seven models added on 11-04 are checked by check_newdev_tables_20261104.py; only these names are skipped
NEWDEV = {'MiMo-V2.6-Distill-9B$^{a}$', 'phi-4', 'Apertus-8B', 'Moonlight-16B-A3B', 'MiMo-7B-RL', 'MiMo-VL-7B', 'ERNIE-4.5-21B-A3B'}
for disp, (pc, po, pl) in paper_rows.items():
    if disp in NEWDEV and disp not in key_of:
        continue
    m = key_of[disp]
    sh, best = geo((m, TABLE_RENDERING.get(m, 'non_thinking')))
    L = len(sh['content'])
    n = {p: sum(x >= 0.8 for x in sh[p]) for p in ('content', 'onset')}
    fmt = {p: (f'{n[p]} ({best[p]})/{L}' if best and best[p] != n[p] else f'{n[p]}/{L}') for p in n}
    c, size = tail_cover(sh, 2 / 3)
    check(f'Table 5 {disp}: content', pc, fmt['content'])
    check(f'Table 5 {disp}: onset', po, fmt['onset'])
    check(f'Table 5 {disp}: last third', pl, f'{c}/{size}')
v4 = {}
for r in csv.DictReader(open(A3 / 'stats/v4_geometry_summary.csv', encoding='utf-8')):
    if r['mode'] == 'non_thinking':
        v4[r['model']] = max(v4.get(r['model'], 0), int(r['last_third_80pct']))
for m, v in v4.items():
    check(f'Table 5 {NAMES[m]}: last third vs v4_geometry_summary.csv (max over positions)', paper_rows[NAMES[m]][2].split('/')[0], v)

# 2. the depth criterion
for frac, label in ((2 / 3, 'last third'), (3 / 4, 'last quarter')):
    for key in SELFREF:
        c, size = tail_cover(geo(key)[0], frac)
        check(f'{label}, self-reference {NAMES[key[0]]} {key[1]}: every layer covered', f'{size}/{size}', f'{c}/{size}')
    worst = max(((tail_cover(geo(k)[0], frac)[0], k) for k in NOINDEX), key=lambda t: t[0])
    if label == 'last third':
        check('last third, models without a self-index: at most 3 layers, Gemma-4-31B thinking',
              '3 (gemma4_31b thinking)', f'{worst[0]} ({worst[1][0]} {worst[1][1]})')
    else:
        results.append((f'last quarter, models without a self-index: most layers covered (not stated in the paper)', '-',
                        f'{worst[0]} ({worst[1][0]} {worst[1][1]})', 'info'))
sh = geo(('gemma4_31b', 'thinking'))[0]
L = len(sh['content'])
check('Gemma-4-31B thinking: content layers, of which in the last third', '11, 3',
      f"{sum(x >= 0.8 for x in sh['content'])}, {sum(sh['content'][l] >= 0.8 for l in range(math.ceil(2 * L / 3), L))}")
sh = geo(('seed_oss_36b', 'thinking'))[0]
on = [l for l, x in enumerate(sh['onset']) if x >= 0.8]
check('Seed-OSS-36B thinking: onset layers = its first 17 and no later', 'L0-L16', f'L{on[0]}-L{on[-1]}' if on == list(range(len(on))) else on)


# 3-5. body-level attribution
def bodies(d):
    return [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(str(d / 'role150_*.json')))]


def attr(recs, orders, roles=('assistant', 'user', 'system')):
    P = {r: [sum(x['roles'][r]['by_order'][o]['self'] for o in orders) / len(orders) for x in recs] for r in roles}
    mean = {r: sum(v) / len(v) for r, v in P.items()}
    ext = [r for r in roles if r != 'assistant']
    diff = [P['assistant'][i] - max(P[r][i] for r in ext) for i in range(len(recs))]
    return mean, max(mean[r] for r in ext), sum(diff) / len(diff), sum(d > 0 for d in diff), len(recs)


FIRST = {0: 'self', 1: 'system', 2: 'user'}  # the option in position A in each order
g46 = bodies(M8 / 'glm46v_flash/non_thinking/geo_attr')
first = {o: {r: sum(x['roles'][r]['by_order'][o][FIRST[o]] for x in g46) / len(g46) for r in ('assistant', 'user', 'system')} for o in range(3)}
o_best = max(range(3), key=lambda o: min(first[o].values()))
check(f'GLM-4.6V-Flash: first option in one order (o{o_best}), min over roles', '0.99', f'{min(first[o_best].values()):.2f}')
results.append(('  GLM-4.6V-Flash first-option probability by order (A/U/S)', '-',
                '; '.join(f"o{o} " + '/'.join(f'{first[o][r]:.2f}' for r in ('assistant', 'user', 'system')) for o in range(3)), 'info'))
check('GLM-4.6V-Flash: excess of P(you|own) over external (o0+o1)', '0.01', f'{attr(g46, (0, 1))[2]:.2f}')
g47 = bodies(M8 / 'glm47_flash/non_thinking/geo_attr')
_, _, ex, pos, n = attr(g47, (0, 1))
check('GLM-4.7-Flash: excess (o0+o1), bodies positive', '0.05 (130 of 150)', f'{ex:.2f} ({pos} of {n})')
check('GLM-4.7-Flash: excess when "you" is the last option (o2)', '-0.02', f'{attr(g47, (2,))[2]:.2f}')
results.append(('  GLM-4.7-Flash excess by order o0/o1/o2 (README: +0.039/+0.058/-0.021)', '-',
                '/'.join(f'{attr(g47, (o,))[2]:+.3f}' for o in range(3)), 'info'))

fp8 = bodies(M8 / 'qwen38_27b/non_thinking/geo_attr_simfp8')
mean, other, ex, pos, n = attr(fp8, (0, 1))
check('Qwen3.8-27B simulated FP8: own vs at most other, bodies positive', '0.765 vs 0.065, 150 of 150', f"{mean['assistant']:.3f} vs {other:.3f}, {pos} of {n}")
s = json.load(open(M8 / 'qwen38_27b/non_thinking/geo_attr_simfp8/_run_summary.json', encoding='utf-8'))
check('Qwen3.8-27B simulated FP8: self-index layers (content, onset)', '64/64, 64/64',
      ', '.join(f"{sum(x >= 0.8 for x in s[p]['share_A_lowest'])}/{s['n_layers']}" for p in ('content', 'onset')))
nat = bodies(M8 / 'qwen38_flashnext/non_thinking/natural')
mean, other, ex, pos, n = attr(nat, (0, 1), roles=('assistant', 'user'))  # 10-26: text changed from the three-order 0.13 vs 0.09
check('Qwen3.8-Flash-Next natural order (o0+o1): P(you) own vs User', '0.07 vs 0.07', f"{mean['assistant']:.2f} vs {mean['user']:.2f}")
mean, other, ex, pos, n = attr(nat, (0, 1, 2), roles=('assistant', 'user'))
results.append(('  Flash-Next natural order, all three orders (the old text)', '-', f"{mean['assistant']:.2f} vs {mean['user']:.2f}", 'info'))
en = bodies(M8 / 'qwen38_27b/non_thinking/geo_attr_en')
mean, other, ex, pos, n = attr(en, (0, 1))
check('Qwen3.8-27B English: own vs at most other (o0+o1)', '0.68 vs 0.08', f"{mean['assistant']:.2f} vs {other:.2f}")
s = json.load(open(M8 / 'qwen38_27b/non_thinking/geo_attr_en/_run_summary.json', encoding='utf-8'))
check('Qwen3.8-27B English: content layers', '64/64', f"{sum(x >= 0.8 for x in s['content']['share_A_lowest'])}/{s['n_layers']}")

w = max(len(r[0]) for r in results)
for name, paper, got, flag in results:
    print(f'{name:{w}s} | paper {paper} | recomputed {got} | {flag}')
bad = [r for r in results if r[3] == 'MISMATCH']
print(f'\n{sum(r[3] == "OK" for r in results)} OK, {len(bad)} MISMATCH, {sum(r[3] == "info" for r in results)} info')
