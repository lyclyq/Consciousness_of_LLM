"""Recompute the numbers of the P2 paper that were computed by hand (user 10-26: every number has a script), mechanism section and
appendix. Each line: number | paper | recomputed | OK / MISMATCH / NOTE, and whether the paper phrase is in the current EN tex.
Standard library only. Sources and rules:
 1. Sec 4.4 "removal closes 0.99--1.00 of the gap ..., installation 0.63--0.67": stats/v4_mechanism.json, results.{qwen_sp,qwen_th,
    oss_th}.all; gap = base_A.stat - base_U.stat; removal = -swU8_A.diff / gap, installation = swA8_U.diff / gap (diff = item-paired).
 2. Sec 4.4 gpt-oss 64 dimensions (-0.75, +0.58 against +0.51, 0.76 of the gap against 0.67): runs/pro/thinking_swap/
    gpt_oss_20b_all_k64/summary.json (swU64_A, swA64_U you_minus_base.mean; gap = base_A - base_U you_share) and oss_th.all.swA8_U.diff.
 3. App A.2 group ranges: stats/v4_ownstep.csv, the rows of Tables tab:self_ownstep and tab:self_ownstep_ctrl (not qwen35_122b_fp8_M,
    deepseek_flash_thinkm); without a self-index = the five models of Sec 4.5 ("Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.5-27B, Gemma-4-31B,
    and Seed-OSS-36B ... have neither level"); range = max and min of own_minus_user per group and mode.
 4. App A.4 (sec:abl_reasoning_subspace): runs/pro/reasoning_subspace120/problems/*.json (60 held-out problems) and
    runs/analysis/reasoning_subspace120_summary.json (its bootstrap intervals). Wrong steps: problems whose user-written wrong step is
    not adopted (baseline adoption index < 0.5) and problems in which installing k = 64/128/256 dimensions moves the adoption index
    by more than 0.1. Correct steps: cells k >= 32 and the full replacement at L31/L39 (the set whose L47 shift spans 70--88%);
    installation = user <- assistant, removal = assistant <- user; change of P(correct digit); accuracy = baseline mean
    P(correct digit) of a correct step.
 5. Sec 4.2 "97.7% with probability above 0.5": compliant_150 raw tgz, s3/attrgeo150/bodies, mode default, turns attr (A = self)
    and attr2 (B = self), role assistant, P = exp(native letter log-prob); 150 bodies x 2 orders.
 6. Sec 3.2 model facts: p2_small_checks_20261014/results/pilot_identity.json (identity record of a pilot; model facts only).
usage: check_mechanism_hand_numbers_20261026.py   (run from evidence/p2_crossmodel_wave3_20261019)"""
import csv
import glob
import json
import math
import re
import statistics as st
import tarfile

TEX = ''.join(open(f'../../paper/source/self_en_{p}.tex', encoding='utf-8').read() for p in ('body', 'appendix'))
SMALL = '../p2_small_checks_20261014/results/'
out = []


def line(name, paper, got, ok, phrase=None, note=''):
    flag = {True: 'OK', False: 'MISMATCH', None: 'NOTE'}[ok]
    intex = '' if phrase is None else (' [in tex]' if phrase in TEX else ' [PHRASE NOT IN TEX]')
    out.append(f'{name:58s} | paper {paper:<14s} | recomputed {got:<22s} | {flag}{intex}' + (f'  ({note})' if note else ''))


def rng(xs, fmt='{:.2f}'):
    return f'{fmt.format(min(xs))}--{fmt.format(max(xs))}'


# 1. gap fractions of the all-layer replacement
M = json.load(open('stats/v4_mechanism.json', encoding='utf-8'))['results']
rem, ins = {}, {}
for m in ('qwen_sp', 'qwen_th', 'oss_th'):
    a = M[m]['all']
    gap = a['base_A']['stat'] - a['base_U']['stat']
    rem[m], ins[m] = -a['swU8_A']['diff'] / gap, a['swA8_U']['diff'] / gap
out.append('1. Gap closed, all layers, 8 dims: ' + ', '.join(f'{m} removal {rem[m]:.3f} installation {ins[m]:.3f}' for m in rem))
line('removal closes ... of the gap', '0.99--1.00', rng(rem.values()), rng(rem.values()) == '0.99--1.00', '0.99--1.00 of the gap')
line('installation closes', '0.63--0.67', rng(ins.values()), rng(ins.values()) == '0.63--0.67', 'installation 0.63--0.67')

# 2. gpt-oss, 64 dimensions at every layer
K = json.load(open('runs/pro/thinking_swap/gpt_oss_20b_all_k64/summary.json', encoding='utf-8'))
gap64 = K['base_A']['you_share'] - K['base_U']['you_share']
r64, i64, i8 = K['swU64_A']['you_minus_base']['mean'], K['swA64_U']['you_minus_base']['mean'], M['oss_th']['all']['swA8_U']['diff']
gap8 = M['oss_th']['all']['base_A']['stat'] - M['oss_th']['all']['base_U']['stat']
out.append(f'2. gpt-oss k64: gap {gap64:.4f} (8-dim run {gap8:.4f}); random 64 dims: removal {K["rswU64_A"]["you_minus_base"]["mean"]:+.4f}, '
           f'installation {K["rswA64_U"]["you_minus_base"]["mean"]:+.4f}')
line('k64 removal', '-0.75', f'{r64:+.2f}', round(r64, 2) == -0.75, '(\\(-0.75\\))')
line('k64 installation vs 8 dims', '+0.58 / +0.51', f'{i64:+.2f} / {i8:+.2f}', (round(i64, 2), round(i8, 2)) == (0.58, 0.51), '\\(+0.58\\) against \\(+0.51\\)')
line('k64 / 8-dim share of the gap', '0.76 / 0.67', f'{i64 / gap64:.2f} / {i8 / gap8:.2f}', (round(i64 / gap64, 2), round(i8 / gap8, 2)) == (0.76, 0.67),
     '0.76 of the gap against 0.67')

# 3. App A.2: own-step preference grouped by self-index
NO_INDEX = {'Qwen3.6-27B', 'Qwen3.6-35B-A3B', 'Qwen3.5-27B', 'Gemma-4-31B', 'Seed-OSS-36B'}
ROWS = {('qwen38_27b_8models', 'single_pass'): 'Qwen3.8-27B', ('exaone4_32b_en_M', 'single_pass'): 'EXAONE-4.0.1-32B',
        ('qwen36_35b_8models', 'single_pass'): 'Qwen3.6-35B-A3B', ('glm46v_flash_8models', 'single_pass'): 'GLM-4.6V-Flash',
        ('gemma4_31b_8models', 'single_pass'): 'Gemma-4-31B', ('qwen36_27b_8models', 'single_pass'): 'Qwen3.6-27B',
        ('granite41_30b_en_M', 'single_pass'): 'Granite-4.1-30B', ('granite42_30b_en_M', 'single_pass'): 'Granite-4.2-30B',
        ('qwen35_27b_M', 'single_pass'): 'Qwen3.5-27B', ('qwen38_flashnext_8models', 'single_pass'): 'Qwen3.8-Flash-Next',
        ('glm47_flash_8models', 'single_pass'): 'GLM-4.7-Flash', ('seed_oss_36b_M', 'single_pass'): 'Seed-OSS-36B',
        ('qwen38_27b', 'thinking'): 'Qwen3.8-27B', ('gpt_oss_20b_mxfp4_en', 'thinking'): 'gpt-oss-20b', ('gpt_oss_120b_en', 'thinking'): 'gpt-oss-120b',
        ('exaone4_32b_en', 'thinking'): 'EXAONE-4.0.1-32B', ('glm53_flash_thinkm', 'thinking'): 'GLM-5.3-Flash',
        ('flashnext', 'thinking'): 'Qwen3.8-Flash-Next', ('qwen36_35b_a3b', 'thinking'): 'Qwen3.6-35B-A3B', ('gemma4_31b_en', 'thinking'): 'Gemma-4-31B',
        ('r1_distill_llama70b', 'thinking'): 'R1-Distill-Llama-70B', ('glm47_flash', 'thinking'): 'GLM-4.7-Flash', ('qwen36_27b', 'thinking'): 'Qwen3.6-27B',
        ('qwen35_27b', 'thinking'): 'Qwen3.5-27B', ('granite42_30b_en', 'thinking'): 'Granite-4.2-30B', ('seed_oss_36b', 'thinking'): 'Seed-OSS-36B',
        ('glm46v_flash', 'thinking'): 'GLM-4.6V-Flash'}
csvrows = {(r['model'], r['mode']): float(r['own_minus_user']) for r in csv.DictReader(open('stats/v4_ownstep.csv', encoding='utf-8-sig'))}
missing = [k for k in ROWS if k not in csvrows]
assert not missing, missing
out.append(f'3. own-step rows used: {sum(m == "single_pass" for _, m in ROWS)} without thinking, {sum(m == "thinking" for _, m in ROWS)} with '
           f'(paper tables: 1 + 11 and 6 + 9); not in the paper: {sorted(set(csvrows) - set(ROWS))}')
for mode, paper_idx, paper_no, ph in (('single_pass', '+0.14 / -0.12', '+0.07 / -0.17', 'without thinking, from \\(+0.14\\) to \\(-0.12\\)'),
                                      ('thinking', '+0.14 / -0.81', '+0.07 / -0.39', 'with thinking, from \\(+0.14\\) to \\(-0.81\\)')):
    for grp, paper in (('with self-index', paper_idx), ('without self-index', paper_no)):
        v = [csvrows[k] for k, name in ROWS.items() if k[1] == mode and (name in NO_INDEX) == (grp == 'without self-index')]
        got = f'{max(v):+.2f} / {min(v):+.2f}'
        line(f'own-step {mode}, {grp} (max / min)', paper, got, got == paper, ph if grp == 'with self-index' else None)

# 4. App A.4: self-index subspace of a reasoning step
recs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob('runs/pro/reasoning_subspace120/problems/*.json'))]
S = json.load(open('runs/analysis/reasoning_subspace120_summary.json', encoding='utf-8'))
assert len(recs) == 60 and {d['split'] for d in recs} == {'confirmation'}
idx = {}
for d in recs:
    for r in d['results']:
        key = ('base', r['step'], r['role']) if r['cond'] == 'baseline' else (r['step'], r['recipient'], r['layer'], r['method'], r['k'])
        idx[(d['id'],) + key] = r
ids = [d['id'] for d in recs]
not_adopted = {i for i in ids if idx[(i, 'base', 'wrong', 'user')]['adoption_index'] < 0.5}
counts, inside, share = [], [], []
for k in (64, 128, 256):
    for l in (31, 39):
        dd = {i: idx[(i, 'wrong', 'user', l, 'svd', k)]['adoption_index'] - idx[(i, 'base', 'wrong', 'user')]['adoption_index'] for i in ids}
        big = {i for i in ids if abs(dd[i]) > 0.1}
        counts.append(len(big))
        inside.append(len(big & not_adopted))
        share.append(sum(dd[i] for i in not_adopted) / sum(dd.values()))
gapd = [idx[(i, 'base', 'wrong', 'assistant')]['adoption_index'] - idx[(i, 'base', 'wrong', 'user')]['adoption_index'] for i in ids]
out.append(f'4. user-written wrong step not adopted (adoption index < 0.5): {len(not_adopted)} problems; role gap > 0.1 in {sum(g > 0.1 for g in gapd)}; '
           f'installation k64/128/256 x L31/L39 moves the index by > 0.1 in {counts} problems, of which {inside} are among the not-adopted '
           f'ones, which carry {min(share):.2f}--{max(share):.2f} of the mean change')
line('"concentrated ... in the 8--10 problems"', '8--10', f'{min(counts)}--{max(counts)} (not adopted: {len(not_adopted)})',
     min(counts) == 8 and max(counts) == 10, 'in the 8--10 problems', 'rule not recorded; |change| > 0.1 in the installation cells')

CELLS = [('svd', k) for k in (32, 64, 128, 256)] + [('full', 0)]


def summ(rec, l, m, k):
    return next(c for c in S['cells'] if c['step'] == 'correct' and c['recipient'] == rec and c['layer'] == l and c['method'] == m and c['k'] == k)


inst = [summ('user', l, m, k) for m, k in CELLS for l in (31, 39)]
mx = max(abs(c['d_p_correct']) for c in inst)
line('correct step, installation: max |change P(correct)|', '0.018', f'{mx:.4f}', round(mx, 3) == 0.018, 'changes by at most 0.018')
line('  ... intervals include zero', 'yes', str(all(c['d_p_correct_ci'][0] <= 0 <= c['d_p_correct_ci'][1] for c in inst)),
     all(c['d_p_correct_ci'][0] <= 0 <= c['d_p_correct_ci'][1] for c in inst))
mv = [c['moved_toward_donor']['47'] for c in inst]
line('  ... L47 states moved toward the own step', '70--88%', f'{100 * min(mv):.0f}--{100 * max(mv):.0f}%',
     (round(100 * min(mv)), round(100 * max(mv))) == (70, 88), '70--88\\%', f'k >= 64 only: {100 * min(mv[2:]):.0f}--{100 * max(mv[2:]):.0f}%')
rm = [summ('assistant', l, m, k) for m, k in CELLS for l in (31, 39)]
drop = [-c['d_p_correct'] for c in rm]
line('correct step, removal: drop of P(correct)', '0.03--0.08', rng(drop), rng(drop) == '0.03--0.08', 'correct digit by 0.03--0.08')
med, n90, n01, n03, flip = [], [], [], [], []
for m, k in CELLS:
    for l in (31, 39):
        dd = [idx[(i, 'correct', 'assistant', l, m, k)]['p_correct_digit'] - idx[(i, 'base', 'correct', 'assistant')]['p_correct_digit'] for i in ids]
        med.append(st.median(dd))
        tot, c, n = sum(dd), 0.0, 0
        for x in sorted(dd, key=lambda x: -abs(x)):
            c, n = c + x, n + 1
            if abs(c) >= 0.9 * abs(tot):
                break
        n90.append(n)
        n01.append(sum(x < -0.1 for x in dd))
        n03.append(sum(x < -0.3 for x in dd))
        flip.append(sum(idx[(i, 'correct', 'assistant', l, m, k)]['top_id'] != idx[(i, 'base', 'correct', 'assistant')]['top_id'] for i in ids))
line('  ... median change zero', '0', f'|median| <= {max(abs(x) for x in med):.4f}', max(abs(x) for x in med) < 0.002, '(median change zero)')
n1 = [sum(abs(idx[(i, 'correct', 'assistant', l, m, k)]['p_correct_digit'] - idx[(i, 'base', 'correct', 'assistant')]['p_correct_digit']) > 0.1
           for i in ids) for m, k in CELLS for l in (31, 39)]
# 10-26: the text said "concentrated in 5--7", which no rule reproduces (90% of change 5--10, drop > 0.1 6--11, drop > 0.3 3--7);
# replaced by the |change| > 0.1 count of the sentence before (fix_paper_reasoning_concentration_20261026.py)
line('  ... |change| > 0.1 in only N of the 60 problems', '7--12', f'{min(n1)}--{max(n1)}', (min(n1), max(n1)) == (7, 12),
     'more than 0.1 in only 7--12 of the 60 problems',
     f'per cell (k32 L31, L39, ..., full L39) {n1}; 90% of change {min(n90)}--{max(n90)}, drop > 0.1 {min(n01)}--{max(n01)}, '
     f'drop > 0.3 {min(n03)}--{max(n03)}, answer changes {min(flip)}--{max(flip)}')
acc = {r: st.mean(idx[(i, 'base', 'correct', r)]['p_correct_digit'] for i in ids) for r in ('user', 'assistant')}
line('user-written correct step as accurate as own', '0.97 / 0.95', f'{acc["user"]:.2f} / {acc["assistant"]:.2f}',
     (round(acc['user'], 2), round(acc['assistant'], 2)) == (0.97, 0.95), '(0.97 and 0.95)', 'mean P(correct digit), user / own')

# 5. Sec 4.2: own messages claimed with probability above 0.5
T = tarfile.open(SMALL + 'compliant_150_20261015/raw/compliant_150_per_body.tgz')
ps = []
for mem in T.getmembers():
    if re.fullmatch(r's3/attrgeo150/bodies/[^/]+\.json', mem.name):
        for c in json.load(T.extractfile(mem))['conditions']:
            if c['mode'] == 'default' and c['turn'] in ('attr', 'attr2'):
                ps.append(math.exp(c['roles']['assistant']['native_letter_logprob']['A' if c['turn'] == 'attr' else 'B']))
n5 = sum(p > 0.5 for p in ps)
line('own messages claimed with P > 0.5', '97.7%', f'{100 * n5 / len(ps):.1f}% ({n5}/{len(ps)})', round(100 * n5 / len(ps), 1) == 97.7,
     '97.7\\% with probability above 0.5', f'mean P(you) {st.mean(ps):.3f}')

# 6. Sec 3.2: model facts
I = json.load(open(SMALL + 'pilot_identity.json', encoding='utf-8'))
lt = I['layer_types']
full = [i for i, t in enumerate(lt) if t == 'full_attention']
line('parameters', '27,356,728,560', f'{sum(I["dtype_parameter_counts"].values()):,}', sum(I['dtype_parameter_counts'].values()) == 27356728560, '27,356,728,560 parameters',
     f'dtypes {list(I["dtype_parameter_counts"])}, quantized {I["quantized"]}')
line('architecture', 'Qwen3_5ForC.', I['architecture'][0][:22], I['architecture'] == ['Qwen3_5ForConditionalGeneration'], 'Qwen3\\_5ForConditionalGeneration')
line('blocks: linear / full, full at 3, 7, ..., 63', '64: 48 / 16', f'{len(lt)}: {len(lt) - len(full)} / {len(full)}',
     len(lt) == 64 and len(full) == 16 and full == list(range(3, 64, 4)), '48 linear-attention and 16 full-attention blocks')
line('revision 1d4bf0f2ff60, 248,320-entry vocabulary', 'see text', 'not in pilot_identity.json', None, None,
     'this file records neither; the revision is noted only in the wave-3 README')

print('\n'.join(out))
print('\nall checks OK' if not any(' | MISMATCH' in s for s in out) else '\nMISMATCH FOUND (see lines above)')
