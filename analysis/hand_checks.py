"""Checks of the numbers that the text states without a generated table: each is recomputed from the archived records and
printed as "paper | recomputed | OK / MISMATCH". With --paper-source (the folder of the paper's LaTeX source) the checks also
report whether the phrase of the number is in the current English text and read the cells of Table tab:self_crossmodel from it.

mechanism (stats/check_mechanism_hand_numbers_20261026.txt), standard library only. Sources and rules:
 1. "removal closes 0.99--1.00 of the gap ..., installation 0.63--0.67": v4_mechanism.json (causal_tests.py),
    results.{qwen_sp,qwen_th,oss_th}.all; gap = base_A.stat - base_U.stat; removal = -swU8_A.diff / gap, installation =
    swA8_U.diff / gap (diff = item-paired).
 2. gpt-oss 64 dimensions (-0.75, +0.58 against +0.51, 0.76 of the gap against 0.67): the run thinking_swap/gpt_oss_20b_all_k64
    (summary.json: swU64_A, swA64_U you_minus_base.mean; gap = base_A - base_U you_share) and oss_th.all.swA8_U.diff.
 3. Own-step group ranges: v4_ownstep.csv (ownstep.py), models without a self-index = Qwen3.6-27B, Qwen3.6-35B-A3B,
    Qwen3.5-27B, Gemma-4-31B and Seed-OSS-36B; range = max and min of own_minus_user per group and mode. (A table of an earlier
    version of the appendix; skipped when rows are missing from the package.)
 4. Self-index subspace of a reasoning step (an earlier appendix): runs reasoning_subspace120 (60 held-out problems) and
    runs/analysis/reasoning_subspace120_summary.json; not in the package, so skipped.
 5. "97.7% with probability above 0.5": compliant_150_per_body.tgz, attrgeo150/bodies, mode default, turns attr (A = self) and
    attr2 (B = self), role assistant, P = exp(native letter log-prob); 150 bodies x 2 orders.
 6. Model facts of Qwen3.8-27B: p2_small_checks_20261014/results/pilot_identity.json (identity record of a pilot; model facts only).
crossmodel (stats/check_crossmodel_hand_numbers_20261026.txt): cross-model numbers recomputed from the archived per-layer shares
  and per-body records; never fits a rule to a value. Shared rule: a layer singles out the own message if share_A_lowest >= 0.8;
  the last third is layers >= ceil(2L/3), the last quarter layers >= ceil(3L/4); "content or onset" = union of the two
  positions, a layer counted once; four-stream models (Qwen3.8-Flash-Next, GLM-5.3-Flash): mean_of_streams
  share_A_lowest_by_layer, strongest stream = the stream with the most layers.
 1. Table tab:self_crossmodel, columns content / onset / last third, under the non-thinking rendering (thinking rendering for
    models with only a thinking mode); Qwen3.5-27B and Granite-4.1/4.2 also against v4_geometry_summary.csv (needs the paper source).
 2. Every model and mode that recognizes its own messages singles it out at every layer of the last third, the models without a
    self-index at most 3 of its layers (Gemma-4-31B under the thinking rendering); the same for the last quarter.
 3. Body-level P("you") from the per-body records roles.<role>.by_order[o].<option>: per body the mean over the orders used,
    then the mean over bodies; "own" = P(you | assistant), "other" = max over User and System of the body-mean P(you); "excess" =
    mean over bodies of P(you|A) - max(P(you|U), P(you|S)). Orders: o0 you/user/system, o1 system/you/user, o2 user/system/you;
    the paper uses o0 and o1. GLM-4.6V-Flash: first-option probability per order and role; excess. GLM-4.7-Flash: excess over
    o0+o1 and its bodies > 0; excess in o2 alone ("you" last).
 4. Qwen3.8-27B simulated FP8: own, other, bodies with excess > 0, index layers. Qwen3.8-Flash-Next natural order (roles
    assistant and user only): own and User P(you).
 5. Qwen3.8-27B English: own and other (o0+o1), content layers.
traces (stats/trace_counts.txt): reasoning-trace counts of the causal-test text ("They are sufficient where nothing else names
  the source"):
  - Qwen3.8-27B thinking, thinking_swap/qwen38_27b_all: a trace "describes an assistant message" if it mentions the assistant
    ("assistant", case-insensitive, not followed by ")", or 助手) after the answer option is removed from the trace ("AI
    assistant" / "AI Assistant" / "AI 助手" / "AI助手", the option text of the attribution question, which every trace quotes);
    paper: 102 of 150 for the edited User message (swA8_U), none unedited (base_U), 3 after the random-subspace edit (rswA8_U).
  - gpt-oss-20b, thinking_swap/gpt_oss_20b_all: traces for the edited System message (swA8_S) containing the exact phrase "the
    developer message" (case-insensitive); paper: 147 of 150 (149 contain "developer" at all, which the text does not claim).

usage: hand_checks.py --results RES --out OUT [--paper-source DIR]   (needs v4_mechanism.json, v4_ownstep.csv and
       v4_geometry_summary.csv in OUT)
"""
import csv
import glob
import json
import math
import os
import re
import statistics as st
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import M8, SC, W3, Roots, arguments, run_path, write_text  # noqa: E402


def mechanism_checks(A3, S3, SMALL, TEX):
    """A3: the wave-3 archive in the results package; S3: output stats folder (v4_mechanism.json, v4_ownstep.csv);
    SMALL: results folder of the two-level archive; TEX: text of the English paper source or None."""
    out = []

    def line(name, paper, got, ok, phrase=None, note=''):
        flag = {True: 'OK', False: 'MISMATCH', None: 'NOTE'}[ok]
        intex = '' if phrase is None or TEX is None else (' [in tex]' if phrase in TEX else ' [PHRASE NOT IN TEX]')
        out.append(f'{name:58s} | paper {paper:<14s} | recomputed {got:<22s} | {flag}{intex}' + (f'  ({note})' if note else ''))


    def rng(xs, fmt='{:.2f}'):
        return f'{fmt.format(min(xs))}--{fmt.format(max(xs))}'


    # 1. gap fractions of the all-layer replacement
    M = json.load(open(S3 / 'v4_mechanism.json', encoding='utf-8'))['results']
    rem, ins = {}, {}
    for m in ('qwen_sp', 'qwen_th', 'oss_th'):
        a = M[m]['all']
        gap = a['base_A']['stat'] - a['base_U']['stat']
        rem[m], ins[m] = -a['swU8_A']['diff'] / gap, a['swA8_U']['diff'] / gap
    out.append('1. Gap closed, all layers, 8 dims: ' + ', '.join(f'{m} removal {rem[m]:.3f} installation {ins[m]:.3f}' for m in rem))
    line('removal closes ... of the gap', '0.99--1.00', rng(rem.values()), rng(rem.values()) == '0.99--1.00', '0.99--1.00 of the gap')
    line('installation closes', '0.63--0.67', rng(ins.values()), rng(ins.values()) == '0.63--0.67', 'installation 0.63--0.67')

    # 2. gpt-oss, 64 dimensions at every layer
    K = json.load(open(run_path(A3, 'thinking_swap/gpt_oss_20b_all_k64') / 'summary.json', encoding='utf-8'))
    gap64 = K['base_A']['you_share'] - K['base_U']['you_share']
    r64, i64, i8 = K['swU64_A']['you_minus_base']['mean'], K['swA64_U']['you_minus_base']['mean'], M['oss_th']['all']['swA8_U']['diff']
    gap8 = M['oss_th']['all']['base_A']['stat'] - M['oss_th']['all']['base_U']['stat']
    out.append(f'2. gpt-oss k64: gap {gap64:.4f} (8-dim run {gap8:.4f}); random 64 dims: removal {K["rswU64_A"]["you_minus_base"]["mean"]:+.4f}, '
               f'installation {K["rswA64_U"]["you_minus_base"]["mean"]:+.4f}')
    line('k64 removal', '-0.75', f'{r64:+.2f}', round(r64, 2) == -0.75, '(\\(-0.75\\))')
    line('k64 installation vs 8 dims', '+0.58 / +0.51', f'{i64:+.2f} / {i8:+.2f}', (round(i64, 2), round(i8, 2)) == (0.58, 0.51), '\\(+0.58\\) against \\(+0.51\\)')
    line('k64 / 8-dim share of the gap', '0.76 / 0.67', f'{i64 / gap64:.2f} / {i8 / gap8:.2f}', (round(i64 / gap64, 2), round(i8 / gap8, 2)) == (0.76, 0.67),
         '0.76 of the gap against 0.67')

    # 3. own-step preference grouped by self-index (a table of an earlier appendix)
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
    csvrows = {(r['model'], r['mode']): float(r['own_minus_user']) for r in csv.DictReader(open(S3 / 'v4_ownstep.csv', encoding='utf-8-sig'))}
    missing = [k for k in ROWS if k not in csvrows]
    if missing:  # rows of measures that are not in the results package
        out.append(f'3. skipped: needs inputs not in the package (own-step rows {missing})')
    else:
        out.append(f'3. own-step rows used: {sum(m == "single_pass" for _, m in ROWS)} without thinking, {sum(m == "thinking" for _, m in ROWS)} with '
                   f'(paper tables: 1 + 11 and 6 + 9); not in the paper: {sorted(set(csvrows) - set(ROWS))}')
        for mode, paper_idx, paper_no, ph in (('single_pass', '+0.14 / -0.12', '+0.07 / -0.17', 'without thinking, from \\(+0.14\\) to \\(-0.12\\)'),
                                              ('thinking', '+0.14 / -0.81', '+0.07 / -0.39', 'with thinking, from \\(+0.14\\) to \\(-0.81\\)')):
            for grp, paper in (('with self-index', paper_idx), ('without self-index', paper_no)):
                v = [csvrows[k] for k, name in ROWS.items() if k[1] == mode and (name in NO_INDEX) == (grp == 'without self-index')]
                got = f'{max(v):+.2f} / {min(v):+.2f}'
                line(f'own-step {mode}, {grp} (max / min)', paper, got, got == paper, ph if grp == 'with self-index' else None)

    # 4. self-index subspace of a reasoning step (an earlier appendix)
    if not run_path(A3, 'reasoning_subspace120/problems').exists():
        out.append('4. skipped: needs inputs not in the package (runs reasoning_subspace120/problems, '
                   'runs/analysis/reasoning_subspace120_summary.json)')
    else:
        recs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(str(run_path(A3, 'reasoning_subspace120/problems') / '*.json')))]
        S = json.load(open(A3 / 'runs' / 'analysis' / 'reasoning_subspace120_summary.json', encoding='utf-8'))
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
        # the count of problems with |change| > 0.1 (the rule of the sentence before)
        line('  ... |change| > 0.1 in only N of the 60 problems', '7--12', f'{min(n1)}--{max(n1)}', (min(n1), max(n1)) == (7, 12),
             'more than 0.1 in only 7--12 of the 60 problems',
             f'per cell (k32 L31, L39, ..., full L39) {n1}; 90% of change {min(n90)}--{max(n90)}, drop > 0.1 {min(n01)}--{max(n01)}, '
             f'drop > 0.3 {min(n03)}--{max(n03)}, answer changes {min(flip)}--{max(flip)}')
        acc = {r: st.mean(idx[(i, 'base', 'correct', r)]['p_correct_digit'] for i in ids) for r in ('user', 'assistant')}
        line('user-written correct step as accurate as own', '0.97 / 0.95', f'{acc["user"]:.2f} / {acc["assistant"]:.2f}',
             (round(acc['user'], 2), round(acc['assistant'], 2)) == (0.97, 0.95), '(0.97 and 0.95)', 'mean P(correct digit), user / own')

    # 5. own messages claimed with probability above 0.5
    T = tarfile.open(SMALL / 'compliant_150_20261015' / 'raw' / 'compliant_150_per_body.tgz')
    ps = []
    for mem in T.getmembers():
        if re.fullmatch(r's3/attrgeo150/bodies/[^/]+\.json', mem.name):
            for c in json.load(T.extractfile(mem))['conditions']:
                if c['mode'] == 'default' and c['turn'] in ('attr', 'attr2'):
                    ps.append(math.exp(c['roles']['assistant']['native_letter_logprob']['A' if c['turn'] == 'attr' else 'B']))
    n5 = sum(p > 0.5 for p in ps)
    line('own messages claimed with P > 0.5', '97.7%', f'{100 * n5 / len(ps):.1f}% ({n5}/{len(ps)})', round(100 * n5 / len(ps), 1) == 97.7,
         '97.7\\% with probability above 0.5', f'mean P(you) {st.mean(ps):.3f}')

    # 6. model facts
    I = json.load(open(SMALL / 'pilot_identity.json', encoding='utf-8'))
    lt = I['layer_types']
    full = [i for i, t in enumerate(lt) if t == 'full_attention']
    line('parameters', '27,356,728,560', f'{sum(I["dtype_parameter_counts"].values()):,}', sum(I['dtype_parameter_counts'].values()) == 27356728560, '27,356,728,560 parameters',
         f'dtypes {list(I["dtype_parameter_counts"])}, quantized {I["quantized"]}')
    line('architecture', 'Qwen3_5ForC.', I['architecture'][0][:22], I['architecture'] == ['Qwen3_5ForConditionalGeneration'], 'Qwen3\\_5ForConditionalGeneration')
    line('blocks: linear / full, full at 3, 7, ..., 63', '64: 48 / 16', f'{len(lt)}: {len(lt) - len(full)} / {len(full)}',
         len(lt) == 64 and len(full) == 16 and full == list(range(3, 64, 4)), '48 linear-attention and 16 full-attention blocks')
    line('revision 1d4bf0f2ff60, 248,320-entry vocabulary', 'see text', 'not in pilot_identity.json', None, None,
         'this file records neither; the revision is noted only in the wave-3 README')

    text = '\n'.join(out) + ('\n\nall checks OK' if not any(' | MISMATCH' in s for s in out) else '\n\nMISMATCH FOUND (see lines above)')
    print(text)
    return text


def crossmodel_checks(A8, A3, S3, TEX):
    """A8, A3: the two archives in the results package; S3: output stats folder (v4_geometry_summary.csv); TEX: text of
    the English paper body (for the cells of Table tab:self_crossmodel) or None."""
    M8, R = A8 / 'models', A3 / 'runs'
    NAMES = {'qwen38_27b': 'Qwen3.8-27B', 'qwen38_flashnext': 'Qwen3.8-Flash-Next', 'gpt_oss_120b': 'gpt-oss-120b', 'gpt_oss_20b': 'gpt-oss-20b',
             'glm53_flash': 'GLM-5.3-Flash', 'exaone4_32b': 'EXAONE-4.0.1-32B', 'glm46v_flash': 'GLM-4.6V-Flash', 'glm47_flash': 'GLM-4.7-Flash',
             'r1_llama70b': 'R1-Distill-Llama-70B', 'seed_oss_36b': 'Seed-OSS-36B', 'gemma4_31b': 'Gemma-4-31B', 'qwen36_27b': 'Qwen3.6-27B',
             'qwen36_35b': 'Qwen3.6-35B-A3B', 'qwen35_27b': 'Qwen3.5-27B', 'granite41_30b': 'Granite-4.1-30B', 'granite42_30b': 'Granite-4.2-30B'}
    STREAM = 'stream'
    GEO = {  # (model, rendering) -> run summary with per-layer share_A_lowest, or (STREAM, file) for the four-stream models
        ('qwen38_27b', 'non_thinking'): M8 / 'qwen38_27b/non_thinking/geo_attr/_run_summary.json',
        ('qwen38_27b', 'thinking'): M8 / 'qwen38_27b/thinking/geo/_run_summary.json',
        ('gpt_oss_120b', 'thinking'): run_path(A3, 'suite_think/gpt_oss_120b_en/summary.json'),
        ('gpt_oss_20b', 'thinking'): run_path(A3, 'suite_think/gpt_oss_20b_mxfp4_en/summary.json'),
        ('qwen38_flashnext', 'non_thinking'): (STREAM, run_path(A3, 'flashnext_stream_20261020/flashnext_stream_geometry_non_20261020.json')),
        ('qwen38_flashnext', 'thinking'): (STREAM, run_path(A3, 'flashnext_stream_20261020/flashnext_stream_geometry_think_20261020.json')),
        ('glm53_flash', 'thinking'): (STREAM, run_path(A3, 'suite_think/glm53_flash_stream_geometry.json')),
        ('exaone4_32b', 'non_thinking'): run_path(A3, 'suite/exaone4_32b_en/summary.json'),
        ('exaone4_32b', 'thinking'): run_path(A3, 'suite_think/exaone4_32b_en/summary.json'),
        ('r1_llama70b', 'thinking'): M8 / 'r1_llama70b/thinking/geo/_run_summary.json',
        ('granite42_30b', 'non_thinking'): run_path(A3, 'suite/granite42_30b_en/summary.json'),
        ('granite41_30b', 'non_thinking'): run_path(A3, 'suite/granite41_30b_en/summary.json'),
        ('seed_oss_36b', 'non_thinking'): run_path(A3, 'suite/seed_oss_36b/summary.json'),
        ('seed_oss_36b', 'thinking'): run_path(A3, 'suite_think/seed_oss_36b/summary.json'),
        ('qwen35_27b', 'non_thinking'): run_path(A3, 'suite/qwen35_27b/summary.json')}
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
    if TEX is None:
        results.append(('Table 5: skipped (the paper source was not given, --paper-source)', '-', '-', 'info'))
    else:
        tex = TEX
        tab = tex[tex.index('\\label{tab:self_crossmodel}'):]
        tab = tab[tab.index('\\midrule'):tab.index('\\bottomrule')]
        paper_rows = {}
        for line in tab.splitlines()[1:]:
            cells = [c.strip() for c in line.rstrip('\\ ').split('&')]
            if len(cells) > 3:
                paper_rows[cells[0]] = cells[1:4]
        key_of = {v: k for k, v in NAMES.items()}
        for disp, (pc, po, pl) in paper_rows.items():
            if disp not in key_of:  # rows added to the table after this check was written (OLMo-2: v5_olmo_*.csv)
                results.append((f'Table 5 {disp}: not covered by this check', '-', '-', 'info'))
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
        for r in csv.DictReader(open(S3 / 'v4_geometry_summary.csv', encoding='utf-8')):
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
    mean, other, ex, pos, n = attr(nat, (0, 1), roles=('assistant', 'user'))  # the first two option orders, as the text
    check('Qwen3.8-Flash-Next natural order (o0+o1): P(you) own vs User', '0.07 vs 0.07', f"{mean['assistant']:.2f} vs {mean['user']:.2f}")
    mean, other, ex, pos, n = attr(nat, (0, 1, 2), roles=('assistant', 'user'))
    results.append(('  Flash-Next natural order, all three orders (the old text)', '-', f"{mean['assistant']:.2f} vs {mean['user']:.2f}", 'info'))
    en = bodies(M8 / 'qwen38_27b/non_thinking/geo_attr_en')
    mean, other, ex, pos, n = attr(en, (0, 1))
    check('Qwen3.8-27B English: own vs at most other (o0+o1)', '0.68 vs 0.08', f"{mean['assistant']:.2f} vs {other:.2f}")
    s = json.load(open(M8 / 'qwen38_27b/non_thinking/geo_attr_en/_run_summary.json', encoding='utf-8'))
    check('Qwen3.8-27B English: content layers', '64/64', f"{sum(x >= 0.8 for x in s['content']['share_A_lowest'])}/{s['n_layers']}")

    w = max(len(r[0]) for r in results)
    lines = [f'{name:{w}s} | paper {paper} | recomputed {got} | {flag}' for name, paper, got, flag in results]
    bad = [r for r in results if r[3] == 'MISMATCH']
    lines.append(f'\n{sum(r[3] == "OK" for r in results)} OK, {len(bad)} MISMATCH, {sum(r[3] == "info" for r in results)} info')
    text = '\n'.join(lines)
    print(text)
    return text


def trace_counts(A3):
    QWEN, GPTOSS = str(run_path(A3, 'thinking_swap/qwen38_27b_all')), str(run_path(A3, 'thinking_swap/gpt_oss_20b_all'))

    def items(d, cond):
        xs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(d + '/items/*.json'))]
        return [x for x in xs if x['cond'] == cond]

    def describes_assistant(s):
        s_en = s.replace('AI assistant', '').replace('AI Assistant', '')
        s_zh = s.replace('AI 助手', '').replace('AI助手', '')
        return bool(re.search(r'assistant(?! *\))', s_en, re.I)) or '助手' in s_zh


    rows = []
    for cond, paper in (('swA8_U', 102), ('base_U', 0), ('rswA8_U', 3)):
        xs = items(QWEN, cond)
        got = sum(describes_assistant(x.get('gen', '')) for x in xs)
        rows.append((f'Qwen3.8-27B thinking {cond}: describes an assistant message', paper, got, len(xs)))
        plain = sum(bool(re.search('assistant', x.get('gen', ''), re.I)) or '助手' in x.get('gen', '') for x in xs)
        rows.append((f'  (without removing the option text: {plain}/{len(xs)}, not the paper rule)', None, None, None))
    xs = items(GPTOSS, 'swA8_S')
    g = [x.get('gen', '').lower() for x in xs]
    rows.append(('gpt-oss-20b swA8_S: "the developer message"', 147, sum('the developer message' in s for s in g), len(xs)))
    rows.append((f'  (any "developer": {sum("developer" in s for s in g)}/{len(xs)}, not claimed)', None, None, None))

    ok = True
    for name, paper, got, n in rows:
        if paper is None:
            print(name)
            continue
        flag = 'OK' if paper == got else 'MISMATCH'
        ok &= paper == got
        print(f'{name:62s} paper {paper:>3}/150  recomputed {got:>3}/{n}  {flag}')
    print('all match' if ok else 'MISMATCH FOUND')
    return ok


def main(argv=None):
    ap = arguments(__doc__)
    ap.add_argument('--paper-source', help='folder with self_en_body.tex and self_en_appendix.tex (optional)')
    args = ap.parse_args(argv)
    r = Roots(args)
    A3, S3 = r.res(W3), r.out(W3 + '/stats')
    body = tex_all = None
    if args.paper_source:
        P = Path(args.paper_source)
        body = (P / 'self_en_body.tex').read_text(encoding='utf-8')
        tex_all = ''.join(open(P / f'self_en_{p}.tex', encoding='utf-8').read() for p in ('body', 'appendix'))
    print('== mechanism')
    write_text(S3 / 'check_mechanism_hand_numbers_20261026.txt', mechanism_checks(A3, S3, r.res(SC + '/results'), tex_all) + '\n')
    print('== crossmodel')
    write_text(S3 / 'check_crossmodel_hand_numbers_20261026.txt', crossmodel_checks(r.res(M8), A3, S3, body) + '\n')
    print('== traces')
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        trace_counts(A3)
    print(buf.getvalue(), end='')
    write_text(S3 / 'trace_counts.txt', buf.getvalue())


if __name__ == '__main__':
    main()
