"""DRAFT figure self_kl_scatter (11-10, not in the paper): the separation of the model's own message in next-token prediction
(Section sec:res_nexttoken) against the share of message-content layers that carry the self-index, as a continuous relation.
Spec of 11-10 (second version; the first, with the content-or-onset union on x and a pass/fail reading of y = 1, is kept in
drafts_20261110/superseded_union_x/).

(a) Main probe, the 25 models of the main comparison, one KL setting per model (dict MAIN of
    evidence/p2_modeswitch_20261107/code/local/make_kl_index_20261108.py: non-thinking rendering where the template has one).
    x = share of decoder layers at which the own message is singled out (smallest distance ratio R_A) in at least 80% of the
        held-out bodies on MESSAGE CONTENT, the content column of Table tab:self_crossmodel, from the run summaries read by
        release/p2_release_20261102/github/analysis/fig_crossmodel.py (the script of the 25-model Figure fig:self_crossmodel), in
        the rendering of the KL setting (non-thinking; thinking for the models with only a thinking mode). Four-stream models
        (Qwen3.8-Flash-Next, GLM-5.3-Flash): mean of the four streams (38/48, 18/45; the table gives the strongest stream in
        parentheses), drawn with a dark marker edge.
    y = the smaller of the two content ratios KL(A||U)/KL(U||S) and KL(A||S)/KL(U||S) (control tokens removed and renormalized),
        modeswitch/ctrl/<setting>/summary.json -> ratios['A-U/U-S_kl_content'], ['A-S/U-S_kl_content'] (Table tab:self_kl),
        with the 95% interval of that ratio (15 wording families resampled) as a vertical bar. Log scale.
    Colors and fill as in Figure fig:self_crossmodel: filled = the layers singled out at content or onset include every layer of
    the last third of the depth.
(b) Natural conversation order: x = the content column of Table tab:self_natural (separate_content of the suite_nat3 summaries)
    and of Table tab:self_natural3 (DeepSeek-V4.1-Flash: mean of its four streams, dark edge; the two Ministral-3-14B versions);
    fill from content or onset in this order, as in Figure fig:self_natural. y as in (a) from the natural-order ratios: for the
    10 models of Table tab:self_natural the full distribution, modeswitch/<setting>/summary.json ->
    orders.natural.ratios['A-U/U-S_kl_body'], ['A-S/U-S_kl_body'] (on which the control tokens of these models carry almost no
    probability); for the 3 models that accept only natural order the content ratios, modeswitch/nat3/<setting>/summary.json ->
    ratios['A-U/U-S_kl_content'], ['A-S/U-S_kl_content'] (Table tab:self_kl_natural). Arrows from the main-probe point of (a)
    (faint) to the natural-order point for the 10 models measured in both orders; squares for the 3 natural-order-only models.
The dashed line y = 1 is a reference only: the own message differs from the User or System message as much as these two differ
from each other. Each panel gives Spearman's rho between x and y over the models shown (ties: mean ranks).
KL summaries are read from the local mirror evidence/p2_modeswitch_20261107/results/{server3,pro}/modeswitch (server3 first, it
holds GLM-5.3-Flash and DeepSeek-V4.1-Flash). Checks (the script stops on a mismatch): the content layer counts against the
content columns of Tables tab:self_crossmodel, tab:self_natural and tab:self_natural3 in paper/source (read only); the
content-or-onset shares and the fill against the archived points of the earlier figure versions (stats/self_crossmodel_points_v6.json,
18 models; stats/self_natural_points_v1.json, 8 models) and, if given, a points file of the 25-model release script (--check).
Outputs (drafts, never paper/): drafts_20261110/self_kl_scatter.pdf/.png and self_kl_scatter_points.json next to this script.
usage: python -I make_fig_kl_scatter_20261110.py [--check POINTS_JSON] [--out DIR]"""
import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent              # .../evidence/p2_crossmodel_wave3_20261019/code/local/figures
ROOT = HERE.parents[4]                               # projects/consciousness
W3 = ROOT / 'evidence' / 'p2_crossmodel_wave3_20261019'
A8 = ROOT / 'evidence' / 'p2_crossmodel_8models_20261018'
MS = ROOT / 'evidence' / 'p2_modeswitch_20261107' / 'results'
MS_ROOTS = (MS / 'server3' / 'modeswitch', MS / 'pro' / 'modeswitch')   # server3 first
PAPER = ROOT / 'paper' / 'source'                   # read only, for the checks of the content counts

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'axes.titlesize': 10.5,
                     'axes.titleweight': 'bold', 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5})

# names and colors of fig_crossmodel.py (release, 11-04) incl. the NEWDEV entries of newdev.py
NAMES = {'qwen38_27b': 'Qwen3.8-27B', 'qwen38_flashnext': 'Qwen3.8-Flash-Next', 'gpt_oss_120b': 'gpt-oss-120b', 'gpt_oss_20b': 'gpt-oss-20b',
         'glm53_flash': 'GLM-5.3-Flash', 'exaone4_32b': 'EXAONE-4.0.1-32B', 'glm46v_flash': 'GLM-4.6V-Flash', 'glm47_flash': 'GLM-4.7-Flash',
         'r1_llama70b': 'R1-Distill-Llama-70B', 'seed_oss_36b': 'Seed-OSS-36B', 'gemma4_31b': 'Gemma-4-31B', 'qwen36_27b': 'Qwen3.6-27B',
         'qwen36_35b': 'Qwen3.6-35B-A3B', 'qwen35_27b': 'Qwen3.5-27B', 'granite41_30b': 'Granite-4.1-30B',
         'granite42_30b': 'Granite-4.2-30B', 'olmo2_7b': 'OLMo-2-7B', 'olmo2_13b': 'OLMo-2-13B',
         'ernie45_21b': 'ERNIE-4.5-21B-A3B', 'mimo7b_rl': 'MiMo-7B-RL', 'mimo_vl7b': 'MiMo-VL-7B', 'mimo_distill9b': 'MiMo-V2.6-Distill-9B',
         'moonlight16b': 'Moonlight-16B-A3B', 'apertus8b': 'Apertus-8B', 'phi4': 'phi-4',
         # natural-order-only models (natural_order.py; Ministral-3-14B-Reasoning is not in Figure fig:self_natural)
         'dsv41_flash': 'DeepSeek-V4.1-Flash', 'ministral3_14b': 'Ministral-3-14B-Instruct', 'ministral3_14b_r': 'Ministral-3-14B-Reasoning'}
COL = {'qwen38_27b': '#2a78d6', 'qwen38_flashnext': '#6a3fb5', 'glm46v_flash': '#1baf7a', 'glm47_flash': '#0f7d58',
       'qwen36_35b': '#9aa0a6', 'qwen36_27b': '#6f767d', 'gemma4_31b': '#c49a00', 'r1_llama70b': '#eb6834',
       'exaone4_32b': '#1098ad', 'seed_oss_36b': '#8c564b', 'gpt_oss_20b': '#a61e4d', 'gpt_oss_120b': '#e64980', 'glm53_flash': '#74b816',
       'qwen35_27b': '#a3b3c6', 'granite42_30b': '#22305e', 'granite41_30b': '#7383c9', 'olmo2_7b': '#b8b85a', 'olmo2_13b': '#6b6b1f',
       'ernie45_21b': '#3b5bdb', 'mimo7b_rl': '#f76707', 'mimo_vl7b': '#fd7e14', 'mimo_distill9b': '#c2255c',
       'moonlight16b': '#495057', 'apertus8b': '#2f9e44', 'phi4': '#1971c2',
       # natural_order.py: two hues not used in Figure fig:self_crossmodel; one more for Ministral-3-14B-Reasoning (new here)
       'dsv41_flash': '#d9480f', 'ministral3_14b': '#5c5f66', 'ministral3_14b_r': '#ae3ec9'}
STREAM_EDGE = '#1a1a1a'   # marker edge of the four-stream models (x = mean of the four streams)

# KL setting (MAIN of make_kl_index_20261108.py) -> (model key, mode of its self-index in Figure fig:self_crossmodel); the mode
# follows the rendering of the KL run (identity.json tmpl_kwargs): non-thinking where the template has a non-thinking mode
MAIN = {'qwen38_nt': ('qwen38_27b', 'non_thinking'), 'flashnext_nt': ('qwen38_flashnext', 'non_thinking'),
        'qwen36_27b': ('qwen36_27b', 'non_thinking'), 'qwen36_35b_a3b': ('qwen36_35b', 'non_thinking'),
        'qwen35_27b': ('qwen35_27b', 'non_thinking'), 'glm46v_flash': ('glm46v_flash', 'non_thinking'),
        'glm47_flash': ('glm47_flash', 'non_thinking'), 'glm53_flash': ('glm53_flash', 'thinking'),
        'r1_distill_llama70b': ('r1_llama70b', 'thinking'), 'gpt_oss_20b_en': ('gpt_oss_20b', 'thinking'),
        'gpt_oss_120b_en': ('gpt_oss_120b', 'thinking'), 'gemma4_31b_en': ('gemma4_31b', 'non_thinking'),
        'seed_oss_36b': ('seed_oss_36b', 'non_thinking'), 'exaone4_32b_en': ('exaone4_32b', 'non_thinking'),
        'granite42_30b_en': ('granite42_30b', 'non_thinking'), 'granite41_30b_en': ('granite41_30b', 'non_thinking'),
        'olmo2_13b_en': ('olmo2_13b', 'non_thinking'), 'olmo2_7b_en': ('olmo2_7b', 'non_thinking'),
        'ernie45_21b_a3b_thinking': ('ernie45_21b', 'thinking'), 'mimo7b_rl_0530': ('mimo7b_rl', 'non_thinking'),
        'mimo_vl7b': ('mimo_vl7b', 'thinking'), 'mimo_v26_nt': ('mimo_distill9b', 'non_thinking'),
        'moonlight16b_a3b_instruct': ('moonlight16b', 'non_thinking'), 'apertus8b_en': ('apertus8b', 'non_thinking'),
        'phi4_en': ('phi4', 'non_thinking')}

# geometry summaries of fig_crossmodel.py, per (model, mode); paths below runs/<group>/ of the wave-3 archive unless marked A8
GEO = {('qwen38_27b', 'non_thinking'): 'A8:qwen38_27b/non_thinking/geo_attr', ('qwen36_27b', 'non_thinking'): 'A8:qwen36_27b/non_thinking/geo_attr',
       ('qwen36_35b', 'non_thinking'): 'A8:qwen36_35b/non_thinking/geo_attr', ('glm46v_flash', 'non_thinking'): 'A8:glm46v_flash/non_thinking/geo_attr',
       ('glm47_flash', 'non_thinking'): 'A8:glm47_flash/non_thinking/geo_attr', ('gemma4_31b', 'non_thinking'): 'A8:gemma4_31b/non_thinking/geo_attr',
       ('r1_llama70b', 'thinking'): 'A8:r1_llama70b/thinking/geo',
       ('seed_oss_36b', 'non_thinking'): 'suite/seed_oss_36b/summary.json', ('exaone4_32b', 'non_thinking'): 'suite/exaone4_32b_en/summary.json',
       ('gpt_oss_20b', 'thinking'): 'suite_think/gpt_oss_20b_mxfp4_en/summary.json',
       ('gpt_oss_120b', 'thinking'): 'suite_think/gpt_oss_120b_en/summary.json',
       ('qwen35_27b', 'non_thinking'): 'suite/qwen35_27b/summary.json', ('granite41_30b', 'non_thinking'): 'suite/granite41_30b_en/summary.json',
       ('granite42_30b', 'non_thinking'): 'suite/granite42_30b_en/summary.json', ('olmo2_7b', 'non_thinking'): 'suite/olmo2_7b_en/summary.json',
       ('olmo2_13b', 'non_thinking'): 'suite/olmo2_13b_en/summary.json',
       # NEWDEV geo of newdev.py
       ('ernie45_21b', 'thinking'): 'suite/ernie45_21b_a3b_thinking/summary.json', ('mimo7b_rl', 'non_thinking'): 'suite/mimo7b_rl_0530/summary.json',
       ('mimo_vl7b', 'thinking'): 'suite/mimo_vl7b_rl_2508/summary.json',
       ('mimo_distill9b', 'non_thinking'): 'suite/mimo_v26_distill_qwen9b/summary.json',
       ('moonlight16b', 'non_thinking'): 'suite/moonlight16b_a3b_instruct/summary.json',
       ('apertus8b', 'non_thinking'): 'suite/apertus8b_instruct_2509_en/summary.json', ('phi4', 'non_thinking'): 'suite/phi4_en/summary.json'}
STREAMS = {('qwen38_flashnext', 'non_thinking'): 'flashnext_stream_20261020/flashnext_stream_geometry_non_20261020.json',
           ('glm53_flash', 'thinking'): 'suite_think/glm53_flash_stream_geometry.json'}

# natural order: KL setting -> (model key, natural-order geometry run below runs/<group>/)
NATURAL = {'exaone4_32b_en': ('exaone4_32b', 'suite_nat3/exaone4_32b_en'), 'glm46v_flash': ('glm46v_flash', 'suite_nat3/glm46v_flash'),
           'granite41_30b_en': ('granite41_30b', 'suite_nat3/granite41_30b_en'),
           'moonlight16b_a3b_instruct': ('moonlight16b', 'suite_nat3/moonlight16b_a3b_instruct'),
           'mimo7b_rl_0530': ('mimo7b_rl', 'suite_nat3/mimo7b_rl_0530'), 'phi4_en': ('phi4', 'suite_nat3/phi4_en'),
           'ernie45_21b_a3b_thinking': ('ernie45_21b', 'suite_nat3/ernie45_21b_a3b_thinking'),
           'olmo2_7b_en': ('olmo2_7b', 'suite_nat3/olmo2_7b_en'), 'seed_oss_36b': ('seed_oss_36b', 'suite_nat3/seed_oss_36b'),
           'gemma4_31b_en': ('gemma4_31b', 'suite_nat3/gemma4_31b_en')}
NAT_ONLY = {'dsv41_flash': ('dsv41_flash', 'suite_nat3/dsv41_flash_1017'),
            'ministral3_14b_instruct_en': ('ministral3_14b', 'suite_nat3/ministral3_14b_instruct_en'),
            'ministral3_14b_reasoning_en': ('ministral3_14b_r', 'suite_nat3/ministral3_14b_reasoning_en')}
FOUR_STREAM = {'qwen38_flashnext', 'glm53_flash', 'dsv41_flash'}

# label positions (data coordinates; y in ratio units) and alignment; models not listed get a label just left of the point
LABEL_A = {'glm46v_flash': (0.967, 57.5, 'right'), 'gpt_oss_20b': (0.801, 30.7, 'right'), 'exaone4_32b': (0.967, 14.4, 'right'),
           'gpt_oss_120b': (0.727, 8.56, 'left'), 'mimo_distill9b': (0.967, 4.56, 'right'), 'qwen38_27b': (0.967, 3.42, 'right'),
           'glm47_flash': (0.691, 2.25, 'right'), 'phi4': (0.875, 1.63, 'left'),
           'qwen38_flashnext': (1.08, 0.50, 'right', (0.9, 1.0)), 'glm53_flash': (0.43, 1.10, 'left'),
           'granite42_30b': (0.50, 0.70, 'left'), 'granite41_30b': (0.24, 0.46, 'left'),
           'gemma4_31b': (0.0, 3.2, 'left'), 'olmo2_13b': (0.07, 2.1, 'left', (0.12, 0.0)), 'olmo2_7b': (0.16, 1.5, 'left'),
           # labels of the models near x = 0 below the reference line, in a column right of them, in the order of their y
           'qwen35_27b': (0.24, 0.36, 'left'), 'moonlight16b': (0.24, 0.279, 'left'), 'mimo7b_rl': (0.24, 0.216, 'left'),
           'qwen36_27b': (0.24, 0.168, 'left'), 'apertus8b': (0.24, 0.130, 'left'), 'ernie45_21b': (0.24, 0.101, 'left'),
           'qwen36_35b': (0.24, 0.078, 'left'), 'mimo_vl7b': (0.24, 0.061, 'left'), 'seed_oss_36b': (0.24, 0.047, 'left'),
           'r1_llama70b': (0.24, 0.036, 'left')}
LABEL_B = {'glm46v_flash': (0.86, 3.9, 'right'), 'mimo7b_rl': (0.86, 5.2, 'right'), 'granite41_30b': (0.86, 6.9, 'right'),
           'dsv41_flash': (0.86, 9.2, 'right'), 'moonlight16b': (0.86, 12.2, 'right'), 'exaone4_32b': (0.86, 16.2, 'right'),
           'phi4': (0.60, 2.84, 'right'), 'ernie45_21b': (0.50, 2.0, 'right'), 'gemma4_31b': (0.40, 0.52, 'left'),
           'olmo2_7b': (0.19, 0.36, 'left'), 'seed_oss_36b': (0.08, 0.285, 'left'), 'ministral3_14b_r': (0.30, 0.118, 'left'),
           'ministral3_14b': (0.04, 0.085, 'left')}
REF = (1, 0.80, 0.93, 'top')     # panel, x, y, va of the text at the reference line y = 1
RHO_POS = ((1.07, 140, 'top'), (1.07, 0.036, 'bottom'))   # per panel: x, y (right-aligned) and va of the Spearman annotation
YLIM = (0.03, 150)
ARC = -0.15                      # curvature of the arrows of panel (b) whose two points share x (they bow to the right)
FIGSIZE = (7.4, 4.6)


def longpath(p):
    """extended-length form on Windows (run folders can pass 260 characters)"""
    s = os.path.abspath(str(p))
    if os.name == 'nt' and not s.startswith('\\\\?\\'):
        s = '\\\\?\\UNC\\' + s[2:] if s.startswith('\\\\') else '\\\\?\\' + s
    return s


def load(p):
    with open(longpath(p), encoding='utf-8') as fh:
        return json.load(fh)


def run_path(rel):
    """runs/<group>/<rel> of the wave-3 archive, whichever group holds it (exactly one)"""
    hits = sorted(glob.glob(str(W3 / 'runs' / '*' / rel)))
    if len(hits) != 1:
        raise SystemExit(f'{rel}: found {len(hits)} times under {W3 / "runs"}')
    return Path(hits[0])


def geo_path(spec):
    if spec.startswith('A8:'):
        return A8 / 'models' / spec[3:] / '_run_summary.json'
    return run_path(spec)


def ms_path(sub, setting):
    """modeswitch/<sub>/<setting>/summary.json, server3 mirror first"""
    for r in MS_ROOTS:
        p = (r / sub / setting if sub else r / setting) / 'summary.json'
        if p.exists():
            return p
    raise SystemExit(f'no KL summary for {sub}/{setting}')


def shares(content, onset):
    """content: layers singled out on message content (>= 80% of held-out bodies) and the number of layers; union: share of layers
    singled out at content or onset (a layer counted once) and whether the union covers every layer of the last third"""
    c, o = np.asarray(content, float) >= 0.8, np.asarray(onset, float) >= 0.8
    u = c | o
    L = len(u)
    return dict(n_content=int(c.sum()), n_layers=L, x=float(c.sum()) / L, x_union=float(u.sum()) / L,
                late_full=bool(u[int(np.ceil(2 * L / 3)):].all()))


def geo_main(key):
    if key in STREAMS:
        s = load(run_path(STREAMS[key]))
        return shares(s['content']['mean_of_streams']['share_A_lowest_by_layer'], s['onset']['mean_of_streams']['share_A_lowest_by_layer'])
    s = load(geo_path(GEO[key]))
    return shares(s['content']['share_A_lowest'], s['onset']['share_A_lowest'])


def geo_natural(rel, streams):
    s = load(run_path(rel + '/summary.json'))
    sfx = '_mean_of_streams' if streams else ''
    return shares(s['separate_content' + sfx]['share_A_lowest'], s['separate_onset' + sfx]['share_A_lowest'])


def smaller(ratios, kind):
    """the smaller of A-U/U-S and A-S/U-S (kind 'kl_content' or 'kl_body'), with its interval and which one it is"""
    au, as_ = ratios[f'A-U/U-S_{kind}'], ratios[f'A-S/U-S_{kind}']
    pick, name = (au, 'A-U/U-S') if au['ratio'] <= as_['ratio'] else (as_, 'A-S/U-S')
    return dict(y=pick['ratio'], lo=pick['ci95'][0], hi=pick['ci95'][1], which=name,
                au=[au['ratio']] + list(au['ci95']), as_=[as_['ratio']] + list(as_['ci95']))


def collect():
    P = {}
    for setting, (m, mode) in MAIN.items():
        P[m, 'main'] = dict(setting=setting, mode=mode, ratio_kind='content', **geo_main((m, mode)),
                            **smaller(load(ms_path('ctrl', setting))['ratios'], 'kl_content'))
    for setting, (m, rel) in NATURAL.items():
        P[m, 'natural'] = dict(setting=setting, ratio_kind='full', **geo_natural(rel, False),
                               **smaller(load(ms_path('', setting))['orders']['natural']['ratios'], 'kl_body'))
    for setting, (m, rel) in NAT_ONLY.items():
        P[m, 'natural'] = dict(setting=setting, ratio_kind='content', **geo_natural(rel, m in FOUR_STREAM),
                               **smaller(load(ms_path('nat3', setting))['ratios'], 'kl_content'))
    return P


def table_content(label):
    """{model name: (content layers, layers)} of the content column of a table in paper/source (read only); for the four-stream
    rows 'a (b)/L' the mean of the streams a; for Table tab:self_natural3 the single-pass and geometry-only rows"""
    out = {}
    for f in ('self_en_body.tex', 'self_en_appendix.tex'):
        t = (PAPER / f).read_text(encoding='utf-8')
        i = t.find(r'\label{' + label + '}')
        if i < 0:
            continue
        body = t[i:t.index(r'\end{tabular}', i)]
        for line in body.splitlines():
            cells = [c.strip() for c in line.rstrip('\\ ').split('&')]
            if len(cells) < 3:
                continue
            name = re.sub(r'\$\^\{a\}\$', '', cells[0]).strip()
            col = cells[1]
            if label == 'tab:self_natural3':
                if cells[1] not in ('single pass', 'geometry only'):
                    continue
                col = cells[2]
            mm = re.fullmatch(r'(\d+)(?: \((\d+)\))?/(\d+)', col)
            if mm:
                out[name] = (int(mm.group(1)), int(mm.group(3)))
    if not out:
        raise SystemExit(f'{label}: table not found in {PAPER}')
    return out


def check(P, extra):
    """content counts against the paper tables; content-or-onset shares and fill against the archived points"""
    n = 0
    tab = {'main': table_content('tab:self_crossmodel'), 'natural': {**table_content('tab:self_natural'), **table_content('tab:self_natural3')}}
    for (m, k), v in P.items():
        ref = tab[k].get(NAMES[m])
        if ref is None:
            raise SystemExit(f'{NAMES[m]} not in the {k} table')
        if ref != (v['n_content'], v['n_layers']):
            raise SystemExit(f'content count {NAMES[m]} ({k}): {v["n_content"]}/{v["n_layers"]} here, {ref[0]}/{ref[1]} in the paper table')
        n += 1
    print(f'content check: {n} counts equal to the paper tables')
    refs = [(W3 / 'stats' / 'self_crossmodel_points_v6.json', 'main'), (W3 / 'stats' / 'self_natural_points_v1.json', 'natural')]
    if extra:
        refs.append((Path(extra), 'main'))
    n = 0
    for f, kind in refs:
        pts = load(f)
        for (m, k), v in P.items():
            keys = [f'{m}/{v["mode"]}'] if k == 'main' else []
            if kind == 'natural':
                keys = [f'{m}/natural'] if k == 'natural' else [f'{m}/main']
            elif k != 'main':
                continue
            for key in keys:
                if key in pts:
                    rx, _, rlf = pts[key]
                    if abs(round(v['x_union'], 3) - rx) > 1e-9 or rlf != v['late_full']:
                        raise SystemExit(f'union mismatch {key}: {v["x_union"]:.3f}/{v["late_full"]} here, {rx}/{rlf} in {f.name}')
                    n += 1
    print(f'union/fill check: {n} values equal to the archived points')


def marker(ax, x, y, m, lf, drawn, mk='o', s=46):
    """filled = last third covered; open = not; a ring around a point (nearly) on top of earlier ones (as fig_crossmodel.py);
    four-stream models with a dark edge"""
    ly = math.log10(y)
    n_under = sum(abs(x - x0) < 0.02 and abs(ly - ly0) < 0.05 for x0, ly0 in drawn)
    four = m in FOUR_STREAM
    if n_under:
        ax.scatter(x, y, s=s + 80 * n_under, marker=mk, facecolor='none', edgecolor=STREAM_EDGE if four else COL[m], linewidth=1.5, zorder=5)
    elif lf:
        ax.scatter(x, y, s=s, marker=mk, color=COL[m], edgecolor=STREAM_EDGE if four else 'white', linewidth=1.1 if four else 0.8, zorder=4)
    else:
        ax.scatter(x, y, s=s, marker=mk, facecolor='white', edgecolor=COL[m], linewidth=1.6, zorder=4)
    drawn.append((x, ly))


def label(ax, m, x, y, pos):
    tx, ty, ha, *rp = pos.get(m, (x - 0.03, y, 'right'))
    far = abs(tx - x) > 0.05 or abs(math.log10(ty) - math.log10(y)) > 0.06
    # the leader leaves the label at its aligned end when the point lies on that side, else from the middle of the label;
    # a fourth entry in the label position gives that point on the label explicitly (fractions of its width and height)
    rel = (0.0, 0.5) if ha == 'left' and x <= tx + 0.02 else ((1.0, 0.5) if ha == 'right' and x >= tx - 0.02 else (0.5, 0.5))
    rel = rp[0] if rp else rel
    ax.annotate(NAMES[m], (x, y), xytext=(tx, ty), textcoords='data', ha=ha, va='center', fontsize=7.5, color='#333333', zorder=6,
                arrowprops=dict(arrowstyle='-', color='#9a9a9a', lw=0.6, shrinkA=1.5, shrinkB=3.5, relpos=rel) if far else None)


def errbar(ax, x, lo, hi, m, alpha=0.75):
    ax.plot([x, x], [lo, hi], color=COL[m], lw=1.1, alpha=alpha, solid_capstyle='butt', zorder=2)


def rho(P, kind):
    v = [P[k] for k in P if k[1] == kind]
    r = spearmanr([a['x'] for a in v], [a['y'] for a in v])
    return float(r.statistic), float(r.pvalue), len(v)


def draw(P, out_dir=None, with_labels=True):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True, gridspec_kw={'wspace': 0.08})
    for ax in axes:
        ax.set_yscale('log')
        ax.axhline(1, color='#888888', lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(*YLIM)
        ax.spines[['top', 'right']].set_visible(False)
    ax = axes[0]
    drawn = []
    for setting, (m, mode) in MAIN.items():
        v = P[m, 'main']
        errbar(ax, v['x'], v['lo'], v['hi'], m)
    for setting, (m, mode) in MAIN.items():
        v = P[m, 'main']
        marker(ax, v['x'], v['y'], m, v['late_full'], drawn)
        if with_labels:
            label(ax, m, v['x'], v['y'], LABEL_A)
    ax.set_title('(a) main probe', loc='left')
    ax = axes[1]
    drawn = []
    order = list(NATURAL) + list(NAT_ONLY)
    for setting in order:
        m = (NATURAL.get(setting) or NAT_ONLY.get(setting))[0]
        v = P[m, 'natural']
        errbar(ax, v['x'], v['lo'], v['hi'], m)
        if (m, 'main') in P:
            v0 = P[m, 'main']
            ax.scatter(v0['x'], v0['y'], s=30, facecolor=COL[m] if v0['late_full'] else 'white', edgecolor=COL[m], linewidth=1.2,
                       alpha=0.55, zorder=3)
            # an arrow along the same x would run over the points stacked there: it bows to the right
            ax.annotate('', xy=(v['x'], v['y']), xytext=(v0['x'], v0['y']), zorder=2, annotation_clip=False,
                        arrowprops=dict(arrowstyle='-|>', color=COL[m], lw=1.3, shrinkA=4, shrinkB=5.5, mutation_scale=10,
                                        connectionstyle=f'arc3,rad={ARC if abs(v["x"] - v0["x"]) < 0.02 else 0}'))
    for setting in order:
        m = (NATURAL.get(setting) or NAT_ONLY.get(setting))[0]
        v = P[m, 'natural']
        marker(ax, v['x'], v['y'], m, v['late_full'], drawn, mk='o' if setting in NATURAL else 's', s=46)
        if with_labels:
            label(ax, m, v['x'], v['y'], LABEL_B)
    hs = [plt.Line2D([], [], ls='', marker='o', ms=5, mfc='white', mec='#777777', alpha=0.6, label='main probe'),
          plt.Line2D([], [], ls='', marker='o', ms=7, mfc='#777777', mec='white', label='natural order'),
          plt.Line2D([], [], ls='', marker='s', ms=7, mfc='#777777', mec='white', label='natural order only'),
          plt.Line2D([], [], ls='', marker='o', ms=7, mfc='white', mec='#777777', mew=1.5, label='last third not covered'),
          plt.Line2D([], [], ls='', marker='o', ms=7, mfc='#777777', mec=STREAM_EDGE, mew=1.1, label='four streams: mean')]
    ax.legend(handles=hs, loc='upper left', fontsize=7.4, frameon=False, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title('(b) natural order', loc='left')
    for ax, kind, (tx, ty, va) in zip(axes, ('main', 'natural'), RHO_POS):
        r, p, n = rho(P, kind)
        ax.text(tx, ty, f'Spearman ρ = {r:.2f} ({n} models)', ha='right', va=va, fontsize=8, color='#333333')
    if with_labels:
        i, tx, ty, va = REF
        axes[i].text(tx, ty, 'as different as User from System', ha='center', va=va, fontsize=7.5, color='#666666', style='italic', zorder=6)
    axes[0].set_yticks([0.1, 1, 10, 100])
    axes[0].set_yticklabels(['0.1', '1', '10', '100'])
    axes[0].set_ylabel('own message set apart in next-token\nprediction (smaller KL ratio)')
    fig.supxlabel('self-index: share of layers singled out on message content', fontsize=10, y=-0.01)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, 'self_kl_scatter.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, 'self_kl_scatter.png'), dpi=170, bbox_inches='tight')
    return fig, axes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', help='points json of the 25-model release script (fig_crossmodel.py) to compare the union shares with')
    ap.add_argument('--out', default=str(HERE / 'drafts_20261110'))
    a = ap.parse_args()
    if 'paper' in Path(a.out).resolve().parts:
        raise SystemExit('drafts only: refusing to write into paper/')
    P = collect()
    check(P, a.check)
    draw(P, a.out)
    pts = {f'{m}/{k}': {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for (m, k), v in P.items()}
    for kind in ('main', 'natural'):
        r, p, n = rho(P, kind)
        pts[f'spearman/{kind}'] = dict(rho=round(r, 4), p=p, n=n)
        print(f'Spearman rho {kind}: {r:.4f} (p = {p:.2g}, n = {n})')
    with open(os.path.join(a.out, 'self_kl_scatter_points.json'), 'w', encoding='utf-8') as fh:
        json.dump(pts, fh, indent=1)
    for (m, k), v in P.items():
        print(f"{m + '/' + k:28s} x={v['n_content']}/{v['n_layers']} ({v['x']:.3f}) lf={int(v['late_full'])} y={v['y']:.3f} "
              f"[{v['lo']:.3f}, {v['hi']:.3f}] {v['which']} {v['ratio_kind']}")


if __name__ == '__main__':
    main()
