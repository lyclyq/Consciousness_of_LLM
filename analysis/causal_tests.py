"""Causal tests: the replacement of the self-index coordinates in four models and five readouts, from the per-item results of
the replacement runs (selfindex role-swap runs).

Part 1, the main runs. Readouts: Qwen3.8-27B without thinking (mean P("you") at the response onset); Qwen3.8-27B, gpt-oss-20b and
gpt-oss-120b with thinking (share of finished items answered "you"). For every condition: n, the statistic with a 95% interval
(Wilson for shares, bootstrap over items for mean probabilities), and the item-paired difference from the unedited condition of
the same recipient with a 95% bootstrap interval (pairs: body x option order). Runs: every layer (attribution, content,
arithmetic, 64 dimensions), groups of positions (body, end of the message, question) and layer windows of 8 layers moving by 4
(gpt-oss-20b: its runs on the thirds of the depth are the windows L0-L7, L8-L15, L16-L23).
Outputs: stats/v4_mechanism.json, stats/v4_mechanism_tables.md; figure self_mechanism_where.pdf/.png (groups of positions; layer
windows; missing runs are left out and listed).

Part 1b, the fifth readout: MiMo-V2.6-Distill-Qwen-9B in the default call of its template (share of finished items answered
"you"), with the statistics of part 1. Its runs are in the next-token archive (evidence/p2_modeswitch_20261107,
results/<group>/thinking_swap/mimo_v26_<run>): the paper's procedure r05q_all (the fit with the question, every position; base_A,
base_U, swU8_A, swA8_U, rswU8_A, rswA8_U; no System conditions) for the attribution question, all_k64 for 64 dimensions,
content_all and arith_all (base_A, base_U, swU8_A, swA8_U) for the content and arithmetic questions. Groups of positions and layer
windows were not run for it. The first four readouts are those of part 1.
Outputs: stats/v5_mechanism.json, stats/v5_mechanism_tables.md; figure self_mechanism.pdf/.png (all-layer replacement, the five
readouts).

Part 2, the extensions:
  - Gemma-4-31B without thinking, in natural order (its claim to its own message appears only when the message answers a user
    request; appendix only): every layer, groups of positions, thirds, content and arithmetic questions, and its layer windows
    (8 layers, step 4, every position; 14 windows);
  - the single layers L39 (Qwen) and L30 (Gemma);
  - 64 instead of 8 dimensions at every layer, attribution question, every readout (*_all_k64; their unedited conditions are
    copies of the 8-dimension runs, identical prompts);
  - the secret game (v6 items: v5 plus one System sentence): which role the model takes after "I'll be the setter, number ready,
    the guesser has m tries" was said by itself or by the user, and the content control (the number of tries), 8 dimensions.
Attribution statistics as in part 1 (same functions). Game: share of finished items in which the model acts as the setter (its
reply reads the setter's message as its own), item-paired change on items finished in both conditions; control: share answered
correctly. Outputs: stats/v4_mech_ext.json, stats/v4_mech_ext_tables.md.

usage: causal_tests.py --results RES --out OUT [--item-cache DIR]
"""
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (MS, W3, FIGURES, BASE, Roots, arguments, boot, items, result_path, run_glob, run_path,  # noqa: E402
                    summarize, wilson)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'pdf.fonttype': 42, 'axes.titlesize': 9, 'axes.titleweight': 'bold',
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False})
C_SWAP, C_INST, C_RAND, C_INK, C_GRID, C_SHADE = '#2a78d6', '#e07b39', '#8b929b', '#2b2f36', '#e3e6ea', '#f3f4f6'
# readout colours: model hue as in the cross-model figures (Qwen3.8-27B blue, gpt-oss-20b wine, gpt-oss-120b pink); Qwen without
# thinking a lighter blue; MiMo-V2.6-Distill-Qwen-9B orange
C_RO = {'qwen_sp': '#8dbbef', 'qwen_th': '#2a78d6', 'oss_th': '#a61e4d', 'oss120_th': '#e64980', 'mimo_th': '#f08c00'}
READOUTS = [('qwen_sp', 'Qwen3.8-27B, no thinking', 'P("you")'), ('qwen_th', 'Qwen3.8-27B, thinking', 'share answering "you"'),
            ('oss_th', 'gpt-oss-20b, thinking', 'share answering "you"'), ('oss120_th', 'gpt-oss-120b, thinking', 'share answering "you"')]
ROOT = {'qwen_sp': 'singlepass_swap/qwen38_27b_{}', 'qwen_th': 'thinking_swap/qwen38_27b_{}',
        'oss_th': 'thinking_swap/gpt_oss_20b_{}', 'oss120_th': 'thinking_swap/gpt_oss_120b_{}'}
N_LAYERS = {'qwen_sp': 64, 'qwen_th': 64, 'oss_th': 24, 'oss120_th': 36}
# layer windows: run suffix -> (first, last layer); gpt-oss-20b reuses its third runs (same conditions, settings and fit)
WINDOWS = {ro: {f'window_L{a:02d}_{a + 7:02d}': (a, a + 7) for a in range(0, 57, 4)} for ro in ('qwen_sp', 'qwen_th')}
WINDOWS['oss_th'] = {'third_early': (0, 7), 'window_L04_11': (4, 11), 'third_middle': (8, 15), 'window_L12_19': (12, 19),
                     'third_late': (16, 23)}
WINDOWS['oss120_th'] = {f'window_L{a:02d}_{a + 7:02d}': (a, a + 7) for a in range(0, 29, 4)}
WHERE = ('qwen_sp', 'qwen_th', 'oss_th', 'oss120_th')  # readouts with groups of positions and layer windows
CONDS = [('own', 'base_A', 'unedited'), ('own', 'swU8_A', '← User'), ('own', 'swS8_A', '← System'), ('own', 'rswU8_A', '← random'),
         ('user', 'base_U', 'unedited'), ('user', 'swA8_U', '← own'), ('user', 'rswA8_U', '← random'),
         ('system', 'base_S', 'unedited'), ('system', 'swA8_S', '← own')]
GROUP = {'own': "model's own message", 'user': 'User message', 'system': 'System message'}
# part 1b: the fifth readout, and the names of its runs for the run keys of the others
READOUTS5 = READOUTS + [('mimo_th', 'MiMo-V2.6-Distill-9B, default call', 'share answering "you"')]
MIMO_ROOT = 'thinking_swap/mimo_v26_{}'  # below results/<group>/ of the next-token archive
MIMO_RUN = {'all': 'r05q_all', 'all_content': 'content_all', 'all_arith': 'arith_all', 'all_k64': 'all_k64'}
RUNS = {'all': 'all', 'all_content': 'all_content', 'all_arith': 'all_arith', 'all_k64': 'all_k64',
        **{f'region_{x}': f'region_{x}' for x in ('body', 'end', 'question')}}


def tables_md(R, W, missing, readouts, title):
    """The markdown tables of v4_mechanism_tables.md / v5_mechanism_tables.md (for the text)."""
    L = [title, '', f'missing runs: {", ".join(missing) or "none"}', '']
    for k in RUNS:
        L.append(f'## {k}')
        L.append('| condition | ' + ' | '.join(t for _, t, _ in readouts) + ' |')
        L.append('|---|' + '---|' * len(readouts))
        conds = sorted({c for ro, _, _ in readouts for c in R[ro].get(k, {})})
        for c in conds:
            cells = []
            for ro, _, _ in readouts:
                s = R[ro].get(k, {}).get(c)
                if not s:
                    cells.append('')
                elif 'accuracy' in s:
                    cells.append(f"{s['accuracy']:.3f} ({s['finished']}/{s['n']})")
                else:
                    t = f"{s['stat']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"
                    if 'diff' in s:
                        t += f"; diff {s['diff']:+.3f} [{s['diff_ci'][0]:+.3f}, {s['diff_ci'][1]:+.3f}]"
                    cells.append(t)
            L.append(f'| {c} | ' + ' | '.join(cells) + ' |')
        L.append('')
    L += ['## layer windows (8 layers, every position; change from unedited)', '| readout | window | run | own <- User | User <- own | own <- User, random |',
          '|---|---|---|---|---|---|']
    for ro, WW in W.items():
        for w_, S_ in WW.items():
            f = lambda c: (f"{S_[c]['stat']:.3f}; {S_[c]['diff']:+.3f} [{S_[c]['diff_ci'][0]:+.3f}, {S_[c]['diff_ci'][1]:+.3f}]"  # noqa: E731
                           + (f" ({S_[c]['finished']}/{S_[c]['n']} fin.)" if S_[c].get('finished', S_[c]['n']) != S_[c]['n'] else '')
                           if c in S_ and 'diff' in S_[c] else '')
            L.append(f"| {ro} | {w_} | {S_['run']} | {f('swU8_A')} | {f('swA8_U')} | {f('rswU8_A')} |")
    return '\n'.join(L)


def mechanism_v5(R, W, missing, M, S, F):
    """Part 1b: R, W, missing of part 1; M = next-token archive in the results package; S = output stats folder, F = output
    figure folder. Adds the MiMo readout, writes v5_mechanism.json and v5_mechanism_tables.md, and draws Figure self_mechanism
    with the five readouts."""
    R, missing = {**R, 'mimo_th': {}}, list(missing)
    for k in RUNS:
        if k not in MIMO_RUN:  # groups of positions were not run for this readout
            continue
        d = result_path(M, MIMO_ROOT.format(MIMO_RUN[k]))
        recs = items(d) if d.exists() else []
        if not recs:
            missing.append(f'mimo_th/{k}')
            continue
        R['mimo_th'][k] = summarize(recs, k)
    json.dump({'results': R, 'windows': W, 'missing': missing}, open(S / 'v5_mechanism.json', 'w', encoding='utf-8'), indent=1)
    (S / 'v5_mechanism_tables.md').write_text(
        tables_md(R, W, missing, READOUTS5, '# v5 mechanism numbers (generated by make_v5_mechanism_20261110.py)'), encoding='utf-8')

    # figure: all-layer replacement; one panel per edited message, the five readouts side by side under each condition
    F.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.7), sharey=True, gridspec_kw={'wspace': 0.08, 'width_ratios': [4, 3, 2]})
    w = 0.165
    for ax, g in zip(axes, ('own', 'user', 'system')):
        conds = [(c, lab) for gg, c, lab in CONDS if gg == g]
        for j, (c, lab) in enumerate(conds):
            for i, (ro, _, _) in enumerate(READOUTS5):
                s = R[ro].get('all', {}).get(c)
                if s is None:
                    continue
                x = j + (i - (len(READOUTS5) - 1) / 2) * w
                ax.bar(x, s['stat'], width=w * 0.92, color=C_RO[ro], edgecolor='white', lw=0.4, zorder=3)
                ax.errorbar(x, s['stat'], yerr=[[s['stat'] - s['ci'][0]], [s['ci'][1] - s['stat']]], fmt='none', ecolor=C_INK, lw=0.7,
                            capsize=1.2, zorder=4)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([lab for _, lab in conds], fontsize=7.2)
        ax.set_xlim(-0.55, len(conds) - 0.45)
        ax.set_ylim(0, 1.0)
        ax.set_title(GROUP[g])
        ax.grid(axis='y', color=C_GRID, lw=0.6, zorder=0)
        ax.tick_params(axis='x', length=0)
    axes[0].set_ylabel('claimed as own\n(P("you") / share answering "you")')
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_RO[ro]) for ro, _, _ in READOUTS5]
    fig.legend(handles, [t for _, t, _ in READOUTS5], loc='lower center', ncol=3, frameon=False, fontsize=7.0, bbox_to_anchor=(0.5, -0.13))
    fig.savefig(F / 'self_mechanism.pdf', bbox_inches='tight')
    fig.savefig(F / 'self_mechanism.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('missing runs (five readouts):', ', '.join(missing) or 'none')


def mechanism(A, S, F):
    """Part 1: A = wave-3 archive in the results package, S = output stats folder, F = output figure folder. Returns R, W and the
    missing runs for part 1b."""
    R, missing = {}, []
    runs = RUNS
    W = {}
    for ro, _, _ in READOUTS:
        R[ro] = {}
        for k, suffix in runs.items():
            if k.startswith('region_') and ro not in WHERE:
                continue
            d = run_path(A, ROOT[ro].format(suffix))
            recs = items(d) if d.exists() else []
            if not recs:
                missing.append(f'{ro}/{k}')
                continue
            R[ro][k] = summarize(recs, k)
        if ro in WINDOWS:
            W[ro] = {}
            for suffix, (a, b) in WINDOWS[ro].items():
                d = run_path(A, ROOT[ro].format(suffix))
                recs = items(d) if d.exists() else []
                if not recs:
                    missing.append(f'{ro}/{suffix}')
                    continue
                W[ro][f'L{a}-L{b}'] = {'layers': [a, b], 'run': suffix, **summarize(recs, suffix)}
    S.mkdir(parents=True, exist_ok=True)
    json.dump({'results': R, 'windows': W, 'missing': missing}, open(S / 'v4_mechanism.json', 'w', encoding='utf-8'), indent=1)

    # markdown tables (for the text)
    (S / 'v4_mechanism_tables.md').write_text(
        tables_md(R, W, missing, READOUTS, '# v4 mechanism numbers (generated by make_v4_mechanism.py)'), encoding='utf-8')
    # the all-layer figure (self_mechanism.pdf) shows the five readouts and is drawn in part 1b (mechanism_v5)

    # figure 2: where. Top: one group of positions, every layer (bars; 'all' = every position and layer, the reference). Bottom:
    # every position, one window of 8 layers (lines at the window centre, 95% bands; dashed: every layer). Item-paired change,
    # own <- User (removal), User <- own (installation), own <- User in a random subspace.
    cols = [('region_body', 'body'), ('region_end', 'end'), ('region_question', 'question'), ('all', 'all')]
    fig, axes = plt.subplots(2, 4, figsize=(7.3, 4.9), sharey='row', gridspec_kw={'wspace': 0.1, 'hspace': 0.42})
    rd = {ro: t.replace(', ', '\n') for ro, t, _ in READOUTS}
    for ax, ro in zip(axes[0], WHERE):
        for j, (k, lab) in enumerate(cols):
            xx = j + (0.4 if k == 'all' else 0)
            for off, c, col in ((-0.2, 'swU8_A', C_SWAP), (0.2, 'swA8_U', C_INST)):
                s = R[ro].get(k, {}).get(c)
                if not s or 'diff' not in s:
                    continue
                ax.bar(xx + off, s['diff'], width=0.38, color=col, zorder=3)
                ax.errorbar(xx + off, s['diff'], yerr=[[s['diff'] - s['diff_ci'][0]], [s['diff_ci'][1] - s['diff']]], fmt='none',
                            ecolor=C_INK, lw=0.7, capsize=1.5, zorder=4)
        ax.axhline(0, color='#9aa1a9', lw=0.7, zorder=2)
        ax.axvline(len(cols) - 1 - 0.3, color=C_GRID, lw=0.8, ls=':', zorder=1)
        ax.set_xticks([j + (0.4 if k == 'all' else 0) for j, (k, _) in enumerate(cols)])
        ax.set_xticklabels([lab for _, lab in cols], fontsize=6.6, rotation=30, ha='right', rotation_mode='anchor')
        ax.set_xlim(-0.6, len(cols) - 0.2)
        ax.set_ylim(-1.0, 0.75)
        ax.set_title(rd[ro])
        ax.grid(axis='y', color=C_GRID, lw=0.6, zorder=0)
        ax.tick_params(axis='x', length=0)
    for ax, ro in zip(axes[1], WHERE):
        nl = N_LAYERS[ro]
        ax.axvspan(math.ceil(2 * nl / 3) - 0.5, nl - 0.5, color=C_SHADE, zorder=0, lw=0)
        ax.text(nl - 1, 0.97, 'last third', ha='right', va='top', fontsize=6.2, color='#6b7280')
        ax.axhline(0, color='#9aa1a9', lw=0.7, zorder=1)
        for c, col, mk in (('swU8_A', C_SWAP, 'o'), ('swA8_U', C_INST, 's'), ('rswU8_A', C_RAND, '^')):
            pts = sorted((S_['layers'], S_[c]) for S_ in W.get(ro, {}).values() if c in S_ and 'diff' in S_[c])
            if not pts:
                continue
            x = [(a + b) / 2 for (a, b), _ in pts]
            ax.fill_between(x, [s['diff_ci'][0] for _, s in pts], [s['diff_ci'][1] for _, s in pts], color=col, alpha=0.18, lw=0, zorder=2)
            ax.plot(x, [s['diff'] for _, s in pts], color=col, lw=1.5, marker=mk, ms=3.2, mec='white', mew=0.5, zorder=3)
            a = R[ro].get('all', {}).get(c)
            if a and 'diff' in a and c != 'rswU8_A':
                ax.axhline(a['diff'], color=col, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(-0.5, nl - 0.5)
        ax.set_ylim(-1.05, 1.0)
        ax.set_xticks(range(0, nl + 1, 16 if nl > 40 else (8 if nl > 30 else 4)))
        ax.grid(axis='y', color=C_GRID, lw=0.6, zorder=0)
    axes[0][0].set_ylabel('positions edited (all layers)\nchange in "you" vs unedited', fontsize=7.4)
    axes[1][0].set_ylabel('layers edited (all positions)\nchange in "you" vs unedited', fontsize=7.4)
    fig.text(0.5, 0.035, 'layer (centre of the 8-layer window)', ha='center', fontsize=7.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_SWAP), plt.Rectangle((0, 0), 1, 1, color=C_INST),
               plt.Line2D([], [], color=C_RAND, marker='^', lw=1.5, ms=3.2), plt.Line2D([], [], color='#6b7280', lw=0.8, ls=(0, (3, 2)))]
    fig.legend(handles, ["model's own message ← User coordinates", 'User message ← own coordinates', 'own ← User, random subspace',
                         'every layer'], loc='lower center', ncol=4, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.045))
    fig.savefig(F / 'self_mechanism_where.pdf', bbox_inches='tight')
    fig.savefig(F / 'self_mechanism_where.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('missing:', missing)
    return R, W, missing


# ---------------------------------------------------------------------------------------------------------------------------
# Part 2: extensions

READ = {'qwen_sp': ('Qwen3.8-27B, no thinking', 'singlepass_swap/qwen38_27b_{}', 64),
        'qwen_th': ('Qwen3.8-27B, thinking', 'thinking_swap/qwen38_27b_{}', 64),
        'oss_th': ('gpt-oss-20b, thinking', 'thinking_swap/gpt_oss_20b_{}', 24),
        'oss120_th': ('gpt-oss-120b, thinking', 'thinking_swap/gpt_oss_120b_{}', 36),
        'gemma_sp': ('Gemma-4-31B, no thinking, natural order', 'singlepass_swap/gemma4_31b_nat_{}', 60)}
GAMES = {'qwen_sp': 'games_swap/qwen38_27b_sp_s6_{}', 'qwen_th': 'games_swap/qwen38_27b_s6_{}',
         'oss_th': 'games_swap/gpt_oss_20b_s6_{}', 'oss120_th': 'games_swap/gpt_oss_120b_s6_{}',
         'gemma_sp': 'games_swap/gemma4_31b_nat_sp_s6_{}'}
SINGLE = {'qwen_sp': ('L39', 39), 'gemma_sp': ('L30', 30)}


def game_summary(d):
    recs = [r for r in items(d) if r.get('game') == 'secret']
    by = {}
    for r in recs:
        by.setdefault(r['id'], {})[r['cond']] = r
    out = {}
    for c in sorted({r['cond'] for r in recs}):
        rs = [r for r in recs if r['cond'] == c]
        fin = [r for r in rs if r['finished']]
        s = {'n': len(rs), 'finished': len(fin)}
        if 'correct' in rs[0]['answer']:
            s['accuracy'] = sum(r['answer']['correct'] for r in fin) / max(1, len(fin))
            out[c] = s
            continue
        k = sum(r['answer']['reads_as'] == 'assistant' for r in fin)
        s.update(stat=k / max(1, len(fin)), ci=list(wilson(k, len(fin))),
                 roles={a: sum(r['answer']['reads_as'] == a for r in fin) for a in ('assistant', 'user', None)})
        base = {'assistant': 'base_A', 'user': 'base_U'}[rs[0]['recipient']]
        if c != base:
            v = lambda r: float(r['answer']['reads_as'] == 'assistant')  # noqa: E731
            dd = [v(x[c]) - v(x[base]) for x in by.values() if c in x and base in x and x[c]['finished'] and x[base]['finished']]
            if dd:
                s.update(diff=sum(dd) / len(dd), diff_ci=list(boot(dd, seed=1)), n_pairs=len(dd))
        out[c] = s
    return out


def get(A, pattern, suffix, fn=summarize):
    d = run_path(A, pattern.format(suffix))
    if not d.exists() or os.path.exists(d / 'NOT_USED.txt'):
        return None
    recs = items(d) if fn is summarize else None
    return fn(recs, suffix) if fn is summarize else fn(d)


def fmt(s, key='diff'):
    if s is None:
        return ''
    if 'accuracy' in s:
        return f"acc {s['accuracy']:.2f} ({s['finished']}/{s['n']})"
    t = f"{s['stat']:.2f}"
    if 'finished' in s and s['finished'] != s['n']:
        t += f" ({s['finished']}/{s['n']} fin.)"
    if key in s:
        t += f"; {s[key]:+.2f} [{s[key + '_ci'][0]:+.2f}, {s[key + '_ci'][1]:+.2f}]"
    return t


def mechanism_ext(A, S):
    """Part 2: A = wave-3 archive in the results package, S = output stats folder."""
    R = {'gemma': {}, 'windows': {}, 'single': {}, 'k64': {}, 'games': {}}
    for k in ('all', 'all_content', 'all_arith', 'region_body', 'region_end', 'region_question', 'third_early', 'third_middle',
              'third_late', 'third_late_content', 'third_late_arith'):
        s = get(A, READ['gemma_sp'][1], k)
        if s:
            R['gemma'][k] = s
    for ro in ('gemma_sp',):
        R['windows'][ro] = {}
        for d in run_glob(A, READ[ro][1].format('window_L*')):
            if os.path.exists(os.path.join(d, 'NOT_USED.txt')):
                continue
            a, b = map(int, re.search(r'window_L(\d+)_(\d+)$', d).groups())
            R['windows'][ro][f'L{a}-L{b}'] = {'layers': [a, b], **summarize(items(d), 'window')}
    for ro, (lab, layer) in SINGLE.items():
        s = get(A, READ[ro][1], lab)
        if s:
            R['single'][ro] = {'layer': layer, **s}
    for ro in READ:
        s = get(A, READ[ro][1], 'all_k64')
        if s:
            R['k64'][ro] = s
        R['games'][ro] = {}
        for q in ('main', 'control'):
            s = get(A, GAMES[ro], q, fn=game_summary)
            if s:
                R['games'][ro][q] = s
    S.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(S / 'v4_mech_ext.json', 'w', encoding='utf-8'), indent=1)

    L = ['# v4 mechanism, extensions (generated by make_v4_mech_ext.py)', '']
    L += ['## Gemma-4-31B, natural order, single pass', '| run | condition | value; change [95%] |', '|---|---|---|']
    for k, S_ in R['gemma'].items():
        for c, s in S_.items():
            L.append(f'| {k} | {c} | {fmt(s)} |')
    L += ['', '## layer windows (8 layers, every position)', '| readout | window | own <- User | User <- own | own <- User, random |',
          '|---|---|---|---|---|']
    for ro, W in R['windows'].items():
        for w, S_ in W.items():
            L.append(f"| {ro} | {w} | {fmt(S_.get('swU8_A'))} | {fmt(S_.get('swA8_U'))} | {fmt(S_.get('rswU8_A'))} |")
    L += ['', '## single layer', '| readout | condition | value; change |', '|---|---|---|']
    for ro, S_ in R['single'].items():
        for c, s in S_.items():
            if isinstance(s, dict):
                L.append(f"| {ro} L{S_['layer']} | {c} | {fmt(s)} |")
    L += ['', '## 64 dimensions, every layer, attribution', '| readout | condition | value; change |', '|---|---|---|']
    for ro, S_ in R['k64'].items():
        for c, s in S_.items():
            L.append(f'| {ro} | {c} | {fmt(s)} |')
    L += ['', '## secret game v6 (share acting as the setter; control = accuracy)', '| readout | run | condition | value; change |',
          '|---|---|---|---|']
    for ro, G in R['games'].items():
        for q, S_ in G.items():
            for c, s in S_.items():
                L.append(f'| {ro} | {q} | {c} | {fmt(s)} |')
    (S / 'v4_mech_ext_tables.md').write_text('\n'.join(L) + '\n', encoding='utf-8')

    print(open(S / 'v4_mech_ext_tables.md', encoding='utf-8').read())


def main(argv=None):
    args = arguments(__doc__, item_cache=True).parse_args(argv)
    r = Roots(args)
    A = r.res(W3)
    S = r.out(W3 + '/stats')
    F = r.out(FIGURES)
    R, W, missing = mechanism(A, S, F)
    mechanism_v5(R, W, missing, r.res(MS), S, F)
    mechanism_ext(A, S)


if __name__ == '__main__':
    main()
