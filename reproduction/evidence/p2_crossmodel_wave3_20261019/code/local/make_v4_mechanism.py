"""Numbers and figures of the causal tests in paper v4 (three models, four readouts), computed from the per-item results in the
wave-3 archive (crossmodel_role_swap.py runs). Readouts: Qwen3.8-27B without thinking (mean P("you") at the response onset),
Qwen3.8-27B, gpt-oss-20b and gpt-oss-120b with thinking (share of finished items answered "you"). For every condition: n, the
statistic with a 95% interval (Wilson for shares, bootstrap over items for mean probabilities), and the item-paired difference
from the unedited condition of the same recipient with a 95% bootstrap interval (pairs: body x option order).
10-27 (user: the layer-window maps replace the thirds; gpt-oss-120b added): runs every layer (attribution, content, arithmetic,
64 dimensions) for all four readouts; layer windows of 8 layers moving by 4 (gpt-oss-20b: its third runs are the windows L0-L7,
L8-L15, L16-L23). The 10-26 version (thirds) is make_v4_mechanism_1026.py.
10-29 (paper v5, user 10-28: groups of positions are required, gpt-oss-120b filled in): groups of positions and the eight layer
windows L0-L7 ... L28-L35 of gpt-oss-120b as well, so all four readouts have both; the localisation figure has four columns. The
10-27 version (three readouts) is make_v4_mechanism_1027.py.
Outputs (ARCHIVE/stats): v4_mechanism.json, v4_mechanism_tables.md; figures (OUT_DIR): self_mechanism.pdf/.png (all-layer
replacement) and self_mechanism_where.pdf/.png (groups of positions; layer windows; missing runs are left out and listed).
usage: make_v4_mechanism.py ARCHIVE OUT_DIR"""
import glob
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'pdf.fonttype': 42, 'axes.titlesize': 9, 'axes.titleweight': 'bold',
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False})
C_SWAP, C_INST, C_RAND, C_INK, C_GRID, C_SHADE = '#2a78d6', '#e07b39', '#8b929b', '#2b2f36', '#e3e6ea', '#f3f4f6'
# readout colours: model hue as in the cross-model figures (Qwen3.8-27B blue, gpt-oss-20b wine, gpt-oss-120b pink); Qwen without
# thinking a lighter blue
C_RO = {'qwen_sp': '#8dbbef', 'qwen_th': '#2a78d6', 'oss_th': '#a61e4d', 'oss120_th': '#e64980'}
READOUTS = [('qwen_sp', 'Qwen3.8-27B, no thinking', 'P("you")'), ('qwen_th', 'Qwen3.8-27B, thinking', 'share answering "you"'),
            ('oss_th', 'gpt-oss-20b, thinking', 'share answering "you"'), ('oss120_th', 'gpt-oss-120b, thinking', 'share answering "you"')]
ROOT = {'qwen_sp': 'runs/server3/singlepass_swap/qwen38_27b_{}', 'qwen_th': 'runs/server3/thinking_swap/qwen38_27b_{}',
        'oss_th': 'runs/pro/thinking_swap/gpt_oss_20b_{}', 'oss120_th': 'runs/pro/thinking_swap/gpt_oss_120b_{}'}
N_LAYERS = {'qwen_sp': 64, 'qwen_th': 64, 'oss_th': 24, 'oss120_th': 36}
# layer windows: run suffix -> (first, last layer); gpt-oss-20b reuses its third runs (same conditions, settings and fit)
WINDOWS = {ro: {f'window_L{a:02d}_{a + 7:02d}': (a, a + 7) for a in range(0, 57, 4)} for ro in ('qwen_sp', 'qwen_th')}
WINDOWS['oss_th'] = {'third_early': (0, 7), 'window_L04_11': (4, 11), 'third_middle': (8, 15), 'window_L12_19': (12, 19),
                     'third_late': (16, 23)}
WINDOWS['oss120_th'] = {f'window_L{a:02d}_{a + 7:02d}': (a, a + 7) for a in range(0, 29, 4)}
WHERE = ('qwen_sp', 'qwen_th', 'oss_th', 'oss120_th')  # readouts with groups of positions and layer windows
BASE = {'assistant': 'base_A', 'user': 'base_U', 'system': 'base_S'}
CONDS = [('own', 'base_A', 'unedited'), ('own', 'swU8_A', '← User'), ('own', 'swS8_A', '← System'), ('own', 'rswU8_A', '← random'),
         ('user', 'base_U', 'unedited'), ('user', 'swA8_U', '← own'), ('user', 'rswA8_U', '← random'),
         ('system', 'base_S', 'unedited'), ('system', 'swA8_S', '← own')]
GROUP = {'own': "model's own message", 'user': 'User message', 'system': 'System message'}


def items(d):
    """Per-item records of a run. Reading tens of thousands of item files is slow on Windows; if env P2_ITEM_CACHE names a
    directory, the records of each run (without string fields over 2,000 characters, i.e. the generated text, which no statistic
    here reads) are cached there, keyed by the run path and refreshed when the number or the newest time of its item files changes."""
    fs = sorted(glob.glob(os.path.join(d, 'items', '*.json')))
    cache = os.environ.get('P2_ITEM_CACHE')
    if not cache:
        return [json.load(open(f, encoding='utf-8')) for f in fs]
    sig = [len(fs), max((os.path.getmtime(f) for f in fs), default=0.0)]
    p = os.path.join(cache, hashlib.sha1(os.path.abspath(d).encode()).hexdigest() + '.json')
    if os.path.exists(p):
        c = json.load(open(p, encoding='utf-8'))
        if c['sig'] == sig:
            return c['recs']
    recs = [{k: v for k, v in json.load(open(f, encoding='utf-8')).items() if not (isinstance(v, str) and len(v) > 2000)} for f in fs]
    os.makedirs(cache, exist_ok=True)
    json.dump({'path': os.path.abspath(d), 'sig': sig, 'recs': recs}, open(p, 'w', encoding='utf-8'))
    return recs


def wilson(k, n):
    if n == 0:
        return (float('nan'),) * 2
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def boot(v, reps=4000, seed=0):
    rng = random.Random(seed)
    m = [sum(rng.choice(v) for _ in v) / len(v) for _ in range(reps)]
    m.sort()
    return m[int(0.025 * reps)], m[int(0.975 * reps) - 1]


def val(r):
    return r['p']['self'] if 'p' in r else float(r.get('on_answer') == 'self')


def summarize(recs, key):
    out = {}
    for c in sorted({r['cond'] for r in recs}):
        rs = [r for r in recs if r['cond'] == c]
        if 'p' in rs[0]:
            v = [r['p']['self'] for r in rs]
            m = sum(v) / len(v)
            lo, hi = boot(v)
            s = {'n': len(rs), 'stat': m, 'ci': [lo, hi], 'p_user': sum(r['p']['user'] for r in rs) / len(rs),
                 'p_system': sum(r['p']['system'] for r in rs) / len(rs)}
        else:
            fin = [r for r in rs if r.get('finished', True)]
            k = sum(r.get('on_answer') == 'self' for r in fin)
            s = {'n': len(rs), 'finished': len(fin), 'stat': k / max(1, len(fin)), 'ci': list(wilson(k, len(fin))),
                 'answers': {a: sum(r.get('on_answer') == a for r in fin) for a in ('self', 'user', 'system', None)}}
            if rs[0].get('content'):
                s = {'n': len(rs), 'finished': len(fin), 'accuracy': sum(r['content']['correct'] for r in fin) / max(1, len(fin))}
        out[c] = s
    for c, s in out.items():
        rs = [r for r in recs if r['cond'] == c]
        b = BASE.get(rs[0]['recipient'])
        if b is None or b == c or b not in out or 'accuracy' in s:
            continue
        bb = {(r['id'], tuple(r['order'])): r for r in recs if r['cond'] == b}
        d = [val(r) - val(bb[(r['id'], tuple(r['order']))]) for r in rs if (r['id'], tuple(r['order'])) in bb
             and r.get('finished', True) and bb[(r['id'], tuple(r['order']))].get('finished', True)]
        if d:
            s['diff'] = sum(d) / len(d)
            s['diff_ci'] = list(boot(d, seed=1))
            s['n_pairs'] = len(d)
    return out


def main():
    A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    R, missing = {}, []
    runs = {'all': 'all', 'all_content': 'all_content', 'all_arith': 'all_arith', 'all_k64': 'all_k64'}
    runs.update({f'region_{x}': f'region_{x}' for x in ('body', 'end', 'question')})
    W = {}
    for ro, _, _ in READOUTS:
        R[ro] = {}
        for k, suffix in runs.items():
            if k.startswith('region_') and ro not in WHERE:
                continue
            d = A / ROOT[ro].format(suffix)
            recs = items(d) if d.exists() else []
            if not recs:
                missing.append(f'{ro}/{k}')
                continue
            R[ro][k] = summarize(recs, k)
        if ro in WINDOWS:
            W[ro] = {}
            for suffix, (a, b) in WINDOWS[ro].items():
                d = A / ROOT[ro].format(suffix)
                recs = items(d) if d.exists() else []
                if not recs:
                    missing.append(f'{ro}/{suffix}')
                    continue
                W[ro][f'L{a}-L{b}'] = {'layers': [a, b], 'run': suffix, **summarize(recs, suffix)}
    (A / 'stats').mkdir(exist_ok=True)
    json.dump({'results': R, 'windows': W, 'missing': missing}, open(A / 'stats' / 'v4_mechanism.json', 'w', encoding='utf-8'), indent=1)

    # markdown tables (for the text)
    L = ['# v4 mechanism numbers (generated by make_v4_mechanism.py)', '', f'missing runs: {", ".join(missing) or "none"}', '']
    for k in runs:
        L.append(f'## {k}')
        L.append('| condition | ' + ' | '.join(t for _, t, _ in READOUTS) + ' |')
        L.append('|---|' + '---|' * len(READOUTS))
        conds = sorted({c for ro, _, _ in READOUTS for c in R[ro].get(k, {})})
        for c in conds:
            cells = []
            for ro, _, _ in READOUTS:
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
        for w_, S in WW.items():
            f = lambda c: (f"{S[c]['stat']:.3f}; {S[c]['diff']:+.3f} [{S[c]['diff_ci'][0]:+.3f}, {S[c]['diff_ci'][1]:+.3f}]"  # noqa: E731
                           + (f" ({S[c]['finished']}/{S[c]['n']} fin.)" if S[c].get('finished', S[c]['n']) != S[c]['n'] else '')
                           if c in S and 'diff' in S[c] else '')
            L.append(f"| {ro} | {w_} | {S['run']} | {f('swU8_A')} | {f('swA8_U')} | {f('rswU8_A')} |")
    (A / 'stats' / 'v4_mechanism_tables.md').write_text('\n'.join(L), encoding='utf-8')

    # figure 1: all-layer replacement; one panel per edited message, the four readouts side by side under each condition
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.7), sharey=True, gridspec_kw={'wspace': 0.08, 'width_ratios': [4, 3, 2]})
    w = 0.2
    for ax, g in zip(axes, ('own', 'user', 'system')):
        conds = [(c, lab) for gg, c, lab in CONDS if gg == g]
        for j, (c, lab) in enumerate(conds):
            for i, (ro, _, _) in enumerate(READOUTS):
                s = R[ro].get('all', {}).get(c)
                if s is None:
                    continue
                x = j + (i - 1.5) * w
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
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_RO[ro]) for ro, _, _ in READOUTS]
    fig.legend(handles, [t for _, t, _ in READOUTS], loc='lower center', ncol=4, frameon=False, fontsize=7.0, bbox_to_anchor=(0.5, -0.07))
    fig.savefig(OUT / 'self_mechanism.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'self_mechanism.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

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
        ax.set_xticklabels([lab for _, lab in cols], fontsize=6.6, rotation=30, ha='right', rotation_mode='anchor')  # 11-01: "end question" ran together
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
            pts = sorted((S['layers'], S[c]) for S in W.get(ro, {}).values() if c in S and 'diff' in S[c])
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
    fig.savefig(OUT / 'self_mechanism_where.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'self_mechanism_where.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('missing:', missing)


if __name__ == '__main__':
    main()
