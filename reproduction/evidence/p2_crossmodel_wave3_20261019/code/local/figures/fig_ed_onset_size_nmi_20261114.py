"""Extended Data Figs. 2 and 3 of the NMI version (11-14). Read-only on the archives.
ED Fig. 2  a self-index at the response onset only. Share of held-out texts in which the Assistant rendering has the smallest
           distance ratio, by layer, on the message text and at the response onset, for Qwen3.8-27B (main probe, reference),
           DeepSeek-R1-Distill-Llama-70B (main probe), and Gemma-4-31B (main probe and natural order). Sources: the geometry
           summaries of the runs (share_A_lowest per layer): 8-model archive models/<m>/<mode>/geo*/_run_summary.json and
           wave-3 runs/server3/suite_nat3/gemma4_31b_en/summary.json (separate renderings).
ED Fig. 3  the levels against model size (total parameters, and parameters active per token) and against the published GPQA score. Parameters: Table tab:self_models of the long
           version (total parameters as stated by the developers). GPQA: stats/v6_benchmarks_20261104.csv, rows whose verbatim
           label names GPQA (GPQA or GPQA Diamond, as published; settings differ between developers). Levels: main-probe groups of
           evidence/p2_words_along_20261110/results/final_20261113/final_tables_20261113.json. Statistic: Spearman rho with the
           level (0, 1, 2), two-sided permutation test, 20,000 permutations, seed 20261114.
Nature figure guide: Arial, text 5-7 pt, panel letters 8 pt bold, 180 mm wide.
usage (from projects/consciousness): uv run --offline --no-project --with numpy --with matplotlib python -I fig_ed_onset_size_nmi_20261114.py [--copy-to-paper]"""
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42})
ROOT = Path('.').resolve()
EV = ROOT / 'evidence'
A8, A3 = EV / 'p2_crossmodel_8models_20261018/models', EV / 'p2_crossmodel_wave3_20261019'
OUT = A3 / 'figures_nmi_20261114'
FT = json.loads((EV / 'p2_words_along_20261110/results/final_20261113/final_tables_20261113.json').read_text(encoding='utf-8'))
INK, MUTED, GRID = '#2b2f36', '#6b7280', '#e4e4e0'
C_OWN, C_EXT = '#2a78d6', '#e07b39'
GROUPS = ['neither', 'self-index only', 'self-reference']
GLAB = ['Neither', 'Index\nonly', 'Self-\nreference']


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=5.8, colors=MUTED, length=2)


def letter(ax, a, title):
    kw = dict(xy=(0, 1), xycoords='axes fraction', textcoords='offset points', ha='left', va='baseline', annotation_clip=False)
    ax.annotate(a, xytext=(-2, 6), fontsize=8, fontweight='bold', color=INK, **kw)
    ax.annotate(title, xytext=(8, 6), fontsize=6.6, fontweight='bold', color=INK, **kw)


def ed_fig2():
    j = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))  # noqa: E731
    q = j(A8 / 'qwen38_27b/non_thinking/geo_attr/_run_summary.json')
    r = j(A8 / 'r1_llama70b/thinking/geo/_run_summary.json')
    g = j(A8 / 'gemma4_31b/non_thinking/geo_attr/_run_summary.json')
    gn = j(A3 / 'runs/server3/suite_nat3/gemma4_31b_en/summary.json')
    # title in two lines (model, then order and the layer counts): a one-line title ran into the next panel, and counts placed
    # inside the panel sat on the curves (user 11-14: overlapping text)
    panels = [('a', 'Qwen3.8-27B', 'main probe', q['content']['share_A_lowest'], q['onset']['share_A_lowest']),
              ('b', 'DeepSeek-R1-Distill-Llama-70B', 'main probe', r['content']['share_A_lowest'], r['onset']['share_A_lowest']),
              ('c', 'Gemma-4-31B', 'main probe', g['content']['share_A_lowest'], g['onset']['share_A_lowest']),
              ('d', 'Gemma-4-31B', 'natural order', gn['separate_content']['share_A_lowest'], gn['separate_onset']['share_A_lowest'])]
    fig, axs = plt.subplots(1, 4, figsize=(7.08, 2.0), sharey=True)
    counts = {}
    for ax, (a, model, order, c, o) in zip(axs, panels):
        c, o = np.array(c, float), np.array(o, float)
        x = np.arange(len(c))
        ax.axhline(0.8, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.plot(x, c, color=INK, lw=1.1, zorder=3)
        ax.plot(x, o, color=C_OWN, lw=1.1, zorder=4)
        ax.set_xlim(-1, len(c))
        ax.set_ylim(-0.04, 1.06)
        ax.set_xlabel('layer', fontsize=6, color=INK, labelpad=1.5)
        style(ax)
        title = f'{model}, {order}'
        counts[title] = {'layers': int(len(c)), 'text_80': int((c >= 0.8).sum()), 'onset_80': int((o >= 0.8).sum())}
        kw = dict(xy=(0, 1), xycoords='axes fraction', textcoords='offset points', ha='left', va='baseline', annotation_clip=False)
        ax.annotate(a, xytext=(-2, 15), fontsize=8, fontweight='bold', color=INK, **kw)
        ax.annotate(model, xytext=(8, 15), fontsize=6.4, fontweight='bold', color=INK, **kw)
        ax.annotate(f"{order}: text {counts[title]['text_80']}/{len(c)}, onset {counts[title]['onset_80']}/{len(c)}", xytext=(8, 6),
                    fontsize=5.8, color=INK, **kw)
    axs[0].set_ylabel('share of held-out texts with\nthe own message singled out', fontsize=6, color=INK)
    h = [Line2D([], [], color=INK, lw=1.1, label='on the message text'), Line2D([], [], color=C_OWN, lw=1.1, label='at the response onset'),
         Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3, 2)), label='criterion, 80% of the texts')]
    fig.legend(handles=h, loc='upper center', ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, labelcolor=INK)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.72, bottom=0.16, wspace=0.12)
    fig.savefig(OUT / 'ed_fig2_onset.pdf')
    fig.savefig(OUT / 'ed_fig2_onset.png', dpi=220)
    plt.close(fig)
    return counts


def rank(a):
    a = np.asarray(a, float)
    o = a.argsort()
    r = np.empty(len(a))
    r[o] = np.arange(len(a))
    for v in np.unique(a):
        m = a == v
        r[m] = r[m].mean()
    return r


def rho_p(x, y, rng):
    r0 = float(np.corrcoef(rank(x), rank(y))[0, 1])
    perm = np.array([np.corrcoef(rank(rng.permutation(x)), rank(y))[0, 1] for _ in range(20000)])
    return r0, float((np.abs(perm) >= abs(r0)).mean())


def ed_fig3():
    tex = (ROOT / 'paper/source/self_en_body.tex').read_text(encoding='utf-8')
    t = tex[tex.index('\\label{tab:self_models}'):]
    t = t[t.index('\\midrule') + 8:t.index('\\bottomrule')]
    params = {}
    for line in t.split('\\\\'):
        c = [x.strip() for x in line.replace('\n', ' ').split('&')]
        m = re.match(r'([\d.]+)B(?:/([\d.]+)B)?', c[3]) if len(c) >= 4 else None
        if m:   # "total" or "total/active per token" (mixture-of-experts models)
            params[re.sub(r'\$\^\{[a-z]\}\$', '', c[0]).strip()] = (float(m.group(1)), float(m.group(2) or m.group(1)))
    alias = {'MiMo-V2.6-Distill-Qwen-9B': 'MiMo-V2.6-Distill-9B', 'DeepSeek-R1-Distill-Llama-70B': 'R1-Distill-Llama-70B', 'MiMo-VL-7B-RL-2508': 'MiMo-VL-7B',
             'MiMo-7B-RL': 'MiMo-7B-RL-0530'}   # 11-15: the CSV names the 0530 release; it was dropped before for want of this alias
    # 11-15 (user): a model with two published GPQA scores takes the score of the mode in which it recognizes its own messages
    GPQA_SETTING = {'EXAONE-4.0.1-32B': '32B Reasoning Mode'}
    rows = [r for r in FT['rows']['main'] if r['group'] in GROUPS]
    gp = {}
    for r in csv.DictReader(open(A3 / 'stats/v6_benchmarks_20261104.csv', encoding='utf-8')):
        if r['benchmark_label_verbatim'].startswith('GPQA') and r['value']:
            if r['model'] in GPQA_SETTING and r['setting'] != GPQA_SETTING[r['model']]:
                continue
            assert r['model'] not in gp, ('two GPQA rows, choose one in GPQA_SETTING', r['model'])
            gp[r['model']] = float(r['value'])
    data = []
    for r in rows:
        nm = r['name']
        p = params.get(nm, params.get(alias.get(nm, nm)))
        assert p is not None, (nm, sorted(params))
        g = gp.get(nm, gp.get(alias.get(nm, nm)))
        data.append({'name': nm, 'level': GROUPS.index(r['group']), 'params_b': p[0], 'active_b': p[1], 'gpqa': g})
    rng = np.random.default_rng(20261114)
    fig, axs = plt.subplots(1, 3, figsize=(7.08, 2.05))
    stats = {}
    for ax, key, a, title, yl in ((axs[0], 'params_b', 'a', 'Total parameters', 'parameters (billions, log scale)'),
                                  (axs[1], 'active_b', 'b', 'Parameters active per token', 'parameters (billions, log scale)'),
                                  (axs[2], 'gpqa', 'c', 'Published GPQA score', 'score (%)')):
        d = [(x['level'], x[key]) for x in data if x[key] is not None]
        lv, v = np.array([a_ for a_, _ in d], float), np.array([b for _, b in d], float)
        r0, p = rho_p(lv, np.log10(v) if key != 'gpqa' else v, rng)
        stats[key] = {'n': len(d), 'spearman': r0, 'p_two_sided': p}
        for g in range(3):
            ys = v[lv == g]
            xs = g + (np.linspace(-0.22, 0.22, len(ys))[rng.permutation(len(ys))] if len(ys) > 1 else np.zeros(len(ys)))
            ax.plot(xs, ys, 'o', ms=3.0, mfc=INK, mec='white', mew=0.4, alpha=0.8, zorder=3)
            if len(ys):
                ax.plot([g - 0.3, g + 0.3], [np.median(ys)] * 2, color=C_OWN, lw=1.3, zorder=4, solid_capstyle='butt')
        if key != 'gpqa':
            ax.set_yscale('log')
            tk = [10, 30, 100, 300] if key == 'params_b' else [3, 10, 30]
            ax.set_yticks(tk)
            ax.set_yticklabels([str(x) for x in tk])
            ax.minorticks_off()
        ax.set_xlim(-0.6, 2.6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(GLAB, fontsize=5.8, color=INK, linespacing=0.95)
        ax.tick_params(axis='x', length=0, pad=2)
        ax.set_ylabel(yl, fontsize=6, color=INK, labelpad=2)
        style(ax)
        letter(ax, a, title)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1.6 if key != 'gpqa' else hi + 0.14 * (hi - lo))
        ax.text(0.03, 0.985, f"ρ = {r0:+.2f}, P = {p:.2f}, n = {len(d)}", transform=ax.transAxes, fontsize=5.8, color=INK, ha='left', va='top')
    fig.subplots_adjust(left=0.075, right=0.99, top=0.87, bottom=0.16, wspace=0.36)
    fig.savefig(OUT / 'ed_fig3_size.pdf')
    fig.savefig(OUT / 'ed_fig3_size.png', dpi=220)
    plt.close(fig)
    return {'models': data, 'tests': stats}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = ed_fig2()
    s = ed_fig3()
    (OUT / 'ed_fig2_fig3.json').write_text(json.dumps({'ed_fig2_layers_meeting_criterion': c, 'ed_fig3': s}, ensure_ascii=False, indent=1), encoding='utf-8')
    if '--copy-to-paper' in sys.argv:
        for f in ('ed_fig2_onset.pdf', 'ed_fig3_size.pdf'):
            shutil.copyfile(OUT / f, ROOT / 'paper/nmi/ed' / f)
    print(json.dumps(c), json.dumps(s['tests']))


if __name__ == '__main__':
    main()
