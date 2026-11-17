"""Fig. 3 of the NMI version (11-14): recognizing one's own messages, and its nesting in the self-index. Redrawn from the scatter
self_crossmodel.pdf of the long version in the row-per-model form of Figs. 2, 4 and 5 (Nature figure guide: Arial, text 5-7 pt,
panel letters 8 pt bold lowercase, 180 mm wide).
  a, b  rate of "you" to the attribution question, one row per model (25, main probe), grouped as in Fig. 2a: blue = the model's
        own message, orange = an external message (the larger of User and System), joined by a line. a without thinking
        (probability of "you", two option orders), b with thinking (share of finished items answered "you"). A model without
        that mode has an empty lane.
  c     the nesting: the 25 models by whether they carry a self-index and whether they recognize their own messages, one dot per
        model; P from the hypergeometric distribution of that table (one-sided), 0.004 as in the paper.
Values: Table tab:self_crossmodel of the long version (paper/source/self_en_body.tex, checked against the archives by
check_crossmodel_hand_numbers_20261103.py and check_newdev_tables_20261104.py), read from the tex. Groups and row order: the
main-probe rows of evidence/p2_words_along_20261110/results/final_20261113/final_tables_20261113.json (as Fig. 2a).
usage (from projects/consciousness): uv run --offline --no-project --with numpy --with matplotlib python -I fig_nesting_nmi_20261114.py [--copy-to-paper]"""
import json
import re
import shutil
import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42})
ROOT = Path('.').resolve()
TEX = ROOT / 'paper/source/self_en_body.tex'
FT = json.loads((ROOT / 'evidence/p2_words_along_20261110/results/final_20261113/final_tables_20261113.json').read_text(encoding='utf-8'))
OUT = ROOT / 'evidence/p2_crossmodel_wave3_20261019/figures_nmi_20261114'
INK, MUTED, GRID, BAND = '#2b2f36', '#6b7280', '#e4e4e0', '#f4f4f1'
C_OWN, C_EXT = '#2a78d6', '#e07b39'
GL = {'self-reference': 'Self-ref.', 'self-index only': 'Index only', 'neither': 'Neither'}
GORDER = ['self-reference', 'self-index only', 'neither']
ROWH, GAP = 0.105, 0.5


def table_rows():
    """rows of Table tab:self_crossmodel: name, own/other without thinking, own/User/System with thinking (None = no such mode)"""
    t = TEX.read_text(encoding='utf-8')
    t = t[t.index('\\label{tab:self_crossmodel}'):]
    t = t[t.index('\\midrule') + 8:t.index('\\bottomrule')]
    out = []
    for line in t.split('\\\\'):
        c = [x.strip() for x in line.replace('\n', ' ').split('&')]
        if len(c) != 11:
            continue
        f = lambda x: None if x == '--' else float(x)  # noqa: E731
        name = re.sub(r'\$\^\{a\}\$', '', c[0]).strip()
        out.append({'name': name, 'non': (f(c[4]), f(c[5])), 'think': (f(c[7]), None if f(c[8]) is None else max(f(c[8]), f(c[9])))})
    return out


def main():
    rows = table_rows()
    ft = [r for r in FT['rows']['main'] if r['group'] in GL]
    assert len(rows) == len(ft) == 25, (len(rows), len(ft))
    alias = {'MiMo-V2.6-Distill-9B': 'MiMo-V2.6-Distill-Qwen-9B', 'R1-Distill-Llama-70B': 'DeepSeek-R1-Distill-Llama-70B',
             'MiMo-VL-7B': 'MiMo-VL-7B-RL-2508'}   # short forms of the table
    grp = {g['name']: g['group'] for g in ft}
    for r in rows:
        r['group'] = grp.pop(alias.get(r['name'], r['name']))
    assert not grp, grp
    # y layout: groups as in Fig. 2a, table order within a group
    y, ypos, spans = 0.0, {}, []
    for g in GORDER:
        rs = [r for r in rows if r['group'] == g]
        y0 = y
        for r in rs:
            ypos[r['name']] = y
            y += 1
        spans.append((g, y0 - 0.5, y - 0.5))
        y += GAP
    y -= GAP
    W, TOP, BOT = 7.08, 0.50, 0.42
    H = ROWH * (y + 0.6) + TOP + BOT
    fig = plt.figure(figsize=(W, H))
    L, wcol, gapc = 1.42, 1.62, 0.14
    axs = [fig.add_axes([(L + k * (wcol + gapc)) / W, BOT / H, wcol / W, (H - TOP - BOT) / H]) for k in range(2)]
    for ax, key, letter, title, xl in zip(axs, ('non', 'think'), 'ab', ('Without thinking', 'With thinking'),
                                         ('probability of “you”', 'share of answers “you”')):
        for g, a, b in spans:
            ax.axhspan(a, b, color=BAND, zorder=0, lw=0)
        ax.set_ylim(y - 0.4, -0.6)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['0', '0.5', '1'])
        ax.tick_params(axis='x', labelsize=6, colors=MUTED, length=2, pad=1.5)
        ax.tick_params(axis='y', length=0)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel(xl, fontsize=6.3, color=INK, labelpad=2)
        kw = dict(xy=(0, 1), xycoords='axes fraction', textcoords='offset points', ha='left', va='baseline', annotation_clip=False)
        ax.annotate(letter, xytext=(0, 5), fontsize=8, fontweight='bold', color=INK, **kw)
        ax.annotate(title, xytext=(9, 5), fontsize=7, fontweight='bold', color=INK, **kw)
        for r in rows:
            yy = ypos[r['name']]
            own, ext = r[key]
            if own is None:
                continue
            ax.plot([ext, own], [yy, yy], color=INK, lw=0.7, alpha=0.35, zorder=2)
            ax.plot(ext, yy + 0.12, 'o', ms=3.0, mfc=C_EXT, mec='white', mew=0.4, alpha=0.9, zorder=3)
            ax.plot(own, yy - 0.12, 'o', ms=3.2, mfc=C_OWN, mec='white', mew=0.4, alpha=0.9, zorder=4)
    axs[0].set_yticks([ypos[r['name']] for r in rows])
    axs[0].set_yticklabels([r['name'] for r in rows], fontsize=6, color=INK)
    axs[1].set_yticks([])
    for g, a, b in spans:
        axs[0].text(-0.80, (a + b) / 2, GL[g], rotation=90, va='center', ha='center', fontsize=5.6, color=MUTED,
                    transform=axs[0].get_yaxis_transform(), clip_on=False)
    h = [Line2D([], [], marker='o', ls='', ms=3.4, mfc=C_OWN, mec='white', label="the model's own message"),
         Line2D([], [], marker='o', ls='', ms=3.2, mfc=C_EXT, mec='white', label='an external message (the larger of User and System)')]
    fig.legend(handles=h, loc='upper left', ncol=2, fontsize=6.3, frameon=False, bbox_to_anchor=(L / W - 0.012, 1.0), handletextpad=0.2,
               columnspacing=1.2, labelcolor=INK, borderpad=0.1)
    # c: the nesting, one dot per model
    x0 = L + 2 * wcol + gapc + 0.32
    cx = fig.add_axes([x0 / W, BOT / H, (W - x0 - 0.05) / W, (H - TOP - BOT) / H])
    cx.set_xlim(0, 2.45)
    cx.set_ylim(0, 2.75)
    cx.axis('off')
    cx.annotate('c', xy=(0, 1), xycoords='axes fraction', xytext=(0, 5), textcoords='offset points', fontsize=8, fontweight='bold',
                color=INK, ha='left', va='baseline', annotation_clip=False)
    cx.annotate('The two levels are nested', xy=(0, 1), xycoords='axes fraction', xytext=(9, 5), textcoords='offset points',
                fontsize=7, fontweight='bold', color=INK, ha='left', va='baseline', annotation_clip=False)
    n = {(i, s): sum((r['group'] != 'neither') == i and (r['group'] == 'self-reference') == s for r in rows) for i in (True, False) for s in (True, False)}
    assert n[(False, True)] == 0 and sum(n.values()) == 25
    k_ref, k_idx = n[(True, True)], n[(True, True)] + n[(True, False)]
    p = comb(k_idx, k_ref) / comb(25, k_ref)   # all k_ref recognizing models among the k_idx models with a self-index
    cw, ch, gx, gy = 0.92, 0.95, 0.45, 0.42
    cols = ((True, 'recognizes its\nown messages'), (False, 'does not'))
    rws = ((True, 'self-index'), (False, 'no self-index'))
    for j, (s, lab) in enumerate(cols):
        cx.text(gx + cw * (j + 0.5), gy + 2 * ch + 0.06, lab, ha='center', va='bottom', fontsize=6, color=INK, linespacing=1.05)
    for i, (ix, lab) in enumerate(rws):
        cx.text(gx - 0.07, gy + ch * (1.5 - i), lab, ha='center', va='center', fontsize=6, color=INK, rotation=90)
        for j, (s, _) in enumerate(cols):
            k = n[(ix, s)]
            empty = k == 0
            cx.add_patch(FancyBboxPatch((gx + cw * j + 0.03, gy + ch * (1 - i) + 0.03), cw - 0.06, ch - 0.06,
                                        boxstyle='round,pad=0,rounding_size=0.03', fc='white' if empty else BAND,
                                        ec=C_OWN if empty else GRID, lw=1.0 if empty else 0.8, ls='--' if empty else '-'))
            xc, yc = gx + cw * (j + 0.5), gy + ch * (1.5 - i)
            cx.text(xc, yc + 0.27, str(k), ha='center', va='center', fontsize=7, fontweight='bold', color=INK)
            per = 4
            for m in range(k):
                rr, cc = divmod(m, per)
                nrow = min(per, k - rr * per)
                cx.plot(xc + (cc - (nrow - 1) / 2) * 0.115, yc + 0.02 - rr * 0.13, 'o', ms=3.4,
                        mfc=C_OWN if s else INK, mec='white', mew=0.4, alpha=0.9 if s else 0.55)
            if empty:
                cx.text(xc, yc - 0.08, 'no model', ha='center', va='center', fontsize=5.6, color=MUTED, style='italic')
    cx.text(gx + cw, gy - 0.12, f'25 models, one dot each.  P = {p:.3f}', ha='center', va='top', fontsize=6.3, color=INK)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'fig3_nesting.pdf')
    fig.savefig(OUT / 'fig3_nesting.png', dpi=220)
    json.dump({'rows': rows, 'counts': {f'index={i},selfref={s}': k for (i, s), k in n.items()}, 'p_one_sided': p},
              open(OUT / 'fig3_nesting.json', 'w', encoding='utf-8'), indent=1)
    if '--copy-to-paper' in sys.argv:
        shutil.copyfile(OUT / 'fig3_nesting.pdf', ROOT / 'paper/nmi/figures/fig3_nesting.pdf')
    print(f'written fig3_nesting: {W * 25.4:.0f} x {H * 25.4:.0f} mm, P = {p:.4f}, counts {n}')


if __name__ == '__main__':
    main()
