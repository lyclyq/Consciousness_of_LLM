"""Extended Data Fig. 1 of the NMI version (11-14): is a role as such special, or only the model's own role?
The comparison with ordinary words is split as own - words = (own - external roles) + (external roles - words). One dot per model,
by level of the model (neither level, self-index only, self-reference); bars are group medians.
  a  main probe, 25 models;  b  natural order, the 13 models whose template keeps a System message.
  columns: next-token divergence, external roles relative to ordinary words (role-general part) and the model's own message
  relative to the external roles (part specific to the own role), both as log10 of the ratio; share kept, the same two parts as
  differences. Each panel carries the Spearman correlation with the level and its one-sided permutation P.
Values and tests: results/final_20261113/group_decomposition_20261113.json (group_decomposition_20261113.py); nothing is recomputed.
Nature figure guide: Arial, text 5-7 pt, panel letters 8 pt bold, 180 mm wide.
usage (from projects/consciousness): uv run --offline --no-project --with numpy --with matplotlib python -I fig_ed_decomposition_nmi_20261114.py [--copy-to-paper]"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42})
ROOT = Path('.').resolve()
WA = ROOT / 'evidence/p2_words_along_20261110'
D = json.loads((WA / 'results/final_20261113/group_decomposition_20261113.json').read_text(encoding='utf-8'))
OUT = WA / 'results/final_20261113/figs'
INK, MUTED, GRID, BAND = '#2b2f36', '#6b7280', '#e4e4e0', '#f4f4f1'
C_ROLE, C_OWN = '#6b7280', '#2a78d6'
GROUPS = ['neither', 'self-index only', 'self-reference']
GLAB = ['Neither', 'Index\nonly', 'Self-\nreference']
COLS = [('kl', 'log10 U||S / words (role-general)', 'External roles vs ordinary words', 'ratio of divergences (log10)', C_ROLE),
        ('kl', 'log10 self / U||S (self-specific)', 'Own message vs external roles', 'ratio of divergences (log10)', C_OWN),
        ('retention', 'U-S - words (role-general)', 'External roles vs ordinary words', 'difference in share kept', C_ROLE),
        ('retention', 'self - U-S (self-specific)', 'Own message vs external roles', 'difference in share kept', C_OWN)]


def main():
    W, H = 7.08, 4.25
    fig, axs = plt.subplots(2, 4, figsize=(W, H))
    rng = np.random.default_rng(20261114)
    for i, (order, rowlab) in enumerate((('main', 'Main probe'), ('natural', 'Natural order'))):
        for j, (kind, q, title, yl, col) in enumerate(COLS):
            ax = axs[i, j]
            pts = [(GROUPS.index(m['group']), m[q]) for m in D['per_model'][order] if m.get(q) is not None and m[q] == m[q] and m['group'] in GROUPS]   # NaN: no System message in this order
            t = D['tests'][order][kind][q]
            assert t['n'] == len(pts), (order, q, t['n'], len(pts))
            ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
            for g in range(3):
                ys = np.array([y for x, y in pts if x == g])
                if not len(ys):
                    continue
                xs = g + np.linspace(-0.22, 0.22, len(ys))[rng.permutation(len(ys))] if len(ys) > 1 else np.array([g])
                ax.plot(xs, ys, 'o', ms=3.0, mfc=col, mec='white', mew=0.4, alpha=0.85, zorder=3)
                ax.plot([g - 0.3, g + 0.3], [np.median(ys)] * 2, color=INK, lw=1.3, zorder=4, solid_capstyle='butt')
            ax.set_xlim(-0.6, 2.6)
            ax.set_xticks(range(3))
            ax.set_xticklabels(GLAB, fontsize=5.8, color=INK, linespacing=0.95)
            ax.tick_params(axis='y', labelsize=5.8, colors=MUTED, length=2)
            ax.tick_params(axis='x', length=0, pad=2)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            for s in ('left', 'bottom'):
                ax.spines[s].set_color(GRID)
            ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.set_ylabel(yl, fontsize=6, color=INK, labelpad=2)
            p = t['p_spearman']
            ptxt = f'{p:.4f}' if p < 0.001 else f'{p:.3f}' if p < 0.1 else f'{p:.2f}'
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.16 * (hi - lo))   # room for the statistic above the points
            ax.text(0.03, 0.985, f"ρ = {t['spearman_group']:+.2f}, P = {ptxt}", transform=ax.transAxes, fontsize=5.8, color=INK,
                    ha='left', va='top')
            ax.set_title(title, fontsize=6.0, color=INK, fontweight='bold', pad=3, loc='left')
    fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.085, wspace=0.42, hspace=0.62)
    for i, (letter, lab) in enumerate((('a', 'Main probe, 25 models'), ('b', 'Natural order, 13 models whose template keeps a System message'))):
        y = axs[i, 0].get_position().y1 + 0.052
        fig.text(0.004, y, letter, fontsize=8, fontweight='bold', color=INK, ha='left', va='bottom')
        fig.text(0.022, y, lab, fontsize=6.6, fontweight='bold', color=INK, ha='left', va='bottom')
    for x0, x1, lab in ((0, 1, 'Next-token divergence'), (2, 3, 'Share kept along the message')):
        p0, p1 = axs[0, x0].get_position(), axs[0, x1].get_position()
        fig.text((p0.x0 + p1.x1) / 2, 0.978, lab, fontsize=7, fontweight='bold', color=INK, ha='center', va='center')
        fig.add_artist(plt.Line2D([p0.x0, p1.x1], [0.958, 0.958], color=INK, lw=0.6))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'ed_fig1_decomposition.pdf')
    fig.savefig(OUT / 'ed_fig1_decomposition.png', dpi=220)
    if '--copy-to-paper' in sys.argv:
        (ROOT / 'paper/nmi/ed').mkdir(exist_ok=True)
        shutil.copyfile(OUT / 'ed_fig1_decomposition.pdf', ROOT / 'paper/nmi/ed/ed_fig1_decomposition.pdf')
    print(f'written ed_fig1_decomposition: {W * 25.4:.0f} x {H * 25.4:.0f} mm')


if __name__ == '__main__':
    main()
