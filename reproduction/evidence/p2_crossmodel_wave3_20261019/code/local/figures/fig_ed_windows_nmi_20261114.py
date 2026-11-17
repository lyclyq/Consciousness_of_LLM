"""Extended Data Fig. 4 of the NMI version (11-14): where the index coordinates are read. Redrawn from self_mechanism_where.pdf
of the long version (make_v4_mechanism.py) in the NMI format (Arial, text 5-7 pt, panel letters 8 pt bold, 180 mm wide), the
same values. Read-only.
  a  every layer, one group of positions edited (the message text, the tokens that end the message, the following question), and
     all positions for reference: item-paired change in the rate of "you" from the unedited message, removal (the model's own
     message receives User coordinates) and installation (a User message receives the model's own coordinates), 95% intervals.
  b  every position, one window of eight layers moving by four, plotted at the centre of the window; dashed, the edit of every
     layer; shaded, the last third of the layers; open triangles, removal in a random subspace of the same size.
Readouts: Qwen3.8-27B without and with thinking, gpt-oss-20b, gpt-oss-120b (the readouts with these runs).
Source: stats/v4_mechanism.json (results.<readout>.{region_body, region_end, region_question, all}, windows.<readout>), the same
file as in the release copy (sha256 8e4e68d9...).
usage (from projects/consciousness): uv run --offline --no-project --with numpy --with matplotlib python -I fig_ed_windows_nmi_20261114.py [--copy-to-paper]"""
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42})
ROOT = Path('.').resolve()
A3 = ROOT / 'evidence/p2_crossmodel_wave3_20261019'
OUT = A3 / 'figures_nmi_20261114'
D = json.loads((A3 / 'stats/v4_mechanism.json').read_text(encoding='utf-8'))
R, W = D['results'], D['windows']
INK, MUTED, GRID, SHADE = '#2b2f36', '#6b7280', '#e4e4e0', '#f2f2ee'
C_REM, C_INS = '#2a78d6', '#e07b39'
RO = [('qwen_sp', 'Qwen3.8-27B, no thinking', 64), ('qwen_th', 'Qwen3.8-27B, thinking', 64),
      ('oss_th', 'gpt-oss-20b', 24), ('oss120_th', 'gpt-oss-120b', 36)]
COLS = [('region_body', 'text'), ('region_end', 'end'), ('region_question', 'question'), ('all', 'all')]   # explained in the caption


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=5.8, colors=MUTED, length=2)


def main():
    fig, axs = plt.subplots(2, 4, figsize=(7.08, 4.1), sharey='row', gridspec_kw={'wspace': 0.1, 'hspace': 0.55})
    for ax, (ro, name, _) in zip(axs[0], RO):
        for j, (k, _) in enumerate(COLS):
            xx = j + (0.35 if k == 'all' else 0)
            for off, c, col in ((-0.19, 'swU8_A', C_REM), (0.19, 'swA8_U', C_INS)):
                s = R[ro].get(k, {}).get(c)
                if not s or 'diff' not in s:
                    continue
                ax.bar(xx + off, s['diff'], width=0.36, color=col, zorder=3)
                ax.errorbar(xx + off, s['diff'], yerr=[[s['diff'] - s['diff_ci'][0]], [s['diff_ci'][1] - s['diff']]], fmt='none',
                            ecolor=INK, lw=0.6, capsize=1.3, zorder=4)
        ax.axhline(0, color=MUTED, lw=0.6, zorder=2)
        ax.axvline(len(COLS) - 1 - 0.32, color=GRID, lw=0.8, ls=':', zorder=1)
        ax.set_xticks([j + (0.35 if k == 'all' else 0) for j, (k, _) in enumerate(COLS)])
        ax.set_xticklabels([lab for _, lab in COLS], fontsize=5.8, color=INK, linespacing=0.95)
        ax.tick_params(axis='x', length=0, pad=2)
        ax.set_xlim(-0.6, len(COLS) - 0.25)
        ax.set_ylim(-1.0, 0.75)
        style(ax)
        ax.set_title(name, fontsize=6.4, color=INK, fontweight='bold', pad=4)
    for ax, (ro, name, nl) in zip(axs[1], RO):
        ax.axvspan(math.ceil(2 * nl / 3) - 0.5, nl - 0.5, color=SHADE, zorder=0, lw=0)
        ax.text(nl - 1, 0.95, 'last third', ha='right', va='top', fontsize=5.6, color=MUTED)
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        for c, col, mk, mfc in (('swU8_A', C_REM, 'o', C_REM), ('swA8_U', C_INS, 's', C_INS), ('rswU8_A', INK, '^', 'white')):
            pts = sorted((S['layers'], S[c]) for S in W.get(ro, {}).values() if c in S and 'diff' in S[c])
            if not pts:
                continue
            x = [(a + b) / 2 for (a, b), _ in pts]
            ax.fill_between(x, [s['diff_ci'][0] for _, s in pts], [s['diff_ci'][1] for _, s in pts], color=col, alpha=0.16, lw=0, zorder=2)
            ax.plot(x, [s['diff'] for _, s in pts], color=col, lw=1.1 if c != 'rswU8_A' else 0.8, marker=mk, ms=3.0, mfc=mfc, mec=col if c == 'rswU8_A' else 'white',
                    mew=0.6, zorder=3)
            a = R[ro].get('all', {}).get(c)
            if a and 'diff' in a and c != 'rswU8_A':
                ax.axhline(a['diff'], color=col, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(-0.5, nl - 0.5)
        ax.set_ylim(-1.05, 1.0)
        ax.set_xticks(range(0, nl + 1, 16 if nl > 40 else (8 if nl > 30 else 4)))
        ax.set_xlabel('layer (centre of the window)', fontsize=6, color=INK, labelpad=1.5)
        style(ax)
    axs[0][0].set_ylabel('change in the rate of “you”\n(every layer, one group of positions)', fontsize=6, color=INK)
    axs[1][0].set_ylabel('change in the rate of “you”\n(every position, eight layers)', fontsize=6, color=INK)
    for row, (letter, title) in enumerate((('a', 'Groups of positions'), ('b', 'Windows of layers'))):
        p = axs[row][0].get_position()
        y = p.y1 + (0.045 if row == 0 else 0.022)
        fig.text(0.004, y, letter, fontsize=8, fontweight='bold', color=INK, ha='left', va='bottom')
        fig.text(0.024, y, title, fontsize=6.8, fontweight='bold', color=INK, ha='left', va='bottom')
    h = [Patch(color=C_REM, label="removal: own message ← User coordinates"), Patch(color=C_INS, label='installation: User message ← own coordinates'),
         Line2D([], [], color=INK, lw=0.8, marker='^', ms=3.0, mfc='white', label='removal in a random subspace'),
         Line2D([], [], color=INK, lw=0.7, ls=(0, (3, 2)), label='every layer')]
    fig.legend(handles=h, loc='upper center', ncol=4, fontsize=5.9, frameon=False, bbox_to_anchor=(0.53, 1.0), columnspacing=1.3,
               handlelength=1.6, labelcolor=INK)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.85, bottom=0.1)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'ed_fig4_windows.pdf')
    fig.savefig(OUT / 'ed_fig4_windows.png', dpi=220)
    if '--copy-to-paper' in sys.argv:
        shutil.copyfile(OUT / 'ed_fig4_windows.pdf', ROOT / 'paper/nmi/ed/ed_fig4_windows.pdf')
    print('written ed_fig4_windows')


if __name__ == '__main__':
    main()
