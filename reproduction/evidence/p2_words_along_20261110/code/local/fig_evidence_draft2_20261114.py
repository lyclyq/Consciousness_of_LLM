"""Draft 2 of the evidence figure (11-14, after the user's comment on draft 1: the ordinary-word position on a shared axis was hard
to read; show two magnitudes per readout instead). One row per model, rows grouped as in the paper. Five aligned columns:
  a  geometry: share of layers with the self-index (filled: message content, open: response onset);
  b  next-token KL, self against external roles: min(A||U, A||S) / U||S (1 = equal), 95% interval of that ratio;
  c  next-token KL in units of one ordinary word (median of 32 word pairs = 1): self and external roles (U||S);
  d  retention, self minus external roles: share kept of self minus share kept of U-S (0 = equal), 95% interval;
  e  retention relative to ordinary words: self minus word median (95% interval) and U-S minus word median.
Orders without a System message have no external-role values (b, d empty; c, e self only).
Inputs as in fig_evidence_draft_20261114.py. usage: fig_evidence_draft2_20261114.py main|natural OUT_PREFIX"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_evidence_draft_20261114 import (AN, C_ROLE, C_SELF, EV, FT, GEO_MAIN, GEO_NAT, GLAB, GRID, INK, MUTED, BAND,  # noqa: E402
                                         ROW_ORDER_MAIN, SHORT)

REF = '#8d8d89'
DY = 0.17      # vertical offset within a row: self / content above, external roles / onset below
ALPHA = 0.85   # marker opacity, so that close points stay visible
WORD = '#b9b9b4'   # ordinary word pairs: range of the 32 pairs (grey bar), median = the dashed reference line


def kl_row(src):
    s = json.loads((EV / src / 'summary.json').read_text(encoding='utf-8'))['kinds']['kl_content']
    tu = s.get('to_US') or {}
    if 'A-U' not in tu:
        wa = s['word_median_to_AU_12']
        return {'self_ext': None, 'self_w': 1 / wa['ratio'], 'role_w': None, 'wlo': wa['min'] / wa['ratio'], 'whi': wa['max'] / wa['ratio']}
    pair = 'A-U' if tu['A-U']['ratio'] <= tu['A-S']['ratio'] else 'A-S'
    wu = s['word_median_to_US_12']
    w = wu['ratio']
    # word-pair range relative to the median word pair (the median is the reference line, 1)
    return {'self_ext': tu[pair]['ratio'], 'self_ext_ci': tu[pair]['ci95'], 'self_w': tu[pair]['ratio'] / w, 'role_w': 1 / w,
            'wlo': wu['min'] / w, 'whi': wu['max'] / w}


def main():
    order, out = sys.argv[1], sys.argv[2]
    rows = [r for r in FT['rows'][order] if r['group'] in GLAB]
    if order == 'main':
        rows.sort(key=lambda r: ROW_ORDER_MAIN.index(r['tag']))
    geo = GEO_MAIN if order == 'main' else GEO_NAT
    groups = [g for g in GLAB if any(r['group'] == g for r in rows)]
    y, ypos, spans = 0.0, {}, []
    for g in groups:
        y0 = y
        for r in rows:
            if r['group'] == g:
                ypos[r['tag']] = y
                y += 1
        spans.append((g, y0 - 0.5, y - 0.5))
        y += 0.9
    n = len(rows)
    H = 0.235 * n + 2.4
    fig, axs = plt.subplots(1, 5, figsize=(7.2, H), sharey=True, gridspec_kw={'width_ratios': [0.72, 1, 1, 1, 1], 'wspace': 0.09})
    for ax in axs:
        for i, (g, a, b) in enumerate(spans):
            ax.axhspan(a, b, color=BAND if i % 2 == 0 else 'white', zorder=0, lw=0)
        ax.set_ylim(y - 0.9 - 0.4, -0.6)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=6.6, colors=MUTED, length=2.5)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    axs[0].set_yticks([ypos[r['tag']] for r in rows])
    # natural order: the templates without a System message after a user turn take the self-index (and so the group) from the
    # main probe; mark them, their natural-order group is borrowed
    borrowed = {t for t in ROW_ORDER_MAIN if order == 'natural' and t not in geo}
    axs[0].set_yticklabels([SHORT.get(r['name'], r['name']) + (' †' if r['tag'] in borrowed else '') for r in rows], fontsize=6.9, color=INK)
    if borrowed:
        fig.text(0.01, 0.004, '† template keeps no System message after a user turn: self-index and group taken from the main probe.',
                 fontsize=5.8, color=MUTED, ha='left', va='bottom')
    for g, a, b in spans:
        axs[0].text(-1.72, (a + b) / 2, GLAB[g], rotation=90, va='center', ha='center', fontsize=7.2, color=MUTED,
                    transform=axs[0].get_yaxis_transform(), clip_on=False)

    def none(ax, yy, txt='no System message'):
        ax.text(0.5, yy, txt, fontsize=5.2, color=MUTED, ha='center', va='center', style='italic', transform=ax.get_yaxis_transform())

    for r in rows:
        yy = ypos[r['tag']]
        # a geometry
        ax = axs[0]
        if r['tag'] in geo:
            c, o, of = geo[r['tag']]
            # content slightly above, onset slightly below the row, so equal values do not hide each other
            ax.plot([c / of, o / of], [yy - DY, yy + DY], color=INK, lw=0.8, alpha=0.3, zorder=2)
            ax.plot(c / of, yy - DY, 'o', ms=4.3, mfc=INK, mec='none', alpha=ALPHA, zorder=3)
            ax.plot(o / of, yy + DY, 'o', ms=4.3, mfc='none', mec=INK, mew=0.9, alpha=ALPHA, zorder=4)
        else:
            none(ax, yy, 'main probe only')
        # b, c next-token KL
        k = kl_row(r['kl']['src'])
        if k['self_ext'] is None:
            none(axs[1], yy)
        else:
            axs[1].plot(k['self_ext_ci'], [yy, yy], color=INK, lw=0.9, zorder=3)
            axs[1].plot(k['self_ext'], yy, 'D', ms=4.0, mfc=INK, mec='white', mew=0.6, zorder=4)
        # c: self against ordinary words only (user 11-14); external roles against words go to Extended Data
        axs[2].plot(k['self_w'], yy, 'o', ms=4.6, mfc=C_SELF, mec='none', alpha=ALPHA, zorder=5)
        # d, e retention
        u = AN[order][r['tag']]
        q = u['word_share_quantiles']
        med = q[2]
        if u.get('advantage_vs_US'):
            a = u['advantage_vs_US']
            axs[3].plot(a['ci'], [yy, yy], color=INK, lw=0.9, zorder=3)
            axs[3].plot(a['value'], yy, 'D', ms=4.0, mfc=INK, mec='white', mew=0.6, zorder=4)
        else:
            none(axs[3], yy)
        # e: self against ordinary words only
        a = u['advantage_vs_word_median']
        axs[4].plot(a['ci'], [yy, yy], color=C_SELF, lw=0.9, alpha=ALPHA, zorder=4)
        axs[4].plot(a['value'], yy, 'o', ms=4.6, mfc=C_SELF, mec='none', alpha=ALPHA, zorder=5)

    axs[0].set_xlim(-0.08, 1.08)
    axs[0].set_xticks([0, 0.5, 1])
    axs[0].set_xticklabels(['0', '0.5', '1'])
    axs[0].set_xlabel('share of layers\nwith the self-index', fontsize=6.8, color=INK)
    for ax, lim, ticks in ((axs[1], (0.03, 150), [0.1, 1, 10, 100]), (axs[2], (0.05, 300), [0.1, 1, 10, 100])):
        ax.set_xscale('log')
        ax.set_xlim(*lim)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.axvline(1, color=REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
    axs[1].set_xlabel('KL ratio,\nself ÷ external', fontsize=6.8, color=INK)
    axs[2].set_xlabel('KL ÷ KL of the\nmedian word pair', fontsize=6.8, color=INK)
    axs[3].set_xlim(-0.62, 0.62)
    axs[3].set_xticks([-0.5, 0, 0.5])
    axs[3].set_xticklabels(['−0.5', '0', '+0.5'])
    axs[4].set_xlim(-0.5, 0.85)
    axs[4].set_xticks([-0.4, 0, 0.4, 0.8])
    axs[4].set_xticklabels(['−0.4', '0', '+0.4', '+0.8'])
    for ax in axs[3:]:
        ax.axvline(0, color=REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
    axs[3].set_xlabel('share kept,\nself − external', fontsize=6.8, color=INK)
    axs[4].set_xlabel('share kept − share\nof the median word pair', fontsize=6.8, color=INK)
    subt = ['', 'self vs external', 'self vs ordinary words', 'self vs external', 'self vs ordinary words']
    for ax, t, st in zip(axs, 'abcde', subt):
        ax.set_title(t, fontsize=8.2, color=INK, loc='left', fontweight='bold', pad=11)
        if st:
            ax.annotate(st, xy=(0.6, 1.0), xycoords='axes fraction', xytext=(0, 11), textcoords='offset points', fontsize=6.2,
                        color=INK, ha='center', va='bottom', annotation_clip=False)
    # name the reference lines
    for ax, x, lab in ((axs[1], 1, 'equal'), (axs[2], 1, 'ordinary words'), (axs[3], 0, 'equal'), (axs[4], 0, 'ordinary words')):
        ax.annotate(lab, xy=(x, 1.0), xycoords=('data', 'axes fraction'), xytext=(0, 1.5), textcoords='offset points',
                    fontsize=5.6, color=MUTED, ha='center', va='bottom', style='italic', annotation_clip=False)
    top = 1 - 1.27 / H
    fig.subplots_adjust(left=0.262, right=0.985, top=top, bottom=0.82 / H)
    # headers over the column pairs
    def header(a0, a1, text):
        p0, p1 = a0.get_position(), a1.get_position()
        yh = top + 0.30 / H
        fig.text((p0.x0 + p1.x1) / 2, yh + 0.07 / H, text, ha='center', va='bottom', fontsize=7.8, color=INK, fontweight='bold')
        fig.add_artist(Line2D([p0.x0 + 0.012, p1.x1], [yh, yh], color=INK, lw=0.7))
    header(axs[0], axs[0], 'Geometry')
    header(axs[1], axs[2], 'Next-token KL')
    header(axs[3], axs[4], 'Retention along the message')
    h = [Line2D([], [], marker='o', ls='', ms=4.4, mfc=INK, mec='white', label='a: layers with the self-index on message content'),
         Line2D([], [], marker='o', ls='', ms=4.4, mfc='none', mec=INK, mew=0.9, label='a: layers with the self-index at the response onset'),
         Line2D([], [], marker='D', ls='-', lw=0.9, ms=4.0, color=INK, mfc=INK, mec='white',
                label='b, d: self relative to external roles (95% interval)'),
         Line2D([], [], marker='o', ls='-', lw=0.9, ms=4.8, color=C_SELF, mfc=C_SELF, mec='white',
                label='c, e: self relative to ordinary words (e: 95% interval)')]
    fig.legend(handles=h, loc='upper center', ncol=2, fontsize=6.3, frameon=False, bbox_to_anchor=(0.52, 1.0), columnspacing=1.4,
               handletextpad=0.4, labelcolor=INK, labelspacing=0.35)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out + '.png', dpi=220, facecolor='white')
    fig.savefig(out + '.pdf', facecolor='white')
    print('written', Path(out).name, 'rows', n)


if __name__ == '__main__':
    main()
