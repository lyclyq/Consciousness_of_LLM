"""Combined evidence figure for the main text (11-14, user: the main-probe, natural-order and pooled views in one long figure).
Three blocks stacked, sharing the five columns of fig_evidence_draft2_20261114.py:
  A  main probe, 25 models (groups of the paper);
  B  natural order, 17 models (groups of the paper appendix; † = template keeps no System message after a user turn, so the
     self-index and the group come from the main probe);
  C  both orders weighted equally, the 14 models measured in both: models with the same group in both orders, then the models
     whose group changes with the order (values: mean of the two orders, KL ratios geometric mean; open = main probe only).
Columns: a geometry (content filled, onset open); b KL self ÷ external; c KL self ÷ median word pair; d retention self − external;
e retention self − median word pair. Intervals 95% (family resampling) in A and B; none in C (the orders were resampled apart).
11-14 figure-format pass (Nature figure guide; the version before it is in tmp/retired_20261114_fig2_before_nmi_format): Arial,
text 5-7 pt, the three blocks are the panels a, b, c (8 pt bold lowercase), the five columns carry no letters and are named by
their headers, rows closer together so that the figure is 180 mm wide and at most 170 mm high. No value, group or order changes.
usage: fig_evidence_combined_20261114.py OUT_PREFIX"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_evidence_draft_20261114 import AN, BAND, C_SELF, FT, GEO_MAIN, GEO_NAT, GRID, INK, MUTED, ROW_ORDER_MAIN, SHORT  # noqa: E402
from fig_evidence_draft2_20261114 import REF, kl_row  # noqa: E402

GL = {'self-reference': 'Self-ref.', 'selfref-one': 'Self-ref.,\none order', 'self-index only': 'Index only',
      'index-one': 'Index,\none order', 'neither': 'Neither'}
# block C (user 11-14): self-reference in both orders, self-reference in one order, a self-index in both orders (no
# self-reference), a self-index in one order, neither level in either order
GORDER = ['self-reference', 'selfref-one', 'self-index only', 'index-one', 'neither']
BANDED = {'self-reference', 'self-index only', 'neither'}
ROWH, GAP, BLOCKGAP = 0.082, 0.35, 2.2   # inches per row; gaps in rows (target: 170 mm high, Nature figure guide)
FS = 5.3
TOP, BOT = 0.74, 0.50                    # inches above and below the axes
plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42})


def order_rows(order):
    geo = GEO_MAIN if order == 'main' else GEO_NAT
    rows = [r for r in FT['rows'][order] if r['group'] in GL]
    out = []
    for r in rows:
        k = kl_row(r['kl']['src'])
        u = AN[order][r['tag']]
        g = geo.get(r['tag'])
        gs = g or GEO_MAIN[r['tag']]   # sort key of a borrowed row: its main-probe geometry, as its group
        out.append({'tag': r['tag'], 'label': SHORT.get(r['name'], r['name']) + (' †' if order == 'natural' and g is None else ''),
                    'group': r['group'], 'a': (g[0] / g[2], g[1] / g[2]) if g else None, 'geo_key': (gs[0] + gs[1]) / (2 * gs[2]),
                    'b': (k['self_ext'], k.get('self_ext_ci')) if k['self_ext'] is not None else None,
                    'c': (k['self_w'], PC_ORDER[order][r['tag']]['ci95']),
                    'd': (u['advantage_vs_US']['value'], u['advantage_vs_US']['ci']) if u.get('advantage_vs_US') else None,
                    'e': (u['advantage_vs_word_median']['value'], u['advantage_vs_word_median']['ci']), 'open': set()})
    out.sort(key=sort_key)
    return out


def sort_key(r):
    """user 11-14: within a group, descending; key = self-index coverage (mean of the content and onset shares of layers),
    ties broken by the KL ratio self ÷ external (column b)"""
    return (GORDER.index(r['group']), -round(r['geo_key'], 6), -(np.log10(r['b'][0]) if r['b'] else -9))


def pooled_rows():
    R = {o: {r['tag']: r for r in order_rows(o)} for o in ('main', 'natural')}
    out = []
    for t, rm in R['main'].items():
        if t not in R['natural']:
            continue
        rn = R['natural'][t]
        sr = [x['group'] == 'self-reference' for x in (rm, rn)]
        ix = [x['group'] in ('self-reference', 'self-index only') for x in (rm, rn)]
        # user 11-14: "self-reference in one order" needs the self-index in both orders; a model with the index in one order only
        # (Gemma-4-31B: self-reference and index in natural order only) counts as "index in one order"
        g = ('self-reference' if all(sr) else 'selfref-one' if any(sr) and all(ix) else 'self-index only' if all(ix)
             else 'index-one' if any(ix) else 'neither')
        lab = SHORT.get(FT_NAME[t], FT_NAME[t])
        if g in ('selfref-one', 'index-one'):
            lab += ' (natural only)' if (sr if g == 'selfref-one' else ix)[1] else ' (main only)'
        row = {'tag': t, 'label': lab, 'group': g, 'open': set()}
        ac = [x['a'] for x in (rm, rn) if x['a']]
        row['a'] = (float(np.mean([a[0] for a in ac])), float(np.mean([a[1] for a in ac])))
        row['geo_key'] = (rm['geo_key'] + rn['geo_key']) / 2
        for k in 'bcde':
            v = [x[k][0] for x in (rm, rn) if x[k]]
            if len(v) == 1:
                row['open'].add(k)
            val = (10 ** float(np.mean(np.log10(v)))) if k in 'bc' else float(np.mean(v))
            p = PCI[t][k]   # joint family bootstrap over both orders (pooled_ci_20261114.py); its point must equal val
            assert abs(p['value'] - val) <= 1e-6 * max(1.0, abs(val)), (t, k, p['value'], val)
            row[k] = (val, p['ci95'])
        out.append(row)
    out.sort(key=sort_key)
    return out


FT_NAME = {r['tag']: r['name'] for o in ('main', 'natural') for r in FT['rows'][o]}
_PJ = json.loads((Path(__file__).resolve().parents[2] / 'results/final_20261113/pooled_ci_20261114.json').read_text(encoding='utf-8'))
PCI, PC_ORDER = _PJ['models'], _PJ['per_order_c']


def main():
    out = sys.argv[1]
    blocks = [('a', 'Main probe: the message under test opens the conversation (25 models)', order_rows('main')),
              ('b', 'Natural order: the message under test answers a user request (17 models)', order_rows('natural')),
              ('c', 'Both orders, each weighted ½ (the 14 models measured in both)', pooled_rows())]
    # y layout
    y, ypos, spans, btitles = 0.0, {}, [], []
    for bi, (letter, title, rows) in enumerate(blocks):
        if bi:
            y += BLOCKGAP
        btitles.append((letter, title, y - 1.6))
        for g in GORDER:
            rs = [r for r in rows if r['group'] == g]
            if not rs:
                continue
            y0 = y
            for r in rs:
                ypos[(bi, r['tag'])] = y
                y += 1
            spans.append((bi, g, y0 - 0.5, y - 0.5))
            y += GAP
    y -= GAP
    H = ROWH * (y + 3.0) + TOP + BOT   # the axes span y + 3.0 rows (ylim below)
    fig, axs = plt.subplots(1, 5, figsize=(7.08, H), sharey=True, gridspec_kw={'width_ratios': [0.72, 1, 1, 1, 1], 'wspace': 0.09})
    for ax in axs:
        for i, (bi, g, a, b) in enumerate(spans):
            ax.axhspan(a, b, color=BAND if g in BANDED else 'white', zorder=0, lw=0)
        ax.set_ylim(y + 0.6, -2.4)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=FS, colors=MUTED, length=2)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    ticks, labels = [], []
    for bi, (_, _, rows) in enumerate(blocks):
        for r in rows:
            ticks.append(ypos[(bi, r['tag'])])
            labels.append(r['label'])
    axs[0].set_yticks(ticks)
    axs[0].set_yticklabels(labels, fontsize=FS, color=INK)
    from matplotlib.transforms import blended_transform_factory
    btr = blended_transform_factory(fig.transFigure, axs[0].transData)
    for bi, g, a, b in spans:
        if b - a < 1.5:   # one-row group: a rotated label would run into its neighbours; write it across, at the far left
            fig.text(0.012, (a + b) / 2, GL[g].replace('\n', ' '), va='center', ha='left', fontsize=5.0, color=MUTED,
                     transform=btr)
        else:
            axs[0].text(-2.35, (a + b) / 2, GL[g], rotation=90, va='center', ha='center', fontsize=5.0, color=MUTED,
                        transform=axs[0].get_yaxis_transform(), clip_on=False)
    # block titles: figure-level text above all axes (x in figure coordinates, y in data rows), on a white strip
    for letter, title, yt in btitles:
        fig.text(0.01, yt + 0.25, letter, fontsize=8, color=INK, fontweight='bold', ha='left', va='center', transform=btr, zorder=20,
                 bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.2})
        fig.text(0.032, yt + 0.25, title, fontsize=6.6, color=INK, fontweight='bold', ha='left', va='center',
                 transform=btr, zorder=20, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.2})
    DY = 0.21
    for bi, (_, _, rows) in enumerate(blocks):
        for r in rows:
            yy = ypos[(bi, r['tag'])]
            if r['a']:
                c, o = r['a']
                axs[0].plot([c, o], [yy - DY, yy + DY], color=INK, lw=0.7, alpha=0.3, zorder=2)
                axs[0].plot(c, yy - DY, 'o', ms=2.5, mfc=INK, mec=INK, mew=0.6, alpha=0.85, zorder=3)
                axs[0].plot(o, yy + DY, 'o', ms=2.5, mfc='none', mec=INK, mew=0.7, alpha=0.85, zorder=4)
            else:
                axs[0].text(0.5, yy, 'main probe only', fontsize=5.0, color=MUTED, ha='center', va='center', style='italic',
                            transform=axs[0].get_yaxis_transform())
            for ax, k, mk, col in ((axs[1], 'b', 'D', INK), (axs[2], 'c', 'o', C_SELF), (axs[3], 'd', 'D', INK), (axs[4], 'e', 'o', C_SELF)):
                if r[k] is None:
                    if k in 'bd':
                        ax.text(0.5, yy, 'no System message', fontsize=5.0, color=MUTED, ha='center', va='center', style='italic',
                                transform=ax.get_yaxis_transform())
                    continue
                v, ci = r[k]
                if ci:
                    ax.plot(ci, [yy, yy], color=col, lw=0.8, alpha=0.9, zorder=3)
                op = k in r['open']
                ax.plot(v, yy, mk, ms=3.0 if mk == 'o' else 2.6, mfc='white' if op else col, mec=col, mew=0.9, alpha=0.9, zorder=4)
    for ax, lim, tk in ((axs[1], (0.03, 150), [0.1, 1, 10, 100]), (axs[2], (0.05, 300), [0.1, 1, 10, 100])):
        ax.set_xscale('log')
        ax.set_xlim(*lim)
        ax.set_xticks(tk)
        ax.set_xticklabels([str(t) for t in tk])
        ax.minorticks_off()
        ax.axvline(1, color=REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
    axs[0].set_xlim(-0.08, 1.08)
    axs[0].set_xticks([0, 0.5, 1])
    axs[0].set_xticklabels(['0', '0.5', '1'])
    axs[3].set_xlim(-0.62, 0.62)
    axs[3].set_xticks([-0.5, 0, 0.5])
    axs[3].set_xticklabels(['−0.5', '0', '+0.5'])
    axs[4].set_xlim(-0.5, 0.85)
    axs[4].set_xticks([-0.4, 0, 0.4, 0.8])
    axs[4].set_xticklabels(['−0.4', '0', '+0.4', '+0.8'])
    for ax in axs[3:]:
        ax.axvline(0, color=REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for ax, t in zip(axs, ['share of layers\nwith the self-index', 'KL ratio,\nself ÷ external', 'KL ÷ KL of the\nmedian word pair',
                           'share kept,\nself − external', 'share kept − share\nof the median word pair']):
        ax.set_xlabel(t, fontsize=FS + 0.3, color=INK, labelpad=1.5)
    subt = ['', 'self vs external', 'self vs ordinary words', 'self vs external', 'self vs ordinary words']
    for ax, st in zip(axs, subt):   # the columns carry no letters (panel letters are the blocks a, b, c)
        if st:
            ax.annotate(st, xy=(0.5, 1.0), xycoords='axes fraction', xytext=(0, 9), textcoords='offset points', fontsize=FS + 0.3,
                        color=INK, ha='center', va='bottom', annotation_clip=False)
    for ax, x, lb in ((axs[1], 1, 'equal'), (axs[2], 1, 'ordinary words'), (axs[3], 0, 'equal'), (axs[4], 0, 'ordinary words')):
        ax.annotate(lb, xy=(x, 1.0), xycoords=('data', 'axes fraction'), xytext=(0, 1.5), textcoords='offset points',
                    fontsize=5.0, color=MUTED, ha='center', va='bottom', style='italic', annotation_clip=False)
    top = 1 - TOP / H
    fig.subplots_adjust(left=0.27, right=0.985, top=top, bottom=BOT / H)

    def header(a0, a1, text):
        p0, p1 = a0.get_position(), a1.get_position()
        yh = top + 0.25 / H
        fig.text((p0.x0 + p1.x1) / 2, yh + 0.04 / H, text, ha='center', va='bottom', fontsize=7, color=INK, fontweight='bold')
        fig.add_artist(Line2D([p0.x0 + 0.012, p1.x1], [yh, yh], color=INK, lw=0.6))
    header(axs[0], axs[0], 'Geometry')
    header(axs[1], axs[2], 'Next-token KL')
    header(axs[3], axs[4], 'Share kept along the message')
    h = [Line2D([], [], marker='o', ls='', ms=3.6, mfc=INK, mec=INK, label='geometry: layers on the message text'),
         Line2D([], [], marker='o', ls='', ms=3.6, mfc='none', mec=INK, mew=0.8, label='geometry: layers at the response onset'),
         Line2D([], [], marker='D', ls='-', lw=0.8, ms=3.4, color=INK, mfc=INK, label='self relative to external roles (95% interval)'),
         Line2D([], [], marker='o', ls='-', lw=0.8, ms=3.8, color=C_SELF, mfc=C_SELF, label='self relative to ordinary words (95% interval)'),
         Line2D([], [], marker='o', ls='', ms=3.6, mfc='white', mec=INK, mew=0.9, label='c, open: main probe only')]
    fig.legend(handles=h, loc='upper center', ncol=3, fontsize=FS, frameon=False, bbox_to_anchor=(0.6, 1.0), columnspacing=1.2,
               handletextpad=0.4, labelcolor=INK, labelspacing=0.25, borderpad=0.2)
    fig.text(0.01, 0.002, '† no System message after a user turn: self-index and group from the main probe.  c: mean of the two orders '
             '(KL: geometric mean). 95% intervals from one family resample drawn for both orders.', fontsize=5.0, color=MUTED,
             ha='left', va='bottom')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out + '.png', dpi=250, facecolor='white')
    fig.savefig(out + '.pdf', facecolor='white')
    print(f'written {Path(out).name}: {7.08:.2f} x {H:.2f} in ({7.08 * 25.4:.0f} x {H * 25.4:.0f} mm), rows {len(ticks)}')


if __name__ == '__main__':
    main()
