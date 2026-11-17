"""Pooled view of the evidence figure (11-14, user: models measured in both orders, grouped by a group that is the same in both
orders, the two orders weighted equally). Same five columns as fig_evidence_draft2_20261114.py; each value is the mean of the two
orders (KL ratios: geometric mean, i.e. the mean of the logs; shares: arithmetic mean). Where an order has no value (natural order
without a System message: no self-vs-external value, and the geometry of those templates is taken from the main probe) the other
order alone is used, and the mark is drawn open. Models whose group changes with the order are drawn in a separate block and are
not counted in the group statistics. No intervals: the two orders are resampled separately, so a pooled interval is not defined
by the existing runs.
usage: fig_evidence_pooled_20261114.py OUT_PREFIX"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_evidence_draft_20261114 import AN, BAND, C_SELF, FT, GEO_MAIN, GEO_NAT, GRID, INK, MUTED, SHORT  # noqa: E402
from fig_evidence_draft2_20261114 import REF, kl_row  # noqa: E402

GL = {'self-reference': 'Self-reference', 'self-index only': 'Self-index only', 'neither': 'Neither level',
      'changes': 'Group changes\nwith the order'}
ORDER_G = ['self-reference', 'self-index only', 'changes', 'neither']


def rank(a):
    a = np.asarray(a, float)
    r = np.empty(len(a))
    r[a.argsort(kind='mergesort')] = np.arange(len(a))
    for v in np.unique(a):
        m = a == v
        r[m] = r[m].mean()
    return r


def main():
    out = sys.argv[1]
    rows = {o: {r['tag']: r for r in FT['rows'][o]} for o in ('main', 'natural')}
    both = [t for t in rows['main'] if t in rows['natural'] and rows['main'][t]['group'] in GL and rows['natural'][t]['group'] in GL]
    data = []
    for t in both:
        gm, gn = rows['main'][t]['group'], rows['natural'][t]['group']
        g = gm if gm == gn else 'changes'
        vals = {'a_c': [], 'a_o': [], 'b': [], 'c': [], 'd': [], 'e': []}
        for o, geo in (('main', GEO_MAIN), ('natural', GEO_NAT)):
            if t in geo:
                c, on, of = geo[t]
                vals['a_c'].append(c / of)
                vals['a_o'].append(on / of)
            k = kl_row(rows[o][t]['kl']['src'])
            if k['self_ext'] is not None:
                vals['b'].append(np.log10(k['self_ext']))
            vals['c'].append(np.log10(k['self_w']))
            u = AN[o][t]
            if u.get('advantage_vs_US'):
                vals['d'].append(u['advantage_vs_US']['value'])
            vals['e'].append(u['advantage_vs_word_median']['value'])
        pooled = {k: (float(np.mean(v)), len(v)) for k, v in vals.items()}
        for k in ('b', 'c'):
            pooled[k] = (10 ** pooled[k][0], pooled[k][1])
        data.append({'tag': t, 'name': rows['main'][t]['name'], 'group': g, 'main': gm, 'natural': gn, 'v': pooled,
                     'per_order': {k: v for k, v in vals.items()}})
    data.sort(key=lambda d: (ORDER_G.index(d['group']), -d['v']['d'][0] if d['v']['d'][1] else 0))
    # group statistics over the models with the same group in both orders
    stats = {}
    G = {'neither': 0, 'self-index only': 1, 'self-reference': 2}
    cons = [d for d in data if d['group'] in G]
    rng = np.random.default_rng(20261114)
    for k in ('b', 'c', 'd', 'e'):
        x = np.array([np.log10(d['v'][k][0]) if k in 'bc' else d['v'][k][0] for d in cons])
        g = np.array([G[d['group']] for d in cons])
        rho = float(np.corrcoef(rank(g), rank(x))[0, 1])
        p = float(np.mean([np.corrcoef(rank(rng.permutation(g)), rank(x))[0, 1] >= rho - 1e-12 for _ in range(20000)]))
        med = {n: float(np.median(x[g == i])) for n, i in G.items()}
        stats[k] = {'spearman_group': rho, 'p_one_sided': p, 'n': len(x),
                    'medians': {n: (10 ** m if k in 'bc' else m) for n, m in med.items()}}
    # figure
    y, ypos, spans = 0.0, {}, []
    for gname in ORDER_G:
        ms = [d for d in data if d['group'] == gname]
        if not ms:
            continue
        y0 = y
        for d in ms:
            ypos[d['tag']] = y
            y += 1
        spans.append((gname, y0 - 0.5, y - 0.5))
        y += 0.9
    n = len(data)
    H = 0.25 * n + 2.6
    fig, axs = plt.subplots(1, 5, figsize=(7.2, H), sharey=True, gridspec_kw={'width_ratios': [0.72, 1, 1, 1, 1], 'wspace': 0.09})
    for ax in axs:
        for i, (gname, a, b) in enumerate(spans):
            ax.axhspan(a, b, color=BAND if i % 2 == 0 else 'white', zorder=0, lw=0)
        ax.set_ylim(y - 0.9 - 0.4, -0.6)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=6.6, colors=MUTED, length=2.5)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    lab = []
    for d in data:
        s = SHORT.get(d['name'], d['name'])
        if d['group'] == 'changes':
            s += f" ({'none' if d['main'] == 'neither' else d['main']} → {'self-ref.' if d['natural'] == 'self-reference' else 'index'})"
        lab.append(s)
    axs[0].set_yticks([ypos[d['tag']] for d in data])
    axs[0].set_yticklabels(lab, fontsize=6.6, color=INK)
    for gname, a, b in spans:
        axs[0].text(-2.75, (a + b) / 2, GL[gname], rotation=90, va='center', ha='center', fontsize=6.9, color=MUTED,
                    transform=axs[0].get_yaxis_transform(), clip_on=False, linespacing=1.0)
    DY = 0.17
    for d in data:
        yy, v = ypos[d['tag']], d['v']
        if v['a_c'][1]:
            axs[0].plot([v['a_c'][0], v['a_o'][0]], [yy - DY, yy + DY], color=INK, lw=0.8, alpha=0.3, zorder=2)
            axs[0].plot(v['a_c'][0], yy - DY, 'o', ms=4.3, mfc=INK, mec=INK, mew=0.9, alpha=0.85, zorder=3)
            axs[0].plot(v['a_o'][0], yy + DY, 'o', ms=4.3, mfc='none', mec=INK, mew=0.9, alpha=0.85, zorder=4)
        for ax, k, mk, col in ((axs[1], 'b', 'D', INK), (axs[2], 'c', 'o', C_SELF), (axs[3], 'd', 'D', INK), (axs[4], 'e', 'o', C_SELF)):
            val, cnt = v[k]
            if not cnt:
                continue
            ax.plot(val, yy, mk, ms=4.6 if mk == 'o' else 4.0, mfc=col if cnt == 2 else 'white', mec=col, mew=1.0, alpha=0.9, zorder=4)
    for ax, lim, ticks in ((axs[1], (0.03, 150), [0.1, 1, 10, 100]), (axs[2], (0.05, 300), [0.1, 1, 10, 100])):
        ax.set_xscale('log')
        ax.set_xlim(*lim)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.axvline(1, color=REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
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
        ax.axvline(0, color=REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for ax, t in zip(axs, ['share of layers\nwith the self-index', 'KL ratio,\nself ÷ external', 'KL ÷ KL of the\nmedian word pair',
                           'share kept,\nself − external', 'share kept − share\nof the median word pair']):
        ax.set_xlabel(t, fontsize=6.8, color=INK)
    subt = ['', 'self vs external', 'self vs ordinary words', 'self vs external', 'self vs ordinary words']
    for ax, t, st in zip(axs, 'abcde', subt):
        ax.set_title(t, fontsize=8.2, color=INK, loc='left', fontweight='bold', pad=11)
        if st:
            ax.annotate(st, xy=(0.62, 1.0), xycoords='axes fraction', xytext=(0, 11), textcoords='offset points', fontsize=5.9,
                        color=INK, ha='center', va='bottom', annotation_clip=False)
    for ax, x, lb in ((axs[1], 1, 'equal'), (axs[2], 1, 'ordinary words'), (axs[3], 0, 'equal'), (axs[4], 0, 'ordinary words')):
        ax.annotate(lb, xy=(x, 1.0), xycoords=('data', 'axes fraction'), xytext=(0, 1.5), textcoords='offset points',
                    fontsize=5.6, color=MUTED, ha='center', va='bottom', style='italic', annotation_clip=False)
    top = 1 - 1.3 / H
    fig.subplots_adjust(left=0.345, right=0.985, top=top, bottom=1.05 / H)

    def header(a0, a1, text):
        p0, p1 = a0.get_position(), a1.get_position()
        yh = top + 0.30 / H
        fig.text((p0.x0 + p1.x1) / 2, yh + 0.07 / H, text, ha='center', va='bottom', fontsize=7.8, color=INK, fontweight='bold')
        fig.add_artist(Line2D([p0.x0 + 0.012, p1.x1], [yh, yh], color=INK, lw=0.7))
    header(axs[0], axs[0], 'Geometry')
    header(axs[1], axs[2], 'Next-token KL')
    header(axs[3], axs[4], 'Retention along the message')
    h = [Line2D([], [], marker='o', ls='', ms=4.4, mfc=INK, mec=INK, label='a: layers on message content'),
         Line2D([], [], marker='o', ls='', ms=4.4, mfc='none', mec=INK, mew=0.9, label='a: layers at the response onset'),
         Line2D([], [], marker='D', ls='', ms=4.0, mfc=INK, mec=INK, label='b, d: self relative to external roles'),
         Line2D([], [], marker='o', ls='', ms=4.8, mfc=C_SELF, mec=C_SELF, label='c, e: self relative to ordinary words'),
         Line2D([], [], marker='o', ls='', ms=4.6, mfc='white', mec=INK, mew=1.0, label='open: main probe only')]
    fig.legend(handles=h, loc='upper center', ncol=3, fontsize=6.2, frameon=False, bbox_to_anchor=(0.6, 1.0), columnspacing=1.3,
               handletextpad=0.4, labelcolor=INK, labelspacing=0.35)
    st = '; '.join(f"{c}: rho {stats[c]['spearman_group']:+.2f} (p {stats[c]['p_one_sided']:.3f})" for c in 'bcde')
    fig.text(0.01, 0.004, 'Mean of the two conversation orders, each weighted 1/2 (KL: geometric mean). Group statistics over the '
             f"{stats['b']['n']} models with the same group in both orders, Spearman with the group: {st}.",
             fontsize=5.5, color=MUTED, ha='left', va='bottom', wrap=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out + '.png', dpi=220, facecolor='white')
    fig.savefig(out + '.pdf', facecolor='white')
    Path(out + '.json').write_text(json.dumps({'stats': stats, 'models': data}, ensure_ascii=False, indent=1), encoding='utf-8')
    for c in 'bcde':
        s = stats[c]
        print(c, f"rho {s['spearman_group']:+.2f} p {s['p_one_sided']:.4f} n {s['n']}", {k: round(v, 3) for k, v in s['medians'].items()})
    for d in data:
        print(f"  {d['group'][:14]:14s} {d['name'][:26]:26s} " + ' '.join(f"{k}={d['v'][k][0]:.3g}({d['v'][k][1]})" for k in 'bcde'))


if __name__ == '__main__':
    main()
