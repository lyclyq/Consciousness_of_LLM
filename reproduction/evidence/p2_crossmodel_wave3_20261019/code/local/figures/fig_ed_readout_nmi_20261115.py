"""Extended Data Fig. 6 of the NMI version (2026-11-15, user: the schematic with the self-index vector goes into the appendix to aid
understanding). A schematic, no data: how one offset u is read or not read.
Redrawn from panel (a) of make_fig_readout_20261110.py (Figure 2 of the long version) in the format of the NMI figures: Arial,
180 mm wide, no panel letter (a single panel, the title is in the caption), the attribution question in the wording of the Methods.
Panel (b) of the long version is not redrawn: it sorts the models by an earlier criterion (the last third of the layers) with the
earlier counts, and Fig. 3 of the NMI version shows the nesting with the current criteria.
Lanes, from the top: a question that does not concern the source (readout w1 with w1 . u ~ 0, every role read alike), the
attribution question (readout w2 aligned with u, the model's own message read apart, self-reference), and no question at all
(the prediction of the next token on the message itself, w3 aligned with u, which shows the self-index but not self-reference).
Colors follow Fig. 1 (Assistant blue, User orange, System green, self-reference amber). The dot and the approximately-equal sign
are plain characters, as in Fig. 1, so that every glyph is set in Arial.
usage: fig_ed_readout_nmi_20261115.py OUT_DIR   (writes ed_fig6_readout.pdf and .png; run with SOURCE_DATE_EPOCH=0)"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2, LINE = '#0b0b0b', '#52514e', '#c9c7c1'
C = {'A': '#2a78d6', 'U': '#eb6834', 'S': '#1baf7a'}
REF, REF_TXT, REF_FILL, IDX_FILL = '#d9822b', '#b8651b', '#fff6ec', '#eef4fc'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})
T1, T2, T3 = 7.0, 6.2, 6.0          # box titles, box notes, labels inside the model


def box(ax, x0, y0, x1, y1, fc='white', ec=LINE, lw=0.8, r=0.08):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f'round,pad=0.0,rounding_size={r}', fc=fc, ec=ec, lw=lw))


def arrow(ax, p, q, col=INK, lw=1.1, ms=9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, color=col, lw=lw, shrinkA=0, shrinkB=0))


def dots(ax, pts, s=105, z=4):
    for r, (x, y) in pts.items():
        ax.scatter([x], [y], s=s, c=C[r], edgecolors='white', linewidths=1.0, zorder=z)
        ax.text(x, y, r, ha='center', va='center', fontsize=5.6, color='white', fontweight='bold', zorder=z + 1)


def main(out_dir):
    W, H = 7.2, 4.16                   # drawing units, as in the long version; the canvas is 180 mm wide
    FW = 7.08
    fig = plt.figure(figsize=(FW, FW * H / W))
    a = fig.add_axes([0, 0, 1, 1])
    a.set_xlim(0, W)
    a.set_ylim(0, H)
    a.set_aspect('equal')
    a.axis('off')

    y3 = 0.66                          # no question
    y2 = y3 + 1.18                     # a question that concerns the source
    y1 = y2 + 1.39                     # a question that does not
    dy = y2 - 0.78
    # inputs
    box(a, 0.05, y1 - 0.42, 1.80, y1 + 0.42)
    a.text(0.925, y1 + 0.12, 'Input that does not\nconcern the source', ha='center', va='center', fontsize=T1, color=INK, linespacing=1.15)
    a.text(0.925, y1 - 0.25, 'e.g. the last number\nin the message', ha='center', va='center', fontsize=T2, color=INK2, linespacing=1.1)
    box(a, 0.05, y2 - 0.50, 1.80, y2 + 0.50)
    a.text(0.925, y2 + 0.22, 'Input that concerns\nthe source', ha='center', va='center', fontsize=T1, color=INK, linespacing=1.15)
    a.text(0.925, y2 - 0.25, '"Who wrote the message\nbefore this question?"', ha='center', va='center', fontsize=T2, color=INK2,
           linespacing=1.1)
    box(a, 0.05, y3 - 0.42, 1.80, y3 + 0.42)
    a.text(0.925, y3 + 0.17, 'No question', ha='center', va='center', fontsize=T1, color=INK)
    a.text(0.925, y3 - 0.17, 'next token on the\nmessage itself', ha='center', va='center', fontsize=T2, color=INK2, linespacing=1.1)

    # the model: the same content in three roles, the model's own message shifted by u
    box(a, 2.15, 0.12, 5.10, H - 0.08, fc=IDX_FILL, ec=C['A'], lw=1.2)
    a.text(3.625, H - 0.27, 'Model', ha='center', va='center', fontsize=7.5, fontweight='bold', color=C['A'])
    U, S = (2.62, 1.36 + dy), (2.82, 1.20 + dy)
    cx, cy = (U[0] + S[0]) / 2, (U[1] + S[1]) / 2
    A = (cx + 0.52, cy + 0.52)
    arrow(a, (cx + 0.10, cy + 0.10), (A[0] - 0.09, A[1] - 0.09), col=C['A'], lw=1.4)
    a.text(cx + 0.14, cy + 0.42, r'$u$', fontsize=8.5, color=C['A'], ha='center', va='center')
    dots(a, {'U': U, 'S': S, 'A': A})
    a.text(2.95, 1.02 + dy, 'own message shifted by ' + r'$u$', ha='center', va='center', fontsize=T3, color=INK2)
    # readouts as number lines
    for y, lab, xs in ((y1, 'readout $w_1$:  $w_1$ · $u$ ≈ 0', {'U': 4.17, 'A': 4.33, 'S': 4.49}),
                       (y2, 'readout $w_2$:  $w_2$ · $u$ large', {'U': 3.88, 'S': 4.04, 'A': 4.76}),
                       (y3, 'readout $w_3$:  $w_3$ · $u$ large', {'U': 3.88, 'S': 4.04, 'A': 4.66})):
        a.plot([3.68, 4.98], [y, y], color=INK2, lw=0.9, zorder=2)
        for t in (3.68, 4.98):
            a.plot([t, t], [y - 0.05, y + 0.05], color=INK2, lw=0.9, zorder=2)
        a.text(4.33, y + 0.29, lab, ha='center', va='center', fontsize=6.4, color=INK)
        dots(a, {r: (x, y) for r, x in xs.items()}, s=95)
    a.text(4.33, y1 - 0.29, 'all roles read alike', ha='center', va='center', fontsize=T3, color=INK2)
    a.text(4.33, y2 - 0.29, 'own message read apart', ha='center', va='center', fontsize=T3, color=REF_TXT)
    a.text(4.33, y3 - 0.29, 'own message read apart', ha='center', va='center', fontsize=T3, color=C['A'])
    arrow(a, (1.80, y1), (3.60, y1), col=INK2)
    arrow(a, (1.80, y2), (3.60, y2), col=REF, lw=1.3)
    arrow(a, (1.80, y3), (3.60, y3), col=C['A'], lw=1.3)

    # outputs
    box(a, 5.45, y1 - 0.42, 7.15, y1 + 0.42)
    a.text(6.30, y1 + 0.12, 'Output unchanged\nby the source', ha='center', va='center', fontsize=T1, color=INK, linespacing=1.15)
    a.text(6.30, y1 - 0.25, 'the same answer whichever\nrole the message has', ha='center', va='center', fontsize=T2, color=INK2,
           linespacing=1.1)
    box(a, 5.45, y2 - 0.50, 7.15, y2 + 0.50, fc=REF_FILL, ec=REF, lw=1.2)
    a.text(6.30, y2 + 0.20, 'Output follows the source:\nself-reference', ha='center', va='center', fontsize=T1, color=REF_TXT,
           fontweight='bold', linespacing=1.15)
    a.text(6.30, y2 - 0.25, '"You" for its own\nmessages only', ha='center', va='center', fontsize=T2, color=INK2, linespacing=1.1)
    box(a, 5.45, y3 - 0.42, 7.15, y3 + 0.42, fc=IDX_FILL, ec=C['A'], lw=1.2)
    a.text(6.30, y3 + 0.12, 'Next-token prediction\nfollows the source', ha='center', va='center', fontsize=T1, color=C['A'],
           fontweight='bold', linespacing=1.15)
    a.text(6.30, y3 - 0.25, 'shows the self-index,\nnot self-reference', ha='center', va='center', fontsize=T2, color=INK2,
           linespacing=1.1)
    arrow(a, (5.06, y1), (5.43, y1), col=INK2)
    arrow(a, (5.06, y2), (5.43, y2), col=REF, lw=1.3)
    arrow(a, (5.06, y3), (5.43, y3), col=C['A'], lw=1.3)

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'ed_fig6_readout.pdf'))
    fig.savefig(os.path.join(out_dir, 'ed_fig6_readout.png'), dpi=200)
    print('saved', out_dir, round(FW, 2), round(FW * H / W, 2))


if __name__ == '__main__':
    main(sys.argv[1])
