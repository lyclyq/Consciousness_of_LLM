"""20261110 draft (from make_fig_readout_v4.py): panel (a) adds the task-free readout. Figure self_readout (concept only, no data):
how a self-index yields self-reference.
(a) One displacement, three readouts: the model's own message carries a self-index offset u (Equation self_decomposition) over its
    whole representation. Two readouts answer a question: one orthogonal to u gives the same value for every role (output unchanged
    by the source), one aligned with u separates the model's own message (output follows the source). The third needs no question:
    the projection toward the next token on the tokens of the message itself; aligned with u, it sets the own message apart with no
    task, so it shows the self-index (first level), not self-reference (drawn in the Assistant/self-index blue, not the amber of
    self-reference).
(b) Three cases across models (unchanged from v4): no displacement; a displacement without a single-pass readout; a displacement
    and a readout.
Colors follow the overview figure (Assistant blue, User orange, System green, self-reference amber).
The zh paper includes the same English figure (figures/self_readout.pdf), so there is no zh version.
usage: make_fig_readout_20261110.py OUT_DIR"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2, LINE, PALE = '#0b0b0b', '#52514e', '#c9c7c1', '#f3f2ee'
C = {'A': '#2a78d6', 'U': '#eb6834', 'S': '#1baf7a'}
REF, REF_TXT, REF_FILL, IDX_FILL = '#d9822b', '#b8651b', '#fff6ec', '#eef4fc'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans'})


def box(ax, x0, y0, x1, y1, fc='white', ec=LINE, lw=0.8, r=0.08):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f'round,pad=0.0,rounding_size={r}', fc=fc, ec=ec, lw=lw))


def arrow(ax, p, q, col=INK, lw=1.1, ls='-', ms=9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, color=col, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def dots(ax, pts, s=120, z=4):
    for r, (x, y) in pts.items():
        ax.scatter([x], [y], s=s, c=C[r], edgecolors='white', linewidths=1.0, zorder=z)
        ax.text(x, y, r, ha='center', va='center', fontsize=6.4, color='white', fontweight='bold', zorder=z + 1)


def panel(fig, rect, w, h):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax


def main(out_dir):
    W, HA, HB = 7.2, 4.41, 2.55  # v4: HA 3.35; the third lane adds about one inch to panel (a) only
    fig = plt.figure(figsize=(W, HA + HB))
    a = panel(fig, [0, HB / (HA + HB), 1, HA / (HA + HB)], W, HA)
    b = panel(fig, [0, 0, 1, HB / (HA + HB)], W, HB)

    # (a) one displacement, three readouts: two answer a question, one needs none
    a.text(0.05, HA - 0.13, '(a) One displacement, read or not read', fontsize=10, fontweight='bold', va='center')
    y3 = 0.66                         # task-free lane (new)
    y2 = y3 + 1.18                    # question that depends on the source
    y1 = y2 + 1.39                    # question that does not (v4 spacing)
    dy = y2 - 0.78                    # v4 had y2 = 0.78; the geometry keeps its place relative to lanes 1-2
    box(a, 0.05, y1 - 0.42, 1.80, y1 + 0.42)
    a.text(0.925, y1 + 0.12, 'Input that does not\ndepend on the source', ha='center', va='center', fontsize=8.2, color=INK,
           linespacing=1.15)
    a.text(0.925, y1 - 0.27, 'e.g. the last number\nin the message', ha='center', va='center', fontsize=7, color=INK2, linespacing=1.1)
    box(a, 0.05, y2 - 0.50, 1.80, y2 + 0.50)
    a.text(0.925, y2 + 0.22, 'Input that depends\non the source', ha='center', va='center', fontsize=8.2, color=INK, linespacing=1.15)
    a.text(0.925, y2 - 0.25, '"Who wrote the previous\nmessage?"', ha='center', va='center', fontsize=7,
           color=INK2, linespacing=1.1)
    box(a, 0.05, y3 - 0.42, 1.80, y3 + 0.42)
    a.text(0.925, y3 + 0.17, 'No question', ha='center', va='center', fontsize=8.2, color=INK)
    a.text(0.925, y3 - 0.17, 'next token on the\nmessage itself', ha='center', va='center', fontsize=7, color=INK2,
           linespacing=1.1)

    box(a, 2.15, 0.12, 5.10, HA - 0.33, fc=IDX_FILL, ec=C['A'], lw=1.2)
    a.text(3.625, HA - 0.52, 'Model', ha='center', va='center', fontsize=9, fontweight='bold', color=C['A'])
    # geometry: the same content in three roles; the model's own message is displaced by u
    U, S = (2.62, 1.36 + dy), (2.82, 1.20 + dy)
    cx, cy = (U[0] + S[0]) / 2, (U[1] + S[1]) / 2
    A = (cx + 0.52, cy + 0.52)
    arrow(a, (cx + 0.10, cy + 0.10), (A[0] - 0.09, A[1] - 0.09), col=C['A'], lw=1.4)
    a.text(cx + 0.14, cy + 0.42, r'$u$', fontsize=10, color=C['A'], ha='center', va='center')
    dots(a, {'U': U, 'S': S, 'A': A})
    a.text(2.95, 1.02 + dy, 'own message shifted by ' + r'$u$', ha='center', va='center', fontsize=6.8, color=INK2)
    # readouts as number lines
    for y, lab, xs in ((y1, r'readout $w_1$:  $w_1\cdot u \approx 0$', {'U': 4.17, 'A': 4.33, 'S': 4.49}),
                       (y2, r'readout $w_2$:  $w_2\cdot u$ large', {'U': 3.88, 'S': 4.04, 'A': 4.76}),
                       (y3, r'readout $w_3$:  $w_3\cdot u$ large', {'U': 3.88, 'S': 4.04, 'A': 4.66})):
        a.plot([3.68, 4.98], [y, y], color=INK2, lw=0.9, zorder=2)
        for t in (3.68, 4.98):
            a.plot([t, t], [y - 0.05, y + 0.05], color=INK2, lw=0.9, zorder=2)
        a.text(4.33, y + 0.30, lab, ha='center', va='center', fontsize=7.2, color=INK)
        dots(a, {r: (x, y) for r, x in xs.items()}, s=105)
    a.text(4.33, y1 - 0.30, 'all roles read alike', ha='center', va='center', fontsize=6.8, color=INK2)
    a.text(4.33, y2 - 0.30, 'own message read apart', ha='center', va='center', fontsize=6.8, color=REF_TXT)
    a.text(4.33, y3 - 0.30, 'own message read apart', ha='center', va='center', fontsize=6.8, color=C['A'])
    arrow(a, (1.80, y1), (3.60, y1), col=INK2)
    arrow(a, (1.80, y2), (3.60, y2), col=REF, lw=1.3)
    arrow(a, (1.80, y3), (3.60, y3), col=C['A'], lw=1.3)

    box(a, 5.45, y1 - 0.42, 7.15, y1 + 0.42)
    a.text(6.30, y1 + 0.12, 'Output unchanged\nby the source', ha='center', va='center', fontsize=8.2, color=INK, linespacing=1.15)
    a.text(6.30, y1 - 0.26, 'same answer whichever\nrole the message has', ha='center', va='center', fontsize=7, color=INK2, linespacing=1.1)
    box(a, 5.45, y2 - 0.50, 7.15, y2 + 0.50, fc=REF_FILL, ec=REF, lw=1.2)
    a.text(6.30, y2 + 0.20, 'Output follows\nthe source', ha='center', va='center', fontsize=8.2, color=REF_TXT, fontweight='bold',
           linespacing=1.15)
    a.text(6.30, y2 - 0.26, '"I wrote this" for its\nown messages only', ha='center', va='center', fontsize=7, color=INK2,
           linespacing=1.1)
    box(a, 5.45, y3 - 0.42, 7.15, y3 + 0.42, fc=IDX_FILL, ec=C['A'], lw=1.2)
    a.text(6.30, y3 + 0.12, 'Next-token prediction\nfollows the source', ha='center', va='center', fontsize=8.2, color=C['A'], fontweight='bold',
           linespacing=1.15)
    a.text(6.30, y3 - 0.26, 'shows the self-index,\nnot self-reference', ha='center', va='center', fontsize=7, color=INK2,
           linespacing=1.1)
    arrow(a, (5.06, y1), (5.43, y1), col=INK2)
    arrow(a, (5.06, y2), (5.43, y2), col=REF, lw=1.3)
    arrow(a, (5.06, y3), (5.43, y3), col=C['A'], lw=1.3)

    # (b) three cases across models (unchanged from v4)
    b.text(0.05, 2.42, '(b) Three cases across models', fontsize=10, fontweight='bold', va='center')
    # 11-10 review: the caption and Section 7.3 place Granite-4.1-30B (index ends at L36) in this case, so the title names the last third
    cases = [(0.05, 'No self-index in the last third', LINE, 'white', False, None,
              r'$u\approx 0$' + ' in the last third:\nthe own message is not set apart\nthere, with or without thinking'),
             (2.47, 'Self-index, no single-pass readout', C['A'], IDX_FILL, True, 'dashed',  # 11-01: was "without readout"
              r'$u$' + ' is present but a single pass\ndoes not map it to "you";\nthinking can supply the readout'),
             (4.89, 'Self-index and self-reference', C['A'], IDX_FILL, True, 'solid',
              r'$u$' + ' is read in a single pass:\nthe model claims its own\nmessages and only those')]
    for x0, title, ec, fc, shifted, readout, txt in cases:
        x1 = x0 + 2.26
        box(b, x0, 0.42, x1, 2.20, fc=fc, ec=ec, lw=1.1 if shifted else 0.9)
        b.text((x0 + x1) / 2, 2.02, title, ha='center', va='center', fontsize=8.2, fontweight='bold',
               color=C['A'] if shifted else INK2)
        gx, gy = x0 + 0.42, 1.25
        U, S = (gx, gy + 0.10), (gx + 0.18, gy - 0.06)
        if shifted:
            cxx, cyy = gx + 0.09, gy + 0.02
            A = (cxx + 0.40, cyy + 0.40)
            arrow(b, (cxx + 0.09, cyy + 0.09), (A[0] - 0.08, A[1] - 0.08), col=C['A'], lw=1.2, ms=7)
        else:
            A = (gx + 0.20, gy + 0.18)
        dots(b, {'U': U, 'S': S, 'A': A}, s=100)
        if readout:
            col = REF if readout == 'solid' else INK2
            arrow(b, (A[0] + 0.14, A[1]), (x0 + 1.55, A[1]), col=col, lw=1.2 if readout == 'solid' else 0.9,
                  ls='-' if readout == 'solid' else (0, (3, 2)), ms=8)
            b.text(x0 + 1.62, A[1], '"you"', ha='left', va='center', fontsize=8, color=REF_TXT if readout == 'solid' else INK2)
            if readout == 'dashed':
                b.text(x0 + 1.10, A[1] + 0.13, 'thinking', ha='center', va='bottom', fontsize=6.6, color=INK2, style='italic')
        b.text((x0 + x1) / 2, 0.75, txt, ha='center', va='center', fontsize=6.9, color=INK2, linespacing=1.12)
    for x0 in (2.31, 4.73):
        arrow(b, (x0, 1.30), (x0 + 0.16, 1.30), col=INK2, lw=0.9, ms=7)
    b.text(3.6, 0.20, 'The cases are nested: no model shows self-reference without a self-index.',
           ha='center', va='center', fontsize=7.4, color=INK)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'self_readout.{ext}'), dpi=200 if ext == 'png' else None)
    print('saved', out_dir)


if __name__ == '__main__':
    main(sys.argv[1])
