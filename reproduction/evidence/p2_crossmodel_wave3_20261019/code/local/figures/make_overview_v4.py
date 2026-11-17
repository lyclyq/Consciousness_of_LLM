"""Overview figure (two-level account), paper v4. (a) The probe: one native conversation per role (chat template, non-thinking mode),
identical text in the role message. (b) The two levels and the two readouts measured in this paper. (c) How to read the
distance ratio R. Schematic; the numbers quoted are those of Sections 4.1-4.5.
v4 changes (make_overview_v2.py otherwise): the use of the index in reasoning is replaced by the content question of the
causal tests, a readout that does not read the index; the self-reference arrow names the causal test in two models.
v5 (10-29): the arrow names the causal tests of the paper, three models and four readouts, read before the last third of the
layers; the item-level prediction of claim strength ("strength predicted by the index") is gone, as from the paper.
11-01: "needed and sufficient" becomes "needed; installed, they create the claim", as the text after R08 of the review.
11-10: MiMo-V2.6 is the fifth readout. The arrow reads "(four models, five readouts)" for removal and installation, and the position
and layer statement is scoped to the Qwen and gpt-oss readouts, since MiMo has neither test."""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

INK, INK2, LINE, PALE = '#0b0b0b', '#52514e', '#c9c7c1', '#f3f2ee'
C = {'A': '#2a78d6', 'U': '#eb6834', 'S': '#1baf7a'}
NAME = {'A': 'Assistant', 'U': 'User', 'S': 'System'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42})


def box(ax, x, y, w, h, fc, ec=LINE, lw=0.8, r=0.012, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.003,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls))


def arrow(ax, p, q, col=INK, lw=1.2, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=10, color=col, lw=lw, shrinkA=0, shrinkB=0, ls=ls))


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 7.4))
    # (a) probe
    ax = fig.add_axes([0.0, 0.80, 1.0, 0.20])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.01, 0.97, '(a) The probe: the same text in an Assistant, User, or System message', ha='left', va='top',
            fontsize=8.6, fontweight='bold', color=INK)
    y, h = 0.36, 0.30
    toks = [(0.02, 0.085, '<|im_start|>', 'mono'), (0.11, 0.075, 'ROLE', 'role'), (0.19, 0.03, '\\n', 'mono'),
            (0.225, 0.11, 'template\nformatting\n(Assistant only)', 'fmt'), (0.34, 0.20, 'message body\n(same in all runs)', 'body'),
            (0.545, 0.085, '<|im_end|>\\n', 'mono'), (0.635, 0.15, 'User message\n(e.g. who wrote it?)', 'text'),
            (0.79, 0.095, 'generation\nprompt', 'text'), (0.89, 0.09, 'response\nonset', 'text')]
    for x, w, lab, kind in toks:
        if kind == 'role':
            for k, r in enumerate('AUS'):
                yy = y + h - 0.1 * (k + 1) - 0.005 * k
                ax.add_patch(FancyBboxPatch((x, yy), w, 0.095, boxstyle='round,pad=0.002,rounding_size=0.008', fc=C[r], ec='white', lw=0.8))
                ax.text(x + w / 2, yy + 0.047, NAME[r].lower(), ha='center', va='center', fontsize=6, color='white', fontweight='bold')
            continue
        box(ax, x, y, w, h, PALE if kind == 'body' else ('#e9f1fb' if kind == 'fmt' else 'white'))
        ax.text(x + w / 2, y + h / 2, lab, ha='center', va='center', fontsize=5.4 if kind in ('mono', 'fmt') else 6.4, color=INK,
                family='DejaVu Sans Mono' if kind == 'mono' else 'DejaVu Sans', linespacing=1.05)
    ax.text(0.5, 0.2, 'Every input is the chat template\'s own rendering (non-thinking mode). Representations are read at the message content,\n'
            'the end of the message and the response onset, after each of the 64 blocks.', ha='center', va='top', fontsize=6.6, color=INK2)
    # (b) two levels
    bx = fig.add_axes([0.0, 0.31, 1.0, 0.48])
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    bx.axis('off')
    bx.text(0.01, 0.99, '(b) Two levels of self', ha='left', va='top', fontsize=9.2, fontweight='bold', color=INK)
    # ability axis: higher = more abstract ability
    arrow(bx, (0.045, 0.05), (0.045, 0.9), col=INK2, lw=1.4)
    bx.text(0.022, 0.48, 'ability (higher = more abstract)', ha='center', va='center', rotation=90, fontsize=7.6, color=INK2)
    # low level box
    box(bx, 0.09, 0.05, 0.86, 0.2, '#eef4fc', ec=C['A'], lw=1.2)
    bx.text(0.52, 0.205, 'Self-index (low level)', ha='center', va='center', fontsize=9, fontweight='bold', color=C['A'])
    bx.text(0.52, 0.11, 'An inside/outside displacement over the whole representation of the model\'s own message:\n'
            'Assistant is separated from User and System at every layer, from the first block on.', ha='center', va='center',
            fontsize=7.6, color=INK, linespacing=1.25)
    # instinctive use (between the levels) and self-reference (high level)
    box(bx, 0.09, 0.38, 0.33, 0.22, 'white', ec=LINE, lw=1.0)
    bx.text(0.255, 0.555, 'Questions about the content', ha='center', va='center', fontsize=8.6, fontweight='bold', color=INK2)
    bx.text(0.255, 0.455, 'e.g. the last number in the message:\nanswered alike in every role, and\nunchanged when the index is replaced',
            ha='center', va='center', fontsize=7.4, color=INK, linespacing=1.2)
    box(bx, 0.57, 0.64, 0.38, 0.24, '#fff6ec', ec='#d9822b', lw=1.2)
    bx.text(0.76, 0.835, 'Self-reference (high level)', ha='center', va='center', fontsize=8.6, fontweight='bold', color='#b8651b')
    bx.text(0.76, 0.725, 'Recognizing its own past message:\n"I wrote this" (0.77); User and System\nboth assigned to the outside',
            ha='center', va='center', fontsize=7.4, color=INK, linespacing=1.2)
    bx.plot([0.255, 0.255], [0.255, 0.375], color=INK2, lw=1.1, ls=':')
    arrow(bx, (0.76, 0.255), (0.76, 0.635), col='#d9822b')
    bx.text(0.27, 0.315, r'not read ($w\cdot u\approx 0$)', ha='left', va='center', fontsize=7.2, color=INK2)
    bx.text(0.748, 0.45, '8 coordinates per layer needed;\ninstalled, they create the claim\n(four models, five readouts);\n'
            'read at the end of the message,\nbefore the last third of the layers\n(Qwen and gpt-oss readouts)', ha='right', va='center',
            fontsize=7.2, color='#b8651b', linespacing=1.15)
    bx.text(0.09, 0.925, 'Analogy (structure only): corollary discharge, shared by most animals, versus mirror self-recognition, which few pass',
            ha='left', va='center', fontsize=7, color=INK2, style='italic')
    # (c) reading R
    fig.text(0.01, 0.29, '(c) Reading the distance ratio R', ha='left', va='top', fontsize=8.6, fontweight='bold', color=INK)
    cx = fig.add_axes([0.06, 0.05, 0.62, 0.22])
    cx.set_xlim(0, 3.0)
    cx.set_ylim(0, 1.25)
    cx.set_aspect('equal')
    cx.axis('off')
    layouts = [('Assistant singled out', C['A'], '$R_A$ small (observed at every layer)',
                {'A': (0.10, 0.50), 'U': (0.80, 0.80), 'S': (0.84, 0.25)}),
               ('no role singled out', INK2, '$R_A \\approx 1$',
                {'A': (0.12, 0.25), 'U': (0.82, 0.25), 'S': (0.47, 0.856)})]
    for k, (title, col, note, lay) in enumerate(layouts):
        ox = 0.25 + k * 1.45
        pts = {r: (ox + p[0], p[1]) for r, p in lay.items()}
        for r2 in 'US':
            cx.plot(*zip(pts['A'], pts[r2]), color=LINE, lw=1.0, ls='--')
        cx.plot(*zip(pts['U'], pts['S']), color=INK2, lw=1.6)
        for r, p in pts.items():
            cx.add_patch(Circle(p, 0.075, fc=C[r], ec='white', lw=1.2, zorder=3))
            cx.text(p[0], p[1], r, ha='center', va='center', fontsize=7, color='white', fontweight='bold', zorder=4)
        cx.text(ox + 0.47, 1.0, title, ha='center', va='bottom', fontsize=7, color=col, fontweight='bold')
        cx.text(ox + 0.47, 0.02, note, ha='center', va='bottom', fontsize=7, color=INK)
    # 10-30 (supervisor's review R12): R_A < 1 compares U-S with the MEAN of the two Assistant distances; only the smallest
    # of the three ratios identifies the closest pair (fix_three_claims_20261030.py)
    fig.text(0.68, 0.16, 'Three runs give three points\nper token and layer. $R_A$ compares\nthe User-System distance (solid)\n'
             'with the mean of their distances\nto Assistant (dashed). $R_A<1$: User\nand System are closer to each other\n'
             'than to Assistant on average. If $R_A$\nis the smallest of the three ratios,\nUser-System is the closest pair.\n'
             '$R_U$ and $R_S$ are defined\nin the same way.',
             ha='left', va='center', fontsize=6.6, color=INK2, linespacing=1.2)
    fig.savefig(f'{out_dir}/self_overview.pdf')
    fig.savefig(f'{out_dir}/self_overview.png', dpi=150)
    print('saved')


if __name__ == '__main__':
    main(*sys.argv[1:])
