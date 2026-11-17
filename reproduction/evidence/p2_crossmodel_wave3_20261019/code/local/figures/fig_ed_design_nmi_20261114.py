"""Extended Data Fig. 5 of the NMI version (11-14), from make_fig_design.py (figure self_design of the long version): the same
token sequences and colours in the NMI format (Arial for text, DejaVu Sans Mono for template tokens, which has the return symbol, text 5-7 pt, at most 180 mm wide):
shorter box labels, smaller padding, panel letters a, b in bold lowercase. ratio_panel() draws panel c, the distance ratio R (the
panel that left Fig. 1 on 11-14). Writes ed_fig5ab_design.pdf and ed_fig5c_ratio.pdf. The original docstring follows.
Figure self_design (paper v5, user 10-29: the experimental design with the representative templates, Qwen3.8-27B and
gpt-oss-20b, token by token): the main-probe renderings of one message in the three roles, aligned at the first body token, with
what each experiment changes. Template level: the role word (every experiment); the Assistant-only format tokens (removed by the
Qwen3.8 template switch preserve_thinking=False; masked out of attention, with later positions recomputed, in gpt-oss; given the
position of the token before them in position compression). Feature level: the replacement of the self-index coordinates at the
message under test and every later prompt position, at every layer (generation prompt and generated tokens not edited). Reading
positions: the body (mean over its tokens) and the response onset (the last prompt token).
Token sequences: apply_chat_template of each model's own tokenizer with BODY and QUESTION as placeholders, recorded on PRO in
template_scan/design_tokens_20261029.json (Qwen3.8-27B non-thinking rendering; gpt-oss-20b default rendering). The template
tokens around the message under test are drawn one by one; the system preamble of gpt-oss, the User message with the question and
the generation prompt are drawn as one box each (their tokens are identical in the three renderings). Coordinates are points, so
boxes fit their monospace labels.
usage: fig_ed_design_nmi_20261114.py OUT_DIR"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

FS_TOK, FS_BOX = 5.2, 5.8          # font sizes (pt) of template tokens (monospace) and of the summary boxes (sans)
CW_TOK, CW_BOX = 0.602 * FS_TOK, 0.56 * FS_BOX
PAD, GAP, H = 2.4, 1.2, 11.0       # box padding, gap between boxes, box height (pt)
NL, NL2 = '⏎', '⏎⏎'
BOX = {'BODY': 'message text (identical)', 'Q': 'User: question', 'GEN': 'generation prompt', 'SYS': 'system'}
# (header tokens, end tokens after the body); every row continues with the question message and the generation prompt
QWEN = {'Assistant': (['<|im_start|>', 'assistant', NL, '<think>', NL2, '</think>', NL2], ['<|im_end|>', NL]),
        'User': (['<|im_start|>', 'user', NL], ['<|im_end|>', NL]),
        'System': (['<|im_start|>', 'system', NL], ['<|im_end|>', NL])}
OSS = {'Assistant': (['SYS', '<|start|>', 'assistant', '<|channel|>', 'final', '<|message|>'], ['<|end|>']),
       'User': (['SYS', '<|start|>', 'user', '<|message|>'], ['<|end|>']),
       'System': (['SYS', '<|start|>', 'developer', '<|message|>', '#', 'Instructions', NL2], [NL2, '<|end|>'])}
FORMAT = {'qwen': {('Assistant', 3), ('Assistant', 4), ('Assistant', 5), ('Assistant', 6)}, 'oss': {('Assistant', 3), ('Assistant', 4)}}
ROLE_WORDS = {'assistant', 'user', 'system', 'developer'}
C_ROLE, C_FMT, C_BODY, C_Q, C_TOK, C_SYS, C_EDIT = '#cfe0f7', '#f6c08f', '#d9dde3', '#eef0f3', '#ffffff', '#e8e2f4', '#e3f1e6'
INK, MUTED, EDGE = '#2b2f36', '#6b7280', '#9aa1a9'


def wid(t):
    if t in BOX:
        return CW_BOX * len(BOX[t]) + 2 * PAD
    return CW_TOK * len(t) + 2 * PAD


def box(ax, x, y, t, fc, dashed=False):
    w = wid(t)
    ax.add_patch(FancyBboxPatch((x, y - H / 2), w, H, boxstyle='round,pad=0,rounding_size=1.6', fc=fc, ec=EDGE if not dashed else '#c3c8ce',
                                lw=0.5, ls=(0, (2, 1)) if dashed else '-', zorder=2))
    sans = t in BOX
    ax.text(x + w / 2, y, BOX.get(t, t), ha='center', va='center', fontsize=FS_BOX if sans else FS_TOK,
            family='Arial' if sans else 'DejaVu Sans Mono', color=MUTED if dashed else INK, zorder=3)
    return w


def draw(ax, rows, fmt, title):
    dy, xmin, xmax, onset = -20.0, 0.0, 0.0, 0.0
    for r, (role, (head, end)) in enumerate(rows.items()):
        y = r * dy
        x = -sum(wid(t) + GAP for t in head)
        xmin = min(xmin, x)
        ax.text(x - 4, y, role, ha='right', va='center', fontsize=6.6, fontweight='bold', color=INK)
        for i, t in enumerate(head):
            fc = C_SYS if t == 'SYS' else C_FMT if (role, i) in fmt else C_ROLE if t in ROLE_WORDS else C_TOK
            x += box(ax, x, y, t, fc) + GAP
        x = 0.0
        x += box(ax, x, y, 'BODY', C_BODY) + GAP
        for t in end:
            x += box(ax, x, y, t, C_TOK) + GAP
        x += box(ax, x, y, 'Q', C_Q) + GAP
        ax.add_patch(Rectangle((-1.2, y - H / 2 - 2.5), x - GAP + 2.0, H + 5.0, fc=C_EDIT, ec='none', zorder=0))
        x += box(ax, x, y, 'GEN', C_TOK, dashed=True)
        onset = max(onset, x - 3)
        xmax = max(xmax, x)
    yb = (len(rows) - 1) * dy - 11
    ax.annotate('', xy=(0, yb), xytext=(wid('BODY'), yb), arrowprops=dict(arrowstyle='|-|', lw=0.6, color=INK, mutation_scale=2))
    ax.text(wid('BODY') / 2, yb - 3, 'message text:\nmean over its tokens', ha='center', va='top', fontsize=5.8, color=INK)
    ax.annotate('response onset', xy=(onset, 7.5), xytext=(onset, 17), ha='center', va='bottom', fontsize=5.8, color=INK,
                arrowprops=dict(arrowstyle='->', lw=0.6, color=INK))
    ax.text(xmax, yb - 3, 'generation prompt: not edited', ha='right', va='top', fontsize=5.8, color=MUTED)
    ax.set_title(title, loc='left', fontsize=7.0, fontweight='bold')
    ax.axis('off')
    return xmin - 34, xmax + 4, yb - 20, 28


def main(out):
    # one data unit = one point: the axes are placed by hand so that their size in points equals their data span
    fig = plt.figure(figsize=(7.4, 4.4))
    axes = [fig.add_axes([0, 0.5, 1, 0.5]), fig.add_axes([0, 0, 1, 0.5])]
    lims = [draw(axes[0], QWEN, FORMAT['qwen'], r'$\bf{a}$   Qwen3.8-27B, rendering without reasoning'), draw(axes[1], OSS, FORMAT['oss'], r'$\bf{b}$   gpt-oss-20b')]
    x0, x1 = min(l[0] for l in lims), max(l[1] for l in lims)
    hs = [l[3] - l[2] for l in lims]
    legend_h, top = 44.0, 10.0
    W, Ht = x1 - x0, sum(hs) + legend_h + top
    fig.set_size_inches(W / 72, Ht / 72)
    yb = legend_h
    for ax, l, h in zip(axes[::-1], lims[::-1], hs[::-1]):
        ax.set_position([0, yb / Ht, 1, h / Ht])
        yb += h
    for ax, l in zip(axes, lims):
        ax.set_xlim(x0, x1)
        ax.set_ylim(l[2], l[3])
    handles = [Rectangle((0, 0), 1, 1, fc=C_ROLE, ec=EDGE, lw=0.5), Rectangle((0, 0), 1, 1, fc=C_FMT, ec=EDGE, lw=0.5),
               Rectangle((0, 0), 1, 1, fc=C_EDIT, ec='none')]
    labels = ['role word: differs between the three renderings',
              'format that the template adds to Assistant messages only (removed, masked or given earlier positions in the controls)',
              'positions whose self-index coordinates the replacement edits, at every layer']
    fig.legend(handles, labels, loc='lower left', ncol=1, frameon=False, fontsize=5.9, bbox_to_anchor=(0.03, 0.0))
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, 'ed_fig5ab_design.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(out, 'ed_fig5ab_design.png'), dpi=220, bbox_inches='tight')
    print('ab width mm', round(W / 72 * 25.4, 1))
    ratio_panel(out)



def ratio_panel(out):
    """panel c: the distance ratio R_A (from the panel of Fig. 1 that was moved here on 11-14)"""
    C = {'A': '#2a78d6', 'U': '#eb6834', 'S': '#1baf7a'}
    fig = plt.figure(figsize=(7.08, 1.55))
    ax = fig.add_axes([0.0, 0.0, 0.44, 0.86])
    ax.set_xlim(0, 3.0)
    ax.set_ylim(-0.35, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    lays = [('Assistant singled out', '$R_A$ small', {'A': (0.10, 0.50), 'U': (0.80, 0.80), 'S': (0.84, 0.25)}),
            ('no role singled out', '$R_A$ ≈ 1', {'A': (0.12, 0.25), 'U': (0.82, 0.25), 'S': (0.47, 0.856)})]
    for k, (title, note, lay) in enumerate(lays):
        ox = 0.25 + k * 1.45
        pts = {r: (ox + q[0], q[1]) for r, q in lay.items()}
        for r2 in 'US':
            ax.plot(*zip(pts['A'], pts[r2]), color='#9aa1a9', lw=0.9, ls='--')
        ax.plot(*zip(pts['U'], pts['S']), color=INK, lw=1.4)
        for r, q in pts.items():
            ax.add_patch(Circle(q, 0.075, fc=C[r], ec='white', lw=1.0, zorder=3))
            ax.text(q[0], q[1], r, ha='center', va='center', fontsize=6.4, color='white', fontweight='bold', zorder=4)
        ax.text(ox + 0.47, 1.02, title, ha='center', va='bottom', fontsize=6.4, color=INK, fontweight='bold')
        ax.text(ox + 0.47, -0.16, note, ha='center', va='top', fontsize=6.2, color=INK)
    fig.text(0.0, 0.98, r'$\bf{c}$   The distance ratio', fontsize=7.0, fontweight='bold', color=INK, ha='left', va='top')
    fig.text(0.45, 0.5, 'The three renderings of a message give three points per token and layer.\n'
             '$R_A$ divides the User-System distance (solid) by the mean of the distances of\n'
             'the Assistant point to the two (dashed). $R_A < 1$: User and System are closer to\n'
             'each other than to Assistant. A layer carries the self-index when $R_A$ is the\n'
             'smallest of the three ratios in at least 80% of the held-out texts.\n'
             '$R_U$ and $R_S$ are defined in the same way.',
             fontsize=6.0, color=INK, ha='left', va='center', linespacing=1.35)
    fig.savefig(os.path.join(out, 'ed_fig5c_ratio.pdf'))
    fig.savefig(os.path.join(out, 'ed_fig5c_ratio.png'), dpi=220)


if __name__ == '__main__':
    main(sys.argv[1])
