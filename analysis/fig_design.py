"""Figure self_design (fig:self_design): the experimental design with the representative templates, Qwen3.8-27B and gpt-oss-20b,
token by token. The main-probe renderings of one message in the three roles, aligned at the first body token, with what each
experiment changes. Template level: the role word (every experiment); the Assistant-only format tokens (removed by the Qwen3.8
template switch preserve_thinking=False; masked out of attention, with later positions recomputed, in gpt-oss; given the
position of the token before them in position compression). Feature level: the replacement of the self-index coordinates at the
message under test and every later prompt position, at every layer (generation prompt and generated tokens not edited). Reading
positions: the body (mean over its tokens) and the response onset (the last prompt token).
Token sequences: apply_chat_template of each model's own tokenizer with BODY and QUESTION as placeholders, recorded in the run
folder template_scan/design_tokens_20261029.json (Qwen3.8-27B non-thinking rendering; gpt-oss-20b default rendering) and written
out below. The template tokens around the message under test are drawn one by one; the system preamble of gpt-oss, the User
message with the question and the generation prompt are drawn as one box each (their tokens are identical in the three
renderings). Coordinates are points, so boxes fit their monospace labels.
Outputs: paper/source/figures/self_design.pdf/.png.

usage: fig_design.py --out OUT
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, Roots, arguments  # noqa: E402

FS_TOK, FS_BOX = 5.6, 6.0          # font sizes (pt) of template tokens (monospace) and of the summary boxes (sans)
CW_TOK, CW_BOX = 0.602 * FS_TOK, 0.56 * FS_BOX
PAD, GAP, H = 4.0, 1.6, 12.0       # box padding, gap between boxes, box height (pt)
NL, NL2 = '⏎', '⏎⏎'
BOX = {'BODY': 'body (identical tokens)', 'Q': 'User message: question', 'GEN': 'generation prompt', 'SYS': 'system message'}
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
            family='DejaVu Sans' if sans else 'DejaVu Sans Mono', color=MUTED if dashed else INK, zorder=3)
    return w


def draw(ax, rows, fmt, title):
    dy, xmin, xmax, onset = -20.0, 0.0, 0.0, 0.0
    for r, (role, (head, end)) in enumerate(rows.items()):
        y = r * dy
        x = -sum(wid(t) + GAP for t in head)
        xmin = min(xmin, x)
        ax.text(x - 4, y, role, ha='right', va='center', fontsize=7.4, fontweight='bold', color=INK)
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
    ax.text(wid('BODY') / 2, yb - 3, 'message content:\nmean over the body', ha='center', va='top', fontsize=6.0, color=INK)
    ax.annotate('response onset', xy=(onset, 7.5), xytext=(onset, 17), ha='center', va='bottom', fontsize=6.0, color=INK,
                arrowprops=dict(arrowstyle='->', lw=0.6, color=INK))
    ax.text(xmax, yb - 3, 'generation prompt: not edited', ha='right', va='top', fontsize=6.0, color=MUTED)
    ax.set_title(title, loc='left', fontsize=8.0, fontweight='bold')
    ax.axis('off')
    return xmin - 40, xmax + 4, yb - 22, 28


def main_(out):
    # one data unit = one point: the axes are placed by hand so that their size in points equals their data span
    fig = plt.figure(figsize=(7.4, 4.4))
    axes = [fig.add_axes([0, 0.5, 1, 0.5]), fig.add_axes([0, 0, 1, 0.5])]
    lims = [draw(axes[0], QWEN, FORMAT['qwen'], '(a) Qwen3.8-27B, non-thinking rendering'), draw(axes[1], OSS, FORMAT['oss'], '(b) gpt-oss-20b')]
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
              'Assistant-only format: removed by the template switch (a); masked, with later positions recomputed (b); '
              'given the position of the token before it (position compression)',
              'positions whose self-index coordinates the replacement edits, at every layer']
    fig.legend(handles, labels, loc='lower left', ncol=1, frameon=False, fontsize=6.3, bbox_to_anchor=(0.03, 0.0))
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, 'self_design.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(out, 'self_design.png'), dpi=220, bbox_inches='tight')


def main(argv=None):
    args = arguments(__doc__, results=False).parse_args(argv)
    main_(str(Roots(args).out(FIGURES)))


if __name__ == '__main__':
    main()
