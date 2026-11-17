"""Overview figure for the NMI version (Fig. 1), 2026-11-14. From make_overview_v9_20261112.py (approved 11-12).
Only panel (a) changes (user 11-14: the figure must explain how the measures of both orders are obtained):
  (a) now shows the main probe and the natural order as two aligned rows, the ordinary-word control as a third row, and
      where each measure of Fig. 2 is read (geometry: message text and response onset; next-token divergence: every token
      of the message text; share kept: first against last third of the message text).
Panels (b), (c) and (d) are drawn exactly as in v9. Canvas width 7.08 in (180 mm, the NMI limit), height grows with (a).
Run with matplotlib, for example
  uv run --no-project --python 3.12 --with matplotlib python -I make_overview_nmi_20261114.py OUT_DIR
11-14 figure-format pass (Nature figure guide): Arial, panel letters 8 pt bold lowercase without parentheses and in reading
order (a probe, b two levels, c distance ratio, d loop: c and d swapped against v9), panel titles and the two level names at
7 pt, the line "Read in Fig. 2" names the measures without column letters (Fig. 2 has none any more). No element moved.
11-14 re-layout to 180 x 170 mm (user: "可以按照你说的调整"): the distance-ratio panel leaves the figure (Methods / Extended Data;
the four-panel version is in paper/history/fig1_with_distance_ratio_20261114), panel b spans the full width and is flatter, the
loop is panel c, rows and gaps are tighter, and the dotted divider of the loop is darker (user request).
11-15 (user): in panel b the self-reference box is raised 0.12 in above the box of the content questions, so that the ability
axis shows it as the higher level. 11-15 audit: the neutral request reads "Please respond based on the above." as in
the runs, and the eight coordinates per layer "carry" the claim (fewer were not tested, so not "needed"). The analogy line and the human half of panel c are marked as only an aid to understanding.
usage: make_overview_nmi_20261114.py OUT_DIR"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Ellipse, Rectangle
from matplotlib.transforms import Bbox

INK, INK2, LINE, PALE = '#0b0b0b', '#52514e', '#c9c7c1', '#f3f2ee'
C = {'A': '#2a78d6', 'U': '#eb6834', 'S': '#1baf7a'}
NAME = {'A': 'Assistant', 'U': 'User', 'S': 'System'}
ORANGE, ORANGE_D, RED = '#d9822b', '#b8651b', '#c0392b'
GREY, BLUEFILL = '#f4f4f2', '#eef4fc'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})


def ptitle(ax, x, y, letter, title):
    """panel letter (8 pt bold) and panel title (7 pt bold)"""
    ax.text(x, y, letter, ha='left', va='top', fontsize=8, fontweight='bold', color=INK)
    ax.annotate(title, xy=(x, y), xytext=(10, -0.6), textcoords='offset points', ha='left', va='top', fontsize=7, fontweight='bold',
                color=INK, annotation_clip=False)

HA, HBD, HC, GAP1, GAP2 = 1.93, 1.61, 2.96, 0.08, 0.08   # heights of a, b, c and the gaps (inches)
FW, FH = 7.08, HA + GAP1 + HBD + GAP2 + HC              # canvas: 180 mm wide, 170 mm high (Nature figure guide)


def aspect(ax):
    """Inches per x unit over inches per y unit, so that rounded corners come out isotropic."""
    fw, fh = ax.figure.get_size_inches()
    pos = ax.get_position()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (pos.width * fw / (x1 - x0)) / (pos.height * fh / (y1 - y0))


def box(ax, x, y, w, h, fc, ec=LINE, lw=0.8, r=0.012, ls='-', iso=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.003,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls,
                                mutation_aspect=aspect(ax) if iso else 1))


def arrow(ax, p, q, col=INK, lw=1.2, ls='-', rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=10, color=col, lw=lw, shrinkA=0, shrinkB=0, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}'))


def panel_axes(fig, rect, top=1.0):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, top)
    ax.axis('off')
    return ax


def panel_in(fig, left, bottom_in, width, height_in):
    """x in fractions of the panel width, y in inches from the panel bottom."""
    return panel_axes(fig, [left, bottom_in / FH, width, height_in / FH], top=height_in)


def node(ax, x, y, w, h, fc, ec, station, title, body, tc, by=None, fs_s=6.0, fs_t=6.4, fs_b=6.0):
    """A station of the loop: the station name (the same on both sides), how it is realized, and a short body."""
    box(ax, x, y, w, h, fc, ec=ec, lw=1.1, iso=True)
    ax.text(x + w / 2, y + h - 0.105, station, ha='center', va='center', fontsize=fs_s, fontweight='bold', color=INK2)
    ax.text(x + w / 2, y + h - 0.225, title, ha='center', va='center', fontsize=fs_t, fontweight='bold', color=tc)
    if body:
        ax.text(x + w / 2, y + (h - 0.30) / 2 + 0.03 if by is None else by, body, ha='center', va='center', fontsize=fs_b, color=INK,
                linespacing=1.15)


def chip(ax, cx, cy, w, h, label, fc, ec, tc=INK, fs=6.0):
    box(ax, cx - w / 2, cy - h / 2, w, h, fc, ec=ec, lw=0.8, r=0.006, iso=True)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fs, color=tc)


def dot(ax, x, y, role, r=0.052, fs=6.0):
    """A role dot as in panel (d). x is a fraction of the panel width and y is in inches, so an ellipse gives a circle."""
    ax.add_patch(Ellipse((x, y), 2 * r / FW, 2 * r, fc=C[role], ec='white', lw=0.8, zorder=3))
    ax.text(x, y, role, ha='center', va='center', fontsize=fs, color='white', fontweight='bold', zorder=4)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(FW, FH))
    a_bottom = FH - HA
    bd_bottom = a_bottom - GAP1 - HBD
    hc = HC

    # (a) probe: two orders, the ordinary-word control, and where each measure of Fig. 2 is read. y in inches.
    ax = panel_in(fig, 0.0, a_bottom, 1.0, HA)
    ptitle(ax, 0.01, HA - 0.01, 'a', 'The probe: the same text in an Assistant, User or System message, in two conversation orders')
    rh = 0.36                                             # row height (inches)
    X0, XR, XF, XB, XE, XQ, XG = 0.115, 0.225, 0.300, 0.395, 0.700, 0.765, 0.885   # slot left edges
    WB = XE - XB - 0.008                                  # message-text width

    def slot(x, y, w, lab, kind='text', fs=6.0):
        fc = {'body': PALE, 'fmt': '#e9f1fb', 'user': '#fdeee6', 'none': 'white'}.get(kind, 'white')
        box(ax, x, y, w, rh, fc, ec=LINE, ls=':' if kind == 'none' else '-', iso=True)
        ax.text(x + w / 2, y + rh / 2, lab, ha='center', va='center', fontsize=fs, color=INK2 if kind == 'none' else INK, linespacing=1.05)

    def roles(x, y, w, which='AUS'):
        hh = (rh - 0.02) / len(which)
        for k, r in enumerate(which):
            yy = y + rh - (k + 1) * hh - 0.004 * k + 0.01
            ax.add_patch(FancyBboxPatch((x, yy), w, hh - 0.012, boxstyle='round,pad=0.002,rounding_size=0.008', fc=C[r], ec='white',
                                        lw=0.8, mutation_aspect=aspect(ax)))
            ax.text(x + w / 2, yy + (hh - 0.012) / 2, NAME[r], ha='center', va='center', fontsize=5.6, color='white', fontweight='bold')

    def body(y, label):
        box(ax, XB, y, WB, rh, PALE, ec=LINE, iso=True)
        ax.text(XB + WB / 2, y + rh * 0.70, label, ha='center', va='center', fontsize=6.0, color=INK)
        for f in (1 / 3, 2 / 3):
            ax.plot([XB + f * WB] * 2, [y + 0.03, y + rh * 0.48], color=INK2, lw=0.6, ls=':')
        for k in range(3):
            ax.text(XB + (k + 0.5) * WB / 3, y + rh * 0.25, ('first third', 'middle', 'last third')[k], ha='center', va='center',
                    fontsize=5.4, color=INK2)

    rows = [(HA - 0.30 - rh, 'Main probe', 'message opens\nthe conversation', 'main'),
            (HA - 0.30 - 2 * rh - 0.10, 'Natural order', 'message answers\na user request', 'natural')]
    for y, name, sub, kind in rows:
        ax.text(0.01, y + rh * 0.66, name, ha='left', va='center', fontsize=7.0, fontweight='bold', color=INK)
        ax.text(0.01, y + rh * 0.24, sub, ha='left', va='center', fontsize=5.6, color=INK2, linespacing=1.05)
        if kind == 'main':
            slot(X0, y, XR - X0 - 0.008, 'no earlier\nturn', 'none')
        else:
            slot(X0, y, XR - X0 - 0.008, 'User: "Please\nwrite a short\ndescription."', 'user', fs=5.6)
        roles(XR, y, XF - XR - 0.008)
        slot(XF, y, XB - XF - 0.008, 'template\nformatting\n(Assistant\nonly)', 'fmt', fs=5.4)
        body(y, 'message text (identical in all three runs)')
        slot(XE, y, XQ - XE - 0.008, 'end of\nmessage', fs=5.6)
        slot(XQ, y, XG - XQ - 0.008, 'User: question\nor "Please respond\nbased on the above."', 'user', fs=5.4)
        slot(XG, y, 1.0 - XG - 0.01, 'generation\nprompt, then\nresponse onset', fs=5.6)
    # the ordinary-word control (Fig. 2 c, e)
    yw = rows[1][0] - rh - 0.10
    ax.text(0.01, yw + rh * 0.66, 'Ordinary words', ha='left', va='center', fontsize=7.0, fontweight='bold', color=INK)
    ax.text(0.01, yw + rh * 0.24, 'control for the size\nof a difference', ha='left', va='center', fontsize=5.6, color=INK2, linespacing=1.05)
    slot(X0, yw, XR - X0 - 0.008, 'as in the\nprobe order', 'none')
    roles(XR, yw, XF - XR - 0.008, which='U')
    slot(XF, yw, XB - XF - 0.008, '"The following\nis the hospital\'s\n| school\'s record:"', 'user', fs=5.2)
    body(yw, 'message text (identical in both runs)')
    ax.text(XE, yw + rh / 2, 'the two runs differ by one word.\n32 pairs in 8 categories (places, times,\nnames, feelings, occupations, organizations,\n'
            'weather, objects)', ha='left', va='center', fontsize=5.6, color=INK2, linespacing=1.1)
    # where the measures are read (letters = columns of Fig. 2)
    yl = yw - 0.07
    ax.text(0.01, yl, 'Read in Fig. 2:', ha='left', va='top', fontsize=6.2, fontweight='bold', color=INK)
    for x, lab in ((0.115, 'geometry: the message text\nand the response onset'),
                   (0.395, 'next-token divergence: every\ntoken of the message text'),
                   (0.700, 'share kept: the last third\nagainst the first third')):
        ax.text(x, yl, lab, ha='left', va='top', fontsize=5.8, color=INK, linespacing=1.1)

    # (b) two levels as an ability ladder, full width (11-14: flatter, the distance-ratio panel is no longer beside it)
    hb = HBD
    bx = panel_in(fig, 0.0, bd_bottom, 1.0, hb)
    ptitle(bx, 0.01, hb - 0.005, 'b', 'Two levels of self')
    bx.text(0.19, hb - 0.022, 'Analogy, only an aid to understanding: corollary discharge, shared by most animals, versus mirror self-recognition, which few pass',
            ha='left', va='top', fontsize=6.0, color=INK2, style='italic')
    ib, ih = 0.05, 0.40
    qh = sh = 0.46
    qb = ib + ih + 0.30
    sb = qb + 0.12                       # 11-15 (user): self-reference sits a little above the content questions
    arrow(bx, (0.035, ib), (0.035, sb + sh - 0.02), col=INK2, lw=1.3)
    bx.text(0.017, (ib + sb + sh) / 2, 'ability (higher =\nmore abstract)', ha='center', va='center', rotation=90, fontsize=6.0, color=INK2,
            linespacing=1.05)
    box(bx, 0.07, ib, 0.92, ih, BLUEFILL, ec=C['A'], lw=1.2, iso=True)
    bx.text(0.53, ib + ih - 0.12, 'Self-index (low level)', ha='center', va='center', fontsize=7.0, fontweight='bold', color=C['A'])
    bx.text(0.53, ib + 0.12, 'An inside/outside displacement over the model\'s own message: Assistant is apart from User and System at every layer, '
            'from the first block on', ha='center', va='center', fontsize=6.3, color=INK)
    box(bx, 0.07, qb, 0.40, qh, 'white', ec=LINE, lw=1.0, iso=True)
    bx.text(0.27, qb + qh - 0.11, 'Questions about the content', ha='center', va='center', fontsize=7.0, fontweight='bold', color=INK2)
    bx.text(0.27, qb + 0.16, 'e.g. the last number: answered alike in every role,\nunchanged when the index is replaced',
            ha='center', va='center', fontsize=5.9, color=INK, linespacing=1.15)
    bx.plot([0.27, 0.27], [ib + ih + 0.005, qb - 0.005], color=INK2, lw=1.1, ls=':')
    bx.text(0.28, (ib + ih + qb) / 2, 'not read ($w$ · $u$ ≈ 0)', ha='left', va='center', fontsize=6.2, color=INK2)
    box(bx, 0.56, sb, 0.43, sh, '#fff6ec', ec=ORANGE, lw=1.2, iso=True)
    bx.text(0.775, sb + sh - 0.11, 'Self-reference (high level)', ha='center', va='center', fontsize=7.0, fontweight='bold', color=ORANGE_D)
    bx.text(0.775, sb + 0.16, '"I wrote this" (0.77).\nUser and System both assigned to the outside', ha='center', va='center',
            fontsize=6.0, color=INK, linespacing=1.15)
    arrow(bx, (0.775, ib + ih + 0.005), (0.775, sb - 0.005), col=ORANGE)
    bx.text(0.767, (ib + ih + sb) / 2, '8 coordinates per layer carry it.\nInstalled, they create the claim',
            ha='right', va='center', fontsize=5.9, color=ORANGE_D, linespacing=1.15)

    # (c) the loop, full width. The two halves have the same four stations in the same places.
    lx = panel_in(fig, 0.0, 0.0, 1.0, hc)
    ptitle(lx, 0.01, hc - 0.005, 'c', 'The same account as a loop: where the source of a memory entry is marked')
    lx.text(0.01, hc - 0.17, 'Left: a simplified human scheme, only an aid to understanding. Right: the language model.',
            ha='left', va='top', fontsize=6.0, color=INK2, style='italic')
    lx.plot([0.5, 0.5], [0.03, hc - 0.29], color=INK2, lw=1.0, ls=(0, (1.2, 2.2)))   # darker (user 11-14)
    for x0, head, col in ((0.0, 'Human: separate pathways tell the source', INK2),
                          (0.5, 'Language model: one channel, a mark tells the source', C['A'])):
        lx.text(x0 + 0.25, hc - 0.35, head, ha='center', va='center', fontsize=7.0, fontweight='bold', color=col)
    nh, gap = 0.98, 0.36                  # node height, gap between the two rows
    r1 = hc - 0.54 - nh                   # bottom of the top row
    r2 = r1 - gap - nh                    # bottom of the bottom row
    m1, m2, g = r1 + nh / 2, r2 + nh / 2, r1 - gap / 2
    W = 0.164                             # node width
    wo, ho = 0.068, 0.50                  # outside box (world / harness)
    ym = r1 + 0.54                        # height of the inlet
    yo = ym - ho / 2
    L6 = dict(fontsize=6.0, linespacing=1.05)

    # ---------- human side ----------
    xo, x1, x2 = 0.0055, 0.1374, 0.3304
    box(lx, xo, yo, wo, ho, PALE, ec=LINE, lw=0.9, iso=True)
    lx.text(xo + wo / 2, ym, 'world', ha='center', va='center', fontsize=6.4, fontweight='bold', color=INK2)
    arrow(lx, (xo + wo + 0.004, ym), (x1 - 0.003, ym), col=INK2)
    lx.text((xo + wo + x1) / 2, ym + 0.05, 'sensors\nencode', ha='center', va='bottom', color=INK2, **L6)
    node(lx, x1, r1, W, nh, 'white', LINE, '1  Store', 'Short-term memory', None, INK2)
    for lab, tag, yy, fc, ec, tc in (('sensed input', 'outside', ym, '#eceae5', LINE, INK2),
                                     ('thoughts', 'inside', r1 + 0.37, BLUEFILL, C['A'], C['A']),
                                     ('command copy', 'inside', r1 + 0.20, BLUEFILL, C['A'], C['A'])):
        chip(lx, x1 + 0.054, yy, 0.092, 0.10, lab, fc, ec)
        lx.text(x1 + 0.108, yy, tag, ha='left', va='center', color=tc, style='italic', **L6)
    xa, xb = x1 + 0.045, x1 + 0.105
    node(lx, x2, r1, W, nh, GREY, LINE, '2  Source mark', 'given by the pathway',
         'senses = outside\nthoughts and command\ncopies = inside', INK2)
    arrow(lx, (x1 + W + 0.004, m1), (x2 - 0.003, m1), col=INK2)
    node(lx, x2, r2, W, nh, '#fff6ec', ORANGE, '3  Readout', 'of the source',
         '"I did this", "I thought\nthis", "that came from\noutside"', ORANGE_D)
    arrow(lx, (x2 + W / 2, r1 - 0.01), (x2 + W / 2, r2 + nh + 0.01), col=ORANGE)
    node(lx, x1, r2, W, nh, GREY, LINE, '4  Output', 'thought and action',
         'thoughts stay inside,\ncommands go to\nthe body', INK2)
    arrow(lx, (x2 - 0.003, m2), (x1 + W + 0.004, m2), col=ORANGE)
    for xx in (xa, xb):
        arrow(lx, (xx, r2 + nh + 0.01), (xx, r1 - 0.01), col=C['A'], lw=1.2)
    lx.text(xa - 0.007, g, 'thoughts', ha='right', va='center', color=C['A'], **L6)
    lx.text(xb + 0.007, g, 'command copies\n(corollary discharge)', ha='left', va='center', color=C['A'], **L6)
    arrow(lx, (x1 - 0.003, r2 + 0.30), (xo + wo / 2, yo - 0.02), col=INK2, lw=1.0, rad=-0.25)
    lx.text(0.004, r2 + 0.62, 'effectors\nchange\nthe world', ha='left', va='center', color=INK2, **L6)

    # ---------- language model side ----------
    xo, x1, x2 = 0.5055, 0.6374, 0.8304
    box(lx, xo, yo, wo, ho, PALE, ec=LINE, lw=0.9, iso=True)
    lx.text(xo + wo / 2, ym + 0.10, 'harness', ha='center', va='center', fontsize=6.4, fontweight='bold', color=INK2)
    lx.text(xo + wo / 2, ym - 0.08, 'user, tools,\nCLI', ha='center', va='center', color=INK2, **L6)
    arrow(lx, (xo + wo + 0.004, ym), (x1 - 0.003, ym), col=INK2)
    lx.text((xo + wo + x1) / 2, ym + 0.05, 'tokens\nin', ha='center', va='bottom', color=INK2, **L6)
    node(lx, x1, r1, W, nh, 'white', LINE, '1  Store', 'Context', None, INK2)
    lx.text(x1 + W / 2, r1 + nh + 0.06, 'available', ha='center', va='center', fontsize=6.0, color=INK2, style='italic')
    for role, yy in (('U', r1 + 0.585), ('S', r1 + 0.455), ('A', r1 + 0.30)):
        dot(lx, x1 + 0.022, yy, role)
        for k in range(8):
            lx.add_patch(Rectangle((x1 + 0.036 + k * 0.0152, yy - 0.036), 0.0115, 0.072, fc=PALE, ec=LINE, lw=0.5))
    lx.text(x1 + W / 2, r1 + 0.115, 'same token form, only\nthe role header differs', ha='center', va='center', color=INK, fontsize=6.0,
            linespacing=1.15)
    node(lx, x2, r1, W, nh, BLUEFILL, C['A'], '2  Source mark', 'self-index $u$', None, C['A'])
    pu, ps, pa = (x2 + 0.030, r1 + 0.455), (x2 + 0.052, r1 + 0.345), (x2 + 0.112, r1 + 0.565)
    arrow(lx, (x2 + 0.054, r1 + 0.43), (x2 + 0.101, r1 + 0.535), col=C['A'], lw=1.2)
    lx.text(x2 + 0.066, r1 + 0.535, '$u$', ha='center', va='center', fontsize=7.0, color=C['A'])
    for role, p in (('U', pu), ('S', ps), ('A', pa)):
        dot(lx, p[0], p[1], role)
    lx.text(x2 + 0.124, r1 + 0.565, 'own', ha='left', va='center', color=C['A'], style='italic', **L6)
    lx.text(x2 + 0.066, r1 + 0.345, 'outside', ha='left', va='center', color=INK2, style='italic', **L6)
    lx.text(x2 + W / 2, r1 + 0.125, 'the role writes it into\nthe representation', ha='center', va='center', color=INK, fontsize=6.0,
            linespacing=1.15)
    arrow(lx, (x1 + W + 0.004, m1), (x2 - 0.003, m1), col=INK2)
    node(lx, x2, r2, W, nh, '#fff6ec', ORANGE, '3  Readout', 'self-reference',
         '$w$ aligned with $u$\nreads the mark back:\n"you wrote it" (0.77,\nothers at most 0.07)', ORANGE_D)
    lx.text(x2 + W / 2, r2 - 0.07, 'accessible', ha='center', va='center', fontsize=6.0, color=ORANGE_D, style='italic')
    arrow(lx, (x2 + W / 2, r1 - 0.01), (x2 + W / 2, r2 + nh + 0.01), col=ORANGE)
    lx.text(x2 + W / 2 - 0.008, g, 'replacing $u$ removes\nor installs the claim', ha='right', va='center', color=RED, **L6)
    node(lx, x1, r2, W, nh, GREY, LINE, '4  Output', 'tokens',
         'what it writes stays in\nthe context and leaves\nthrough the harness', INK2)
    arrow(lx, (x2 - 0.003, m2), (x1 + W + 0.004, m2), col=ORANGE)
    arrow(lx, (x1 + 0.05, r2 + nh + 0.01), (x1 + 0.05, r1 - 0.01), col=C['A'], lw=1.3)
    lx.text(x1 + 0.043, g, 'kept as an\nAssistant turn', ha='right', va='center', color=C['A'], **L6)
    arrow(lx, (x1 - 0.003, r2 + 0.30), (xo + wo / 2, yo - 0.02), col=INK2, lw=1.0, rad=-0.25)
    lx.text(0.504, r2 + 0.62, 'tokens out:\nreplies,\ntool calls', ha='left', va='center', color=INK2, **L6)

    fig.savefig(f'{out_dir}/fig1_overview.pdf')
    fig.savefig(f'{out_dir}/fig1_overview.png', dpi=170)
    print('saved', FW, round(FH, 3), 'hc', round(hc, 3))


if __name__ == '__main__':
    main(*sys.argv[1:])
