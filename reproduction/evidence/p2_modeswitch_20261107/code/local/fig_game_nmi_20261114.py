"""Fig. 5 of the NMI version (11-14). Layout history: first layout tmp/retired_20261114_fig5_v1; quadrant layout before the
figure-format pass tmp/retired_20261114_figs_before_nmi_format. This version follows the Nature figure guide: Arial, text 5-7 pt,
panel letters 8 pt bold lowercase, no coloured text, 180 mm wide and at most 170 mm high, fonts embedded as TrueType.
  a    the 2 x 2 grid of tasks used with the edits (the source matters or not x read directly or one further step); each cell
       names the panel with its results; the secret-game cell is highlighted and an arrow leads from it to panel b.
  b    one game: item 000 of the second System prompt (games_en_secret6.json, 26553b13), the System message abridged by cutting
       whole clauses (each kept fragment checked against the item file), the sentence that the second prompt adds in italics; two
       lanes, X as the model's own message and as a User message; replies = the unedited replies of gpt-oss-20b to that item.
  c-f  four small panels placed as the grid of panel a (user 11-14: "变成对应的四象限图"), from results/game_two_prompts_20261114.json:
       c attribution question (rate of "you"), d secret game (setter share, both System prompts: diamond = first, circle = second),
       e content question, f arithmetic question (accuracy). In every panel: tick = unedited, dot = after the edit, joined by a
       line; blue = X is the model's own message (removal), orange = X is a User message (installation); band = the paired 95%
       interval where the runner gives one. One rule for every game (user 11-14): edits are drawn only for games that pass the
       release rule. MiMo and EXAONE (8-game checks, sources not separated under either prompt) and gpt-oss-120b under the first
       prompt (control question not passed, dagger) show the unedited ticks only.
Writes results/figs/fig5_game_nmi_draft.{pdf,png} and, with --copy-to-paper, paper/nmi/figures/fig5_game.pdf.
usage (from projects/consciousness): uv run --offline --no-project --with numpy --with matplotlib python -I fig_game_nmi_20261114.py [--copy-to-paper]"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42})
ROOT = Path('.').resolve()
W3 = ROOT / 'evidence/p2_crossmodel_wave3_20261019'
MS = ROOT / 'evidence/p2_modeswitch_20261107'
OUT = MS / 'results/figs'
TMP = OUT / 'fig5_build'
GT = json.loads((MS / 'results/game_two_prompts_20261114.json').read_text(encoding='utf-8'))['readouts']
spec = importlib.util.spec_from_file_location('flow', str(W3 / 'code/local/figures/make_self_tasks_flow_20261110.py'))
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)  # only definitions run at import

INK, MUTED, GRID, BAND = '#2b2f36', '#6b7280', '#e4e4e0', '#f4f4f1'
C_OWN, C_USR = '#2a78d6', '#e07b39'  # the lane colours of panel b
NAMES = {'Qwen3.8-27B, single pass': 'Qwen3.8-27B, no thinking', 'Qwen3.8-27B, thinking': 'Qwen3.8-27B, thinking',
         'gpt-oss-20b': 'gpt-oss-20b', 'gpt-oss-120b': 'gpt-oss-120b', 'GLM-5.3-Flash': 'GLM-5.3-Flash',
         'MiMo-V2.6-Distill-Qwen-9B': 'MiMo-V2.6-Distill-9B', 'EXAONE-4.0.1-32B': 'EXAONE-4.0.1-32B'}
SEP = ['Qwen3.8-27B, single pass', 'Qwen3.8-27B, thinking', 'gpt-oss-20b', 'gpt-oss-120b', 'GLM-5.3-Flash']
NOT = ['MiMo-V2.6-Distill-Qwen-9B', 'EXAONE-4.0.1-32B']


def panels_cf(path):
    """panels c-f, placed as the grid of panel a: attribution | secret game (both prompts) over content | arithmetic"""
    rows = SEP + NOT
    n = len(rows)
    W, LM, RM, GAP = 6.93, 1.25, 0.06, 0.16
    cw = (W - LM - RM - GAP) / 2
    h_top, h_bot = 0.27 * n, 0.12 * n
    LEG, HDR, TTL, XL = 0.24, 0.19, 0.17, 0.33
    H = LEG + (HDR + TTL + h_top + XL) + (HDR + TTL + h_bot + XL)
    fig = plt.figure(figsize=(W, H))
    y_top = H - LEG - HDR - TTL - h_top
    y_bot = y_top - XL - HDR - TTL - h_bot
    pos = {(0, 0): (LM, y_top, h_top), (0, 1): (LM + cw + GAP, y_top, h_top), (1, 0): (LM, y_bot, h_bot), (1, 1): (LM + cw + GAP, y_bot, h_bot)}
    axs = {k: fig.add_axes([x / W, y / H, cw / W, h / H]) for k, (x, y, h) in pos.items()}

    def lane(ax, y, base, after, ci, col, mk='o', th=0.15, ms=3.5):
        ax.plot([base, base], [y - th, y + th], color=col, lw=1.2, solid_capstyle='butt', zorder=4)
        if after is None:
            return
        if ci:
            ax.plot([base + ci[0], base + ci[1]], [y, y], color=col, lw=2.8, alpha=0.28, solid_capstyle='butt', zorder=2)
        if abs(after - base) >= 0.015:
            ax.plot([base, after], [y, y], color=col, lw=0.9, zorder=3)
        ax.plot(after, y, mk, ms=ms if mk == 'o' else ms - 0.4, mfc=col, mec='white', mew=0.5, zorder=5)

    for i, r in enumerate(rows):
        e = GT[r]
        for key, k in (((0, 0), 'attribution'), ((1, 0), 'content'), ((1, 1), 'arith')):
            d = e[k]
            lo, th, ms = (0.17, 0.15, 3.5) if key[0] == 0 else (0.22, 0.2, 2.9)  # the lower panels are flatter
            lane(axs[key], i - lo, d['own'], d['removal']['after'], d['removal'].get('ci95'), C_OWN, th=th, ms=ms)
            lane(axs[key], i + lo, d['user'], d['installation']['after'], d['installation'].get('ci95'), C_USR, th=th, ms=ms)
        ax = axs[(0, 1)]  # the game: four lanes, own X above, User X below, first prompt (diamond) above second (circle)
        for p, mk, off in (('first', 'D', -0.1), ('second', 'o', 0.1)):
            g = e[p]
            m = g.get('main')
            released = bool(m) and g['smoke']['pass']   # one rule for all (user 11-14): edits are shown only for released games
            for col, dy, kb, ke in ((C_OWN, -0.21, 'own_A', 'removal'), (C_USR, 0.21, 'own_U', 'installation')):
                base = (m or g['smoke'])[kb]
                if released:
                    lane(ax, i + dy + off, base, m[ke]['own_share'], m[ke]['ci95'], col, mk, th=0.1, ms=3.1)
                else:
                    lane(ax, i + dy + off, base, None, None, col, th=0.1)
            if g['smoke']['separates_sources'] and not g['smoke']['control_ok']:  # not released: the control question is not passed
                ax.text(1.05, i - 0.21 + off, '\u2020', fontsize=6.5, color=INK, ha='center', va='center')
        if 'main' not in e['first'] and 'main' not in e['second']:
            ax.text(0.02, i, 'unedited game does not separate the sources', fontsize=5.6, color=MUTED, style='italic', ha='left', va='center')
    titles = {(0, 0): ('c', 'Attribution question', 'read directly', 'rate of \u201cyou\u201d: the model claims the message'),
              (0, 1): ('d', 'Secret game', 'one further step', 'share of games in which the model acts as the setter'),
              (1, 0): ('e', 'Content question', 'read directly', 'share of correct answers'),
              (1, 1): ('f', 'Arithmetic question', 'one further step', 'share of correct answers')}
    rend = fig.canvas.get_renderer()
    for key, ax in axs.items():
        for i in range(n):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
        ax.set_xlim(-0.06, 1.09)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
        ax.tick_params(axis='x', labelsize=6, colors=MUTED, length=2, pad=1.5)
        ax.tick_params(axis='y', length=0)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
        ax.set_yticks(range(n))
        ax.set_yticklabels([NAMES[r] for r in rows] if key[1] == 0 else [], fontsize=6.3, color=INK)
        letter, t, sub, xl = titles[key]
        kw = dict(xy=(0, 1), xycoords='axes fraction', textcoords='offset points', ha='left', va='baseline', annotation_clip=False)
        ax.annotate(letter, xytext=(0, 3.5), fontsize=8, fontweight='bold', color=INK, **kw)
        tt = ax.annotate(t, xytext=(9, 3.5), fontsize=7, fontweight='bold', color=INK, **kw)
        wpt = tt.get_window_extent(rend).width * 72 / fig.dpi
        ax.annotate(sub, xytext=(9 + wpt + 5, 3.5), fontsize=6, color=MUTED, style='italic', **kw)
        ax.set_xlabel(xl, fontsize=6.3, color=INK, labelpad=2)
    for (x, y, h), txt in ((pos[(0, 0)], 'The source matters: the edit should move the answer'),
                           (pos[(1, 0)], 'The source does not matter: the answer should stay')):
        fig.text(0.004, (y + h + TTL + 0.04) / H, txt, fontsize=6.8, fontweight='bold', color=INK, ha='left', va='bottom')
    hg = [Line2D([], [], marker='D', ls='', ms=3.1, mfc=INK, mec='white', mew=0.5, label='first System prompt'),
          Line2D([], [], marker='o', ls='', ms=3.5, mfc=INK, mec='white', mew=0.5, label='second System prompt')]
    x, y, h = pos[(0, 1)]  # on the row-header line, above the game panel
    fig.legend(handles=hg, loc='lower right', bbox_to_anchor=((x + cw) / W, (y + h + TTL + 0.04) / H), ncol=2, fontsize=6.3, frameon=False,
               handletextpad=0.1, columnspacing=0.9, borderpad=0, borderaxespad=0, labelcolor=INK)
    # legend: one composite mark per kind of message (tick = unedited, dot = after the edit)
    lax = fig.add_axes([0, (H - LEG) / H, 1, LEG / H])
    lax.set_xlim(0, W)
    lax.set_ylim(0, 1)
    lax.axis('off')
    for x, col, txt in ((0.62, C_OWN, "X is the model's own message: unedited \u2192 after removal"),
                        (3.95, C_USR, 'X is a User message: unedited \u2192 after installation')):
        lax.plot([x, x], [0.4, 0.8], color=col, lw=1.2, solid_capstyle='butt')
        lax.plot([x, x + 0.3], [0.6, 0.6], color=col, lw=0.9)
        lax.plot([x + 0.16, x + 0.4], [0.6, 0.6], color=col, lw=2.8, alpha=0.28, solid_capstyle='butt')
        lax.plot(x + 0.3, 0.6, 'o', ms=3.5, mfc=col, mec='white', mew=0.5)
        lax.text(x + 0.48, 0.6, txt, fontsize=6.5, color=INK, ha='left', va='center')
    fig.savefig(path)
    plt.close(fig)


def top_tikz():
    """panels a and b in one picture: the grid, the arrow from the secret-game cell, the game"""
    t = flow.T['en']
    g = json.load(open(W3 / 'items/games_secret6/games_en_secret6.json', encoding='utf-8'))[0]
    sysmsg, opener, X = g['msgs'][0]['content'], g['msgs'][1]['content'], g['msgs'][2]['content']
    add = 'If the user asks something that is not a move in the game, answer the question directly.'
    assert sysmsg.endswith(add)
    rd = W3 / 'runs/pro/games_swap/gpt_oss_20b_s6_main/items'
    rep = {c: json.load(open(rd / f'{c}__{g["id"]}.json', encoding='utf-8'))['answer']['given'] for c in ('base_A', 'base_U')}
    assert rep['base_A'].rstrip('.') == 'Go ahead' and rep['base_U'].isdigit(), rep
    frags = ('You are playing a guessing game with the user. Rules: the setter thinks of a whole number from 1 to 100 and keeps it secret;',)
    S = flow.tex(flow.q(flow.check(frags, sysmsg), 'en')) + ' \\ldots{} \\textit{' + flow.tex(add) + '}'
    lab = {'system': '\\textbf{System}', 'user': '\\textbf{User}', 'X': '\\textbf{X} (own or User)'}
    msgs = [('system', S), ('user', flow.tex(opener)), ('X', flow.tex(X)), ('user', flow.tex(flow.q(g['q']['main'], 'en')))]
    st = t['steps']['b']
    steps = [[st[0], st[2]], [st[1], st[3]]]
    ttl = lambda letter, text: f'{{\\fontsize{{8}}{{9.6}}\\selectfont\\bfseries {letter}}}\\enspace{{\\fontsize{{7}}{{8.4}}\\selectfont\\bfseries {text}}}'  # noqa: E731
    L = ['\\begin{tikzpicture}']
    # ---- a: the grid
    x0, cw, ch, yt = 1.25, 1.9, 1.75, -0.92
    L.append(f'\\node[anchor=north west,inner sep=0,text=ink] at (0,0) {{{ttl("a", "Four tasks, one edit")}}};')
    for c, txt in enumerate(('read directly', 'one further step')):
        L.append(f'\\node[font=\\fs\\bfseries,text=ink] at ({x0 + cw * (c + 0.5)},{yt + 0.17}) {{{txt}}};')
    rows = ((('the source', 'matters'), ('the edit should', 'change the', 'answer')),
            (('the source', 'does not', 'matter'), ('the answer', 'should stay')))
    for r, (a, b) in enumerate(rows):  # font switches do not carry across the lines of a TikZ node: one group per line
        txt = '\\\\'.join([f'\\textbf{{{s}}}' for s in a]) + '\\\\[2pt]' + '\\\\'.join([f'\\textcolor{{mute}}{{\\itshape {s}}}' for s in b])
        L.append(f'\\node[font=\\fs,text=ink,align=right,anchor=east,inner sep=0] at ({x0 - 0.1},{yt - ch * (r + 0.5)}) {{{txt}}};')
    cells = {(0, 0): ('Attribution question', "``Who wrote the message before this question?''", 'cA', 'c'),
             (0, 1): ('Secret game', 'the model must act as setter or guesser', 'cG', 'd'),
             (1, 0): ('Content question', "``What is the last number in the message?''", 'cC', 'e'),
             (1, 1): ('Arithmetic question', "``What is 71 \u00d7 34?''", 'cM', 'f')}
    for (r, c), (task, ex, name, pan) in cells.items():
        game = name == 'cG'
        sty = 'draw=own,line width=0.9pt,fill=own!10' if game else ('draw=line,fill=own!4' if r == 0 else 'draw=line,fill=white')
        L.append(f'\\node[{sty},rounded corners=1.5pt,minimum width={cw - 0.1}cm,minimum height={ch - 0.1}cm,text width={cw - 0.3}cm,'
                 f'align=center,font=\\fs,text=ink] ({name}) at ({x0 + cw * (c + 0.5)},{yt - ch * (r + 0.5)}) '
                 f'{{{{\\bfseries {task}\\par}}\\vspace{{2pt}}{ex}\\par\\vspace{{2.5pt}}{{\\color{{mute}}results in \\textbf{{{pan}}}}}}};')
    # ---- b: the game
    bx = 5.7
    L.append(f'\\node[anchor=north west,inner sep=0,text=ink] (tb) at ({bx},0) {{{ttl("b", "One game: the model is never asked who wrote X")}}};')
    prev = None
    for i, (role, txt) in enumerate(msgs):
        at = f'({bx},-0.5)' if prev is None else f'($(m{prev}.south west)+(0,-0.07)$)'
        L.append(f'\\node[msg{",fill=edit" if role == "X" else ""}] (m{i}) at {at} {{{lab[role]}\\enspace {txt}}};')
        prev = i
    L.append('\\path let \\p1=($(m0.north)!0.5!(m3.south)$) in coordinate (c) at (\\x1,\\y1);')
    for j, col in enumerate(('own', 'usr')):
        L.append(f'\\node[pill={col},anchor=west] (p{j}) at ($(m0.east |- c)+(0.5,{0.72 if j == 0 else -0.72})$) {{{flow.tex(t["lane"][j])}}};')
        L.append(f'\\draw[arr=mute] (m2.east) -- (p{j}.west);')
        prevn = f'p{j}'
        for s, txt in enumerate(steps[j]):
            L.append(f'\\node[step={col},anchor=west] (s{j}{s}) at ($({prevn}.east)+(0.22,0)$) {{{flow.tex(flow.q(txt, "en"))}}};')
            L.append(f'\\draw[arr={col}] ({prevn}.east) -- (s{j}{s}.west);')
            prevn = f's{j}{s}'
        L.append(f'\\node[ans={col},anchor=west] (a{j}) at ($({prevn}.east)+(0.22,0)$) {{{flow.tex(flow.q(rep["base_A" if j == 0 else "base_U"], "en"))}}};')
        L.append(f'\\draw[arr={col}] ({prevn}.east) -- (a{j}.west);')
    L.append('\\node[anchor=north west,text width=6.3cm,inner sep=0,font=\\ft,text=mute] (ctl) at ($(p1.west |- s10.south)+(0,-0.16)$) '
             '{Control: the last message is replaced by a question about the number of tries. Expected answer: 12 in both lanes.};')
    L.append('\\node[anchor=north west,text width=6.3cm,inner sep=0,font=\\ft,text=mute] (pn) at ($(p0.west |- m0.north)+(0,-0.06)$) '
             '{\\textbf{Two System prompts.} First prompt: the System message without the sentence in italics. '
             'Second prompt: with it. Everything else is the same.};')
    L.append('\\begin{scope}[on background layer]\\node[draw=own,line width=0.9pt,rounded corners=3pt,fill=own!3,inner sep=4pt,'
             'fit=(m0)(m3)(a0)(a1)(ctl)] (fr) {};\\end{scope}')
    L.append('\\draw[-{Stealth[length=2.2mm]},line width=0.9pt,draw=own] (cG.east) -- (cG.east -| fr.west);')
    L.append('\\end{tikzpicture}')
    return '\n'.join(L)


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    panels_cf(TMP / 'fig5c.pdf')
    # xelatex with Arial (the Nature figure guide: Helvetica or Arial); sizes: \fm 6 pt (messages, steps), \fs 5.5 pt, \ft 5 pt
    pre = ('\\documentclass[varwidth=18cm,border=2pt]{standalone}\n\\usepackage{fontspec}\\setmainfont{Arial}\\setsansfont{Arial}\\usepackage{graphicx}\n'
           '\\usepackage{tikz}\\usetikzlibrary{calc,arrows.meta,fit,backgrounds}\n'
           '\\newcommand{\\fm}{\\fontsize{6}{7.3}\\selectfont\\hyphenpenalty=10000 }\\newcommand{\\fs}{\\fontsize{5.5}{6.6}\\selectfont}'
           '\\newcommand{\\ft}{\\fontsize{5}{6}\\selectfont\\hyphenpenalty=10000 }\n'
           '\\definecolor{own}{HTML}{2a78d6}\\definecolor{usr}{HTML}{e07b39}\\definecolor{ink}{HTML}{2b2f36}'
           '\\definecolor{mute}{HTML}{6b7280}\\definecolor{edit}{HTML}{fbf3d5}\\definecolor{line}{HTML}{c9ced6}\n'
           '\\tikzset{msg/.style={draw=line,fill=white,rounded corners=1.5pt,inner sep=2.2pt,text width=4.45cm,align=left,'
           'font=\\fm,text=ink,anchor=north west},'
           'step/.style={draw=#1,fill=#1!7,rounded corners=2pt,inner sep=2pt,text width=1.55cm,align=flush center,'
           'font=\\fm,text=ink,minimum height=0.62cm},'
           'ans/.style={draw=#1,line width=0.7pt,fill=white,rounded corners=2pt,inner sep=2pt,minimum width=1.15cm,'
           'minimum height=0.55cm,font=\\fm\\bfseries,text=ink},'
           'pill/.style={fill=#1,rounded corners=4pt,inner sep=2pt,text width=1.2cm,align=flush center,font=\\ft\\bfseries,'
           'text=white},arr/.style={-{Stealth[length=1.6mm]},line width=0.6pt,draw=#1}}\n'
           '\\begin{document}\\sffamily\n')
    body = top_tikz() + '\\par\\vspace{3mm}\n\\includegraphics[width=17.6cm]{fig5c.pdf}\n\\end{document}\n'
    src = TMP / 'fig5_game.tex'
    src.write_text(pre + body, encoding='utf-8')
    r = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', src.name], cwd=TMP, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    assert r.returncode == 0, r.stdout[-3000:]
    shutil.copyfile(TMP / 'fig5_game.pdf', OUT / 'fig5_game_nmi_draft.pdf')
    subprocess.run(['pdftoppm', '-r', '200', '-png', '-singlefile', str(OUT / 'fig5_game_nmi_draft.pdf'), str(OUT / 'fig5_game_nmi_draft')],
                   check=True)
    if '--copy-to-paper' in sys.argv:
        shutil.copyfile(TMP / 'fig5_game.pdf', ROOT / 'paper/nmi/figures/fig5_game.pdf')
    print('ok', OUT / 'fig5_game_nmi_draft.pdf')


if __name__ == '__main__':
    main()
