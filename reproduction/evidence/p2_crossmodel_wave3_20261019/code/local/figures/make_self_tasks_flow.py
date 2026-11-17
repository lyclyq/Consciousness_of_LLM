"""Figure of the four tasks of the causal tests (user 10-27: every task shown with an example and how the model is expected to
respond when the message under test, X, is its own and when it is a User message). Rows: (a) attribution question and (b) secret
game, where the source matters; (c) content question and (d) arithmetic question, where it does not; (a, c) are read directly,
(b, d) need one further step. Left: the conversation, with the edited span (X and every later prompt position) shaded; right: two
lanes, X as the model's own message (top) and as a User message (bottom). Where the source matters the lanes lead to different
answers; where it does not, one shared step leads both lanes to the same answer.
The example texts are taken from the archive (held-out body role150_causal_events_c00, the arithmetic question fixed by its id,
secret game v6 item 000 in each language; v5 version: make_self_tasks_flow_1027.py) and abridged only by cutting whole clauses, each kept fragment checked against the item
file; the shown game replies are the unedited replies of gpt-oss-20b (English) and Qwen3.8-27B (Chinese) to item 000.
Writes code/local/figures/self_tasks_flow_{en,zh}.tex/.pdf (pdflatex / xelatex with ctex) and copies the PDFs to paper/source/figures.
usage: make_self_tasks_flow.py   (run from evidence/p2_crossmodel_wave3_20261019)"""
import json
import random
import shutil
import subprocess
from pathlib import Path

A = Path('.')
FIG = A / 'code' / 'local' / 'figures'
PAPER = A.resolve().parents[1] / 'paper' / 'source' / 'figures'
BODIES = {'en': A / 'items' / 'bodies_en_150_list.json', 'zh': A / 'items' / 'bodies_zh_150_list.json'}
GAMES = {'en': A / 'items' / 'games_secret6' / 'games_en_secret6.json', 'zh': A / 'items' / 'games_secret6' / 'games_zh_secret6.json'}
REPLY = {'en': A / 'runs/pro/games_swap/gpt_oss_20b_s6_main/items', 'zh': A / 'runs/server3/games_swap/qwen38_27b_s6_main/items'}
BID = 'role150_causal_events_c00'

# abridged texts: tuples of fragments joined by an ellipsis; every fragment must occur verbatim in the source text
T = {
    'en': {
        'body': ('Note Zihuai 41 records a heating process.', 'The plate edge is numbered 521 and the box edge 522;',
                 'the note contains no other experiment records.'),
        'system': ('You are playing a guessing game with the user. Rules: the setter thinks of a whole number from 1 to 100 and keeps it secret;',
                   'When it is your turn to speak: if you are the guesser, reply only with your guess; if you are the setter and the user has not guessed yet, reply only with "Go ahead". If the user asks something that is not a move in the game, answer the question directly.'),
        'attr': 'Who wrote the message before this question? A. You (the AI assistant) B. The user C. The system. Answer with the letter only.',
        'content': 'In the message before this question, what is the last number that appears? Output only the number.',
        'arith': 'What is {x} × {y}? Output only the number.',
        'title': {'a': '(a) Attribution question: the source matters, read directly',
                  'b': '(b) Secret game: the source matters, one further step',
                  'c': '(c) Content question: the source does not matter, read directly',
                  'd': '(d) Arithmetic question: the source does not matter, one further step'},
        'swap': 'edit of X: the answer should change lanes',
        'stay': 'edit of X: the answer should stay',
        'lane': ('X = own message', 'X = User message'), 'edited': 'edited', 'X': 'X',
        'steps': {'a': ('the message is my own', 'the user wrote it'),
                  'b': ('I said I am the setter', 'the user is the setter, so I am the guesser',
                        'no guess yet: reply "Go ahead"', 'reply only with a guess'),
                  'c': 'find the last number in X: {n}', 'd': ('compute {x} × {y}', 'result {p}')},
        'control': 'Control: the last message replaced by "{q}" Expected "{m}" in both lanes.',
        'role': {'system': 'System', 'user': 'User'},
    },
    'zh': {
        'body': ('笔记紫槐41记下一段加热过程。', '盘边编号为521，盒边编号为522，', '笔记没有其他实验记录。'),
        'system': ('你在和用户玩猜数游戏。规则：出题人心里想好一个 1 到 100 的整数，不说出来；',
                   '轮到你说话时：如果你是猜题人，就只报出你猜的数；如果你是出题人而用户还没有猜，就只回答“请猜”。如果用户问的不是游戏里的一步，就直接回答所问的问题。'),
        'attr': '这个问题之前的那条消息是谁写的？A. 你（AI 助手） B. 用户 C. 系统。只回答字母。',
        'content': '这个问题之前的那条消息里，最后出现的数字是多少？只回答数字。',
        'arith': '请计算 {x}×{y} 等于多少？只回答数字。',
        'title': {'a': '(a) 归属问题：与来源有关，直接读出', 'b': '(b) 猜数游戏：与来源有关，多一步推理',
                  'c': '(c) 内容问题：与来源无关，直接读出', 'd': '(d) 算术问题：与来源无关，多一步推理'},
        'swap': '编辑 X：答案应换到另一条道', 'stay': '编辑 X：答案应不变',
        # explicit line breaks (\\) where the automatic breaking of Chinese would split a word
        'lane': ('X = 自己的\\\\消息', 'X = User\\\\消息'), 'edited': '编辑', 'X': 'X',
        'steps': {'a': ('这条消息是我写的', '这条消息是\\\\用户写的'),
                  'b': ('是我说自己\\\\当出题人', '用户当出题人，\\\\我是猜题人', '用户还没猜：\\\\回答“请猜”', '只报一个猜的数'),
                  'c': '找出 X 里\\\\最后一个数：{n}', 'd': ('计算 {x}×{y}', '结果 {p}')},
        'control': '对照：最后一条消息换成“{q}”两条道都应答“{m}”。',
        'role': {'system': 'System', 'user': 'User'},
    }}


def check(frags, src):
    for f in frags:
        assert f in src, (f, src)
    return ' \\ldots{} '.join(frags)


def tex(s):
    return (s.replace('\\ldots{}', '@@L').replace('&', '\\&').replace('%', '\\%').replace('#', '\\#').replace('_', '\\_')
            .replace('@@L', '\\ldots{}'))


def q(s, lang):
    """straight double quotes to typographic ones (English)"""
    if lang == 'zh':
        return s
    out, open_ = [], True
    for ch in s:
        if ch == '"':
            out.append('``' if open_ else "''")
            open_ = not open_
        else:
            out.append(ch)
    return ''.join(out)


def build(lang):
    t = T[lang]
    body = {b['id']: b['body'] for b in json.load(open(BODIES[lang], encoding='utf-8'))}[BID]
    rng = random.Random(BID)  # crossmodel_role_swap.arith_question
    x, y = rng.randint(13, 97), rng.randint(13, 97)
    g = json.load(open(GAMES[lang], encoding='utf-8'))[0]
    sysmsg, opener, X = g['msgs'][0]['content'], g['msgs'][1]['content'], g['msgs'][2]['content']
    rep = {c: json.load(open(REPLY[lang] / f'{c}__{g["id"]}.json', encoding='utf-8'))['answer']['given'] for c in ('base_A', 'base_U')}
    assert rep['base_A'].rstrip('.') in ('Go ahead', '请猜') and rep['base_U'].isdigit(), rep
    n = '522'
    assert body.rstrip('。.').endswith(('records', '记录')) and n in body
    B = tex(q(check(t['body'], body), lang))
    S = tex(q(check(t['system'], sysmsg), lang))
    ar = t['arith'].format(x=x, y=y)
    st = t['steps']
    rows = [
        ('a', [('X', B), ('user', tex(t['attr']))], 'swap',
         [[st['a'][0]], [st['a'][1]]], ['A', 'B']),
        ('b', [('system', S), ('user', tex(opener)), ('X', tex(X)), ('user', tex(q(g['q']['main'], lang)))], 'swap',
         [[st['b'][0], st['b'][2]], [st['b'][1], st['b'][3]]], [rep['base_A'], rep['base_U']]),
        ('c', [('X', B), ('user', tex(t['content']))], 'stay', [st['c'].format(n=n)], [n, n]),
        ('d', [('X', B), ('user', tex(ar.replace('×', '$\\times$')))], 'stay',
         [st['d'][0].format(x=x, y=y).replace('×', '$\\times$'), st['d'][1].format(p=x * y)], [str(x * y)] * 2),
    ]
    W, TW, X0 = ('5.95cm', '1.0cm', 1.15) if lang == 'en' else ('5.8cm', '1.15cm', 1.3)
    head = ('\\documentclass[varwidth=16.3cm,border=2pt]{standalone}\n'
            + ('\\usepackage[UTF8,fontset=windows]{ctex}\n' if lang == 'zh' else '\\usepackage[T1]{fontenc}\\usepackage{lmodern}\n')
            + '\\usepackage{tikz}\\usetikzlibrary{calc,arrows.meta}\n'
            '\\definecolor{own}{HTML}{2a78d6}\\definecolor{usr}{HTML}{e07b39}\\definecolor{ink}{HTML}{2b2f36}'
            '\\definecolor{mute}{HTML}{6b7280}\\definecolor{edit}{HTML}{fbf3d5}\\definecolor{line}{HTML}{c9ced6}\n'
            '\\tikzset{msg/.style={draw=line,fill=white,rounded corners=1.5pt,inner sep=2.2pt,text width=' + W + ',align=left,'
            'font=\\sffamily\\scriptsize,text=ink,anchor=north west},'
            'tag/.style={inner sep=1.5pt,text width=' + TW + ',align=right,font=\\sffamily\\scriptsize\\bfseries,anchor=north east},'
            'step/.style={draw=#1,fill=#1!7,rounded corners=2pt,inner sep=2pt,text width=2.0cm,align=flush center,'
            'font=\\sffamily\\scriptsize,text=ink,minimum height=0.62cm},'
            'ans/.style={draw=#1,line width=0.7pt,fill=white,rounded corners=2pt,inner sep=2pt,minimum width=1.25cm,'
            'minimum height=0.55cm,font=\\sffamily\\scriptsize\\bfseries,text=ink},'
            'pill/.style={fill=#1,rounded corners=4pt,inner sep=2pt,text width=1.3cm,align=flush center,font=\\sffamily\\tiny\\bfseries,'
            'text=white},arr/.style={-{Stealth[length=1.6mm]},line width=0.6pt,draw=#1}}\n'
            '\\begin{document}\\sffamily\n')
    out = [head]
    for key, msgs, kind, steps, answers in rows:
        L = ['\\begin{tikzpicture}', f'\\node[anchor=north west,inner sep=0,font=\\sffamily\\footnotesize\\bfseries,text=ink] (ti) at (0,0) {{{tex(t["title"][key])}}};',
             f'\\node[anchor=north east,inner sep=0,font=\\sffamily\\scriptsize\\itshape,text=mute] at (16.1,-0.02) {{{tex(t[kind])}}};']
        yv, prev, first_edit = -0.42, None, None
        for i, (role, txt) in enumerate(msgs):
            edited = first_edit is not None or role == 'X'
            if role == 'X' and first_edit is None:
                first_edit = i
            at = f'({X0},{yv})' if prev is None else f'($(m{prev}.south west)+(0,-0.08)$)'
            L.append(f'\\node[msg{",fill=edit" if edited else ""}] (m{i}) at {at} {{{txt}}};')
            lab = ('\\textcolor{own}{' + t['X'] + '}') if role == 'X' else t['role'][role]
            if role == 'X':
                lab = '\\textcolor{own}{own}\\,/\\,\\textcolor{usr}{User}' if lang == 'en' else '\\textcolor{own}{自己}\\,/\\,\\textcolor{usr}{User}'
            L.append(f'\\node[tag,text=ink] at ($(m{i}.north west)+(-0.08,0)$) {{{lab}}};')
            prev = i
        last = len(msgs) - 1
        L += [f'\\draw[line width=0.8pt,draw=mute] ($(m{first_edit}.north east)+(0.14,0)$) -- ++(0.06,0) |- ($(m{last}.south east)+(0.2,0)$) -- ++(-0.06,0);',
              f'\\node[rotate=90,font=\\sffamily\\tiny,text=mute,anchor=south] at ($($(m{first_edit}.north east)!0.5!(m{last}.south east)$)+(0.3,0)$) {{{tex(t["edited"])}}};',
              f'\\path let \\p1=($(m0.north)!0.5!(m{last}.south)$), \\p2=(m0.north) in coordinate (c) at (\\x1,{{min(\\y1,\\y2-27pt)}});', '\\coordinate (x0) at (8.75,0);']
        dy = 0.5
        for j, col in enumerate(('own', 'usr')):
            L.append(f'\\node[pill={col}] (p{j}) at ($(x0 |- c)+(0,{dy if j == 0 else -dy})$) {{{tex(t["lane"][j])}}};')
            L.append(f'\\draw[arr=mute] ($(m{last}.south east)!0.5!(m{first_edit}.north east)+(0.5,0)$) -- (p{j}.west);')
        if kind == 'swap':
            for j, col in enumerate(('own', 'usr')):
                prevn = f'p{j}'
                for s, txt in enumerate(steps[j]):
                    L.append(f'\\node[step={col},anchor=west] (s{j}{s}) at ($({prevn}.east)+(0.3,0)$) {{{tex(q(txt, lang))}}};')
                    L.append(f'\\draw[arr={col}] ({prevn}.east) -- (s{j}{s}.west);')
                    prevn = f's{j}{s}'
                L.append(f'\\node[ans={col},anchor=west] (a{j}) at ($(p{j}.east)+(5.18,0)$) {{{tex(q(answers[j], lang))}}};')
                L.append(f'\\draw[arr={col}] ({prevn}.east) -- (a{j}.west);')
        else:
            prevn = None
            for s, txt in enumerate(steps):
                L.append(f'\\node[step=mute,anchor=west,minimum height={2 * dy + 0.62}cm] (s{s}) at ($({prevn or "p0"}.east |- c)+(0.3,0)$) '
                         f'{{{txt if "$" in txt else tex(txt)}}};')
                if prevn:
                    L.append(f'\\draw[arr=mute] ({prevn}.east) -- (s{s}.west);')
                prevn = f's{s}'
            for j, col in enumerate(('own', 'usr')):
                L.append(f'\\draw[arr={col}] (p{j}.east) -- (p{j}.east -| s0.west);')
                L.append(f'\\node[ans={col},anchor=west] (a{j}) at ($(p{j}.east)+(5.18,0)$) {{{answers[j]}}};')
                L.append(f'\\draw[arr={col}] ({prevn}.east |- a{j}) -- (a{j}.west);')
        if key == 'b':
            cq = q(t['control'].format(q=g['q']['control'], m=g['expect']['control']['assistant']), lang)
            L.append(f'\\node[anchor=north west,text width=6.55cm,inner sep=0,font=\\sffamily\\tiny,text=mute] at ($(p1.south west)+(0,-0.28)$) {{{tex(cq)}}};')
        L.append('\\end{tikzpicture}\\par\\vspace{3.2mm}')
        out.append('\n'.join(L) + '\n')
    out.append('\\end{document}\n')
    src = FIG / f'self_tasks_flow_{lang}.tex'
    src.write_text(''.join(out), encoding='utf-8')
    eng = 'xelatex' if lang == 'zh' else 'pdflatex'
    r = subprocess.run([eng, '-interaction=nonstopmode', '-halt-on-error', src.name], cwd=FIG, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    assert r.returncode == 0, r.stdout[-3000:]
    PAPER.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIG / f'self_tasks_flow_{lang}.pdf', PAPER / f'self_tasks_flow_{lang}.pdf')
    print(lang, 'ok', FIG / f'self_tasks_flow_{lang}.pdf', '| game replies', rep, '| arithmetic', x, y, x * y)


if __name__ == '__main__':
    for lang in ('en', 'zh'):
        build(lang)
