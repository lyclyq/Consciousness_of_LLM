"""LaTeX table bodies for the second round of the supervisor's review in the paper (11-01: R03 A, R08, the scaled random control
r06f, and the stricter versions of both self-related tasks), so that no number in them is typed by hand; with `fill` they are
written between the lines "%<<label" and "%>>label" of paper/source/self_{en,zh}_body.tex (tab:self_exchange) and
self_{en,zh}_appendix.tex (the others), bytes in, bytes out, CRLF kept. Number format as make_r05r06_tex_20261031.py.
  tab:self_exchange       main text: change in "you" when a message receives the coordinates of another role; across the
                          boundary from stats/v4_mechanism.json (the paper's runs, as Table tab:self_mechanism), within the
                          outside from stats/r08_20261031.json (the exchange runs <dir>_r08_us)
  tab:self_strict         the stricter tasks, stats/p56full_20261101.json: attribution in a held-out wording and the secret game
                          with X alone edited, both with fit_m (<dir>_p56full_alt, games_swap/<model>_p56full_game_{main,control})
  tab:self_seeds_scaled   the random edits of the five seeds scaled to at least the size of the fitted edit, stats/r06f_20261031.json
  tab:self_rest           the inverted replacement (all but the eight coordinates) against the eight coordinates, the whole state and
                          the random subspace, stats/r03_20261101.json (Part A)
  tab:self_exchange_full  the exchange of the User and System coordinates in full (you, user, system), stats/r08_20261031.json
The numbers quoted in the text are printed for checking.
usage: make_tran2_tex_20261101.py ARCHIVE [fill]"""
import json
import re
import sys
from pathlib import Path

A = Path(sys.argv[1])
SRC = A.resolve().parents[1] / 'paper' / 'source'


def load(name):
    return json.load(open(A / 'stats' / name, encoding='utf-8'))['results']


V4, R08, R03, R6F, P56 = (load('v4_mechanism.json'), load('r08_20261031.json'), load('r03_20261101.json'),
                          load('r06f_20261031.json'), load('p56full_20261101.json'))
TR = load('r08_traces_20261101.json')
RO = ['qsp', 'qth', 'g20', 'g120']
V4K = {'qsp': 'qwen_sp', 'qth': 'qwen_th', 'g20': 'oss_th', 'g120': 'oss120_th'}
SEEDS = [('r06f_s0', 20261022), ('r06f_s1', 20261023), ('r06f_s2', 20261024), ('r06f_s3', 20261025), ('r06f_s4', 20261026)]
L = {'en': {'own': 'own', 'unedited': 'unedited', 'random': 'random', 'scaled': 'random, scaled',
            'in': 'Inside, given the coordinates of an external source', 'out': "Outside, given the model's own coordinates",
            'within': 'Within the outside', 'attr': 'Attribution question, held-out wording',
            'game': 'Secret game, X alone edited', 'ownX': 'own X, unedited', 'userX': 'User X, unedited',
            'control': 'control correct', 'V': 'the eight coordinates', 'C': 'all but the eight coordinates',
            'F': 'the whole state', 'R': 'random subspace', 'yus': 'you / user / system',
            'dyou': 'Change in ``you\'\'',
            'dsrc': 'Change in ``system\'\' (User \\(\\leftarrow\\) System) or ``user\'\' (System \\(\\leftarrow\\) User)',
            'sys_ans': '``system\'\'', 'user_ans': '``user\'\''},
     'zh': {'own': '自己', 'unedited': '未编辑', 'random': '随机', 'scaled': '放大的随机',
            'in': '内：得到外部来源的坐标', 'out': '外：得到模型自己的坐标', 'within': '外部内部',
            'attr': '归属问题，留出措辞', 'game': '猜数游戏，只编辑 X', 'ownX': '自己的 X，未编辑', 'userX': 'User 的 X，未编辑',
            'control': '对照答对', 'V': '8 个坐标', 'C': '8 个坐标以外的全部', 'F': '整个状态', 'R': '随机子空间',
            'yus': '你 / 用户 / 系统', 'dyou': '“你”的变化',
            'dsrc': '“系统”（User \\(\\leftarrow\\) System）或“用户”（System \\(\\leftarrow\\) User）的变化',
            'sys_ans': '“系统”', 'user_ans': '“用户”'}}


def sg(x, d=2):
    """signed number in math mode; 0.00 without sign"""
    s = f'{x:+.{d}f}'
    if s in ('+0.00', '-0.00', '+0.000', '-0.000'):
        return f'\\({abs(x):.{d}f}\\)'
    return f'\\({s}\\)'


def ch(s, ci=True, d=2):
    """change with interval, -0.76 [-0.77, -0.74]; the interval under 'diff_ci' or 'ci'"""
    if s is None or 'diff' not in s:
        return '--'
    lo, hi = s.get('diff_ci') or s['ci']
    d_ = 3 if abs(s['diff']) < 0.005 and s['diff'] != 0 else d
    t = sg(s['diff'], d_)
    return t + f' [{sg(lo, d_)}, {sg(hi, d_)}]' if ci else t


def lev(x, n, fin=None):
    t = f'{x:.2f}'
    if fin is not None and fin != n:
        t += f' ({fin}/{n})'
    return t


def row(lab, cells):
    return f'{lab} & ' + ' & '.join(cells) + ' \\\\'


def head(text):
    return f'\\multicolumn{{5}}{{l}}{{\\emph{{{text}}}}} \\\\'


def arrow(a, b):
    return f'{a} \\(\\leftarrow\\) {b}'


def block_exchange(lang):
    t = L[lang]
    own = t['own']
    rows = [('in', [(arrow(own, 'User'), 'v4', 'swU8_A'), (arrow(own, 'System'), 'v4', 'swS8_A')]),
            ('out', [(arrow('User', own), 'v4', 'swA8_U'), (arrow('System', own), 'v4', 'swA8_S')]),
            ('within', [(arrow('User', 'System'), 'r08', 'swS8_U'), (arrow('System', 'User'), 'r08', 'swU8_S')])]
    out = []
    for i, (g, rs) in enumerate(rows):
        if i:
            out.append('\\midrule')
        out.append(head(t[g]))
        for lab, src, c in rs:
            vals = [V4[V4K[ro]]['all'][c]['diff'] if src == 'v4' else R08[ro][c]['self']['diff'] for ro in RO]
            out.append(row(f'\\quad {lab}', [sg(v) for v in vals]))
    return out


def block_strict(lang):
    t = L[lang]
    own = t['own']
    out = [head(t['attr'])]
    T1 = {ro: P56['T1'][ro]['full'] for ro in RO}
    for c, lab in (('base_A', f"{own}, {t['unedited']}"), ('base_U', f"User, {t['unedited']}")):
        out.append(row(lab, [lev(T1[ro]['levels'][c]['you'], T1[ro]['levels'][c]['n'], T1[ro]['levels'][c].get('finished'))
                             for ro in RO]))
    for c, lab in (('swU8_A', arrow(own, 'User')), ('swA8_U', arrow('User', own)), ('rswU8_A', f"{arrow(own, 'User')}, {t['random']}")):
        out.append(row(f'\\quad {lab}', [ch(T1[ro][c]) for ro in RO]))
    out += ['\\midrule', head(t['game'])]
    T3 = {ro: P56['T3'][ro]['full'] for ro in RO}
    for c, lab in (('base_A', t['ownX']), ('base_U', t['userX'])):
        out.append(row(lab, [lev(T3[ro]['main'][c]['stat'], T3[ro]['main'][c]['n'], T3[ro]['main'][c]['finished']) for ro in RO]))
    for c, lab in (('swU8_A', arrow(own, 'User')), ('swA8_U', arrow('User', own)), ('rswU8_A', f"{arrow(own, 'User')}, {t['random']}")):
        out.append(row(f'\\quad {lab}', [ch(T3[ro]['main'][c]) for ro in RO]))
    cells = []
    for ro in RO:
        acc = [v['accuracy'] for v in T3[ro]['control'].values()]
        cells.append(f'{min(acc):.2f}' if min(acc) == max(acc) else f'{min(acc):.2f}--{max(acc):.2f}')
    out.append(row(t['control'], cells))
    return out


def block_seeds_scaled(lang):
    t = L[lang]
    own = t['own']
    out = []
    for i, (c, lab) in enumerate((('frswU8_A', f"{arrow(own, 'User')}, {t['scaled']}"),
                                  ('frswA8_U', f"{arrow('User', own)}, {t['scaled']}"))):
        if i:
            out.append('\\midrule')
        out.append(f'\\multicolumn{{5}}{{l}}{{{lab}}} \\\\')
        for run, seed in SEEDS:
            out.append(row(f'\\quad {seed}', [ch(R6F[ro][run]['stats'][c]) for ro in RO]))
    return out


def block_rest(lang):
    t = L[lang]
    own = t['own']
    out = []
    for i, (base, conds, lab) in enumerate((('base_A', ['swU8_A', 'cswU8_A', 'fswU8_A', 'rswU8_A'], arrow(own, 'User')),
                                            ('base_U', ['swA8_U', 'cswA8_U', 'fswA8_U', 'rswA8_U'], arrow('User', own)))):
        if i:
            out.append('\\midrule')
        b = {ro: R03[ro]['levels'][base] for ro in RO}
        name = f"{own}, {t['unedited']}" if base == 'base_A' else f"User, {t['unedited']}"
        out.append(row(name, [lev(b[ro]['you'], b[ro]['n'], b[ro].get('finished')) for ro in RO]))
        out.append(f'\\multicolumn{{5}}{{l}}{{{lab}}} \\\\')
        for c, key in zip(conds, ('V', 'C', 'F', 'R')):
            out.append(row(f"\\quad {t[key]}", [ch(R03[ro][c]) for ro in RO]))
    return out


def block_exchange_full(lang):
    t = L[lang]
    out = []
    for base, lab in (('base_U', f"User, {t['unedited']}"), ('base_S', f"System, {t['unedited']}")):
        lv = {ro: R08[ro]['levels'][base] for ro in RO}
        out.append(row(lab, [f"{lv[ro]['self']:.2f} / {lv[ro]['user']:.2f} / {lv[ro]['system']:.2f}" for ro in RO]))
    out += ['\\midrule', f"\\multicolumn{{5}}{{l}}{{{t['dyou']}}} \\\\"]
    pairs = (('swS8_U', arrow('User', 'System')), ('rswS8_U', f"{arrow('User', 'System')}, {t['random']}"),
             ('swU8_S', arrow('System', 'User')), ('rswU8_S', f"{arrow('System', 'User')}, {t['random']}"))
    for c, lab in pairs:
        out.append(row(f'\\quad {lab}', [ch(R08[ro][c]['self'] | {'diff_ci': R08[ro][c]['self']['ci']}) for ro in RO]))
    out += ['\\midrule', f"\\multicolumn{{5}}{{l}}{{{t['dsrc']}}} \\\\"]
    for c, lab in pairs:
        ans = 'system' if c.endswith('S8_U') else 'user'
        out.append(row(f'\\quad {lab}', [ch(R08[ro][c][ans] | {'diff_ci': R08[ro][c][ans]['ci']}) for ro in RO]))
    return out


BODY = {'tab:self_exchange': block_exchange}
APP = {'tab:self_strict': block_strict, 'tab:self_seeds_scaled': block_seeds_scaled, 'tab:self_rest': block_rest,
       'tab:self_exchange_full': block_exchange_full}


def text_numbers():
    f2 = lambda xs: ', '.join(f'{x:+.2f}' for x in xs)  # noqa: E731
    print('exchange, change in you: User<-System', f2(R08[ro]['swS8_U']['self']['diff'] for ro in RO),
          '| System<-User', f2(R08[ro]['swU8_S']['self']['diff'] for ro in RO))
    print('exchange: user reading of a System message given User coords', f2(R08[ro]['swU8_S']['user']['diff'] for ro in RO),
          '| system reading of a User message given System coords', f2(R08[ro]['swS8_U']['system']['diff'] for ro in RO))
    print('exchange random, largest |change| in any answer:',
          max(abs(R08[ro][c][a]['diff']) for ro in RO for c in ('rswS8_U', 'rswU8_S') for a in ('self', 'user', 'system')))
    for k in ('g20', 'g120'):
        r = TR[k]['r08_us']
        print(f'{k} developer message: System unedited', r['base_S']['developer_message'], '/', r['base_S']['n'],
              '| System<-User', r['swU8_S']['developer_message'], '| random', r['rswU8_S']['developer_message'],
              '| paper run System<-own', TR[k]['all']['swA8_S']['developer_message'])
    print('rest (C) removal', f2(R03[ro]['cswU8_A']['diff'] for ro in RO), '| installation', f2(R03[ro]['cswA8_U']['diff'] for ro in RO))
    print('whole state (F) removal', f2(R03[ro]['fswU8_A']['diff'] for ro in RO), '| installation', f2(R03[ro]['fswA8_U']['diff'] for ro in RO))
    print('F edited levels: own<-User', [round(R03[ro]['levels']['fswU8_A']['you'], 3) for ro in RO],
          '| User<-own', [round(R03[ro]['levels']['fswA8_U']['you'], 3) for ro in RO])
    print('V removal', f2(R03[ro]['swU8_A']['diff'] for ro in RO), '| installation', f2(R03[ro]['swA8_U']['diff'] for ro in RO))
    print('qth C removal bounds', R03['qth']['cswU8_A'].get('bounds'), 'finished', R03['qth']['levels']['cswU8_A'])
    for c in ('frswU8_A', 'frswA8_U'):
        print(c, 'largest |change| over five seeds', round(max(abs(R6F[ro][r]['stats'][c]['diff']) for ro in RO for r, _ in SEEDS), 3))
    lrm = [R6F[ro][r]['dose'][c]['layer_ratio_min'] for ro in RO for r, _ in SEEDS for c in ('frswU8_A', 'frswA8_U')]
    rmed = [R6F[ro][r]['dose'][c]['ratio_median'] for ro in RO for r, _ in SEEDS for c in ('frswU8_A', 'frswA8_U')]
    print('scaled random: smallest per-layer ratio', round(min(lrm), 4), '| per-item all-layer ratio median, range over runs',
          round(min(rmed), 2), '-', round(max(rmed), 2))
    T1 = {ro: P56['T1'][ro]['full'] for ro in RO}
    print('held-out wording: unedited own', [round(T1[ro]['levels']['base_A']['you'], 2) for ro in RO],
          '| removal', f2(T1[ro]['swU8_A']['diff'] for ro in RO), '| edited own', [round(T1[ro]['levels']['swU8_A']['you'], 3) for ro in RO],
          '| installation', f2(T1[ro]['swA8_U']['diff'] for ro in RO), '| random', f2(T1[ro]['rswU8_A']['diff'] for ro in RO))
    T3 = {ro: P56['T3'][ro]['full']['main'] for ro in RO}
    print('game X alone: removal', f2(T3[ro]['swU8_A']['diff'] for ro in RO), '| installation', f2(T3[ro]['swA8_U']['diff'] for ro in RO),
          '| random', f2(T3[ro]['rswU8_A']['diff'] for ro in RO))
    acc = [v['accuracy'] for ro in RO for v in P56['T3'][ro]['full']['control'].values()]
    print('game X alone: control correct', min(acc), '-', max(acc))


for lang in ('en', 'zh'):
    text = []
    for lab, f in {**BODY, **APP}.items():
        text += [f'%% {lab}'] + f(lang) + ['%% end']
    (A / 'stats' / f'tran2_tex_{lang}.tex').write_text('\n'.join(text) + '\n', encoding='utf-8')
    sys.stdout.buffer.write(('\n'.join(text) + '\n').encode('utf-8'))
text_numbers()
if len(sys.argv) > 2 and sys.argv[2] == 'fill':
    for lang in ('en', 'zh'):
        for part, blocks in (('body', BODY), ('appendix', APP)):
            p = SRC / f'self_{lang}_{part}.tex'
            raw = p.read_bytes()
            crlf = b'\r\n' in raw
            s = raw.decode('utf-8').replace('\r\n', '\n')
            for lab, f in blocks.items():
                pat = re.compile(rf'(^%<<{re.escape(lab)}\n)(.*?)(^%>>{re.escape(lab)}$)', re.S | re.M)
                assert len(pat.findall(s)) == 1, (p.name, lab)
                body = '\n'.join(f(lang))
                s = pat.sub(lambda m: m.group(1) + body + '\n' + m.group(3), s)
                print('filled', p.name, lab)
            p.write_bytes((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
