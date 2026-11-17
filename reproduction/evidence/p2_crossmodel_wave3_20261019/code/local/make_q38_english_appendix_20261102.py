"""Numbers and LaTeX table bodies of the English ablation of Qwen3.8-27B in Appendix B (user 11-02: the language protocol and the
ablation only in the appendix; "结果还是要放进附录原来位置，但是单次前向游戏得展示为什么不计入"; the cross-language transfer
"只做归属题单次向前的中英文互相影响" plus the games). English runs: PRO 11-02, runs/pro/qwen38_en_20261102 (fits on the English
discovery bodies, as-run scripts of the Chinese runs, hash-checked). Chinese columns: the paper's own numbers, read from
stats/v4_mechanism.json and stats/v4_mech_ext.json, so that they equal Tables tab:self_mechanism and tab:self_game. English
columns: the same functions (make_v4_mechanism.summarize, make_v4_mech_ext.game_summary) on the English per-item results.
Cross-language transfer (single pass, all runs on PRO): the change with the other language's fit over the change with the fit of
the items' own language (Chinese items: refit on the Chinese discovery bodies on PRO, singlepass_swap/qwen38_27b_zh_all_pro).
Subspace overlap: runs/pro/qwen38_en_20261102/xlang_overlap_20261102.json (xlang_overlap_20261102.py, run on PRO).
Outputs: ARCHIVE/stats/q38_english_appendix_20261102.json and q38_english_tex_20261102_{en,zh}.tex (blocks "%% <label>" ... "%% end").
Needs numpy and matplotlib (imported by make_v4_mechanism): uv run --with numpy --with matplotlib python ... ARCHIVE
usage: make_q38_english_appendix_20261102.py ARCHIVE"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_v4_mech_ext as ext  # noqa: E402
import make_v4_mechanism as mech  # noqa: E402

A = Path(sys.argv[1]).resolve()
EN = A / 'runs/pro/qwen38_en_20261102'


def long_items(d):
    """make_v4_mechanism.items with the extended-length prefix: the English item paths exceed the Windows path limit."""
    p = Path('\\\\?\\' + str(Path(d).resolve())) if sys.platform == 'win32' else Path(d)
    return [json.loads(f.read_text(encoding='utf-8')) for f in sorted((p / 'items').glob('*.json'))]


mech.items = ext.items = long_items  # game_summary looks the name up in its own module

M = json.load(open(A / 'stats' / 'v4_mechanism.json', encoding='utf-8'))['results']
E = json.load(open(A / 'stats' / 'v4_mech_ext.json', encoding='utf-8'))
ZH = {'sp': M['qwen_sp'], 'th': M['qwen_th']}  # all, all_content, all_arith, all_k64: what make_v4_mech_tex.py prints
ZG = {'sp': E['games']['qwen_sp'], 'th': E['games']['qwen_th']}

ENR = {'sp': {}, 'th': {}}
for mode, base in (('sp', 'singlepass_swap/qwen38_27b_en_{}'), ('th', 'thinking_swap/qwen38_27b_en_{}')):
    for k in ('all', 'all_content', 'all_arith', 'all_k64'):
        d = EN / base.format(k)
        if d.exists():
            ENR[mode][k] = mech.summarize(long_items(d), k)
EG = {'sp': {q: ext.game_summary(EN / f'games_swap/qwen38_27b_en_sp_s6_{q}') for q in ('main', 'control')},
      'th': {q: ext.game_summary(EN / f'games_swap/qwen38_27b_en_s6_{q}') for q in ('main', 'control')}}

# cross-language transfer, single pass, PRO
X = {'zh_items': {'own_fit': mech.summarize(long_items(EN / 'singlepass_swap/qwen38_27b_zh_all_pro'), 'all'),
                  'other_fit': mech.summarize(long_items(EN / 'singlepass_swap/qwen38_27b_xl_enfit_zh'), 'all'),
                  'own_game': ext.game_summary(EN / 'games_swap/qwen38_27b_zh_pro_sp_s6_main'),
                  'other_game': ext.game_summary(EN / 'games_swap/qwen38_27b_xl_enfit_zh_sp_s6_main')},
     'en_items': {'own_fit': ENR['sp']['all'], 'other_fit': mech.summarize(long_items(EN / 'singlepass_swap/qwen38_27b_xl_zhfit_en'), 'all'),
                  'own_game': EG['sp']['main'], 'other_game': ext.game_summary(EN / 'games_swap/qwen38_27b_xl_zhfit_en_sp_s6_main')}}
XC = [('swU8_A', 'fit'), ('swS8_A', 'fit'), ('rswU8_A', 'fit'), ('swA8_U', 'fit'), ('swA8_S', 'fit'), ('rswA8_U', 'fit'), ('swA8_U', 'game')]
ratios = {}
for side, S in X.items():
    ratios[side] = {}
    for c, kind in XC:
        o, t = S[f'own_{kind}'].get(c, {}), S[f'other_{kind}'].get(c, {})
        if 'diff' in o and 'diff' in t:
            ratios[side][f'{c}_{kind}'] = {'own': o['diff'], 'other': t['diff'], 'ratio': t['diff'] / o['diff'] if abs(o['diff']) > 0.05 else None}
OV = json.load(open(EN / 'xlang_overlap_20261102.json', encoding='utf-8'))

json.dump({'en': ENR, 'en_games': EG, 'xlang': X, 'xlang_ratios': ratios, 'overlap': {k: {kk: v for kk, v in OV[k].items() if kk != 'by_layer'}
           for k in ('k8', 'k64')}}, open(A / 'stats' / 'q38_english_appendix_20261102.json', 'w', encoding='utf-8'), indent=1)

T = {'en': {'own': 'own', 'User': 'User', 'System': 'System', 'unedited': 'unedited', 'random': 'random', 'k64': '64 dims',
            'copy': 'own (copy)', 'content': 'Content, correct', 'arith': 'Arithmetic, correct', 'control': 'Control, correct',
            'attr': 'Attribution', 'game': 'Secret game', 'X': 'X', 'zh_items': 'Chinese items', 'en_items': 'English items'},
     'zh': {'own': '自己', 'User': 'User', 'System': 'System', 'unedited': '未编辑', 'random': '随机', 'k64': '64 维',
            'copy': '自己（复制）', 'content': '内容题，答对', 'arith': '算术题，答对', 'control': '对照题，答对',
            'attr': '归属题', 'game': '猜数游戏', 'X': 'X', 'zh_items': '中文题目', 'en_items': '英文题目'}}


def _sg(x, d=2):
    s = f'{x:+.{d}f}'
    return f'\\({abs(x):.{d}f}\\)' if s in ('+0.00', '-0.00', '+0.000', '-0.000') else f'\\({s}\\)'


def _ch(s, ci=True, d=2):
    if s is None or 'diff' not in s:
        return '--'
    d_ = 3 if abs(s['diff']) < 0.005 and s['diff'] != 0 else d
    return _sg(s['diff'], d_) + (f" [{_sg(s['diff_ci'][0], d_)}, {_sg(s['diff_ci'][1], d_)}]" if ci else '')


def _val(s):
    if s is None:
        return '--'
    t = f"{s['stat']:.2f}"
    return t + (f" ({s['finished']}/{s['n']})" if s.get('finished', s['n']) != s['n'] else '')


def _acc(runs, conds=None):
    xs = [s['accuracy'] for c, s in runs.items() if 'accuracy' in s and (conds is None or c in conds)]
    if not xs:
        return '--'
    lo, hi = min(xs), max(xs)
    return f'{lo:.2f}' if f'{lo:.2f}' == f'{hi:.2f}' else f'{lo:.2f}--{hi:.2f}'


# the four columns: no thinking English, Chinese; thinking English, Chinese
COLS = [(ENR['sp'], EG['sp']), (ZH['sp'], ZG['sp']), (ENR['th'], EG['th']), (ZH['th'], ZG['th'])]


def build(lang):
    t = T[lang]
    out = ['%% tab:self_english']
    rows = [('h', t['attr']), (f"{t['own']}, {t['unedited']}", 'all', 'base_A', 'v'),
            (f"{t['own']} \\(\\leftarrow\\) {t['User']}", 'all', 'swU8_A', 'c'),
            (f"{t['own']} \\(\\leftarrow\\) {t['System']}", 'all', 'swS8_A', 'c'),
            (f"{t['own']} \\(\\leftarrow\\) {t['User']}, {t['random']}", 'all', 'rswU8_A', 'c'),
            (f"{t['own']} \\(\\leftarrow\\) {t['copy']}", 'all', 'idA8_A', 'c'),
            (f"{t['User']}, {t['unedited']}", 'all', 'base_U', 'v'), (f"{t['User']} \\(\\leftarrow\\) {t['own']}", 'all', 'swA8_U', 'c'),
            (f"{t['User']} \\(\\leftarrow\\) {t['own']}, {t['random']}", 'all', 'rswA8_U', 'c'),
            (f"{t['User']} \\(\\leftarrow\\) {t['own']}, {t['k64']}", 'all_k64', 'swA64_U', 'c'),
            (f"{t['User']} \\(\\leftarrow\\) {t['own']}, {t['k64']}, {t['random']}", 'all_k64', 'rswA64_U', 'c'),
            (f"{t['System']}, {t['unedited']}", 'all', 'base_S', 'v'), (f"{t['System']} \\(\\leftarrow\\) {t['own']}", 'all', 'swA8_S', 'c')]
    for r in rows:
        if r[0] == 'h':
            out.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{r[1]}}}}} \\\\")
            continue
        lab, run, c, kind = r
        cells = []
        for j, (R, _) in enumerate(COLS):
            if run == 'all_k64' and j == 0:  # 64 dims: thinking only in English (user 11-02: "行 补 64维 思考")
                cells.append('--')
                continue
            cells.append((_val if kind == 'v' else _ch)(R.get(run, {}).get(c)))
        out.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')  # no \quad: with it the EN table was 7.1pt too wide (11-03)
    out.append(f"{t['content']} & " + ' & '.join(_acc(R.get('all_content', {})) for R, _ in COLS) + ' \\\\')
    out.append(f"{t['arith']} & " + ' & '.join(_acc(R.get('all_arith', {})) for R, _ in COLS) + ' \\\\')
    out.append('\\midrule')
    out.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{t['game']}}}}} \\\\")
    for lab, c, kind in ((f"{t['own']} X, {t['unedited']}", 'base_A', 'v'), (f"{t['own']} \\(\\leftarrow\\) {t['User']}", 'swU8_A', 'c'),
                         (f"{t['own']} \\(\\leftarrow\\) {t['User']}, {t['random']}", 'rswU8_A', 'c'),
                         (f"{t['User']} X, {t['unedited']}", 'base_U', 'v'), (f"{t['User']} \\(\\leftarrow\\) {t['own']}", 'swA8_U', 'c'),
                         (f"{t['User']} \\(\\leftarrow\\) {t['own']}, {t['random']}", 'rswA8_U', 'c')):
        out.append(f'{lab} & ' + ' & '.join((_val if kind == 'v' else _ch)(G.get('main', {}).get(c)) for _, G in COLS) + ' \\\\')
    out.append(f"{t['control']} & " + ' & '.join(_acc(G.get('control', {})) for _, G in COLS) + ' \\\\')
    out.append('%% end')
    # transfer: one block per item language; change with the fit of the same language, of the other language, ratio. The random
    # subspaces are the same stored subspaces in both fits (same seed per layer), so their rows are identical and left out.
    out.append('%% tab:self_english_transfer')
    lab = {'swU8_A': f"{t['own']} \\(\\leftarrow\\) {t['User']}", 'swS8_A': f"{t['own']} \\(\\leftarrow\\) {t['System']}",
           'swA8_U': f"{t['User']} \\(\\leftarrow\\) {t['own']}", 'swA8_S': f"{t['System']} \\(\\leftarrow\\) {t['own']}"}
    for i, (side, name) in enumerate((('zh_items', t['zh_items']), ('en_items', t['en_items']))):
        if i:
            out.append('\\midrule')
        out.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{name}}}}} \\\\")
        S = X[side]
        for c, kind in XC:
            if c.startswith('rsw'):
                continue
            r = ratios[side].get(f'{c}_{kind}', {}).get('ratio')
            row = lab[c] if kind == 'fit' else f"{t['game']}: {lab[c]}"
            out.append(f"\\quad {row} & {_ch(S[f'own_{kind}'].get(c))} & {_ch(S[f'other_{kind}'].get(c))} & "
                       f"{f'{r:.2f}' if r is not None else '--'} \\\\")
    out.append('%% end')
    return '\n'.join(out) + '\n'


for lang in ('en', 'zh'):
    (A / 'stats' / f'q38_english_tex_20261102_{lang}.tex').write_text(build(lang), encoding='utf-8')
print(build('en'))
print('ratios', json.dumps({s: {k: (round(v['own'], 3), round(v['other'], 3), v['ratio'] and round(v['ratio'], 2)) for k, v in r.items()}
                            for s, r in ratios.items()}))
print('overlap k8', {k: v for k, v in OV['k8'].items() if k != 'by_layer'})
