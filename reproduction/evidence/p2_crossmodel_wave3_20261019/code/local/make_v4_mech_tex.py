"""LaTeX table bodies of the causal tests in paper v4 (10-27), from stats/v4_mechanism.json (make_v4_mechanism.py) and
stats/v4_mech_ext.json (make_v4_mech_ext.py), so that no number in these tables is typed by hand. One file per language,
stats/v4_mech_tex_{en,zh}.tex, with one block per table between lines "%% <label>" and "%% end"; the blocks are pasted into the
paper source (tabular bodies only; captions and headers are in the source).
  tab:self_mech_questions  the four questions per readout (attribution and game: removal, installation; content, arithmetic:
                           lowest-highest share correct over the six conditions)
  tab:self_mechanism       attribution, every layer: unedited values, changes for 8 and 64 dimensions and random subspaces
  tab:self_game            secret game v6: unedited share acting as setter, changes, control
  tab:self_mechanism_where groups of positions (appendix; all four readouts from 10-29; 10-28 version make_v4_mech_tex_1028.py)
  tab:self_mechanism_windows layer windows (appendix; all four readouts from 10-29)
  tab:self_mech_gemma      Gemma-4-31B, natural order, no thinking (appendix)
usage: make_v4_mech_tex.py ARCHIVE"""
import json
import sys
from pathlib import Path

A = Path(sys.argv[1])
M = json.load(open(A / 'stats' / 'v4_mechanism.json', encoding='utf-8'))
E = json.load(open(A / 'stats' / 'v4_mech_ext.json', encoding='utf-8'))
R, W = M['results'], M['windows']
RO = ['qwen_sp', 'qwen_th', 'oss_th', 'oss120_th']
L = {'en': {'own': 'own', 'User': 'User', 'System': 'System', 'unedited': 'unedited', 'random': 'random', 'k64': '64 dims',
            'body': 'body', 'end': 'end of message', 'question': 'question', 'all': 'all', 'control': 'control correct',
            'ro': {'qwen_sp': 'Qwen3.8-27B, no thinking', 'qwen_th': 'Qwen3.8-27B, thinking', 'oss_th': 'gpt-oss-20b, thinking',
                   'oss120_th': 'gpt-oss-120b, thinking'}, 'fin': 'fin.', 'of': 'of'},
     'zh': {'own': '自己', 'User': 'User', 'System': 'System', 'unedited': '未编辑', 'random': '随机', 'k64': '64 维',
            'body': '正文', 'end': '消息结束部分', 'question': '问题', 'all': '全部', 'control': '对照答对',
            'ro': {'qwen_sp': 'Qwen3.8-27B，不思考', 'qwen_th': 'Qwen3.8-27B，思考', 'oss_th': 'gpt-oss-20b，思考',
                   'oss120_th': 'gpt-oss-120b，思考'}, 'fin': '完成', 'of': '/'}}


def sg(x, d=2):
    """signed number in math mode; 0.00 without sign"""
    s = f'{x:+.{d}f}'
    if s in ('+0.00', '-0.00', '+0.000', '-0.000'):
        return f'\\({abs(x):.{d}f}\\)'
    return f'\\({s.replace("-", "-")}\\)'


def ch(s, ci=True, d=2):
    """change with interval: -0.76 [-0.77, -0.74]"""
    if s is None or 'diff' not in s:
        return '--'
    d_ = 3 if abs(s['diff']) < 0.005 and s['diff'] != 0 else d
    t = sg(s['diff'], d_)
    if ci:
        t += f" [{sg(s['diff_ci'][0], d_)}, {sg(s['diff_ci'][1], d_)}]"
    return t


def val(s):
    if s is None:
        return '--'
    t = f"{s['stat']:.2f}"
    if s.get('finished', s['n']) != s['n']:
        t += f" ({s['finished']}/{s['n']})"
    return t


def acc_range(runs, conds=None):
    xs = [s['accuracy'] for c, s in runs.items() if 'accuracy' in s and (conds is None or c in conds)]
    if not xs:
        return '--'
    lo, hi = min(xs), max(xs)
    return f'{lo:.2f}' if f'{lo:.2f}' == f'{hi:.2f}' else f'{lo:.2f}--{hi:.2f}'


def gch(s, ci=True):
    if s is None or 'diff' not in s:
        return '--'
    t = sg(s['diff'])
    if ci:
        t += f" [{sg(s['diff_ci'][0])}, {sg(s['diff_ci'][1])}]"
    return t


def gval(s):
    if s is None:
        return '--'
    t = f"{s['stat']:.2f}"
    if s['finished'] != s['n']:
        t += f" ({s['finished']}/{s['n']})"
    return t


def build(lang):
    t = L[lang]
    out = []
    # summary of the four questions
    out.append('%% tab:self_mech_questions')
    for ro in RO:
        a = R[ro].get('all', {})
        g = E['games'].get(ro, {}).get('main', {})
        out.append(f"{t['ro'][ro]} & {ch(a.get('swU8_A'), False)} & {ch(a.get('swA8_U'), False)} & {gch(g.get('swU8_A'), False)} & "
                   f"{gch(g.get('swA8_U'), False)} & {acc_range(R[ro].get('all_content', {}))} & {acc_range(R[ro].get('all_arith', {}))} \\\\")
    out.append('%% end')
    # attribution detail
    out.append('%% tab:self_mechanism')
    rows = [(f"{t['own']}, {t['unedited']}", 'all', 'base_A', 'v'), (f"{t['own']} \\(\\leftarrow\\) {t['User']}", 'all', 'swU8_A', 'c'),
            (f"{t['own']} \\(\\leftarrow\\) {t['System']}", 'all', 'swS8_A', 'c'),
            (f"{t['own']} \\(\\leftarrow\\) {t['User']}, {t['random']}", 'all', 'rswU8_A', 'c'),
            (f"{t['own']} \\(\\leftarrow\\) {t['User']}, {t['k64']}", 'all_k64', 'swU64_A', 'c'),
            (f"{t['own']} \\(\\leftarrow\\) {t['User']}, {t['k64']}, {t['random']}", 'all_k64', 'rswU64_A', 'c'), None,
            (f"{t['User']}, {t['unedited']}", 'all', 'base_U', 'v'), (f"{t['User']} \\(\\leftarrow\\) {t['own']}", 'all', 'swA8_U', 'c'),
            (f"{t['User']} \\(\\leftarrow\\) {t['own']}, {t['random']}", 'all', 'rswA8_U', 'c'),
            (f"{t['User']} \\(\\leftarrow\\) {t['own']}, {t['k64']}", 'all_k64', 'swA64_U', 'c'),
            (f"{t['User']} \\(\\leftarrow\\) {t['own']}, {t['k64']}, {t['random']}", 'all_k64', 'rswA64_U', 'c'), None,
            (f"{t['System']}, {t['unedited']}", 'all', 'base_S', 'v'), (f"{t['System']} \\(\\leftarrow\\) {t['own']}", 'all', 'swA8_S', 'c')]
    for r in rows:
        if r is None:
            out.append('\\midrule')
            continue
        lab, run, c, kind = r
        cells = [(val if kind == 'v' else ch)(R[ro].get(run, {}).get(c)) for ro in RO]
        out.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
    out.append('%% end')
    # game
    out.append('%% tab:self_game')
    G = {ro: E['games'].get(ro, {}) for ro in RO}
    for r in ((f"{t['own']} X, {t['unedited']}", 'base_A', 'v'), (f"{t['own']} \\(\\leftarrow\\) {t['User']}", 'swU8_A', 'c'),
              (f"{t['own']} \\(\\leftarrow\\) {t['User']}, {t['random']}", 'rswU8_A', 'c'), None,
              (f"{t['User']} X, {t['unedited']}", 'base_U', 'v'), (f"{t['User']} \\(\\leftarrow\\) {t['own']}", 'swA8_U', 'c'),
              (f"{t['User']} \\(\\leftarrow\\) {t['own']}, {t['random']}", 'rswA8_U', 'c'), None):
        if r is None:
            out.append('\\midrule')
            continue
        lab, c, kind = r
        out.append(f'{lab} & ' + ' & '.join((gval if kind == 'v' else gch)(G[ro].get('main', {}).get(c)) for ro in RO) + ' \\\\')
    out.append(f"{t['control']}, {t['unedited']} & " + ' & '.join(acc_range(G[ro].get('control', {}), ('base_A', 'base_U')) for ro in RO) + ' \\\\')
    out.append(f"{t['control']}, {'edited' if lang == 'en' else '编辑后'} & " + ' & '.join(
        acc_range(G[ro].get('control', {}), ('swU8_A', 'swA8_U', 'rswU8_A', 'rswA8_U')) for ro in RO) + ' \\\\')
    out.append('%% end')
    # positions (appendix)
    out.append('%% tab:self_mechanism_where')
    for k, lab in (('region_body', t['body']), ('region_end', t['end']), ('region_question', t['question']), ('all', t['all'])):
        for i, (c, cl) in enumerate((('swU8_A', f"{t['own']} \\(\\leftarrow\\) {t['User']}"), ('swA8_U', f"{t['User']} \\(\\leftarrow\\) {t['own']}"),
                                     ('rswU8_A', t['random']))):
            out.append(f"{lab if i == 0 else ''} & {cl} & " + ' & '.join(ch(R[ro].get(k, {}).get(c)) for ro in RO) + ' \\\\')
        if k != 'all':
            out.append('\\midrule')
    out.append('%% end')
    # windows (appendix)
    out.append('%% tab:self_mechanism_windows')
    for j, ro in enumerate(RO):
        out.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{t['ro'][ro]}}}}} \\\\")
        for w_, S in sorted(W.get(ro, {}).items(), key=lambda kv: kv[1]['layers'][0]):
            out.append(f"{w_.replace('-', '--')} & {ch(S.get('swU8_A'))} & {ch(S.get('swA8_U'))} & {ch(S.get('rswU8_A'))} \\\\")
        if j < len(RO) - 1:
            out.append('\\midrule')
    out.append('%% end')
    # Gemma (appendix)
    out.append('%% tab:self_mech_gemma')
    # columns: attribution (8 dims), attribution (64 dims), secret game, content correct, arithmetic correct
    gm, gk, gg = E['gemma'].get('all', {}), E['k64'].get('gemma_sp', {}), E['games'].get('gemma_sp', {}).get('main', {})
    gc, ga = E['gemma'].get('all_content', {}), E['gemma'].get('all_arith', {})
    k64 = {'base_A': 'base_A', 'base_U': 'base_U', 'swU8_A': 'swU64_A', 'rswU8_A': 'rswU64_A', 'swA8_U': 'swA64_U', 'rswA8_U': 'rswA64_U'}
    for lab, c, kind in ((f"{t['own']}, {t['unedited']}", 'base_A', 'v'), (f"{t['own']} \\(\\leftarrow\\) {t['User']}", 'swU8_A', 'c'),
                         (f"{t['own']} \\(\\leftarrow\\) {t['System']}", 'swS8_A', 'c'),
                         (f"{t['own']} \\(\\leftarrow\\) {t['User']}, {t['random']}", 'rswU8_A', 'c'),
                         (f"{t['User']}, {t['unedited']}", 'base_U', 'v'), (f"{t['User']} \\(\\leftarrow\\) {t['own']}", 'swA8_U', 'c'),
                         (f"{t['User']} \\(\\leftarrow\\) {t['own']}, {t['random']}", 'rswA8_U', 'c'),
                         (f"{t['System']}, {t['unedited']}", 'base_S', 'v'), (f"{t['System']} \\(\\leftarrow\\) {t['own']}", 'swA8_S', 'c')):
        f8 = val if kind == 'v' else ch
        fg = gval if kind == 'v' else gch
        acc = lambda S: f"{S[c]['accuracy']:.2f}" if c in S else '--'  # noqa: E731
        out.append(f"{lab} & {f8(gm.get(c))} & {f8(gk.get(k64[c])) if c in k64 else '--'} & {fg(gg.get(c)) if c in gg else '--'} & "
                   f"{acc(gc)} & {acc(ga)} \\\\")
    out.append('%% end')
    out.append('%% tab:self_mech_gemma_windows')
    for w_, S in sorted(E['windows'].get('gemma_sp', {}).items(), key=lambda kv: kv[1]['layers'][0]):
        out.append(f"{w_.replace('-', '--')} & {ch(S.get('swU8_A'))} & {ch(S.get('swA8_U'))} & {ch(S.get('rswU8_A'))} \\\\")
    out.append('%% end')
    p = A / 'stats' / f'v4_mech_tex_{lang}.tex'
    p.write_text('\n'.join(out).replace('\\(-', '\\(-') + '\n', encoding='utf-8')
    return p


for lang in ('en', 'zh'):
    print(build(lang))
