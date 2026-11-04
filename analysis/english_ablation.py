"""English ablation of the causal tests of Qwen3.8-27B (Appendix B; the paragraph "The coordinates are shared between languages" of
Section 5.1): the same tests run with the English items, and the transfer of the fitted subspaces between the two languages.

Runs (results package, W3/runs/pro/qwen38_en_20261102): the English fits on the discovery bodies (single pass and thinking), the
replacement at every layer with eight dimensions (attribution, content, arithmetic) in both modes, installation with 64 dimensions
with thinking, and the secret game in both modes; for the transfer, a refit on the Chinese discovery bodies on the same machine,
its runs on the Chinese items, and each language's subspace applied to the other language's items (single pass; attribution and
the game).
  - English columns: the statistics of causal_tests.py (common.summarize; causal_tests.game_summary) on the English runs.
  - Chinese columns: the paper's own numbers, read from stats/v4_mechanism.json and stats/v4_mech_ext.json (written by
    causal_tests.py earlier in the same run, else the archived copies), so that they equal Tables tab:self_mechanism and
    tab:self_game.
  - Transfer ratio: the change with the other language's subspace over the change with the subspace of the items' own language
    (shown only where the latter is above 0.05 in size).
  - Subspace overlap: per layer, ||B_zh^T B_en||_F^2 / k for the first k columns of the two single-pass fits (k = 8 and 64),
    against k / d for independent random subspaces; recomputed from the fit.pt files when torch is installed, else read from the
    archived xlang_overlap_20261102.json.
  - Reasoning traces (rule of hand_checks.trace_counts): a trace "describes an assistant message" if it mentions the assistant
    after the option text "AI assistant" is removed; for the English and the Chinese thinking runs, 8 and 64 dimensions, with the
    answers of the items whose trace does.
Outputs: W3/stats/q38_english_appendix_20261102.json, W3/stats/q38_english_tex_20261102_{en,zh}.tex (tabular bodies of
tab:self_english and tab:self_english_transfer, blocks "%% <label>" ... "%% end"), W3/stats/trace_counts_q38en_20261102.txt and,
with torch, W3/runs/pro/qwen38_en_20261102/xlang_overlap_20261102.json."""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import W3, Roots, arguments, items, prefer, run_path, summarize, write_text  # noqa: E402
from causal_tests import game_summary  # noqa: E402

EN = 'qwen38_en_20261102'
T = {'en': {'own': 'own', 'User': 'User', 'System': 'System', 'unedited': 'unedited', 'random': 'random', 'k64': '64 dims',
            'copy': 'own (copy)', 'content': 'Content, correct', 'arith': 'Arithmetic, correct', 'control': 'Control, correct',
            'attr': 'Attribution', 'game': 'Secret game', 'zh_items': 'Chinese items', 'en_items': 'English items'},
     'zh': {'own': '自己', 'User': 'User', 'System': 'System', 'unedited': '未编辑', 'random': '随机', 'k64': '64 维',
            'copy': '自己（复制）', 'content': '内容题，答对', 'arith': '算术题，答对', 'control': '对照题，答对',
            'attr': '归属题', 'game': '猜数游戏', 'zh_items': '中文题目', 'en_items': '英文题目'}}
XC = [('swU8_A', 'fit'), ('swS8_A', 'fit'), ('rswU8_A', 'fit'), ('swA8_U', 'fit'), ('swA8_S', 'fit'), ('rswA8_U', 'fit'), ('swA8_U', 'game')]


def sg(x, d=2):
    s = f'{x:+.{d}f}'
    return f'\\({abs(x):.{d}f}\\)' if s in ('+0.00', '-0.00', '+0.000', '-0.000') else f'\\({s}\\)'


def ch(s, ci=True, d=2):
    if s is None or 'diff' not in s:
        return '--'
    d_ = 3 if abs(s['diff']) < 0.005 and s['diff'] != 0 else d
    return sg(s['diff'], d_) + (f" [{sg(s['diff_ci'][0], d_)}, {sg(s['diff_ci'][1], d_)}]" if ci else '')


def val(s):
    if s is None:
        return '--'
    t = f"{s['stat']:.2f}"
    return t + (f" ({s['finished']}/{s['n']})" if s.get('finished', s['n']) != s['n'] else '')


def acc(runs, conds=None):
    xs = [s['accuracy'] for c, s in runs.items() if 'accuracy' in s and (conds is None or c in conds)]
    if not xs:
        return '--'
    lo, hi = min(xs), max(xs)
    return f'{lo:.2f}' if f'{lo:.2f}' == f'{hi:.2f}' else f'{lo:.2f}--{hi:.2f}'


def tex(lang, COLS, X, ratios):
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
            if run == 'all_k64' and j == 0:  # 64 dimensions: with the English items, thinking only
                cells.append('--')
                continue
            cells.append((val if kind == 'v' else ch)(R.get(run, {}).get(c)))
        out.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
    out.append(f"{t['content']} & " + ' & '.join(acc(R.get('all_content', {})) for R, _ in COLS) + ' \\\\')
    out.append(f"{t['arith']} & " + ' & '.join(acc(R.get('all_arith', {})) for R, _ in COLS) + ' \\\\')
    out.append('\\midrule')
    out.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{t['game']}}}}} \\\\")
    for lab, c, kind in ((f"{t['own']} X, {t['unedited']}", 'base_A', 'v'), (f"{t['own']} \\(\\leftarrow\\) {t['User']}", 'swU8_A', 'c'),
                         (f"{t['own']} \\(\\leftarrow\\) {t['User']}, {t['random']}", 'rswU8_A', 'c'),
                         (f"{t['User']} X, {t['unedited']}", 'base_U', 'v'), (f"{t['User']} \\(\\leftarrow\\) {t['own']}", 'swA8_U', 'c'),
                         (f"{t['User']} \\(\\leftarrow\\) {t['own']}, {t['random']}", 'rswA8_U', 'c')):
        out.append(f'{lab} & ' + ' & '.join((val if kind == 'v' else ch)(G.get('main', {}).get(c)) for _, G in COLS) + ' \\\\')
    out.append(f"{t['control']} & " + ' & '.join(acc(G.get('control', {})) for _, G in COLS) + ' \\\\')
    out.append('%% end')
    # transfer: one block per item language; the random subspaces are the same stored subspaces in both fits (left out)
    out.append('%% tab:self_english_transfer')
    lab = {'swU8_A': f"{t['own']} \\(\\leftarrow\\) {t['User']}", 'swS8_A': f"{t['own']} \\(\\leftarrow\\) {t['System']}",
           'swA8_U': f"{t['User']} \\(\\leftarrow\\) {t['own']}", 'swA8_S': f"{t['System']} \\(\\leftarrow\\) {t['own']}"}
    for i, side in enumerate(('zh_items', 'en_items')):
        if i:
            out.append('\\midrule')
        out.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{t[side]}}}}} \\\\")
        S = X[side]
        for c, kind in XC:
            if c.startswith('rsw'):
                continue
            r = ratios[side].get(f'{c}_{kind}', {}).get('ratio')
            row = lab[c] if kind == 'fit' else f"{t['game']}: {lab[c]}"
            out.append(f"\\quad {row} & {ch(S[f'own_{kind}'].get(c))} & {ch(S[f'other_{kind}'].get(c))} & "
                       f"{f'{r:.2f}' if r is not None else '--'} \\\\")
    out.append('%% end')
    return '\n'.join(out) + '\n'


def overlap(D, archived):
    """Overlap of the Chinese and English single-pass subspaces; recomputed with torch, else the archived file."""
    try:
        import torch
    except ImportError:
        return json.loads(Path(archived).read_text(encoding='utf-8')), 'archived (torch not installed)'
    zh, en = (torch.load(D / f'singlepass_ablate/qwen38_27b_{x}/fit.pt', map_location='cpu', weights_only=False) for x in ('zh', 'en'))
    keys = lambda V: sorted(V) if isinstance(V, dict) else list(range(len(V)))  # noqa: E731
    layers = keys(zh['V'])
    d = zh['V'][layers[0]].shape[0]
    out = {'zh_fit': 'singlepass_ablate/qwen38_27b_zh/fit.pt', 'en_fit': 'singlepass_ablate/qwen38_27b_en/fit.pt'}
    for k in (8, 64):
        ov = [float((zh['V'][l][:, :k].double().T @ en['V'][l][:, :k].double()).pow(2).sum() / k) for l in layers]
        rr = [float((zh['R'][l][:, :k].double().T @ en['R'][l][:, :k].double()).pow(2).sum() / k) for l in layers]
        third = len(layers) // 3
        out[f'k{k}'] = {'by_layer': [round(x, 4) for x in ov], 'median': round(statistics.median(ov), 4), 'min': round(min(ov), 4),
                        'max': round(max(ov), 4), 'min_layer': layers[ov.index(min(ov))], 'max_layer': layers[ov.index(max(ov))],
                        'median_by_third': [round(statistics.median(ov[i * third:(i + 1) * third if i < 2 else None]), 4) for i in range(3)],
                        'random_expected': round(k / d, 5), 'stored_random_overlap_median': round(statistics.median(rr), 4)}
    out['hidden'], out['n_layers'] = d, len(layers)
    return out, 'recomputed from the fits'


def describes_assistant(s):
    s_en = s.replace('AI assistant', '').replace('AI Assistant', '')
    s_zh = s.replace('AI 助手', '').replace('AI助手', '')
    return bool(re.search(r'assistant(?! *\))', s_en, re.I)) or '助手' in s_zh


def traces(A3):
    lines = []
    for lang, d8 in (('en', f'{EN}/thinking_swap/qwen38_27b_en_all'), ('zh', 'thinking_swap/qwen38_27b_all')):
        for d, conds in ((d8, ('base_U', 'swA8_U', 'rswA8_U')), (d8 + '_k64', ('swA64_U', 'rswA64_U'))):
            recs = items(str(run_path(A3, d)), cache=False)
            for c in conds:
                xs = [x for x in recs if x['cond'] == c]
                hit = [x for x in xs if describes_assistant(x.get('gen', ''))]
                lines.append(f'{lang} {c:8s} describes an assistant message {len(hit):>3}/{len(xs)}; answers of those '
                             f'{dict(Counter(x.get("on_answer") for x in hit))}; all answers {dict(Counter(x.get("on_answer") for x in xs))}')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    args = arguments(__doc__).parse_args(argv)
    r = Roots(args)
    A3 = r.res(W3)
    D = run_path(A3, EN)
    S = r.out(W3 + '/stats')
    M = json.loads(prefer(r, W3 + '/stats/v4_mechanism.json').read_text(encoding='utf-8'))['results']
    E = json.loads(prefer(r, W3 + '/stats/v4_mech_ext.json').read_text(encoding='utf-8'))
    ZH = {'sp': M['qwen_sp'], 'th': M['qwen_th']}
    ZG = {'sp': E['games']['qwen_sp'], 'th': E['games']['qwen_th']}
    ENR = {'sp': {}, 'th': {}}
    for mode, base in (('sp', 'singlepass_swap/qwen38_27b_en_{}'), ('th', 'thinking_swap/qwen38_27b_en_{}')):
        for k in ('all', 'all_content', 'all_arith', 'all_k64'):
            d = D / base.format(k)
            if d.exists():
                ENR[mode][k] = summarize(items(str(d)), k)
    EG = {'sp': {q: game_summary(str(D / f'games_swap/qwen38_27b_en_sp_s6_{q}')) for q in ('main', 'control')},
          'th': {q: game_summary(str(D / f'games_swap/qwen38_27b_en_s6_{q}')) for q in ('main', 'control')}}
    X = {'zh_items': {'own_fit': summarize(items(str(D / 'singlepass_swap/qwen38_27b_zh_all_pro')), 'all'),
                      'other_fit': summarize(items(str(D / 'singlepass_swap/qwen38_27b_xl_enfit_zh')), 'all'),
                      'own_game': game_summary(str(D / 'games_swap/qwen38_27b_zh_pro_sp_s6_main')),
                      'other_game': game_summary(str(D / 'games_swap/qwen38_27b_xl_enfit_zh_sp_s6_main'))},
         'en_items': {'own_fit': ENR['sp']['all'], 'other_fit': summarize(items(str(D / 'singlepass_swap/qwen38_27b_xl_zhfit_en')), 'all'),
                      'own_game': EG['sp']['main'], 'other_game': game_summary(str(D / 'games_swap/qwen38_27b_xl_zhfit_en_sp_s6_main'))}}
    ratios = {}
    for side, Sd in X.items():
        ratios[side] = {}
        for c, kind in XC:
            o, t = Sd[f'own_{kind}'].get(c, {}), Sd[f'other_{kind}'].get(c, {})
            if 'diff' in o and 'diff' in t:
                ratios[side][f'{c}_{kind}'] = {'own': o['diff'], 'other': t['diff'],
                                               'ratio': t['diff'] / o['diff'] if abs(o['diff']) > 0.05 else None}
    OV, how = overlap(D, D / 'xlang_overlap_20261102.json')
    if how.startswith('recomputed'):
        write_text(r.out(W3 + f'/runs/pro/{EN}') / 'xlang_overlap_20261102.json', json.dumps(OV, indent=1))
    json.dump({'en': ENR, 'en_games': EG, 'xlang': X, 'xlang_ratios': ratios,
               'overlap': {k: {kk: v for kk, v in OV[k].items() if kk != 'by_layer'} for k in ('k8', 'k64')}},
              open(S / 'q38_english_appendix_20261102.json', 'w', encoding='utf-8'), indent=1)
    COLS = [(ENR['sp'], EG['sp']), (ZH['sp'], ZG['sp']), (ENR['th'], EG['th']), (ZH['th'], ZG['th'])]
    for lang in ('en', 'zh'):
        write_text(S / f'q38_english_tex_20261102_{lang}.tex', tex(lang, COLS, X, ratios))
    write_text(S / 'trace_counts_q38en_20261102.txt', traces(A3))
    print(tex('en', COLS, X, ratios))
    print('overlap k8 (' + how + '):', {k: v for k, v in OV['k8'].items() if k != 'by_layer'})
    print(open(S / 'trace_counts_q38en_20261102.txt', encoding='utf-8').read())


if __name__ == '__main__':
    main()
