"""LaTeX table bodies of the causal tests, so that no number in these tables is typed by hand. Each output file holds one or
more blocks between the lines "%% <label>" and "%% end"; in the paper source the same bodies stand between the comment lines
%<<label and %>>label (tabular bodies only; captions and headers are in the source). The numbers that the text quotes are
printed for checking. Inputs are the outputs of causal_tests.py, causal_controls.py and thresholds_orders.py in OUT.

mech_tex -> stats/v4_mech_tex_{en,zh}.tex (from v4_mechanism.json and v4_mech_ext.json)
  tab:self_mech_questions  the four questions per readout (attribution and game: removal, installation; content, arithmetic:
                           lowest-highest share correct over the six conditions)
  tab:self_mechanism       attribution, every layer: unedited values, changes for 8 and 64 dimensions and random subspaces
  tab:self_game            secret game: unedited share acting as setter, changes, control
  tab:self_mechanism_where groups of positions (appendix)
  tab:self_mechanism_windows layer windows (appendix)
  tab:self_mech_gemma, tab:self_mech_gemma_windows  Gemma-4-31B, natural order, no thinking (not in the current paper)
question_tex -> stats/r05r06_tex_{en,zh}.tex (from r05r06_20261031.json)
  tab:self_question  attribution without the question: unedited own and User messages; removal (own <- User) and installation
                     (User <- own) for the two subspaces (fit_q: fitted with the question, as in the main runs; fit_m: on the
                     message alone) crossed with the edited positions (all; the message alone), runs r05q_all, r05q_msg, r05m_all,
                     r05m_msg
  tab:self_seeds     random subspaces of five consecutive base seeds (20261022, the seed of tab:self_mechanism, run again in
                     r05q_all; 20261023..20261026 in r06_s1..4), every layer and position
  also prints the overlap of the fitted subspaces fit_q and fit_m (median over layers, from r05r06_fits/<readout>/fit_diag.json)
fd_tex -> stats/r03b_tex_{en,zh}.tex (from r03b_layers_20261101.json)
  tab:self_fd  the first-order sensitivity of the attribution answer by band of layers (first, middle, last third; the
               installation band of the layer windows): the summed slope along the own-User difference at the User message and
               at the model's own message, the share carried by the eight coordinates with its 95% interval, given only where
               the summed slope is positive with its 95% interval above zero, and the median share of the squared norm of the
               difference held by the eight coordinates
cluster_threshold_tex -> stats/r09_r04_tex_{en,zh}.tex
  tab:self_cluster    the causal tests with 95% intervals over the 15 held-out wording families (r09_cluster_20261101.json;
                      attribution: the seven conditions of the main runs; content question: the four edited conditions)
  tab:self_threshold  the self-index criterion at thresholds of 70%, 80% (the paper) and 90%, with the last third and the last
                      quarter of the depth (r04_threshold_20261101.json, from thresholds_orders.py)
exchange_tex -> stats/tran2_tex_{en,zh}.tex
  tab:self_exchange       main text: change in "you" when a message receives the coordinates of another role; across the
                          boundary from v4_mechanism.json (the paper's runs, as Table tab:self_mechanism), within the outside from
                          r08_20261031.json (the exchange runs <dir>_r08_us)
  tab:self_strict         the stricter tasks, p56full_20261101.json: attribution in a held-out wording and the secret game with X
                          alone edited, both with fit_m
  tab:self_seeds_scaled   the random edits of the five seeds scaled to at least the size of the fitted edit, r06f_20261031.json
  tab:self_rest           the inverted replacement (all but the eight coordinates) against the eight coordinates, the whole state
                          and the random subspace, r03_20261101.json (part A)
  tab:self_exchange_full  the exchange of the User and System coordinates in full (you, user, system), r08_20261031.json
ablation_tex -> stats/v5_appendix_tex_en.tex (from the run summaries)
  tab:self_ablation_2x2  attribution under the format ablation (Qwen3.8-27B template switch, both modes; gpt-oss-20b and
                         gpt-oss-120b, channel tag masked with positions recomputed): unedited claims and item-paired changes
                         (qwen38_nothink/{singlepass_swap,thinking_swap}/qwen38_27b_nothink_all,
                         qwen38_nothink/thinking_swap/gpt_oss_{20b,120b}_mask)
  tab:self_positions     geometry with and without position compression: layers at which the own message is singled out in at
                         least 80% of the held-out bodies, content and onset (qwen38_nothink/suite_mask/qwen38_27b_*,
                         qwen38_nothink/suite_think/gpt_oss_*, suite_mask/gpt_oss_*_compress)

usage: causal_tables.py --results RES --out OUT [--only NAME[,NAME...]]
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import W3, Roots, arguments, run_path  # noqa: E402


def mech_tex(A, S):
    M = json.load(open(S / 'v4_mechanism.json', encoding='utf-8'))
    E = json.load(open(S / 'v4_mech_ext.json', encoding='utf-8'))
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
            for w_, S_ in sorted(W.get(ro, {}).items(), key=lambda kv: kv[1]['layers'][0]):
                out.append(f"{w_.replace('-', '--')} & {ch(S_.get('swU8_A'))} & {ch(S_.get('swA8_U'))} & {ch(S_.get('rswU8_A'))} \\\\")
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
            acc = lambda S_: f"{S_[c]['accuracy']:.2f}" if c in S_ else '--'  # noqa: E731
            out.append(f"{lab} & {f8(gm.get(c))} & {f8(gk.get(k64[c])) if c in k64 else '--'} & {fg(gg.get(c)) if c in gg else '--'} & "
                       f"{acc(gc)} & {acc(ga)} \\\\")
        out.append('%% end')
        out.append('%% tab:self_mech_gemma_windows')
        for w_, S_ in sorted(E['windows'].get('gemma_sp', {}).items(), key=lambda kv: kv[1]['layers'][0]):
            out.append(f"{w_.replace('-', '--')} & {ch(S_.get('swU8_A'))} & {ch(S_.get('swA8_U'))} & {ch(S_.get('rswU8_A'))} \\\\")
        out.append('%% end')
        p = S / f'v4_mech_tex_{lang}.tex'
        p.write_text('\n'.join(out).replace('\\(-', '\\(-') + '\n', encoding='utf-8')
        return p

    for lang in ('en', 'zh'):
        print(build(lang))


def question_tex(A, S):
    R = json.load(open(S / 'r05r06_20261031.json', encoding='utf-8'))['results']
    RO = ['qsp', 'qth', 'g20', 'g120']
    FITS = ['r05q_all', 'r05q_msg', 'r05m_all', 'r05m_msg']
    SEEDS = [('r05q_all', 20261022), ('r06_s1', 20261023), ('r06_s2', 20261024), ('r06_s3', 20261025), ('r06_s4', 20261026)]
    L = {'en': {'own': 'own', 'unedited': 'unedited', 'random': 'random',
                'fits': ['with question, all', 'with question, message', 'message, all', 'message, message']},
         'zh': {'own': '自己', 'unedited': '未编辑', 'random': '随机',
                'fits': ['含问题，全部', '含问题，消息', '消息，全部', '消息，消息']}}

    def sg(x, d=2):
        """signed number in math mode; 0.00 without sign"""
        s = f'{x:+.{d}f}'
        if s in ('+0.00', '-0.00', '+0.000', '-0.000'):
            return f'\\({abs(x):.{d}f}\\)'
        return f'\\({s}\\)'

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

    def row(lab, cells):
        return f'{lab} & ' + ' & '.join(cells) + ' \\\\'

    def block_question(lang):
        t, out = L[lang], []
        for base, cond, head in (('base_A', 'swU8_A', f"{t['own']} \\(\\leftarrow\\) User"),
                                 ('base_U', 'swA8_U', f"User \\(\\leftarrow\\) {t['own']}")):
            if base == 'base_U':
                out.append('\\midrule')
            lab = f"{t['own']}, {t['unedited']}" if base == 'base_A' else f"User, {t['unedited']}"
            out.append(row(lab, [val(R[ro]['r05q_all']['stats'].get(base)) for ro in RO]))
            out.append(f'\\multicolumn{{5}}{{l}}{{{head}}} \\\\')
            for run, rl in zip(FITS, t['fits']):
                out.append(row(f'\\quad {rl}', [ch(R[ro][run]['stats'].get(cond)) for ro in RO]))
        return out

    def block_seeds(lang):
        t, out = L[lang], []
        for cond, head in (('rswU8_A', f"{t['own']} \\(\\leftarrow\\) User, {t['random']}"),
                           ('rswA8_U', f"User \\(\\leftarrow\\) {t['own']}, {t['random']}")):
            if cond == 'rswA8_U':
                out.append('\\midrule')
            out.append(f'\\multicolumn{{5}}{{l}}{{{head}}} \\\\')
            for run, seed in SEEDS:
                out.append(row(f'\\quad {seed}', [ch(R[ro][run]['stats'].get(cond)) for ro in RO]))
        return out

    def text_numbers():
        """the numbers that the text of sec:abl_question / app:mech_question quotes"""
        print('readout | removal, message fit + message edit (gap closed) | installation, same (gap) | main condition rerun: removal, installation')
        for ro in RO:
            s0 = R[ro]['r05q_all']['stats']
            gap = s0['base_A']['stat'] - s0['base_U']['stat']
            m = R[ro]['r05m_msg']['stats']
            print(f"{ro} | {m['swU8_A']['diff']:+.2f} ({-m['swU8_A']['diff'] / gap:.2f}) | {m['swA8_U']['diff']:+.2f} {m['swA8_U']['diff_ci']} "
                  f"({m['swA8_U']['diff'] / gap:.2f}) | {s0['swU8_A']['diff']:+.2f}, {s0['swA8_U']['diff']:+.2f}")
        for run in ('r05q_msg', 'r05q_all'):
            xs = [R[ro][run]['stats']['swA8_U']['diff'] for ro in RO]
            print(f'installation, {run}: {min(xs):+.2f} to {max(xs):+.2f}')
        for cond in ('rswU8_A', 'rswA8_U'):
            print(f'{cond}: largest |change| over the five seeds', max(abs(R[ro][r]['stats'][cond]['diff']) for ro in RO for r, _ in SEEDS))
        ov = {}
        for ro in RO:
            d = json.load(open(run_path(A, f'r05r06_fits/{ro}/fit_diag.json'), encoding='utf-8'))['layers']
            ov[ro] = statistics.median(x['overlap_q_m'] for x in d.values())
        print('overlap of the 8-dim spaces of fit_q and fit_m, median over layers:', {k: round(v, 2) for k, v in ov.items()})

    BLOCKS = {'tab:self_question': block_question, 'tab:self_seeds': block_seeds}
    for lang in ('en', 'zh'):
        text = []
        for lab, f in BLOCKS.items():
            text += [f'%% {lab}'] + f(lang) + ['%% end']
        (S / f'r05r06_tex_{lang}.tex').write_text('\n'.join(text) + '\n', encoding='utf-8')
        print('\n'.join(text))
    text_numbers()


def fd_tex(A, S):
    R = json.load(open(S / 'r03b_layers_20261101.json', encoding='utf-8'))['results']
    RO = ['qsp', 'qth', 'g20', 'g120']
    BANDS = ['first third', 'middle third', 'last third', 'install band']
    L = {'en': {'ro': {'qsp': 'Qwen3.8-27B, no thinking', 'qth': 'Qwen3.8-27B, thinking', 'g20': 'gpt-oss-20b, thinking',
                       'g120': 'gpt-oss-120b, thinking'},
                'band': {'first third': 'first third', 'middle third': 'middle third', 'last third': 'last third',
                         'install band': 'installation band'}},
         'zh': {'ro': {'qsp': 'Qwen3.8-27B，不思考', 'qth': 'Qwen3.8-27B，思考', 'g20': 'gpt-oss-20b，思考', 'g120': 'gpt-oss-120b，思考'},
                'band': {'first third': '前三分之一', 'middle third': '中三分之一', 'last third': '最后三分之一', 'install band': '装入层段'}}}

    def num(x):
        s = f'{x:.2f}'
        return f'\\({"0.00" if s in ("-0.00", "0.00") else s}\\)'

    def sl(x):
        s = f'{x:+.2f}'
        return f'\\({s}\\)' if s not in ('+0.00', '-0.00') else '\\(0.00\\)'

    def share(b):
        if not (b['sum_mean_D_total'] > 0 and b['sum_ci'][0] > 0):
            return '--'
        return f"{num(b['V_share'])} [{num(b['V_share_ci'][0])}, {num(b['V_share_ci'][1])}]"

    def block(lang):
        t, out = L[lang], []
        for i, ro in enumerate(RO):
            if i:
                out.append('\\midrule')
            out.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{t['ro'][ro]}}}}} \\\\")
            for bn in BANDS:
                u, a = R[ro]['bands']['U'][bn], R[ro]['bands']['A'][bn]
                lay = f"L{u['layers'][0]}--L{u['layers'][1]}"
                out.append(f"\\quad {t['band'][bn]} ({lay}) & {sl(u['sum_mean_D_total'])} & {share(u)} & {sl(a['sum_mean_D_total'])} & "
                           f"{share(a)} & {num(u['median_size_share_V'])} \\\\")
        return out

    def text_numbers():
        q = R['qsp']['bands']['U']
        tot = sum(q[b]['sum_mean_D_total'] for b in ('first third', 'middle third', 'last third'))
        print('qsp User end: middle third V share %.2f (size %.2f), slope %.2f of %.2f; last third V share %.2f (size %.2f)' % (
            q['middle third']['V_share'], q['middle third']['median_size_share_V'], q['middle third']['sum_mean_D_total'], tot,
            q['last third']['V_share'], q['last third']['median_size_share_V']))
        for ro in RO:
            print(ro, {ep: {bn: share(R[ro]['bands'][ep][bn]).replace('\\(', '').replace('\\)', '') for bn in BANDS} for ep in ('U', 'A')})

    for lang in ('en', 'zh'):
        body = block(lang)
        (S / f'r03b_tex_{lang}.tex').write_text('\n'.join(['%% tab:self_fd'] + body + ['%% end']) + '\n', encoding='utf-8')
        print('\n'.join(body))
    text_numbers()


def cluster_threshold_tex(A, S):
    R09 = json.load(open(S / 'r09_cluster_20261101.json', encoding='utf-8'))['results']
    R04 = json.load(open(S / 'r04_threshold_20261101.json', encoding='utf-8'))
    RO = ['qsp', 'qth', 'g20', 'g120']
    from thresholds_orders import SELFREF  # the models and modes with self-reference (tab:self_crossmodel), one list for both scripts
    TS = ('0.7', '0.8', '0.9')
    L = {'en': {'own': 'own', 'random': 'random', 'attr': "Attribution: change in ``you''",
                'content': 'Content question: change in the share answered correctly',
                'rows': ['Models with a self-index', '\\quad with self-reference', '\\quad without self-reference', 'Models with neither level',
                         'Models with self-reference but no self-index',
                         f'Model--modes with self-reference (of {len(SELFREF)}) whose own message is singled out at every layer of the last third',
                         '\\quad at every layer of the last quarter',
                         'Most layers of the last third at which a model without a self-index singles out its own message'],
                'of': lambda a, b: f'{a}/{b}'},
         'zh': {'own': '自己', 'random': '随机', 'attr': '归属问题：“你”的变化', 'content': '内容问题：答对比例的变化',
                'rows': ['具有自我索引的模型', '\\quad 其中有自指', '\\quad 其中没有自指', '两层都没有的模型', '有自指而没有自我索引的模型',
                         f'有自指的模型--模式（共 {len(SELFREF)} 个）中，自己的消息在最后三分之一的每一层都被单独分出的',
                         '\\quad 在最后四分之一的每一层都被单独分出的', '没有自我索引的模型在最后三分之一中单独分出自己消息的最多层数'],
                'of': lambda a, b: f'{a}/{b}'}}

    def sg(x, d=2):
        """signed number in math mode; 0.00 without sign"""
        s = f'{x:+.{d}f}'
        if s in ('+0.00', '-0.00', '+0.000', '-0.000'):
            return f'\\({abs(x):.{d}f}\\)'
        return f'\\({s}\\)'

    def cl(s):
        """change with the family interval, -0.76 [-0.78, -0.73]"""
        v, (lo, hi) = s['change'], s['cluster_ci']
        d = 3 if abs(v) < 0.005 and v != 0 else 2
        return f'{sg(v, d)} [{sg(lo, d)}, {sg(hi, d)}]'

    def row(lab, cells):
        return f'{lab} & ' + ' & '.join(cells) + ' \\\\'

    def head(text):
        return f'\\multicolumn{{5}}{{l}}{{\\emph{{{text}}}}} \\\\'

    def arrow(a, b):
        return f'{a} \\(\\leftarrow\\) {b}'

    def block_cluster(lang):
        t = L[lang]
        own, rnd = t['own'], t['random']
        attr = [('own <- User', arrow(own, 'User')), ('own <- System', arrow(own, 'System')),
                ('own <- User, random', f"{arrow(own, 'User')}, {rnd}"), ('own <- own (copy)', arrow(own, own)),
                ('User <- own', arrow('User', own)), ('User <- own, random', f"{arrow('User', own)}, {rnd}"),
                ('System <- own', arrow('System', own))]
        out = [head(t['attr'])]
        for key, lab in attr:
            out.append(row(lab, [cl(R09[ro]['attribution'][key]) for ro in RO]))
        out += ['\\midrule', head(t['content'])]
        for c, lab in (('swU8_A', arrow(own, 'User')), ('swS8_A', arrow(own, 'System')), ('rswU8_A', f"{arrow(own, 'User')}, {rnd}"),
                       ('swA8_U', arrow('User', own))):
            out.append(row(lab, [cl(R09[ro]['content'][c]) for ro in RO]))
        return out

    def threshold_counts(t):
        M = R04['models']
        cls = {m: v['by_rendering'][v['classification_rendering']][t] for m, v in M.items()}
        srm = {m for m, _ in SELFREF}
        full = lambda x: x[0] == x[1]  # noqa: E731
        noidx = [(v[t]['last_third'][0], v[t]['last_third'][1], m, mode) for m in M if not cls[m]['index']
                 for mode, v in M[m]['by_rendering'].items()]
        return {'index': sum(c['index'] for c in cls.values()), 'index_sr': sum(cls[m]['index'] for m in srm),
                'index_only': sum(cls[m]['index'] and m not in srm for m in M), 'neither': sum(not cls[m]['index'] and m not in srm for m in M),
                'sr_noindex': sum(not cls[m]['index'] for m in srm),
                'third': sum(full(M[m]['by_rendering'][mode][t]['last_third']) for m, mode in SELFREF),
                'quarter': sum(full(M[m]['by_rendering'][mode][t]['last_quarter']) for m, mode in SELFREF),
                'noidx_max': max(noidx), 'index_models': sorted(m for m in M if cls[m]['index'])}

    def block_threshold(lang):
        t = L[lang]
        c = {k: threshold_counts(k) for k in TS}
        keys = ['index', 'index_sr', 'index_only', 'neither', 'sr_noindex', 'third', 'quarter']
        out = [row(lab, [str(c[k][key]) for k in TS]) for lab, key in zip(t['rows'][:5], keys[:5])]
        out.append('\\midrule')
        out += [row(lab, [str(c[k][key]) for k in TS]) for lab, key in zip(t['rows'][5:7], keys[5:7])]
        out.append(row(t['rows'][7], [str(c[k]['noidx_max'][0]) if c[k]['noidx_max'][0] == 0 else t['of'](*c[k]['noidx_max'][:2]) for k in TS]))
        return out

    APP = {'tab:self_cluster': block_cluster, 'tab:self_threshold': block_threshold}

    def text_numbers():
        f2 = lambda xs: ', '.join(f'{x:+.2f}' for x in xs)  # noqa: E731
        a = {ro: R09[ro]['attribution'] for ro in RO}
        print('family CI end nearest zero, removal User:', f2(a[ro]['own <- User']['cluster_ci'][1] for ro in RO),
              '| removal System:', f2(a[ro]['own <- System']['cluster_ci'][1] for ro in RO),
              '| installation User:', f2(a[ro]['User <- own']['cluster_ci'][0] for ro in RO))
        rnd = [x for ro in RO for k in ('own <- User, random', 'User <- own, random') for x in a[ro][k]['cluster_ci']]
        print('random family intervals: range', min(rnd), max(rnd), '| contain 0:',
              [(ro, k) for ro in RO for k in ('own <- User, random', 'User <- own, random')
               if not a[ro][k]['cluster_ci'][0] <= 0 <= a[ro][k]['cluster_ci'][1]])
        worst = []
        for ro in RO[1:]:
            for k, s in a[ro].items():
                worst.append((round(max(abs(s['bounds'][0] - s['change']), abs(s['bounds'][1] - s['change'])), 3), ro, k, s['finished'], s['pairs'],
                              [round(x, 2) for x in s['bounds']], round(s['change'], 3)))
        worst.sort(reverse=True)
        print('largest distance of a bound from the change:', worst[:3])
        cont = [abs(R09[ro]['content'][c]['change']) for ro in RO for c in R09[ro]['content']]
        ari = [abs(R09[ro]['arith'][c]['change']) for ro in RO for c in R09[ro]['arith']]
        print('content: largest |change|', round(max(cont), 3), '| arithmetic: largest |change|', max(ari))
        for k in TS:
            c = threshold_counts(k)
            print('threshold', k, {x: c[x] for x in c if x != 'index_models'}, 'index models:', c['index_models'])
        M = R04['models']
        for k in TS:
            for m, mode in SELFREF:
                d = M[m]['by_rendering'][mode][k]
                if d['last_third'][0] != d['last_third'][1] or d['last_quarter'][0] != d['last_quarter'][1]:
                    st = M[m].get('streams', {}).get(mode, {})
                    print('  not full at', k, m, mode, 'third', d['last_third'], 'quarter', d['last_quarter'],
                          '| streams third', [v[k]['last_third'][0] for v in st.values()], 'quarter', [v[k]['last_quarter'][0] for v in st.values()])
        for k in ('0.7',):
            for m in ('gemma4_31b', 'olmo2_13b'):
                v = M[m]
                print('  70%:', m, v['by_rendering'][v['classification_rendering']][k])

    for lang in ('en', 'zh'):
        text = []
        for lab, f in APP.items():
            text += [f'%% {lab}'] + f(lang) + ['%% end']
        (S / f'r09_r04_tex_{lang}.tex').write_text('\n'.join(text) + '\n', encoding='utf-8')
        print('\n'.join(text))
    text_numbers()


def exchange_tex(A, S):
    def load(name):
        return json.load(open(S / name, encoding='utf-8'))['results']

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
        (S / f'tran2_tex_{lang}.tex').write_text('\n'.join(text) + '\n', encoding='utf-8')
        print('\n'.join(text))
    text_numbers()


def ablation_tex(A, S):
    def summ(p):
        return json.load(open(run_path(A, p) / 'summary.json', encoding='utf-8'))

    def sg(x):
        s = f'{x:+.2f}'
        return f'\\({abs(x):.2f}\\)' if s in ('+0.00', '-0.00') else f'\\({s}\\)'

    def stat(v):
        """unedited value: P(you) for single-pass runs, share answered 'you' with thinking"""
        x = v['p_self'] if 'p_self' in v else v['you_share']
        return f'{x:.2f}'

    def ch(v):
        if v is None:
            return '--'
        m = v.get('you_minus_base') or v.get('p_minus_base')
        if not m:
            return '--'
        lo, hi = m['ci95']
        return f"{sg(m['mean'])} [{sg(lo)}, {sg(hi)}]"

    def block_ablation():
        runs = [('qwen38_nothink/singlepass_swap/qwen38_27b_nothink_all', 'sp'), ('qwen38_nothink/thinking_swap/qwen38_27b_nothink_all', 'th'),
                ('qwen38_nothink/thinking_swap/gpt_oss_20b_mask', 'th'), ('qwen38_nothink/thinking_swap/gpt_oss_120b_mask', 'th')]
        S_ = [summ(p) for p, _ in runs]
        rows = [('own, unedited', 'base_A', 'v'), ('own \\(\\leftarrow\\) User', 'swU8_A', 'c'), ('own \\(\\leftarrow\\) System', 'swS8_A', 'c'),
                ('own \\(\\leftarrow\\) User, random', 'rswU8_A', 'c'), ('own \\(\\leftarrow\\) own (copy)', 'idA8_A', 'c'), None,
                ('User, unedited', 'base_U', 'v'), ('User \\(\\leftarrow\\) own', 'swA8_U', 'c'), ('User \\(\\leftarrow\\) own, random', 'rswA8_U', 'c'),
                None, ('System, unedited', 'base_S', 'v'), ('System \\(\\leftarrow\\) own', 'swA8_S', 'c')]
        out = []
        for r in rows:
            if r is None:
                out.append('\\midrule')
                continue
            lab, c, kind = r
            cells = []
            for s in S_:
                v = s.get(c)
                cells.append('--' if v is None else stat(v) if kind == 'v' else ch(v))
            out.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
        return out

    def block_positions():
        G = 'qwen38_nothink/suite_mask/'
        pairs = [('Qwen3.8-27B, non-thinking rendering', G + 'qwen38_27b_default', G + 'qwen38_27b_compress'),
                 ('Qwen3.8-27B, thinking rendering', G + 'qwen38_27b_think_default', G + 'qwen38_27b_think_compress_v2'),
                 ('gpt-oss-20b', 'qwen38_nothink/suite_think/gpt_oss_20b_mxfp4_en', 'suite_mask/gpt_oss_20b_mxfp4_en_compress'),
                 ('gpt-oss-120b', 'qwen38_nothink/suite_think/gpt_oss_120b_en', 'suite_mask/gpt_oss_120b_en_compress')]
        out = []
        for name, d, c in pairs:
            a, b = summ(d), summ(c)
            L = a['n_layers']
            out.append(f"{name} & {a['content']['layers_A_lowest_80']}/{L} & {b['content']['layers_A_lowest_80']}/{L} & "
                       f"{a['onset']['layers_A_lowest_80']}/{L} & {b['onset']['layers_A_lowest_80']}/{L} \\\\")
        return out

    BLOCKS = {'tab:self_ablation_2x2': block_ablation, 'tab:self_positions': block_positions}
    text = []
    for lab, f in BLOCKS.items():
        text += [f'%% {lab}'] + f() + ['%% end']
    (S / 'v5_appendix_tex_en.tex').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print('\n'.join(text))


STEPS = [('mech_tex', mech_tex), ('question_tex', question_tex), ('fd_tex', fd_tex), ('cluster_threshold_tex', cluster_threshold_tex),
         ('exchange_tex', exchange_tex), ('ablation_tex', ablation_tex)]


def main(argv=None):
    ap = arguments(__doc__)
    ap.add_argument('--only', help='comma-separated subset of: ' + ', '.join(n for n, _ in STEPS))
    args = ap.parse_args(argv)
    r = Roots(args)
    A, S = r.res(W3), r.out(W3 + '/stats')
    only = set(args.only.split(',')) if args.only else None
    for name, fn in STEPS:
        if only is None or name in only:
            print(f'== {name}', flush=True)
            fn(A, S)


if __name__ == '__main__':
    main()
