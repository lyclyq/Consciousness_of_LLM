"""LaTeX table bodies of R05 / R06 (supervisor's review) for the appendix "Further Causal Tests" (paper revision of 10-31), from
stats/r05r06_20261031.json (analyze_r05r06_20261031.py), so that no number in them is typed by hand; with `fill` they are written
between the lines "%<<label" and "%>>label" of paper/source/self_en_appendix.tex and self_zh_appendix.tex (bytes in, bytes out,
CRLF kept). Number format as make_v4_mech_tex.py (sg, ch, val, copied here: that script writes its outputs when imported).
  tab:self_question  attribution without the question: unedited own and User messages; removal (own <- User) and installation
                     (User <- own) for the two subspaces (fit_q: fitted with the question, as in the main runs; fit_m: on the
                     message alone) crossed with the edited positions (all; the message alone), PRO runs r05q_all, r05q_msg,
                     r05m_all, r05m_msg
  tab:self_seeds     random subspaces of five consecutive base seeds (20261022, the seed of tab:self_mechanism, run again in
                     r05q_all; 20261023..20261026 in r06_s1..4), every layer and position
The numbers quoted in the text of sec:abl_question and app:mech_question are printed for checking.
usage: make_r05r06_tex_20261031.py ARCHIVE [fill]"""
import json
import re
import statistics
import sys
from pathlib import Path

A = Path(sys.argv[1])
R = json.load(open(A / 'stats' / 'r05r06_20261031.json', encoding='utf-8'))['results']
SRC = A.resolve().parents[1] / 'paper' / 'source'
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
        d = json.load(open(A / 'runs' / 'pro' / 'r05r06_fits' / ro / 'fit_diag.json', encoding='utf-8'))['layers']
        ov[ro] = statistics.median(x['overlap_q_m'] for x in d.values())
    print('overlap of the 8-dim spaces of fit_q and fit_m, median over layers:', {k: round(v, 2) for k, v in ov.items()})


BLOCKS = {'tab:self_question': block_question, 'tab:self_seeds': block_seeds}
for lang in ('en', 'zh'):
    text = []
    for lab, f in BLOCKS.items():
        text += [f'%% {lab}'] + f(lang) + ['%% end']
    (A / 'stats' / f'r05r06_tex_{lang}.tex').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print('\n'.join(text))
text_numbers()
if len(sys.argv) > 2 and sys.argv[2] == 'fill':
    for lang in ('en', 'zh'):
        p = SRC / f'self_{lang}_appendix.tex'
        raw = p.read_bytes()
        crlf = b'\r\n' in raw
        s = raw.decode('utf-8').replace('\r\n', '\n')
        for lab, f in BLOCKS.items():
            pat = re.compile(rf'(^%<<{re.escape(lab)}\n)(.*?)(^%>>{re.escape(lab)}$)', re.S | re.M)
            assert len(pat.findall(s)) == 1, (p.name, lab)
            body = '\n'.join(f(lang))
            s = pat.sub(lambda m: m.group(1) + body + '\n' + m.group(3), s)
            print('filled', p.name, lab)
        p.write_bytes((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
