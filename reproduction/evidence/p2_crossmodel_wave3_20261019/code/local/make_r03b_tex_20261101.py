"""Table tab:self_fd (appendix app:mech_fd, 11-01): R03 Part B, the first-order sensitivity of the attribution answer, by band of
layers, from stats/r03b_layers_20261101.json (analyze_r03b_layers_20261101.py), so that no number is typed by hand; with `fill`
the body is written between "%<<tab:self_fd" and "%>>tab:self_fd" of paper/source/self_{en,zh}_appendix.tex (bytes in, bytes out,
CRLF kept). Per readout and band (first, middle, last third; the installation band of the layer windows): the summed slope along the
own-User difference at the User message and at the model's own message, the share carried by the eight coordinates with its 95%
interval, given only where the summed slope is positive with its 95% interval above zero (rule fixed here, before filling), and the
median share of the squared norm of the difference held by the eight coordinates. The numbers quoted in the text are printed.
usage: make_r03b_tex_20261101.py ARCHIVE [fill]"""
import json
import re
import sys
from pathlib import Path

A = Path(sys.argv[1])
R = json.load(open(A / 'stats' / 'r03b_layers_20261101.json', encoding='utf-8'))['results']
SRC = A.resolve().parents[1] / 'paper' / 'source'
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
    (A / 'stats' / f'r03b_tex_{lang}.tex').write_text('\n'.join(['%% tab:self_fd'] + body + ['%% end']) + '\n', encoding='utf-8')
    sys.stdout.buffer.write(('\n'.join(body) + '\n').encode('utf-8'))
text_numbers()
if len(sys.argv) > 2 and sys.argv[2] == 'fill':
    for lang in ('en', 'zh'):
        p = SRC / f'self_{lang}_appendix.tex'
        raw = p.read_bytes()
        crlf = b'\r\n' in raw
        s = raw.decode('utf-8').replace('\r\n', '\n')
        pat = re.compile(r'(^%<<tab:self_fd\n)(.*?)(^%>>tab:self_fd$)', re.S | re.M)
        assert len(pat.findall(s)) == 1, p.name
        s = pat.sub(lambda m: m.group(1) + '\n'.join(block(lang)) + '\n' + m.group(3), s)
        p.write_bytes((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
        print('filled', p.name, 'tab:self_fd')
