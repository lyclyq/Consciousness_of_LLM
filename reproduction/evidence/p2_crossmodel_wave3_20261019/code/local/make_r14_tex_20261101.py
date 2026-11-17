"""LaTeX table body for R14 of the supervisor's review in the paper (11-01): tab:self_orders, single-pass self-reference under the
main probe over the first two cyclic option orders (the paper) and over all three, from stats/r14_orders_20261101.json, so that no
number is typed by hand; with `fill` it is written between "%<<tab:self_orders" and "%>>tab:self_orders" of
paper/source/self_{en,zh}_appendix.tex, bytes in, bytes out, CRLF kept. Number format as make_tran2_tex_20261101.py.
usage: make_r14_tex_20261101.py ARCHIVE [fill]"""
import json
import re
import sys
from pathlib import Path

A = Path(sys.argv[1])
SRC = A.resolve().parents[1] / 'paper' / 'source'
R = json.load(open(A / 'stats' / 'r14_orders_20261101.json', encoding='utf-8'))
NAMES = {'qwen38_27b': 'Qwen3.8-27B', 'qwen38_flashnext': 'Qwen3.8-Flash-Next', 'exaone4_32b': 'EXAONE-4.0.1-32B',
         'glm46v_flash': 'GLM-4.6V-Flash', 'glm47_flash': 'GLM-4.7-Flash', 'granite42_30b': 'Granite-4.2-30B',
         'granite41_30b': 'Granite-4.1-30B', 'olmo2_13b': 'OLMo-2-13B', 'olmo2_7b': 'OLMo-2-7B', 'seed_oss_36b': 'Seed-OSS-36B',
         'gemma4_31b': 'Gemma-4-31B', 'qwen36_27b': 'Qwen3.6-27B', 'qwen36_35b': 'Qwen3.6-35B-A3B', 'qwen35_27b': 'Qwen3.5-27B'}
YES = {'en': ('yes', 'no'), 'zh': ('有', '无')}


def sg(x, d=2):
    """signed number in math mode; 0.00 without sign"""
    s = f'{x:+.{d}f}'
    if s in ('+0.00', '-0.00'):
        return f'\\({abs(x):.{d}f}\\)'
    return f'\\({s}\\)'


def block_orders(lang):
    out = []
    for m, name in NAMES.items():
        v = R['results'][m]
        a, b = v['two_orders'], v['three_orders']
        out.append(f"{name} & {YES[lang][0] if v['index'] else YES[lang][1]} & {sg(a[3])} & {a[4]} & {sg(b[3])} & {b[4]} \\\\")
    return out


APP = {'tab:self_orders': block_orders}
for lang in ('en', 'zh'):
    text = []
    for lab, f in APP.items():
        text += [f'%% {lab}'] + f(lang) + ['%% end']
    (A / 'stats' / f'r14_tex_{lang}.tex').write_text('\n'.join(text) + '\n', encoding='utf-8')
    sys.stdout.buffer.write(('\n'.join(text) + '\n').encode('utf-8'))
print('nested', R['nested'], 'changed', R['changed'], 'no-index max', R['noindex_max'])
if len(sys.argv) > 2 and sys.argv[2] == 'fill':
    for lang in ('en', 'zh'):
        p = SRC / f'self_{lang}_appendix.tex'
        raw = p.read_bytes()
        crlf = b'\r\n' in raw
        s = raw.decode('utf-8').replace('\r\n', '\n')
        for lab, f in APP.items():
            pat = re.compile(rf'(^%<<{re.escape(lab)}\n)(.*?)(^%>>{re.escape(lab)}$)', re.S | re.M)
            assert len(pat.findall(s)) == 1, (p.name, lab)
            body = '\n'.join(f(lang))
            s = pat.sub(lambda m: m.group(1) + body + '\n' + m.group(3), s)
            print('filled', p.name, lab)
        p.write_bytes((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
