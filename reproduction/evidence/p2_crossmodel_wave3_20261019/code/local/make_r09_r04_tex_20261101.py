"""LaTeX table bodies for R09 and R04 of the supervisor's review in the paper (11-01), so that no number in them is typed by hand;
with `fill` they are written between the lines "%<<label" and "%>>label" of paper/source/self_{en,zh}_appendix.tex, bytes in,
bytes out, CRLF kept. Number format as make_tran2_tex_20261101.py.
  tab:self_cluster    the causal tests with 95% intervals over the 15 held-out wording families, stats/r09_cluster_20261101.json
                      (attribution: the seven conditions of the main runs; content question: the four edited conditions)
  tab:self_threshold  the self-index criterion at thresholds of 70%, 80% (the paper) and 90%, with the last third and the last
                      quarter of the depth, stats/r04_threshold_20261101.json
The numbers quoted in the text are printed for checking.
usage: make_r09_r04_tex_20261101.py ARCHIVE [fill]"""
import json
import re
import sys
from pathlib import Path

A = Path(sys.argv[1])
SRC = A.resolve().parents[1] / 'paper' / 'source'
R09 = json.load(open(A / 'stats' / 'r09_cluster_20261101.json', encoding='utf-8'))['results']
R04 = json.load(open(A / 'stats' / 'r04_threshold_20261101.json', encoding='utf-8'))
RO = ['qsp', 'qth', 'g20', 'g120']
SELFREF = [('qwen38_27b', 'non_thinking'), ('qwen38_27b', 'thinking'), ('qwen38_flashnext', 'thinking'), ('gpt_oss_20b', 'thinking'),
           ('gpt_oss_120b', 'thinking'), ('glm53_flash', 'thinking'), ('exaone4_32b', 'thinking')]
TS = ('0.7', '0.8', '0.9')
L = {'en': {'own': 'own', 'random': 'random', 'attr': "Attribution: change in ``you''",
            'content': 'Content question: change in the share answered correctly',
            'rows': ['Models with a self-index', '\\quad with self-reference', '\\quad without self-reference', 'Models with neither level',
                     'Models with self-reference but no self-index',
                     'Model--modes with self-reference (of 7) whose own message is singled out at every layer of the last third',
                     '\\quad at every layer of the last quarter',
                     'Most layers of the last third at which a model without a self-index singles out its own message'],
            'of': lambda a, b: f'{a}/{b}'},
     'zh': {'own': '自己', 'random': '随机', 'attr': '归属问题：“你”的变化', 'content': '内容问题：答对比例的变化',
            'rows': ['具有自我索引的模型', '\\quad 其中有自指', '\\quad 其中没有自指', '两层都没有的模型', '有自指而没有自我索引的模型',
                     '有自指的模型--模式（共 7 个）中，自己的消息在最后三分之一的每一层都被单独分出的',
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
    (A / 'stats' / f'r09_r04_tex_{lang}.tex').write_text('\n'.join(text) + '\n', encoding='utf-8')
    sys.stdout.buffer.write(('\n'.join(text) + '\n').encode('utf-8'))
text_numbers()
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
