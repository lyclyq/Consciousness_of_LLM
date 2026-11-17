"""Paper v5 (10-29): LaTeX table bodies for the new parts of the appendix "Further Causal Tests", from the run summaries, so that
no number in them is typed by hand; written between the lines "%<<label" and "%>>label" of paper/source/self_en_appendix.tex
(and, once the Chinese appendix has the same markers, of self_zh_appendix.tex).
  tab:self_ablation_2x2  attribution under the format ablation (Qwen3.8-27B template switch, both modes; gpt-oss-20b and
                         gpt-oss-120b, channel tag masked with positions recomputed): unedited claims and item-paired changes
                         (runs/pro/qwen38_nothink/{singlepass_swap,thinking_swap}/qwen38_27b_nothink_all,
                         thinking_swap/gpt_oss_{20b,120b}_mask)
  tab:self_positions     geometry with and without position compression: layers at which the own message is singled out in at
                         least 80% of the held-out bodies, content and onset (runs/pro/qwen38_nothink/suite_mask/qwen38_27b_*,
                         runs/pro/qwen38_nothink/suite_think/gpt_oss_*, runs/pro/suite_mask/gpt_oss_*_compress); Qwen3.8-27B
                         P("you") with and without compression (runs/pro/singlepass_swap/qwen38_27b_pos_{default,compress})
  tab:self_flashnext     Qwen3.8-Flash-Next attribution, every layer, eight dimensions (runs/server3/thinking_swap/flashnext_s3_all)
usage: make_v5_appendix_tables.py ARCHIVE [fill]   (fill: also write the blocks into the English appendix)"""
import json
import re
import sys
from pathlib import Path

A = Path(sys.argv[1])
R = A / 'runs'
SRC = A.resolve().parents[1] / 'paper' / 'source'


def summ(p):
    return json.load(open(A / p / 'summary.json', encoding='utf-8'))


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
    runs = [('runs/pro/qwen38_nothink/singlepass_swap/qwen38_27b_nothink_all', 'sp'), ('runs/pro/qwen38_nothink/thinking_swap/qwen38_27b_nothink_all', 'th'),
            ('runs/pro/qwen38_nothink/thinking_swap/gpt_oss_20b_mask', 'th'), ('runs/pro/qwen38_nothink/thinking_swap/gpt_oss_120b_mask', 'th')]
    S = [summ(p) for p, _ in runs]
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
        for s in S:
            v = s.get(c)
            cells.append('--' if v is None else stat(v) if kind == 'v' else ch(v))
        out.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
    return out


def block_positions():
    G = 'runs/pro/qwen38_nothink/suite_mask/'
    pairs = [('Qwen3.8-27B, non-thinking rendering', G + 'qwen38_27b_default', G + 'qwen38_27b_compress'),
             ('Qwen3.8-27B, thinking rendering', G + 'qwen38_27b_think_default', G + 'qwen38_27b_think_compress_v2'),
             ('gpt-oss-20b', 'runs/pro/qwen38_nothink/suite_think/gpt_oss_20b_mxfp4_en', 'runs/pro/suite_mask/gpt_oss_20b_mxfp4_en_compress'),
             ('gpt-oss-120b', 'runs/pro/qwen38_nothink/suite_think/gpt_oss_120b_en', 'runs/pro/suite_mask/gpt_oss_120b_en_compress')]
    out = []
    for name, d, c in pairs:
        a, b = summ(d), summ(c)
        L = a['n_layers']
        out.append(f"{name} & {a['content']['layers_A_lowest_80']}/{L} & {b['content']['layers_A_lowest_80']}/{L} & "
                   f"{a['onset']['layers_A_lowest_80']}/{L} & {b['onset']['layers_A_lowest_80']}/{L} \\\\")
    return out


def block_flashnext():
    s = summ('runs/server3/thinking_swap/flashnext_s3_all')
    rows = [('own, unedited', 'base_A', 'v'), ('own \\(\\leftarrow\\) User', 'swU8_A', 'c'), ('own \\(\\leftarrow\\) System', 'swS8_A', 'c'),
            ('own \\(\\leftarrow\\) User, random', 'rswU8_A', 'c'), ('own \\(\\leftarrow\\) own (copy)', 'idA8_A', 'c'),
            ('User, unedited', 'base_U', 'v'), ('User \\(\\leftarrow\\) own', 'swA8_U', 'c'), ('User \\(\\leftarrow\\) own, random', 'rswA8_U', 'c'),
            ('System, unedited', 'base_S', 'v'), ('System \\(\\leftarrow\\) own', 'swA8_S', 'c')]
    out = []
    for lab, c, kind in rows:
        v = s[c]
        val = f"{v['you_share']:.2f}" + (f" ({v['finished']}/{v['n']})" if v['finished'] != v['n'] else '')
        out.append(f"{lab} & {val if kind == 'v' else ch(v)} \\\\")
    return out


# tab:self_flashnext dropped 10-29 (user: the causal tests stay at three models and four readouts; block_flashnext kept for the archive)
BLOCKS = {'tab:self_ablation_2x2': block_ablation, 'tab:self_positions': block_positions}
text = []
for lab, f in BLOCKS.items():
    text += [f'%% {lab}'] + f() + ['%% end']
(A / 'stats' / 'v5_appendix_tex_en.tex').write_text('\n'.join(text) + '\n', encoding='utf-8')
print('\n'.join(text))
if len(sys.argv) > 2 and sys.argv[2] == 'fill':
    p = SRC / 'self_en_appendix.tex'
    s = p.read_text(encoding='utf-8')
    for lab, f in BLOCKS.items():
        pat = re.compile(rf'(^%<<{re.escape(lab)}\n)(.*?)(^%>>{re.escape(lab)}$)', re.S | re.M)
        if pat.search(s):
            body = '\n'.join(f())
            s = pat.sub(lambda m: m.group(1) + body + '\n' + m.group(3), s)
            print('filled', lab)
    p.write_text(s, encoding='utf-8')
