"""Splice the five-readout table bodies of stats/v5_mech_tex_{en,zh}.tex (make_v5_mech_tex_20261110.py) into the paper body between
the markers %<<label and %>>label (tab:self_mech_questions, tab:self_mechanism), and give tab:self_mechanism its fifth column
(MiMo-V2.6-Distill-Qwen-9B, default call) in the column spec and the two header rows. Copies the new self_mechanism figure
(figures_v5_20261110/self_mechanism.pdf) over paper/source/figures. Every replacement must match exactly once.
usage: fix_v5_mimo_tables_20261110.py ARCHIVE PAPER_SOURCE"""
import re
import shutil
import sys
from pathlib import Path

A, S = Path(sys.argv[1]), Path(sys.argv[2])
HEAD = {'en': ('& Qwen3.8-27B & Qwen3.8-27B & gpt-oss-20b & gpt-oss-120b \\\\\nRecipient \\(\\leftarrow\\) donor & no thinking & thinking & thinking & thinking \\\\',
               '& Qwen3.8-27B & Qwen3.8-27B & gpt-oss-20b & gpt-oss-120b & MiMo-V2.6-9B \\\\\nRecipient \\(\\leftarrow\\) donor & no thinking & thinking & thinking & thinking & default call \\\\'),
        'zh': ('& Qwen3.8-27B & Qwen3.8-27B & gpt-oss-20b & gpt-oss-120b \\\\\n接收者 \\(\\leftarrow\\) 供体 & 不思考 & 思考 & 思考 & 思考 \\\\',
               '& Qwen3.8-27B & Qwen3.8-27B & gpt-oss-20b & gpt-oss-120b & MiMo-V2.6-9B \\\\\n接收者 \\(\\leftarrow\\) 供体 & 不思考 & 思考 & 思考 & 思考 & 默认调用 \\\\')}


def blocks(text):
    return {m.group(1): m.group(2) for m in re.finditer(r'%% (\S+)\n(.*?)%% end\n', text, re.S)}


for lang in ('en', 'zh'):
    p = S / f'self_{lang}_body.tex'
    nl = '\r\n' if b'\r\n' in p.read_bytes() else '\n'  # the body files are CRLF: keep their line endings
    t = p.read_text(encoding='utf-8')
    B = blocks((A / 'stats' / f'v5_mech_tex_{lang}.tex').read_text(encoding='utf-8'))
    for lab in ('tab:self_mech_questions', 'tab:self_mechanism'):
        pat = re.compile(r'(%<<' + re.escape(lab) + r'\n)(.*?)(%>>' + re.escape(lab) + r'\n)', re.S)
        assert len(pat.findall(t)) == 1, (lang, lab)
        t = pat.sub(lambda m: m.group(1) + B[lab] + m.group(3), t)
    old, new = HEAD[lang]
    i = t.find('%<<tab:self_mechanism\n')
    j = t.rfind(old, 0, i)
    assert j > 0 and t.count(old) >= 1, (lang, 'header')
    k = t.rfind('\\begin{tabular}{lcccc}', 0, j)
    assert k > 0 and j - k < 200, (lang, 'colspec')
    t = t[:k] + '\\begin{tabular}{lccccc}' + t[k + len('\\begin{tabular}{lcccc}'):j] + new + t[j + len(old):]
    p.write_text(t, encoding='utf-8', newline=nl)
    print(lang, 'ok', repr(nl))
shutil.copyfile(A / 'figures_v5_20261110' / 'self_mechanism.pdf', S / 'figures' / 'self_mechanism.pdf')
print('figure copied')
