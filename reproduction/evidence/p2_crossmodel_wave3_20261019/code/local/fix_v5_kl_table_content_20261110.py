r"""11-10 (user: a relation, not a threshold): Table tab:self_kl shows the content index (layers singled out on message content) in
place of the yes/no self-index group. Refills the table body from stats/v5_appendix_tex.json (make_v5_appendix_tex_20261110.py) and
updates the header cell and the caption sentence in both appendices. Every old string must match exactly once. LF files.
usage: fix_v5_kl_table_content_20261110.py ARCHIVE PAPER_SOURCE"""
import json
import re
import sys
from pathlib import Path

A, S = Path(sys.argv[1]), Path(sys.argv[2])
J = json.load(open(A / 'stats' / 'v5_appendix_tex.json', encoding='utf-8'))
REP = {'en': [('Self-index: as grouped in Section~\\ref{sec:res_crossmodel}.',
               'Content index: layers at which the model\'s own message is singled out on its content (Table~\\ref{tab:self_crossmodel}; four-stream models: mean of the streams, strongest stream in parentheses).'),
              ('Model & Self-index & P(control), A / U / S &', 'Model & Content index & P(control), A / U / S &')],
       'zh': [('自我索引：按第~\\ref{sec:res_crossmodel} 节的分组。',
               '内容索引：模型自己的消息在内容上被单独分出的层（表~\\ref{tab:self_crossmodel}；四流模型为四流的平均，括号内为最强的一条流）。'),
              ('模型 & 自我索引 & P（控制），A / U / S &', '模型 & 内容索引 & P（控制），A / U / S &')]}
for lang in ('en', 'zh'):
    p = S / f'self_{lang}_appendix.tex'
    t = p.read_text(encoding='utf-8')
    for a, b in REP[lang]:
        assert t.count(a) == 1, (lang, a[:40])
        t = t.replace(a, b)
    pat = re.compile(r'(%<<tab:self_kl\n)(.*?)(%>>tab:self_kl\n)', re.S)
    assert len(pat.findall(t)) == 1
    t = pat.sub(lambda m: m.group(1) + '\n'.join(J[f'kl_main_{lang}']) + '\n' + m.group(3), t)
    p.write_text(t, encoding='utf-8', newline='\n')
    print(lang, 'ok')
