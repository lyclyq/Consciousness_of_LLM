"""Paper v4, causal tests (10-27). Two steps on paper/source/self_{en,zh}_{body,appendix}.tex:
  install DRAFT_DIR  replace the results subsection of the causal tests (from its \\subsection to the \\FloatBarrier before the
                     cross-model subsection) and the appendix section of the causal tests (to the end of the appendix file) with
                     res_{lang}.tex and app_{lang}.tex from DRAFT_DIR; every table placeholder @@NAME@@ becomes a pair of comment
                     lines %<<label ... %>>label around the table body; the Chinese single-layer paragraph is carried over from
                     the current Chinese appendix. The four files are first copied to paper/history/source_before_mech_v4_<time>.
  fill               write the table bodies between those markers from stats/v4_mech_tex_{lang}.tex (make_v4_mech_tex.py), so
                     that the tables are regenerated, never typed, whenever the numbers change.
usage: splice_v4_mech_paper.py install DRAFT_DIR | fill   (run from evidence/p2_crossmodel_wave3_20261019)"""
import re
import shutil
import sys
import time
from pathlib import Path

A = Path('.').resolve()
SRC = A.parents[1] / 'paper' / 'source'
HIST = A.parents[1] / 'paper' / 'history'
PH = {'QUESTIONS': 'tab:self_mech_questions', 'MECH': 'tab:self_mechanism', 'GAME': 'tab:self_game', 'WHERE': 'tab:self_mechanism_where',
      'WINDOWS': 'tab:self_mechanism_windows', 'GEMMA': 'tab:self_mech_gemma', 'GWINDOWS': 'tab:self_mech_gemma_windows'}
SPAN = {'en': (('\\subsection{Causal Tests in Two Models}', '\\FloatBarrier\n\\subsection{The Two Levels across Models}'),
               ('\\clearpage\n\\section{Further Causal Tests in Two Models}', None)),
        'zh': (('\\subsection{两个模型中的因果检验}', '\\FloatBarrier\n\\subsection{跨模型的两个层级}'),
               ('\\clearpage\n\\section{两个模型中的更多因果检验}', None))}


def read(p):
    return p.read_text(encoding='utf-8')


def write(p, s):
    p.write_text(s, encoding='utf-8', newline='\r\n')  # the paper source files use CRLF


def markers(s):
    for k, lab in PH.items():
        s = s.replace(f'@@{k}@@', f'%<<{lab}\n%>>{lab}')
    return s


def install(drafts):
    bak = HIST / f'source_before_mech_v4_{time.strftime("%Y%m%d_%H%M%S")}'
    bak.mkdir(parents=True)
    for lang in ('en', 'zh'):
        for part in ('body', 'appendix'):
            shutil.copy2(SRC / f'self_{lang}_{part}.tex', bak / f'self_{lang}_{part}.tex')
    print('backup', bak)
    for lang, ((b0, b1), (a0, _)) in SPAN.items():
        body = read(SRC / f'self_{lang}_body.tex')
        i, j = body.index(b0), body.index(b1)
        assert body.count(b0) == 1 and i < j
        res = markers(read(drafts / f'res_{lang}.tex')).rstrip('\n') + '\n\n'
        write(SRC / f'self_{lang}_body.tex', body[:i] + res + body[j:])
        app = read(SRC / f'self_{lang}_appendix.tex')
        k = app.index(a0)
        new = markers(read(drafts / f'app_{lang}.tex'))
        if lang == 'zh':
            m = re.search(r'\\paragraph\{单一层。\}(.*?)\n\n', app[k:], re.S)
            new = new.replace('@@SINGLE_ZH@@', m.group(1).strip())
        assert '@@' not in res and '@@' not in new, (lang, re.findall(r'@@\w*@@?', res + new)[:5])
        write(SRC / f'self_{lang}_appendix.tex', app[:k] + new.rstrip('\n') + '\n')
        print(lang, 'installed')


def fill():
    for lang in ('en', 'zh'):
        rows = read(A / 'stats' / f'v4_mech_tex_{lang}.tex')
        blocks = dict(re.findall(r'^%% (\S+)\n(.*?)^%% end$', rows, re.S | re.M))
        for part in ('body', 'appendix'):
            p = SRC / f'self_{lang}_{part}.tex'
            s = read(p)
            for lab in PH.values():
                pat = re.compile(rf'(^%<<{re.escape(lab)}\n)(.*?)(^%>>{re.escape(lab)}$)', re.S | re.M)
                if pat.search(s):
                    s = pat.sub(lambda m: m.group(1) + blocks[lab].rstrip('\n') + '\n' + m.group(3), s)
                    print(lang, part, lab, len(blocks[lab].strip().splitlines()), 'rows')
            write(p, s)


if __name__ == '__main__':
    if sys.argv[1] == 'install':
        install(Path(sys.argv[2]))
    else:
        fill()
