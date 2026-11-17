"""arXiv version of the NMI paper (11-14, user: "明天第一步应该是 arxiv nmi 版本", then GitHub and Zenodo). Generated from
paper/nmi/self_nmi.tex, so that every later edit is made once, in the NMI source:
  - the author block of the P2 arXiv versions (memory arxiv-author-blocks; the 11-12 long version, paper/source/self_en.tex),
  - no line numbers, single spacing, PDF metadata with the authors,
  - Data and Code availability with the real addresses: GitHub https://github.com/lyclyq/Consciousness_of_LLM and the Zenodo DOI
    given with --doi (reserved on Zenodo before the arXiv upload; without --doi a red placeholder is printed),
  - no Reporting summary line (a journal item),
  - the Supplementary Information (supplementary_nmi.tex) after the Extended Data.
Then it stages a self-contained arXiv source folder paper/nmi/arxiv_build/src (the tex files, the figures, the bbl, no build
leftovers), compiles it twice with pdflatex there, and zips src into paper/nmi/arxiv_build/arxiv_nmi_<date>.zip.
usage (from projects/consciousness): make_arxiv_nmi_20261114.py [--doi 10.5281/zenodo.NNNNNNN]"""
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

NMI = Path('paper/nmi')
BUILD = NMI / 'arxiv_build'
SRC = BUILD / 'src'
GITHUB = 'https://github.com/lyclyq/Consciousness_of_LLM'
AUTHORS = ('\\author{Yuchen Li \\quad Xintong Li$^{*}$ \\quad Zongqi Fan$^{*}$\\\\[0.35em]Ken-tye Yong$^{\\dagger}$ \\quad Nguyen H. Tran$^{\\dagger}$\\\\[0.5em]'
           '{\\normalsize The University of Sydney, Sydney, Australia}\\\\[0.4em]'
           '{\\small yuli0398@uni.sydney.edu.au\\quad xili0045@uni.sydney.edu.au\\quad zfan0452@uni.sydney.edu.au}}\n\\date{}\n')
MARKS = ('\\begin{center}\\footnotesize $^*$ Equal contribution (joint second authors). $^\\dagger$ Joint corresponding authors: '
         'Ken-tye Yong (ken.yong@sydney.edu.au) and Nguyen H. Tran (nguyen.tran@sydney.edu.au).\\end{center}\n')
CONTRIB = ('\\section*{Author contributions}\n'
           'Y.L. conceived the study and led the project throughout. K.-T.Y. and N.H.T. supervised the work, gave revision comments, '
           'suggested ablation experiments and reviewed the manuscript. X.L. and Z.F. provided equipment, computing resources and '
           'deployment support and contributed revision suggestions.\n\n')


VERB = re.compile(r'\s*\\(begin|end)\{(Rendering|Verbatim|verbatim)\}')


def strip_comments(t):
    """11-15: the arXiv source is public, so drop the full-line comments (editing notes). Only lines whose first non-blank
    character is %, outside verbatim environments. Such a line produces no output in LaTeX, so the PDF text is unchanged."""
    out, verb = [], False
    for line in t.splitlines(keepends=True):
        m = VERB.match(line)
        if m:
            verb = m.group(1) == 'begin'
        elif not verb and line.lstrip().startswith('%'):
            continue
        out.append(line)
    return ''.join(out)


def sub(s, a, b, n=1):
    assert s.count(a) == n, (a[:80], s.count(a))
    return s.replace(a, b)


def main():
    doi = sys.argv[sys.argv.index('--doi') + 1] if '--doi' in sys.argv else None
    s = (NMI / 'self_nmi.tex').read_text(encoding='utf-8')
    s = sub(s, '% Nature Machine Intelligence submission version (double-blind).',
            '% arXiv version, GENERATED from self_nmi.tex by paper/scripts/make_arxiv_nmi_20261114.py. Do not edit here.\n% Source note:')
    # fvextra and the macros of the rendered templates (Supplementary Section 1), as in the preamble of paper/source/self_en.tex
    s = sub(s, '\\usepackage{lineno}\n', '\\usepackage{fvextra}\n'
            '\\DefineVerbatimEnvironment{Rendering}{Verbatim}{fontsize=\\scriptsize,breaklines=true,breakanywhere=true,breaksymbolleft={},'
            'breakanywheresymbolpre={},breakindentnchars=11,commandchars=\\\\\\{\\}}\n'
            '\\newcommand{\\PH}[1]{{\\normalfont\\ensuremath{\\langle}\\textit{#1}\\ensuremath{\\rangle}}}\n'
            '\\newcommand{\\NL}{{\\color{gray}\\ensuremath{\\hookleftarrow}}}\n'
            '\\newcommand{\\Note}[1]{{\\normalfont\\itshape\\color{gray}[#1]}}\n'
            '\\newcommand{\\BS}{\\char92 }\\newcommand{\\LB}{\\char123 }\\newcommand{\\RB}{\\char125 }\n')
    s = sub(s, '\\usepackage{setspace}\n', '')
    s = sub(s, 'pdfauthor={}', 'pdfauthor={Yuchen Li; Xintong Li; Zongqi Fan; Ken-tye Yong; Nguyen H. Tran}')
    s = sub(s, '\\linenumbers\n', '')
    s = sub(s, '\\onehalfspacing\n', '')
    a = s.index('\\begin{center}\n{\\Large\\bfseries')
    b = s.index('\\end{center}', a) + len('\\end{center}')
    s = s[:a] + ('\\title{Where Is the ``Self\'\' of a Large Language Model?\\\\[0.3em]\\large From Self-Index to Self-Reference}\n'
                 + AUTHORS + '\\maketitle\n' + MARKS) + s[b:]
    s = sub(s, '\\subsection*{Reporting summary}\nFurther information on research design is available in the Nature Portfolio Reporting Summary linked to this article.\n\n', '')
    ref = f'\\url{{https://doi.org/{doi}}}' if doi else '\\PENDING{Zenodo DOI}'
    s = sub(s, 'The data of this study are available for review at an anonymized repository \\PENDING{link}.', f'The data of this study are available at Zenodo, {ref}.')
    s = sub(s, 'are available for review at the same anonymized repository \\PENDING{link}, under the MIT licence.',
            f'are available at \\url{{{GITHUB}}} under the MIT licence.')
    # author contributions (user 11-15). Only here: the NMI version is double-blind and gets them on the title page at submission
    s = sub(s, '\\section*{Competing interests}\n', CONTRIB + '\\section*{Competing interests}\n')   # 11-15: contributions, then competing interests
    s = sub(s, '\\bibliography{../source/references}\n', '\\bibliography{references}\n')
    s = sub(s, '\\input{extended_data}\n', '\\input{extended_data}\n\\input{supplementary_nmi}\n')
    assert '\\PENDING{link}' not in s
    if SRC.exists():
        shutil.rmtree(SRC)
    (SRC / 'figures').mkdir(parents=True)
    (SRC / 'ed').mkdir()
    (SRC / 'self_nmi_arxiv.tex').write_text(s, encoding='utf-8')
    for f in ('methods_nmi.tex', 'extended_data.tex', 'supplementary_nmi.tex'):
        shutil.copyfile(NMI / f, SRC / f)
    shutil.copyfile(Path('paper/source/references.bib'), SRC / 'references.bib')
    for f in ('fig1_overview.pdf', 'fig2_evidence.pdf', 'fig3_nesting.pdf', 'fig4_ablation.pdf', 'fig5_game.pdf'):
        shutil.copyfile(NMI / 'figures' / f, SRC / 'figures' / f)
    for f in sorted((NMI / 'ed').iterdir()):
        if f.suffix in ('.pdf', '.tex'):
            shutil.copyfile(f, SRC / 'ed' / f.name)
    for f in SRC.rglob('*.tex'):
        f.write_text(strip_comments(f.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
    run = lambda *c: subprocess.run(list(c), cwd=SRC, capture_output=True, text=True, encoding='utf-8', errors='replace')  # noqa: E731
    r = run('pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'self_nmi_arxiv.tex')
    assert r.returncode == 0, r.stdout[-2500:]
    r = run('bibtex', 'self_nmi_arxiv')
    assert r.returncode == 0, r.stdout[-1500:]
    for _ in range(2):
        r = run('pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'self_nmi_arxiv.tex')
        assert r.returncode == 0, r.stdout[-2500:]
    log = (SRC / 'self_nmi_arxiv.log').read_text(encoding='utf-8', errors='replace')
    undefined = [l for l in log.splitlines() if 'undefined' in l.lower()]
    pdf = BUILD / 'self_nmi_arxiv.pdf'
    shutil.copyfile(SRC / 'self_nmi_arxiv.pdf', pdf)
    keep = {'.tex', '.bbl', '.pdf', '.bib'}
    z = BUILD / f'arxiv_nmi_{date.today():%Y%m%d}.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(SRC.rglob('*')):
            if f.is_file() and f.suffix in keep and f.name != 'self_nmi_arxiv.pdf':
                zf.write(f, f.relative_to(SRC).as_posix())
    names = zipfile.ZipFile(z).namelist()
    print('pdf', pdf, '| zip', z, len(names), 'files |', 'DOI', doi or 'placeholder', '| undefined references:', undefined[:5])


if __name__ == '__main__':
    main()
