"""Install or refresh Appendix A (Rendered Items) in paper/source/self_{lang}_appendix.tex from stats/appendix_a_20261029/
appendix_a_{lang}.tex (make_appendix_a_tex.py). First run: the placeholder paragraph after \\label{app:templates} becomes the block
between the comment lines %<<app_templates and %>>app_templates; later runs rewrite what lies between the markers. The preamble
self_{lang}.tex gets fvextra and the macros of the Rendering environment once.
usage: install_appendix_a.py LANG...   (run from evidence/p2_crossmodel_wave3_20261019)"""
import sys
from pathlib import Path

A = Path('.').resolve()
SRC = A.parents[1] / 'paper' / 'source'
PRE = ('\\usepackage{fvextra}\n'
       '\\DefineVerbatimEnvironment{Rendering}{Verbatim}{fontsize=\\scriptsize,breaklines=true,breakanywhere=true,breaksymbolleft={},'
       'breakanywheresymbolpre={},breakindentnchars=11,commandchars=\\\\\\{\\}}\n'
       '\\newcommand{\\PH}[1]{{\\normalfont\\ensuremath{\\langle}\\textit{#1}\\ensuremath{\\rangle}}}\n'
       '\\newcommand{\\NL}{{\\color{gray}\\ensuremath{\\hookleftarrow}}}\n'
       '\\newcommand{\\Note}[1]{{\\normalfont\\itshape\\color{gray}[#1]}}\n'
       '\\newcommand{\\BS}{\\char92 }\\newcommand{\\LB}{\\char123 }\\newcommand{\\RB}{\\char125 }\n')
# (marker, generated file, placeholder of the first run); Appendix B from make_appendix_b_tex.py (added 10-29)
BLOCKS = [('app_templates', 'appendix_a_20261029/appendix_a_{lang}.tex',
           {'en': '% V5-TODO one rendered main-probe item and one natural-order item per model, from each model\'s own tokenizer\n'
                  'This appendix gives, for every model, the rendering of one item of the main probe and of the natural conversation order.\n'}),
          ('app_config', 'appendix_b_20261029/appendix_b_{lang}.tex',
           {'en': '% V5-TODO per model: checkpoint and revision, weight format, template arguments, decoding, API settings; items; subspace fit;\n'
                  '% bootstrap; no hardware details\nThis appendix lists the settings needed to reproduce every run.\n'})]

for lang in sys.argv[1:]:
    p = SRC / f'self_{lang}_appendix.tex'
    s = p.read_text(encoding='utf-8')
    for mark, src, old in BLOCKS:
        block = (A / 'stats' / src.format(lang=lang)).read_text(encoding='utf-8')
        if f'%<<{mark}\n' in s:
            i = s.index(f'%<<{mark}\n') + len(f'%<<{mark}\n')
            j = s.index(f'%>>{mark}\n')
            s = s[:i] + block + s[j:]
        else:
            assert s.count(old[lang]) == 1, mark
            s = s.replace(old[lang], f'%<<{mark}\n' + block + f'%>>{mark}\n')
    p.write_text(s, encoding='utf-8')
    q = SRC / f'self_{lang}.tex'
    t = q.read_text(encoding='utf-8')
    if '\\DefineVerbatimEnvironment{Rendering}' not in t:
        anchor = '\\usepackage[section]{placeins}\n'
        assert t.count(anchor) == 1
        q.write_text(t.replace(anchor, anchor + PRE), encoding='utf-8')
    print(lang, 'installed')
