"""Word counts of the NMI version (11-14): abstract, main text (after the abstract up to Methods, without equations, comments,
citations, labels and figure environments), Methods, and figure legends. Also lists semicolons outside comments and internal codes
that must not appear (v5, v6, s5, s6, developer message).
usage: count_words_nmi_20261114.py paper/nmi/self_nmi.tex"""
import re
import sys


def clean(s):
    s = re.sub(r'(?<!\\)%.*', '', s)
    s = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', ' ', s, flags=re.S)
    s = re.sub(r'\\(cite|label|ref|includegraphics)(\[[^\]]*\])?\{[^}]*\}', ' ', s)
    s = re.sub(r'\\[a-zA-Z]+\*?', ' ', s)
    return re.sub(r'[{}\\$~]', ' ', s)


def words(s):
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'.\-]*", clean(s)))


def main():
    t = open(sys.argv[1], encoding='utf-8').read()
    ab = t.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
    rest = t.split(r'\end{abstract}')[1]
    main_text, after = rest.split(r'\section*{Methods}', 1)
    main_text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', ' ', main_text, flags=re.S)
    methods = after.split(r'\bibliographystyle')[0]
    legends = ' '.join(re.findall(r'\\caption\{(.*?)\}\s*\\label', t, flags=re.S))
    print(f'abstract {words(ab)} | main text {words(main_text)} | methods {words(methods)} | legends {words(legends)}')
    body = re.sub(r'(?<!\\)%.*', '', t)
    for i, line in enumerate(body.split('\n'), 1):
        if ';' in line:
            print(f'semicolon, line {i}: {line.strip()[:100]}')
        for m in re.finditer(r'\b(v5|v6|s5|s6|developer message)\b', line):
            print(f'internal code {m.group(0)!r}, line {i}')


if __name__ == '__main__':
    main()
