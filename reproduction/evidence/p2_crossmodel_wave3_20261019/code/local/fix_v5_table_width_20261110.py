r"""11-10: the tables that a fifth readout or the new KL columns made wider than the text (overfull boxes of 7-80 pt in the first build
of this revision) are scaled to the text width: \resizebox{\linewidth}{!}{\begin{tabular}...\end{tabular}}. Both languages; the
first tabular after each \label is wrapped, once (skipped if already wrapped). Keeps the line endings of each file.
usage: fix_v5_table_width_20261110.py PAPER_SOURCE"""
import sys
from pathlib import Path

S = Path(sys.argv[1])
TABLES = {'body': ['tab:self_mech_questions', 'tab:self_mechanism'],
          'appendix': ['tab:self_kl', 'tab:self_kl_swap', 'tab:self_seeds', 'tab:self_cluster', 'tab:self_question', 'tab:self_wording']}
for lang in ('en', 'zh'):
    for part, labels in TABLES.items():
        p = S / f'self_{lang}_{part}.tex'
        nl = '\r\n' if b'\r\n' in p.read_bytes() else '\n'
        t = p.read_text(encoding='utf-8')
        for lab in labels:
            i = t.index('\\label{' + lab + '}')
            b = t.index('\\begin{tabular}', i)
            if t[max(0, b - 40):b].rstrip().endswith('\\resizebox{\\linewidth}{!}{'):
                continue
            e = t.index('\\end{tabular}', b) + len('\\end{tabular}')
            assert t.count('\\label{' + lab + '}') == 1 and '\\begin{table}' not in t[b:e], (lang, lab)
            t = t[:b] + '\\resizebox{\\linewidth}{!}{' + t[b:e] + '}' + t[e:]
        p.write_text(t, encoding='utf-8', newline=nl)
        print(lang, part, 'ok')
