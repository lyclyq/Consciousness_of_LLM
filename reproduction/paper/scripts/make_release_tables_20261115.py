"""Table-only copies of the long version for the public release (11-15).
Five released scripts parse tables of the long version (paper/source/self_{en,zh}_{body,appendix}.tex):
  fig_nesting_nmi_20261114.py and check_crossmodel_hand_numbers_20261103.py   Table tab:self_crossmodel (English)
  fig_ed_onset_size_nmi_20261114.py                                            Table tab:self_models (English)
  check_newdev_tables_20261104.py      tab:self_crossmodel (both languages), the natural-order tables and the own-step row
  make_supplementary_nmi_20261114.py   tab:self_config, tab:self_models, tab:self_crossmodel, tab:self_items and the template
                                       renderings (the block between %<<app_templates and %>>app_templates)
The release holds, under the same four file names, only these tables (verbatim, in their order in the source) and the template
block, so that the five scripts run unchanged. The text of the long version is not part of the release.
--list prints, for every table of the four files, its labels and whether it is kept.
usage (from projects/consciousness): make_release_tables_20261115.py OUT_DIR [--list]"""
import re
import sys
from pathlib import Path

SRC = Path('paper/source')
BEGIN = re.compile(r'\\begin\{(table\*?|sidewaystable\*?)\}')
LABEL = re.compile(r'\\label\{([^}]+)\}')
# the tables each file must keep (labels): body, the models and the two levels across models; appendix, the checkpoints, the
# items, the natural-order rows and the Apertus two-role row checked by check_newdev_tables_20261104.py, and its own-step row
KEEP = {'body': {'tab:self_models', 'tab:self_crossmodel'},
        'appendix': {'tab:self_config', 'tab:self_items', 'tab:self_natural', 'tab:self_natural2', 'tab:self_ownstep'}}


def tables(s):
    out = []
    for m in BEGIN.finditer(s):
        env = m.group(1)
        end = '\\end{' + env + '}'
        b = s.index(end, m.start()) + len(end)
        out.append((m.start(), b, s[m.start():b]))
    return out


def main():
    out = Path(sys.argv[1])
    listing = '--list' in sys.argv
    for f in ('self_en_body.tex', 'self_en_appendix.tex', 'self_zh_body.tex', 'self_zh_appendix.tex'):
        s = (SRC / f).read_text(encoding='utf-8')
        keep = []
        for a, b, t in tables(s):
            labs = set(LABEL.findall(t))
            take = bool(labs & KEEP['body' if 'body' in f else 'appendix'])
            if listing:
                print(f, sorted(labs), len(t), 'KEEP' if take else '')
            if take:
                keep.append(t)
        block = ''
        if '%<<app_templates' in s:
            block = s[s.index('%<<app_templates'):s.index('%>>app_templates') + len('%>>app_templates')]
        if listing:
            continue
        head = ('% Release copy of ' + f + ' of the long version: only the tables that the released scripts parse'
                + (' and the template renderings' if block else '') + ',\n% verbatim. The text of the long version is not part of the release.'
                + ' Written by paper/scripts/make_release_tables_20261115.py.\n\n')
        (out / 'paper/source').mkdir(parents=True, exist_ok=True)
        body = '\n\n'.join(keep) + ('\n\n' + block if block else '') + '\n'
        open(out / 'paper/source' / f, 'w', encoding='utf-8', newline='\n').write(head + body)
        print(f, len(keep), 'tables', 'and the template block' if block else '', len(body), 'characters of', len(s))


if __name__ == '__main__':
    main()
