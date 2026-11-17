"""Supplementary Information of the NMI version (11-14): the tables that Methods cites, taken verbatim from the long version
(paper/source, identical to the 11-14 frozen copy) by line ranges located through their labels, with the references that point
into the long version rewritten for this paper. Only tables and the rendered templates are taken, not the prose of the long
appendix, which predates the NMI results (it would carry statements that no longer hold, such as the game separating the sources
in every readout).
  Supplementary Table 1  checkpoints and template arguments (tab:self_config)
  Supplementary Table 2  the models (tab:self_models)
  Supplementary Table 3  the two levels across the 25 models (tab:self_crossmodel), the values of Fig. 3
  Supplementary Table 4  the English item texts (tab:self_items), without the row of an experiment that is not in this paper
  Supplementary Table 5  classification and statistics at three thresholds (11-15, from threshold_grid_20261115.json)
  Supplementary Section 1  the rendered input of one item for every model, main probe and natural order (app:templates)
Writes paper/nmi/supplementary_nmi.tex. Fails if a reference is left that this paper does not define.
usage (from projects/consciousness): make_supplementary_nmi_20261114.py"""
import re
from pathlib import Path

SRC = Path('paper/source')
BODY = (SRC / 'self_en_body.tex').read_text(encoding='utf-8')
APP = (SRC / 'self_en_appendix.tex').read_text(encoding='utf-8')
OUT = Path('paper/nmi/supplementary_nmi.tex')
REFMAP = {  # long-version label -> text in this paper
    'sec:setup_crossmodel': 'Methods', 'sec:setup': 'Methods', 'sec:setup_swap': 'Methods', 'sec:setup_compute': 'Methods',
    'app:natural': 'Fig.~2b', 'app:config': 'Supplementary Table~1', 'tab:self_models': 'Supplementary Table~2',
    'tab:self_config': 'Supplementary Table~1', 'tab:self_crossmodel': 'Supplementary Table~3', 'tab:self_items': 'Supplementary Table~4',
    'app:templates': 'Supplementary Section~1', 'app:templates_main': 'Supplementary Section~1.1', 'app:templates_natural': 'Supplementary Section~1.2',
}
OWN = set()


def table(text, label):
    i = text.index('\\label{' + label + '}')
    a = text.rindex('\\begin{table}', 0, i)
    b = text.index('\\end{table}', i) + len('\\end{table}')
    return text[a:b]


def mapped(lab):
    if lab not in REFMAP:
        raise SystemExit(f'unmapped reference {lab}')
    return REFMAP[lab]


def fix(t):
    # "Table~\ref{x}", "Section~\ref{x}", "Appendix~\ref{x}" -> the mapped text, which already names the kind
    t = re.sub(r'(?:Tables?|Sections?|Appendix|Figure|Fig\.)~\\ref\{([^}]+)\}', lambda m: mapped(m.group(1)), t)
    t = re.sub(r'\\ref\{([^}]+)\}', lambda m: mapped(m.group(1)), t)
    # labels keep working inside this file, prefixed so that they cannot clash with the main paper
    t = re.sub(r'\\label\{([^}]+)\}', lambda m: '\\label{si:' + m.group(1) + '}', t)
    return t


GRID = Path('evidence/p2_words_along_20261110/results/final_20261113/threshold_grid_20261115.json')


def pfmt(p):
    if p < 1e-4:   # no permutation of 20,000 reached the observed value
        return '\\(<\\)0.0001'
    return f'{p:.4f}' if p < 0.01 else f'{p:.3f}'


def table_thresholds():
    """11-15: Supplementary Table 5, the classification and the statistics of the paper at three thresholds (threshold_grid_20261115.py)."""
    import json
    g = json.loads(GRID.read_text(encoding='utf-8'))
    rows_a, rows_b = [], []   # a classification, b correlations (two blocks, so that the table fits the text width)
    for t in ('70', '80', '90'):
        r = g['rows'][t]
        mc, nc = r['main_counts'], r['natural_counts']
        cell = lambda d: f"{d['rho']:.2f} ({pfmt(d['P'])})"  # noqa: E731
        rows_a.append(f"{t}\\% & {mc['self-reference']} / {mc['self-index only']} / {mc['neither']} & {pfmt(r['main_nesting_P'])} & "
                      f"{nc['self-reference']} / {nc['self-index only']} / {nc['neither']} \\\\")
        rows_b.append(f"{t}\\% & {cell(r['main_kl_self_ext'])} & {cell(r['main_share_self_ext'])} & "
                      f"{cell(r['natural_kl_self_ext'])} & {cell(r['natural_share_self_ext'])} & {cell(r['pooled_kl_b'])} & {cell(r['pooled_share_d'])} \\\\")
        assert not r['selfref_without_index']['main'] and not r['selfref_without_index']['natural']
    order = {'main': 'main probe', 'natural': 'natural order'}
    ch = {}
    for c in g['changes']:
        ch.setdefault((c['threshold'], c['order'], c['here']), []).append(c['model'])
    parts = []
    for (t, o, here), ms in ch.items():
        names = ms[0] if len(ms) == 1 else ', '.join(ms[:-1]) + (',' if len(ms) > 2 else '') + ' and ' + ms[-1]
        parts.append(f'at {t}\\% in the {order[o]}, {names} ({here})')
    changed = ('Models whose group differs from that at 80\\%: ' + '; '.join(parts) + '.') if parts else 'No model changes group.'
    five = [t for t in ('70', '80', '90') if g['rows'][t]['pooled_level_counts']['selfref-one'] > 0]
    pooled_note = ('The pooled test uses the four levels of the paper at 80\\% and 90\\%' + (
        f" and five at {', '.join(five)}\\%, where a model recognizes its own messages in one order and carries the index in both" if five else '') + '.')
    cap = ('Classification and statistics at three thresholds of the self-index criterion. A layer carries the self-index when the model\'s '
           'own message is singled out in at least the given share of the 75 held-out texts, and a model has a self-index when a block after '
           'block \\(\\lceil L/3 \\rceil\\) carries it. The paper uses 80\\%. Groups SR / IO / N: self-reference, self-index only, neither. '
           'Nesting \\(P\\): the exact probability that all models with self-reference fall among the models with a self-index if the two '
           'levels were unrelated (main probe). \\(\\rho\\) (\\(P\\)): Spearman correlation of the part specific to the model\'s own role '
           '(divergence ratio self / external, difference in share kept self \\(-\\) external) with the group in each order, and with the levels '
           'of the 14 models measured in both orders (pooled), one-sided permutation \\(P\\) with 20,000 permutations. ' + pooled_note +
           ' No model recognizes its own messages without a self-index at any threshold, in either order. ' + changed)
    return '\n'.join(['\\begin{table}[htbp]', '\\centering', '\\caption{' + cap + '}', '\\label{si:tab:thresholds}', '\\small',
                      '\\begin{tabular}{lccc}', '\\toprule',
                      '\\multicolumn{4}{@{}l}{\\textbf{a}\\enspace Groups} \\\\', '\\midrule',
                      ' & \\multicolumn{2}{c}{Main probe} & Natural order \\\\', '\\cmidrule(lr){2-3}\\cmidrule(l){4-4}',
                      'Threshold & SR / IO / N & nesting \\(P\\) & SR / IO / N \\\\', '\\midrule'] + rows_a +
                     ['\\bottomrule', '\\end{tabular}', '', '\\vspace{3mm}', '\\setlength{\\tabcolsep}{4pt}', '\\begin{tabular}{lcccccc}', '\\toprule',
                      '\\multicolumn{7}{@{}l}{\\textbf{b}\\enspace Rank correlation of the part specific to the model\'s own role, \\(\\rho\\) (\\(P\\))} \\\\',
                      '\\midrule',
                      ' & \\multicolumn{2}{c}{Main probe} & \\multicolumn{2}{c}{Natural order} & \\multicolumn{2}{c}{Both orders} \\\\',
                      '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}',
                      'Threshold & divergence & share kept & divergence & share kept & divergence & share kept \\\\', '\\midrule'] + rows_b +
                     ['\\bottomrule', '\\end{tabular}', '\\end{table}'])


def main():
    t_config = table(APP, 'tab:self_config')
    t_models = table(BODY, 'tab:self_models')
    t_cross = table(BODY, 'tab:self_crossmodel')
    t_items = table(APP, 'tab:self_items')
    t_items = re.sub(r'Final question \(Appendix~\\ref\{app:ownstep\}\) & [^\n]*\\\\\n', '', t_items)
    # 11-14 audit: the option wordings that name the model by role or by pronoun belong to experiments that are not in this paper
    for lab in ('Role-name options, order 1', 'Pronoun options, order 1'):
        i = t_items.index(lab)
        j = t_items.index('\\\\\n', t_items.index('\\\\\n', i) + 3) + 3   # this row and the next (order 2)
        t_items = t_items[:i] + t_items[j:]
    assert 'Role-name options' not in t_items and 'Pronoun options' not in t_items
    assert 'app:ownstep' not in t_items
    a = APP.index('%<<app_templates')
    b = APP.index('%>>app_templates')
    templates = APP[a:b]
    parts = ['% Supplementary Information of the NMI version, generated by paper/scripts/make_supplementary_nmi_20261114.py. Do not edit.',
             '\\clearpage', '\\section*{Supplementary Information}', '\\setcounter{table}{0}', '\\renewcommand{\\tablename}{Supplementary Table}',
             '\\renewcommand{\\thetable}{\\arabic{table}}',
             'The tables below give the checkpoints, the models, the values behind Fig.~3, the item texts, and the classification at three thresholds. '
             'Supplementary Section~1 gives the rendered input of one item for every model. All files are in the released repository.',
             fix(t_config), fix(t_models), fix(t_cross), fix(t_items), table_thresholds(),
             '\\clearpage', '\\subsection*{Supplementary Section 1. Rendered items}',
             fix(templates).replace('\\subsection{Main Probe}', '\\subsubsection*{1.1 Main probe}')
             .replace('\\subsection{Natural Conversation Order}', '\\subsubsection*{1.2 Natural conversation order}')
             .replace('This appendix gives', 'This section gives')]
    s = '\n\n'.join(parts) + '\n'
    # the long version promised the Chinese texts "with the code"; in this paper they are part of the released data
    for a, b in (('The Chinese texts, used for the models tested in Chinese, will be released with the code.',
                  'The Chinese texts, used for the models tested in Chinese, are part of the released data.'),
                 ('and the Chinese ones will be released with the code)', 'and the Chinese ones are part of the released data)')):
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    # 11-14 audit: the role is System for every model (no template-specific role names), and the item table names both game prompts
    for a, b in (('the gpt-oss template renders a System message as a developer message, ',
                  'the gpt-oss template gives a System message a header of its own, with the text under \\texttt{\\# Instructions}, '),
                 # 11-15 audit: the remaining differences of the System rendering, from the rendered items of Supplementary Section 1
                 ('which a System message under test replaces. VL:',
                  'which a System message under test replaces. In the System rendering, the templates of Qwen3.8-27B and Qwen3.8-Flash-Next '
                  'with thinking and of Gemma-4-31B with thinking place the System text in the system turn that carries their reasoning setting, '
                  'the Seed-OSS template places the System turn before its turn with the thinking budget, and the ERNIE-4.5 template wraps the '
                  'System text in \\texttt{<system\\_setting>} before its \\texttt{<global\\_setting>}. VL:'),
                 ('In the game, \\(R\\) (20, 50, 100, 200, 500, or 1,000) and \\(N\\) (5 to 12) vary over the 100 games.',
                  'In the game, \\(R\\) (20, 50, 100, 200, 500, or 1,000) and \\(N\\) (5 to 12) vary over the 100 games. The System message is that of the second System prompt, and the first prompt is the same without its last sentence.')):
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    left = re.findall(r'\\ref\{([^}]+)\}', s)
    assert not left, left
    OUT.write_text(s, encoding='utf-8')
    print('written', OUT, len(s), 'characters')


if __name__ == '__main__':
    main()
