"""Extended Data Tables 1 and 2 of the NMI version (11-14), written as LaTeX from the result files. Nothing is recomputed.
  Table 1  the values behind Figs. 4 and 5c,e,f for the seven configurations (one model in one mode; 'readout' before 11-15):
           a attribution question (unedited rates, removal and installation with eight and with 64 directions, random controls),
           b next-token divergence under the same edits and the two questions that do not concern the source.
           Sources: results/figs/fig_ablation_draft.json (shares of the gap and of the divergence with their 95% intervals,
           written by fig_ablation_draft_20261114.py) and results/game_two_prompts_20261114.json (rates and accuracies).
  Table 2  the secret game under the two System prompts: eight-game check, verdict of the release rule, and for released games the
           changes under removal, installation and the random controls. Source: results/game_two_prompts_20261114.json. A game that
           is not released has no edit values (one rule for every game, user 11-14).
Writes paper/nmi/ed/ed_table1.tex and ed_table2.tex (tabular environments only; captions are in paper/nmi/extended_data.tex).
usage (from projects/consciousness): make_ed_tables_nmi_20261114.py"""
import json
import re
from pathlib import Path

MINUS = re.compile(r'(?<![\w$\\{-])-(\d)')   # a plain hyphen before a number becomes a minus sign

MS = Path('evidence/p2_modeswitch_20261107/results')
OUT = Path('paper/nmi/ed')
AB = {d['name']: d for d in json.loads((MS / 'figs/fig_ablation_draft.json').read_text(encoding='utf-8'))}
GT = json.loads((MS / 'game_two_prompts_20261114.json').read_text(encoding='utf-8'))['readouts']
# row label, name in the ablation file, name in the game file (row order of Fig. 4)
ROWS = [('Qwen3.8-27B, no thinking', 'Qwen3.8-27B, no thinking', 'Qwen3.8-27B, single pass'),
        ('Qwen3.8-27B, thinking', 'Qwen3.8-27B, thinking', 'Qwen3.8-27B, thinking'),
        ('gpt-oss-20b', 'gpt-oss-20b', 'gpt-oss-20b'), ('gpt-oss-120b', 'gpt-oss-120b', 'gpt-oss-120b'),
        ('MiMo-V2.6-Distill-Qwen-9B', 'MiMo-V2.6-Distill-9B', 'MiMo-V2.6-Distill-Qwen-9B'),
        ('EXAONE-4.0.1-32B', 'EXAONE-4.0.1-32B', 'EXAONE-4.0.1-32B'), ('GLM-5.3-Flash', 'GLM-5.3-Flash', 'GLM-5.3-Flash')]


def ci(v, nd=2):
    return f'{v[0]:.{nd}f} [{v[1]:.{nd}f}, {v[2]:.{nd}f}]'


def sg(x, nd=2):
    if abs(x) < 0.5 * 10 ** -nd:
        return f'{0:.{nd}f}'
    return f'$-${abs(x):.{nd}f}' if x < 0 else f'+{x:.{nd}f}'


def table1():
    L = ['\\begin{tabular}{@{}lccccccc@{}}', '\\toprule',
         '\\multicolumn{8}{@{}l}{\\textbf{a}\\enspace Attribution question: rate of ``you\'\', and share of the unedited gap between the model\'s own and the User message that an edit closes}\\\\',
         '\\midrule',
         '& \\multicolumn{2}{c}{Unedited} & \\multicolumn{3}{c}{Eight directions per layer} & \\multicolumn{2}{c}{64 directions per layer} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(l){7-8}',
         'Configuration & own & User & removal & installation & random (rem. / inst.) & fitted (rem. / inst.) & random (rem. / inst.) \\\\', '\\midrule']
    for lab, a, g in ROWS:
        A, at = AB[a]['attr'], GT[g]['attribution']
        L.append(f"{lab} & {at['own']:.2f} & {at['user']:.2f} & {ci(A['removal|8|fit'])} & {ci(A['installation|8|fit'])} & "
                 f"{A['removal|8|random'][0]:.2f} / {A['installation|8|random'][0]:.2f} & {A['removal|64|fit'][0]:.2f} / {A['installation|64|fit'][0]:.2f}{'$^{*}$' if AB[a].get('fit64') == 'paper fit' else ''} & "
                 f"{A['removal|64|random'][0]:.2f} / {A['installation|64|random'][0]:.2f} \\\\")
    L += ['\\midrule',
          '\\multicolumn{8}{@{}l}{\\textbf{b}\\enspace Next-token divergence (share of the difference between the model\'s own and the User message), and accuracy on two questions that do not concern the source}\\\\',
          '\\midrule',
          '& \\multicolumn{2}{c}{Removal: share removed} & \\multicolumn{2}{c}{Installation: share closed} & Content question & \\multicolumn{2}{c}{Arithmetic question} \\\\',
          '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}\\cmidrule(l){7-8}',
          'Configuration & fitted & random & fitted & random & own / User / rem. / inst. & \\multicolumn{2}{c}{own / User / rem. / inst.} \\\\', '\\midrule']
    for lab, a, g in ROWS:
        K, c, ar = AB[a]['kl'], GT[g]['content'], GT[g]['arith']
        q = lambda d: f"{d['own']:.2f} / {d['user']:.2f} / {d['removal']['after']:.2f} / {d['installation']['after']:.2f}"  # noqa: E731
        L.append(f"{lab} & {ci(K['removal'])} & {K['removal_rnd'][0]:.2f} & {ci(K['installation'])} & {K['installation_rnd'][0]:.2f} & {q(c)} & "
                 f"\\multicolumn{{2}}{{c}}{{{q(ar)}}} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    return MINUS.sub(lambda m: '$-$' + m.group(1), '\n'.join(L)) + '\n'


def table2():
    L = ['\\begin{tabular}{@{}llcccccccc@{}}', '\\toprule',
         '& & \\multicolumn{3}{c}{Eight-game check, unedited} & & \\multicolumn{4}{c}{Released games: change in the setter share} \\\\',
         '\\cmidrule(lr){3-5}\\cmidrule(l){7-10}',
         'Configuration & System prompt & setter share, own X / User X & control correct, own / User & release rule & Games & '
         'removal & installation & random (rem. / inst.) & control correct, own / User \\\\', '\\midrule']
    for lab, _, g in ROWS:
        for k, p in enumerate(('first', 'second')):
            e = GT[g][p]
            s = e['smoke']
            fails = (['sources not separated'] if not s['separates_sources'] else []) + (['control question'] if not s['control_ok'] else [])
            rule = 'pass' if s['pass'] else 'fail: ' + ', '.join(fails)
            row = [lab if k == 0 else '', p, f"{s['own_A']:.2f} / {s['own_U']:.2f}", f"{s['control_A']:.2f} / {s['control_U']:.2f}", rule]
            m = e.get('main')
            if m:
                f = lambda d: f"{sg(d['change'])} [{sg(d['ci95'][0])}, {sg(d['ci95'][1])}]"  # noqa: E731
                cf = e.get('control_full')
                row += [str(m['n']), f(m['removal']), f(m['installation']),
                        f"{sg(m['random_removal']['change'])} / {sg(m['random_installation']['change'])}",
                        f"{cf['base_A']:.2f} / {cf['base_U']:.2f}" if cf else '--']
            else:
                row += ['--', '--', '--', '--', '--']
            L.append(' & '.join(row) + ' \\\\')
        if g != ROWS[-1][2]:
            L.append('\\addlinespace[2pt]')
    L += ['\\bottomrule', '\\end{tabular}']
    return MINUS.sub(lambda m: '$-$' + m.group(1), '\n'.join(L)) + '\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'ed_table1.tex').write_text(table1(), encoding='utf-8')
    (OUT / 'ed_table2.tex').write_text(table2(), encoding='utf-8')
    print('written', OUT / 'ed_table1.tex', OUT / 'ed_table2.tex')


if __name__ == '__main__':
    main()
