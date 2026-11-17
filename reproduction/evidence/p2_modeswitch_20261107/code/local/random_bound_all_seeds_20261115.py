"""The random control of the attribution question over all seeds (11-15), for the bounds quoted in the paper: "the same edits
along eight random directions per layer change the rate of 'you' by at most X, against at least Y for the fitted removal".
Eight random directions per layer, all positions, the change of the rate of "you" from the unedited condition (item-paired mean):
removal (rswU8_A, the model's own message receives random coordinates) and installation (rswA8_U).
Seeds: five base seeds in six readouts (r05q_all and r06_s1..s4), one in GLM-5.3-Flash (r05q_all). The four Qwen3.8 and gpt-oss
readouts also have the run with the default fit (paper_all, same random seed as r05q_all), which is included.
Fitted removal: swU8_A of r05q_all, the run of Extended Data Table 1.
Sources: evidence/p2_crossmodel_wave3_20261019/stats/r05r06_20261031.json (Qwen3.8, gpt-oss), the run summaries of MiMo-V2.6,
EXAONE-4.0.1 and GLM-5.3 under evidence/p2_modeswitch_20261107/results.
Writes evidence/p2_modeswitch_20261107/results/random_bound_all_seeds_20261115.{json,md}.
usage (from projects/consciousness): random_bound_all_seeds_20261115.py"""
import json
from pathlib import Path

ST = Path('evidence/p2_crossmodel_wave3_20261019/stats/r05r06_20261031.json')
MS = Path('evidence/p2_modeswitch_20261107/results')
OUT = MS / 'random_bound_all_seeds_20261115'
R05 = {'Qwen3.8-27B, single pass': 'qsp', 'Qwen3.8-27B, thinking': 'qth', 'gpt-oss-20b': 'g20', 'gpt-oss-120b': 'g120'}
RUNS = {'MiMo-V2.6-Distill-Qwen-9B': (MS / 'pro/thinking_swap', 'mimo_v26_', ('r05q_all', 'r06_s1', 'r06_s2', 'r06_s3', 'r06_s4')),
        'EXAONE-4.0.1-32B': (MS / 'pro_exaone32_20261112/thinking_swap', 'exaone32_', ('r05q_all', 'r06_s1', 'r06_s2', 'r06_s3', 'r06_s4')),
        'GLM-5.3-Flash': (MS / 'vast_h200_20261111/m/out/mech', 'glm53_', ('r05q_all',))}


def main():
    r05 = json.loads(ST.read_text(encoding='utf-8'))['results']
    rows = {}
    for name, key in R05.items():
        runs = {}
        for run in ('paper_all', 'r05q_all', 'r06_s1', 'r06_s2', 'r06_s3', 'r06_s4'):
            s = r05[key][run].get('stats', r05[key][run])   # paper_all holds the statistics directly
            runs[run] = {'removal': s['rswU8_A']['diff'], 'installation': s['rswA8_U']['diff'], 'n': s['rswU8_A']['n_pairs']}
        fitted = r05[key]['r05q_all']['stats']['swU8_A']['diff']
        rows[name] = {'runs': runs, 'fitted_removal': fitted}
    for name, (d, pre, names) in RUNS.items():
        runs = {}
        for run in names:
            s = json.loads((d / (pre + run) / 'summary.json').read_text(encoding='utf-8'))
            runs[run] = {'removal': s['rswU8_A']['you_minus_base']['mean'], 'installation': s['rswA8_U']['you_minus_base']['mean'],
                         'n': s['rswU8_A']['you_minus_base']['n_pairs']}
        s = json.loads((d / (pre + 'r05q_all') / 'summary.json').read_text(encoding='utf-8'))
        rows[name] = {'runs': runs, 'fitted_removal': s['swU8_A']['you_minus_base']['mean']}
    for r in rows.values():
        r['max_abs_random_removal'] = max(abs(v['removal']) for v in r['runs'].values())
        r['max_abs_random_installation'] = max(abs(v['installation']) for v in r['runs'].values())
    tot = {'max_abs_random_removal': max(r['max_abs_random_removal'] for r in rows.values()),
           'max_abs_random_installation': max(r['max_abs_random_installation'] for r in rows.values()),
           'max_abs_random_any': max(max(r['max_abs_random_removal'], r['max_abs_random_installation']) for r in rows.values()),
           'min_abs_fitted_removal': min(abs(r['fitted_removal']) for r in rows.values())}
    OUT.with_suffix('.json').write_text(json.dumps({'readouts': rows, 'overall': tot}, indent=1), encoding='utf-8')
    md = ['# The random control over all seeds (eight random directions per layer, all positions)', '',
          'Change of the rate of "you" from the unedited condition. Per readout: the largest absolute change over its runs.', '',
          '| readout | runs | random removal, max abs | random installation, max abs | fitted removal (r05q_all) |', '|---|---|---|---|---|']
    for name, r in rows.items():
        md.append(f"| {name} | {len(r['runs'])} | {r['max_abs_random_removal']:.3f} | {r['max_abs_random_installation']:.3f} | "
                  f"{r['fitted_removal']:+.3f} |")
    md += ['', f"Overall: random at most {tot['max_abs_random_any']:.3f} (removal {tot['max_abs_random_removal']:.3f}, installation "
           f"{tot['max_abs_random_installation']:.3f}); fitted removal at least {tot['min_abs_fitted_removal']:.3f} in absolute value.", '']
    OUT.with_suffix('.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
