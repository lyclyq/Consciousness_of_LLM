"""Secret game under the two System prompts, all seven mechanism readouts (11-14, NMI version: Fig. 5 and the Extended Data table;
user 11-14: both prompts are reported). Reads only the runners' summary.json files; nothing is recomputed from items.
  first prompt  items games_{zh,en}_secret5.json (a712030b / d24c0a2f), folders *_s5_*
  second prompt items games_{zh,en}_secret6.json (4d27d8a5 / 26553b13), folders *_s6_*
Per readout and prompt: the smoke verdict under the release rule fixed in advance (10-26: unedited, setter share >= 0.5 for the
model's own X and <= 0.25 for a User X on finished games, >= 75% finished, control correct in >= 7 of 8 under both unedited
conditions); the full game where it exists (unedited setter shares, removal swU8_A and installation swA8_U as change from the
unedited condition with the runner's paired 95% interval, the random-direction edits rswU8_A / rswA8_U); control accuracy.
Also the attribution question of the same readouts (for panel c of the NMI Fig. 5, beside the game): unedited rate of "you" for
the model's own and a User message, and the rate after removal (swU8_A) and installation (swA8_U) with the paired 95% interval,
all 150 held-out items. Sources as fig_ablation_draft_20261114.py: the Qwen / gpt-oss readouts from the release stats
(r05r06_20261031.json, r05q_all), MiMo, EXAONE and GLM-5.3 from their run summaries (*_r05q_all/summary.json).
And the two questions that do not concern the source (content, arithmetic; 75 items, one option order): accuracy unedited and
after the same two edits, from v4_mechanism.json (all_content, all_arith) and the run summaries (*_content_all, *_arith_all).
One standard for all games (user 11-14): a game that does not pass the release rule is not released, whichever part it fails. Its
full game is not read here and is not reported: the table and the JSON carry its eight-game check only. Its full-game folders are
marked NOT_USED and are not in the paper data package (user 11-14: "这个原始数据也排除于论文数据包").
usage (from projects/consciousness): game_table_two_prompts_20261114.py"""
import json
from pathlib import Path

EV = Path('evidence')
W3 = EV / 'p2_crossmodel_wave3_20261019/runs'
MS = EV / 'p2_modeswitch_20261107/results'
GLM5 = MS / 'vast_h200_20261111/m/out/extra'
GLM6 = MS / 'vast_h200_20261114/m/out/v6'
P6 = MS / 'pro_game_v6_20261114/games_swap'
# readout: {prompt: (smoke_main, smoke_control, main, control)}; None where not run
R = {
    'Qwen3.8-27B, single pass': {p: (W3 / f'server3/games_swap/qwen38_27b_sp_{p}_smoke_main', W3 / f'server3/games_swap/qwen38_27b_sp_{p}_smoke_control',
                                     W3 / f'server3/games_swap/qwen38_27b_sp_{p}_main', W3 / f'server3/games_swap/qwen38_27b_sp_{p}_control') for p in ('s5', 's6')},
    'Qwen3.8-27B, thinking': {p: (W3 / f'server3/games_swap/qwen38_27b_{p}_smoke_main', W3 / f'server3/games_swap/qwen38_27b_{p}_smoke_control',
                                  W3 / f'server3/games_swap/qwen38_27b_{p}_main', W3 / f'server3/games_swap/qwen38_27b_{p}_control') for p in ('s5', 's6')},
    'gpt-oss-20b': {p: (W3 / f'pro/games_swap/gpt_oss_20b_{p}_smoke_main', W3 / f'pro/games_swap/gpt_oss_20b_{p}_smoke_control',
                        W3 / f'pro/games_swap/gpt_oss_20b_{p}_main', W3 / f'pro/games_swap/gpt_oss_20b_{p}_control') for p in ('s5', 's6')},
    'gpt-oss-120b': {p: (W3 / f'pro/games_swap/gpt_oss_120b_{p}_smoke_main', W3 / f'pro/games_swap/gpt_oss_120b_{p}_smoke_control',
                         W3 / f'pro/games_swap/gpt_oss_120b_{p}_main', W3 / f'pro/games_swap/gpt_oss_120b_{p}_control') for p in ('s5', 's6')},
    'GLM-5.3-Flash': {'s5': (GLM5 / 'glm53_s5_smoke_main', GLM5 / 'glm53_s5_smoke_control', GLM5 / 'glm53_s5_main50', None),
                      's6': (GLM6 / 'glm53_s6_smoke_main', GLM6 / 'glm53_s6_smoke_control', GLM6 / 'glm53_s6_main', None)},
    'MiMo-V2.6-Distill-Qwen-9B': {'s5': (MS / 'pro/games_swap/mimo_v26_s5_smoke_main', MS / 'pro/games_swap/mimo_v26_s5_smoke_control', None, None),
                                  's6': (P6 / 'mimo_v26_s6_smoke_main', P6 / 'mimo_v26_s6_smoke_control', None, None)},
    'EXAONE-4.0.1-32B': {'s5': (MS / 'pro_exaone32_20261112/games_swap/exaone32_s5_smoke_main', MS / 'pro_exaone32_20261112/games_swap/exaone32_s5_smoke_control', None, None),
                         's6': (P6 / 'exaone32_s6_smoke_main', P6 / 'exaone32_s6_smoke_control', None, None)},
}


ST = Path('evidence/p2_crossmodel_wave3_20261019/stats')   # 11-15: the archive copy (the same bytes as the 11-02 code-release copy read before)
ATTR = {'Qwen3.8-27B, single pass': 'qsp', 'Qwen3.8-27B, thinking': 'qth', 'gpt-oss-20b': 'g20', 'gpt-oss-120b': 'g120',
        'GLM-5.3-Flash': MS / 'vast_h200_20261111/m/out/mech/glm53_r05q_all', 'MiMo-V2.6-Distill-Qwen-9B': MS / 'pro/thinking_swap/mimo_v26_r05q_all',
        'EXAONE-4.0.1-32B': MS / 'pro_exaone32_20261112/thinking_swap/exaone32_r05q_all'}


def attribution(src, r05):
    if isinstance(src, str):
        s = r05[src]['r05q_all']['stats']
        g = lambda k: {'after': s[k]['stat'], 'change': s[k]['diff'], 'ci95': s[k]['diff_ci']}  # noqa: E731
        return {'n': s['base_A']['n'], 'own': s['base_A']['stat'], 'user': s['base_U']['stat'], 'removal': g('swU8_A'), 'installation': g('swA8_U')}
    s = json.loads((src / 'summary.json').read_text(encoding='utf-8'))
    g = lambda k: {'after': s[k]['you_share'], 'change': s[k]['you_minus_base']['mean'], 'ci95': s[k]['you_minus_base']['ci95']}  # noqa: E731
    return {'n': s['base_A']['n'], 'own': s['base_A']['you_share'], 'user': s['base_U']['you_share'], 'removal': g('swU8_A'), 'installation': g('swA8_U')}


V4K = {'Qwen3.8-27B, single pass': 'qwen_sp', 'Qwen3.8-27B, thinking': 'qwen_th', 'gpt-oss-20b': 'oss_th', 'gpt-oss-120b': 'oss120_th'}


def control_q(name, kind, v4):
    if name in V4K:
        s, k = v4[V4K[name]]['all_' + kind], 'accuracy'
    else:
        s = json.loads((Path(str(ATTR[name]).replace('r05q_all', kind + '_all')) / 'summary.json').read_text(encoding='utf-8'))
        k = 'content_accuracy'  # the runner's key for both questions
    return {'n': s['base_A']['n'], 'own': s['base_A'][k], 'user': s['base_U'][k], 'removal': {'after': s['swU8_A'][k]},
            'installation': {'after': s['swA8_U'][k]}}


def load(d):
    return json.loads((d / 'summary.json').read_text(encoding='utf-8'))['secret'] if d is not None else None


def verdict(m, c):
    a, u = m['base_A'], m['base_U']
    sep = a['finished'] >= 0.75 * a['n'] and u['finished'] >= 0.75 * u['n'] and a['own_share'] >= 0.5 and u['own_share'] <= 0.25
    ctl = all(round(c[k]['accuracy'] * c[k]['finished']) >= 7 for k in ('base_A', 'base_U'))
    return {'pass': bool(sep and ctl), 'separates_sources': bool(sep), 'control_ok': bool(ctl),
            'own_A': a['own_share'], 'own_U': u['own_share'], 'control_A': c['base_A']['accuracy'], 'control_U': c['base_U']['accuracy']}


def edit(s, k):
    v = s[k]
    o = v.get('own_minus_base') or {}
    return {'own_share': v['own_share'], 'change': o.get('mean'), 'ci95': o.get('ci95'), 'n': v['n'], 'finished': v['finished']}


def main():
    out = {}
    r05 = json.loads((ST / 'r05r06_20261031.json').read_text(encoding='utf-8'))['results']
    v4 = json.loads((ST / 'v4_mechanism.json').read_text(encoding='utf-8'))['results']
    for name, ps in R.items():
        out[name] = {'attribution': attribution(ATTR[name], r05), 'content': control_q(name, 'content', v4), 'arith': control_q(name, 'arith', v4)}
        for p, (sm, sc, fm, fc) in ps.items():
            e = {'smoke': verdict(load(sm), load(sc))}
            e['released'] = e['smoke']['pass']
            m = load(fm) if e['released'] else None
            if m:
                e['main'] = {'n': m['base_A']['n'], 'own_A': m['base_A']['own_share'], 'own_U': m['base_U']['own_share'],
                             'removal': edit(m, 'swU8_A'), 'installation': edit(m, 'swA8_U'),
                             'random_removal': edit(m, 'rswU8_A'), 'random_installation': edit(m, 'rswA8_U')}
            c = load(fc) if e['released'] else None
            if c:
                e['control_full'] = {k: c[k]['accuracy'] for k in c}
            out[name]['first' if p == 's5' else 'second'] = e
    dst = MS / 'game_two_prompts_20261114.json'
    dst.write_text(json.dumps({'source': __doc__.split('usage')[0].strip(), 'readouts': out}, ensure_ascii=False, indent=1), encoding='utf-8')
    L = ['| readout | prompt | smoke: own X / User X, control A / U | rule | full n | removal | installation | random rem. / inst. | control (full) A / U |',
         '|---|---|---|---|---|---|---|---|---|']
    for name, ps in out.items():
        for p in ('first', 'second'):
            e = ps[p]
            s = e['smoke']
            row = [name, p, f"{s['own_A']:.2f} / {s['own_U']:.2f}, {s['control_A']:.2f} / {s['control_U']:.2f}",
                   'pass' if s['pass'] else ('fail: sources' if not s['separates_sources'] else 'fail: control')]
            m = e.get('main')
            if m:
                f = lambda d: f"{d['change']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"  # noqa: E731
                row += [str(m['n']), f(m['removal']), f(m['installation']),
                        f"{m['random_removal']['change']:+.2f} / {m['random_installation']['change']:+.2f}"]
            else:
                row += ['—', 'not released', 'not released', 'not released']
            cf = e.get('control_full')
            row.append(f"{cf['base_A']:.2f} / {cf['base_U']:.2f}" if cf else '—')
            L.append('| ' + ' | '.join(row) + ' |')
    (MS / 'game_two_prompts_20261114.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
