"""R14 of the supervisor's review (11-01): the single-pass self-reference of the cross-model comparison with all three cyclic option
orders (balanced) against the first two (the paper), from the archived body-level tables (no new runs). Per model with a non-thinking
mode: P("you") for own, User and System messages, the excess own - max(User, System) (body-level means over the orders, as the paper),
bodies with a positive excess, and whether the excess reaches the 0.10 margin of Definition def:self; the self-index from
stats/r04_threshold_20261101.json (80%, classification rendering). Thinking items exist only in the first two orders (300 per source),
so the balanced summary covers the single pass only.
Outputs (ARCHIVE/stats): r14_orders_20261101.json, r14_orders_20261101.md.
usage: analyze_r14_orders_20261101.py ARCHIVE8 ARCHIVE3"""
import csv
import json
import sys
from pathlib import Path

NAMES = {'qwen38_27b': 'Qwen3.8-27B', 'qwen38_flashnext': 'Qwen3.8-Flash-Next', 'exaone4_32b': 'EXAONE-4.0.1-32B',
         'glm46v_flash': 'GLM-4.6V-Flash', 'glm47_flash': 'GLM-4.7-Flash', 'granite42_30b': 'Granite-4.2-30B',
         'granite41_30b': 'Granite-4.1-30B', 'olmo2_13b': 'OLMo-2-13B', 'olmo2_7b': 'OLMo-2-7B', 'seed_oss_36b': 'Seed-OSS-36B',
         'gemma4_31b': 'Gemma-4-31B', 'qwen36_27b': 'Qwen3.6-27B', 'qwen36_35b': 'Qwen3.6-35B-A3B', 'qwen35_27b': 'Qwen3.5-27B'}
MARGIN = 0.10


def main():
    A8, A3 = Path(sys.argv[1]), Path(sys.argv[2])
    files = [A8 / 'stats/attribution_non_thinking_body_level.csv', A3 / 'stats/attribution_non_thinking_body_level.csv',
             A3 / 'stats/v4_attribution_non_thinking.csv', A3 / 'stats/v5_olmo_attribution_non_thinking.csv']
    rows = {}
    for f in files:
        for r in csv.DictReader(open(f, encoding='utf-8', newline='')):
            if r['model'] in NAMES:
                assert (r['model'], r['orders']) not in rows, (r['model'], r['orders'])
                rows[(r['model'], r['orders'])] = r
    r04 = json.loads((A3 / 'stats/r04_threshold_20261101.json').read_text(encoding='utf-8'))['models']
    out, L = {}, ['# R14: single-pass self-reference, three option orders against the first two (11-01)', '',
                  'Excess = mean over bodies of P(you | own) - max(P(you | User), P(you | System)), probabilities averaged over the orders '
                  'first (paper definition). Self-reference: excess >= 0.10. Self-index: 80% criterion, classification rendering.', '',
                  '| model | self-index | 2 orders: own / User / System | excess | bodies > 0 | 3 orders: own / User / System | excess | '
                  'bodies > 0 | self-reference 2 / 3 |', '|---|---|---|---|---|---|---|---|---|']
    for m, name in NAMES.items():
        two, three = rows[(m, '2_orders_as_paper')], rows[(m, '3_orders')]
        v = r04[m]
        idx = v['by_rendering'][v['classification_rendering']]['0.8']['index']
        f = lambda r: [float(r['p_you_own']), float(r['p_you_user']), float(r['p_you_system']), float(r['own_minus_max_other']),  # noqa: E731
                       int(r['bodies_positive'])]
        a, b = f(two), f(three)
        out[m] = {'index': idx, 'two_orders': a, 'three_orders': b, 'selfref_two': a[3] >= MARGIN, 'selfref_three': b[3] >= MARGIN}
        L.append(f"| {name} | {'yes' if idx else 'no'} | {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f} | {a[3]:+.3f} | {a[4]} | "
                 f"{b[0]:.3f} / {b[1]:.3f} / {b[2]:.3f} | {b[3]:+.3f} | {b[4]} | {'yes' if a[3] >= MARGIN else 'no'} / "
                 f"{'yes' if b[3] >= MARGIN else 'no'} |")
    nested = {k: all(v['index'] for v in out.values() if v[f'selfref_{k}']) for k in ('two', 'three')}
    changed = [NAMES[m] for m, v in out.items() if v['selfref_two'] != v['selfref_three']]
    noidx_max = {k: max((v[f'{k}_orders'][3], NAMES[m]) for m, v in out.items() if not v['index']) for k in ('two', 'three')}
    L += ['', f"Nesting (no self-reference without a self-index): two orders {nested['two']}, three orders {nested['three']}. "
          f"Verdict changed by the third order: {', '.join(changed) or 'none'}. Largest excess without a self-index: two orders "
          f"{noidx_max['two'][0]:+.3f} ({noidx_max['two'][1]}), three orders {noidx_max['three'][0]:+.3f} ({noidx_max['three'][1]})."]
    (A3 / 'stats' / 'r14_orders_20261101.json').write_text(json.dumps({'results': out, 'nested': nested, 'changed': changed,
                                                                      'noindex_max': noidx_max}, indent=1), encoding='utf-8')
    (A3 / 'stats' / 'r14_orders_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
