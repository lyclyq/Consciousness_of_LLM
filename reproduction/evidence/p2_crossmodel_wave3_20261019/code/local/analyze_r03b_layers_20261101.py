"""R03 Part B by layer (11-01, user: "是不是一层的位置没选对"): the first-order sensitivities of runs/pro/<dir>_r03fd split by
depth instead of summed over layers. Per readout and endpoint (A: the model's own message, U: the User message), per computed layer:
mean over items of D along d (total), its part in the fitted subspace (V), the rest (perp), the random part (R) and the unperturbed
rows (zero, the numerical noise), and the median share of the squared norm of d in the subspace. Bands: the thirds of the layers,
and the window band in which installation exceeds +0.10 in the layer-window runs of the paper (Table tab:self_mechanism_windows:
Qwen3.8-27B L24-L47 without and L28-L43 with thinking, gpt-oss-20b L0-L15, gpt-oss-120b L8-L23). Shares are ratios of summed
means (as stats/r03_20261101.md), with a 95% bootstrap interval over items (2000 draws, seed 1).
Outputs (ARCHIVE/stats): r03b_layers_20261101.json, r03b_layers_20261101.md.
usage: analyze_r03b_layers_20261101.py ARCHIVE"""
import json
import random
import sys
from pathlib import Path

RD = [('qsp', 'singlepass_swap/qwen38_27b', 'Qwen3.8-27B, no thinking', 64, (24, 47)),
      ('qth', 'thinking_swap/qwen38_27b', 'Qwen3.8-27B, thinking', 64, (28, 43)),
      ('g20', 'thinking_swap/gpt_oss_20b', 'gpt-oss-20b, thinking', 24, (0, 15)),
      ('g120', 'thinking_swap/gpt_oss_120b', 'gpt-oss-120b, thinking', 36, (8, 23))]
DIRS = ('total', 'V', 'perp', 'R', 'zero')


def share(items, layers, ep, num, den='total', draws=2000):
    def ratio(sample):
        n = sum(sum(it[ep][num][l] for l in layers) for it in sample)
        d = sum(sum(it[ep][den][l] for l in layers) for it in sample)
        return n / d if d else float('nan')
    est = ratio(items)
    rng = random.Random(1)
    bs = sorted(ratio([items[rng.randrange(len(items))] for _ in items]) for _ in range(draws))
    bs = [x for x in bs if x == x]
    return est, (bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]) if bs else (float('nan'), float('nan'))


def sum_ci(items, layers, ep, draws=2000):
    """95% bootstrap interval over items of the band's summed slope along d (mean over items)"""
    vals = [sum(it[ep]['total'][l] for l in layers) for it in items]
    rng = random.Random(1)
    bs = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals) for _ in range(draws))
    return bs[int(0.025 * draws)], bs[int(0.975 * draws) - 1]


def main():
    A = Path(sys.argv[1])
    res, L = {}, ['# R03 Part B by layer (11-01)', '',
                  'Mean over items of the central-difference slope D of f = logit(you letter) - logit(user letter), by computed layer, '
                  'at the model\'s own message (A) and at the User message (U); shares are ratios of summed means over a band of '
                  'layers (95% bootstrap interval over items). Install band: the layers of the windows in which installation exceeds '
                  '+0.10 in the paper\'s layer-window runs. Thinking: the model\'s own reasoning is teacher-forced, so D measures the '
                  'letter read after a fixed reasoning text.', '']
    for key, d, name, nl, band in RD:
        items = []
        for p in sorted((A / 'runs/pro' / f'{d}_r03fd' / 'items').glob('*.json')):
            j = json.load(open(p, encoding='utf-8'))
            lays = j['layers']
            it = {'layers': lays}
            for ep in ('A', 'U'):  # an endpoint without a finished reasoning has no slopes (gpt-oss-20b: one item at A)
                if ep in j['endpoints'] and 'D' in j['endpoints'][ep]:
                    it[ep] = {k: dict(zip(lays, j['endpoints'][ep]['D'][k])) for k in DIRS}
                    it[ep]['f0'] = j['endpoints'][ep]['f0']
            if 'norms' in j:
                it['vshare'] = {l: (j['norms']['V'][i] / j['norms']['total'][i]) ** 2 if j['norms']['total'][i] else 0.0
                                for i, l in enumerate(lays)}
            if 'A' in it or 'U' in it:
                items.append(it)
        lays = items[0]['layers']
        thirds = {'first third': [l for l in lays if l < nl / 3], 'middle third': [l for l in lays if nl / 3 <= l < 2 * nl / 3],
                  'last third': [l for l in lays if l >= 2 * nl / 3], 'install band': [l for l in lays if band[0] <= l <= band[1]],
                  'all layers': lays}
        res[key] = {'n_items': len(items), 'layers': lays, 'by_layer': {}, 'bands': {}}
        for ep in ('A', 'U'):
            its = [it for it in items if ep in it]
            res[key]['by_layer'][ep] = {l: {k: sum(it[ep][k][l] for it in its) / len(its) for k in DIRS} for l in lays}
            res[key]['bands'][ep] = {}
            for bn, bl in thirds.items():
                v, vci = share(its, bl, ep, 'V')
                q, qci = share(its, bl, ep, 'perp')
                tot = sum(res[key]['by_layer'][ep][l]['total'] for l in bl)
                vs = sorted(it['vshare'][l] for it in its if 'vshare' in it for l in bl)
                med_vs = vs[len(vs) // 2]
                res[key]['bands'][ep][bn] = {'layers': [bl[0], bl[-1]] if bl else [], 'sum_mean_D_total': tot,
                                             'sum_ci': sum_ci(its, bl, ep), 'V_share': v,
                                             'V_share_ci': vci, 'perp_share': q, 'perp_share_ci': qci, 'median_size_share_V': med_vs}
        L += [f'## {name} ({len(items)} items; computed layers {lays[0]}-{lays[-1]}, step {lays[1] - lays[0]})', '',
              '| endpoint | band | layers | summed mean D_total | V share | perp share | size share of V (median) |', '|---|---|---|---|---|---|---|']
        for ep in ('A', 'U'):
            for bn, b in res[key]['bands'][ep].items():
                L.append(f"| {ep} | {bn} | {b['layers'][0] if b['layers'] else '-'}-{b['layers'][1] if b['layers'] else '-'} | "
                         f"{b['sum_mean_D_total']:+.2f} | {b['V_share']:.2f} [{b['V_share_ci'][0]:.2f}, {b['V_share_ci'][1]:.2f}] | "
                         f"{b['perp_share']:.2f} [{b['perp_share_ci'][0]:.2f}, {b['perp_share_ci'][1]:.2f}] | {b['median_size_share_V']:.2f} |")
        L += ['', '| layer | A: total | A: V | A: perp | U: total | U: V | U: perp | U: R | U: zero |', '|---|---|---|---|---|---|---|---|---|']
        for l in lays:
            a, u = res[key]['by_layer']['A'][l], res[key]['by_layer']['U'][l]
            L.append(f"| L{l} | {a['total']:+.2f} | {a['V']:+.2f} | {a['perp']:+.2f} | {u['total']:+.2f} | {u['V']:+.2f} | {u['perp']:+.2f} | "
                     f"{u['R']:+.2f} | {u['zero']:+.2f} |")
        L.append('')
    (A / 'stats' / 'r03b_layers_20261101.json').write_text(json.dumps({'results': res}, indent=1), encoding='utf-8')
    (A / 'stats' / 'r03b_layers_20261101.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(x for x in L if not x.startswith('| L')))


if __name__ == '__main__':
    main()
