"""Draft of the ablation (causal-test) figure of the final paper (11-14): the seven readouts of the mechanism group, one row each.
Every column runs the same way: 0 = no effect, 1 = the edit closes the whole gap between the model's own message and a User
message (or removes the whole next-token difference); larger = stronger.
  a  attribution question, removal (own message <- User values): share of the unedited gap removed, fitted coordinates (blue) and
     random coordinates of the same dimension (orange), arrow from 8 to 64 dimensions; 95% interval at 8 dimensions;
  b  attribution question, installation (User message <- own values): share of the gap closed, same marks;
  c  next-token KL, removal: 1 - KL(edited own || User) / KL(own || User), fitted and random, 8 dimensions;
  d  next-token KL, installation: 1 - KL(edited User || own) / KL(User || own), fitted and random, 8 dimensions;
  (e, the questions that do not concern the source, was dropped from the figure on 11-14: the NMI Fig. 5c shows the content and
   the arithmetic question in their own panels, fig_game_nmi_20261114.py. The lowest accuracies stay in the JSON under 'ctrl'.)
Sources: the four Qwen / gpt-oss readouts from the release stats (r05r06_20261031.json: fit_q at all positions, the procedure of
the other three readouts; v4_mechanism.json: 64 dimensions with the paper fit, controls), MiMo-V2.6, EXAONE and GLM-5.3 from their
run summaries; KL replacement from results/*/modeswitch/swap*/summary.json (content KL).
usage: fig_ablation_draft_20261114.py OUT_PREFIX"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42})  # Nature figure guide: Arial, editable text
P = Path(__file__).resolve().parents[4]          # projects/consciousness
MS = P / 'evidence/p2_modeswitch_20261107/results'
ST = P / 'evidence/p2_crossmodel_wave3_20261019/stats'   # 11-15: the archive copy (the same bytes as the 11-02 code-release copy read before)
R05 = json.loads((ST / 'r05r06_20261031.json').read_text(encoding='utf-8'))['results']
V4 = json.loads((ST / 'v4_mechanism.json').read_text(encoding='utf-8'))['results']

READOUTS = [  # name, r05 key, v4 key, run folders (attribution 8, 64, content, arith), swap folder
    ('Qwen3.8-27B, no thinking', 'qsp', 'qwen_sp', None, MS / 'pro/modeswitch/swap/qsp'),
    ('Qwen3.8-27B, thinking', 'qth', 'qwen_th', None, MS / 'pro/modeswitch/swap/qth'),
    ('gpt-oss-20b', 'g20', 'oss_th', None, MS / 'pro/modeswitch/swap/g20'),
    ('gpt-oss-120b', 'g120', 'oss120_th', None, MS / 'pro/modeswitch/swap/g120'),
    ('MiMo-V2.6-Distill-9B', None, None, MS / 'pro/thinking_swap/mimo_v26_', MS / 'pro/modeswitch/swap/m26'),
    ('EXAONE-4.0.1-32B', None, None, MS / 'pro_exaone32_20261112/thinking_swap/exaone32_', MS / 'pro_exaone32_20261112/modeswitch/swap/e32'),
    ('GLM-5.3-Flash', None, None, MS / 'vast_h200_20261111/m/out/mech/glm53_', MS / 'vast_h200_20261111/m/out/mech/swap_glm53'),
]
INK, MUTED, GRID, BAND = '#1f1f1e', '#6b6b68', '#e4e4e0', '#f4f4f1'
C_FIT, C_RND, REF = '#2a78d6', '#eb6834', '#8d8d89'
DY, ALPHA = 0.17, 0.9


def j(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def exact_items(d):
    """11-15: exact rates and paired differences from the per-item records of a run folder (the summary rounds to 4 decimals).
    Rate = share of finished items answered 'self'; paired difference over the items finished in both conditions."""
    recs = {}
    for f in sorted(Path(d, 'items').glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        recs[(r['cond'], r['id'], tuple(r['order']))] = (r.get('finished', True), r.get('on_answer') == 'self')
    conds = sorted({c for c, _, _ in recs})
    rate = {c: (lambda v: sum(v) / len(v))([y for (cc, _, _), (fin, y) in recs.items() if cc == c and fin]) for c in conds}
    diff = {}
    for c in conds:
        if c.startswith('base'):
            continue
        b = 'base_A' if c.endswith('_A') else 'base_U'
        pr = [(y, recs[(b, i, o)][1]) for (cc, i, o), (fin, y) in recs.items() if cc == c and fin and recs[(b, i, o)][0]]
        diff[c] = (sum(e - z for e, z in pr) / len(pr), len(pr))
    return rate, diff


def attribution(r05, v4, run):
    """{(edit, dims, fitted): (share, lo, hi)} with share = |diff| / (base_A - base_U)."""
    out = {}
    if run is None:
        s8 = R05[r05]['r05q_all']['stats']
        s64 = V4[v4]['all_k64']
        for s, dims in ((s8, 8), (s64, 64)):
            gap = s['base_A']['stat'] - s['base_U']['stat']
            for cond, edit, fit in ((f'swU{dims}_A', 'removal', True), (f'swA{dims}_U', 'installation', True),
                                    (f'rswU{dims}_A', 'removal', False), (f'rswA{dims}_U', 'installation', False)):
                c = s[cond]
                sg = -1 if edit == 'removal' else 1
                lo, hi = sorted((sg * c['diff_ci'][0] / gap, sg * c['diff_ci'][1] / gap))
                out[(edit, dims, fit)] = (sg * c['diff'] / gap, lo, hi)
        return out
    for suf, dims in (('r05q_all', 8), ('all_k64', 64)):
        s = j(str(run) + suf + '/summary.json')
        rate, diff = exact_items(str(run) + suf)
        for c, v in rate.items():
            assert abs(v - s[c]['you_share']) <= 5.01e-5, (run, suf, c, v, s[c]['you_share'])
        gap = rate['base_A'] - rate['base_U']
        for cond, edit, fit in ((f'swU{dims}_A', 'removal', True), (f'swA{dims}_U', 'installation', True),
                                (f'rswU{dims}_A', 'removal', False), (f'rswA{dims}_U', 'installation', False)):
            d = s[cond]['you_minus_base']
            mean, npair = diff[cond]
            assert npair == d['n_pairs'] and abs(mean - d['mean']) <= 5.01e-5, (run, suf, cond, mean, d)
            sg = -1 if edit == 'removal' else 1
            lo, hi = sorted((sg * d['ci95'][0] / gap, sg * d['ci95'][1] / gap))
            out[(edit, dims, fit)] = (sg * mean / gap, lo, hi)
    return out


def controls(v4, run):
    if run is None:
        acc = lambda st: min(c['accuracy'] for c in st.values())  # noqa: E731
        return acc(V4[v4]['all_content']), acc(V4[v4]['all_arith'])
    acc = lambda p: min(v['content_accuracy'] for v in j(p).values())  # noqa: E731
    return acc(str(run) + 'content_all/summary.json'), acc(str(run) + 'arith_all/summary.json')


def swap(folder):
    s = j(folder / 'summary.json')
    lf, mv = s['left']['content'], s['moved']['content']
    rm = lambda d: (1 - d['value'], 1 - d['ci95'][1], 1 - d['ci95'][0])  # noqa: E731
    ins = lambda d: (d['value'], d['ci95'][0], d['ci95'][1])  # noqa: E731
    return {'removal': rm(lf['A_U']), 'removal_rnd': rm(lf['A_Ur']), 'installation': ins(mv['U_A']), 'installation_rnd': ins(mv['U_Ar'])}


def main():
    out = sys.argv[1]
    data = []
    for name, r05, v4, run, sw in READOUTS:
        data.append({'name': name, 'attr': attribution(r05, v4, run), 'ctrl': controls(v4, run), 'kl': swap(sw),
                     'fit64': 'paper fit' if run is None else 'fit_q'})
    n = len(data)
    H = 0.36 * n + 1.75
    fig, axs = plt.subplots(1, 4, figsize=(7.08, H), sharey=True, gridspec_kw={'width_ratios': [1.15, 1.15, 1, 1], 'wspace': 0.1})
    for ax in axs:
        for i in range(n):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
        ax.set_ylim(n - 0.5, -0.5)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=6.2, colors=MUTED, length=2.5)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    axs[0].set_yticks(range(n))
    axs[0].set_yticklabels([d['name'] + (' *' if d['fit64'] == 'paper fit' else '') for d in data], fontsize=6.9, color=INK)

    def arrow(ax, y, a8, a64, color, filled):
        ax.plot([a8[1], a8[2]], [y, y], color=color, lw=0.9, alpha=ALPHA, zorder=3)
        if abs(a64[0] - a8[0]) >= 0.06:
            ax.annotate('', xy=(a64[0], y), xytext=(a8[0], y), zorder=4,
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.0, alpha=ALPHA, shrinkA=3, shrinkB=0, mutation_scale=7))
        else:   # 64 dimensions change nothing: a short tick at the 64-dimension value instead of an arrow
            ax.plot(a64[0], y, '|', ms=7, mew=1.2, color=color, alpha=ALPHA, zorder=4)
        ax.plot(a8[0], y, 'o', ms=4.6, mfc=color if filled else 'none', mec=color, mew=1.0, alpha=ALPHA, zorder=5)

    for i, d in enumerate(data):
        A = d['attr']
        for ax, edit in ((axs[0], 'removal'), (axs[1], 'installation')):
            arrow(ax, i - DY, A[(edit, 8, True)], A[(edit, 64, True)], C_FIT, True)
            arrow(ax, i + DY, A[(edit, 8, False)], A[(edit, 64, False)], C_RND, False)
        for ax, key in ((axs[2], 'removal'), (axs[3], 'installation')):
            v, lo, hi = d['kl'][key]
            ax.plot([lo, hi], [i - DY, i - DY], color=C_FIT, lw=0.9, alpha=ALPHA, zorder=3)
            ax.plot(v, i - DY, 'o', ms=4.6, mfc=C_FIT, mec='none', alpha=ALPHA, zorder=4)
            v, lo, hi = d['kl'][key + '_rnd']
            ax.plot([lo, hi], [i + DY, i + DY], color=C_RND, lw=0.9, alpha=ALPHA, zorder=3)
            ax.plot(v, i + DY, 'o', ms=4.6, mfc='none', mec=C_RND, mew=1.0, alpha=ALPHA, zorder=4)

    for ax in axs[:4]:
        ax.axvline(0, color=REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.axvline(1, color=REF, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
    for ax in axs[:2]:
        ax.set_xlim(-0.25, 1.75)
        ax.set_xticks([0, 0.5, 1, 1.5])
        ax.set_xticklabels(['0', '0.5', '1', '1.5'])
    for ax in axs[2:4]:
        ax.set_xlim(-0.15, 1.08)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['0', '0.5', '1'])
    axs[0].set_xlabel('share of the gap\nremoved', fontsize=6.8, color=INK)
    axs[1].set_xlabel('share of the gap\nclosed', fontsize=6.8, color=INK)
    axs[2].set_xlabel('share of the KL\ndifference removed', fontsize=6.8, color=INK)
    axs[3].set_xlabel('share of the KL\ndifference closed', fontsize=6.8, color=INK)
    subt = ['removal', 'installation', 'removal', 'installation']
    for ax, t, st in zip(axs, 'abcd', subt):
        ax.set_title(t, fontsize=8, color=INK, loc='left', fontweight='bold', pad=4)
        ax.annotate(st, xy=(0.58, 1.0), xycoords='axes fraction', xytext=(0, 4), textcoords='offset points', fontsize=6.6,
                    color=INK, ha='center', va='bottom', annotation_clip=False)
    top = 1 - 0.87 / H
    fig.subplots_adjust(left=0.215, right=0.985, top=top, bottom=0.78 / H)

    def header(a0, a1, text):
        p0, p1 = a0.get_position(), a1.get_position()
        yh = top + 0.3 / H
        fig.text((p0.x0 + p1.x1) / 2, yh + 0.07 / H, text, ha='center', va='bottom', fontsize=7, color=INK, fontweight='bold')
        fig.add_artist(Line2D([p0.x0 + 0.01, p1.x1], [yh, yh], color=INK, lw=0.7))
    header(axs[0], axs[1], 'Attribution question')
    header(axs[2], axs[3], 'Next-token KL')
    h = [Line2D([], [], marker='o', ls='-', lw=0.9, ms=4.6, color=C_FIT, mfc=C_FIT, label='fitted self-index coordinates (95% interval)'),
         Line2D([], [], marker='o', ls='-', lw=0.9, ms=4.6, color=C_RND, mfc='none', label='random coordinates, same number of directions'),
         Line2D([], [], marker='$\\rightarrow$', ls='', ms=9, color=INK, label='a, b: 8 → 64 directions (tick: no change)')]
    fig.legend(handles=h, loc='upper center', ncol=3, fontsize=6.3, frameon=False, bbox_to_anchor=(0.55, 1.0), columnspacing=1.3,
               handletextpad=0.4, labelcolor=INK, labelspacing=0.35)
    fig.text(0.01, 0.006, '* 64 directions from an earlier fit by the same procedure. Dashed line, no effect. Dotted line, the whole gap.',
             fontsize=5.8, color=MUTED, ha='left', va='bottom')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out + '.png', dpi=220, facecolor='white')
    fig.savefig(out + '.pdf', facecolor='white')
    json.dump([{**{k: v for k, v in d.items() if k != 'attr'}, 'attr': {f'{a}|{b}|{"fit" if c else "random"}': v for (a, b, c), v in d['attr'].items()}}
               for d in data], open(out + '.json', 'w', encoding='utf-8'), indent=1)
    print('written', Path(out).name)


if __name__ == '__main__':
    main()
