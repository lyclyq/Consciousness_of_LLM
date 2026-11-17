"""Draft figure (11-10) of the attribution-question wording test, in the style of the paper's causal-test bar figure
(self_mechanism.pdf, make_v5_mechanism_20261110.py: fonts, readout colours, bar widths, error bars, grid, legend).
Reads results/wording_summary_all.json (analyze_wording_v2_20261109.py). Two panels: (a) removal ('去除', own message <- User
coordinates, change from base_A) and (b) installation ('植入', User message <- own coordinates, change from base_U). x = the three
wordings of the option naming the model (paper = main "You (the AI assistant)" / 你（AI 助手）, pron = "You" / 你, role = "The AI
assistant" / AI 助手); bars = the seven readouts; height = item-paired change of the option naming the model ('self' 'mean'),
error bars = 95% family-bootstrap interval ('ci_families', 15 wording families). Random edits are not drawn.
Qwen3.8-27B with the English items: the same two blues as the Chinese items, hatched (white diagonal lines).
Drafts only, never written into paper/. usage: python -I make_fig_wording_20261110.py [SUMMARY_JSON] [OUT_DIR]
-> OUT_DIR/self_wording.pdf/.png (default ../p2_crossmodel_wave3_20261019/code/local/figures/drafts_20261110)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
EV = HERE.parents[3]  # evidence/
SUMMARY = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / 'results' / 'wording_summary_all.json'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else EV / 'p2_crossmodel_wave3_20261019' / 'code' / 'local' / 'figures' / 'drafts_20261110'
if 'paper' in {p.name for p in OUT.resolve().parents} | {OUT.resolve().name}:
    sys.exit('refusing to write into paper/: drafts only')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'pdf.fonttype': 42, 'axes.titlesize': 9, 'axes.titleweight': 'bold',
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'hatch.linewidth': 0.7})
C_INK, C_GRID, C_ZERO = '#2b2f36', '#e3e6ea', '#9aa1a9'
# readout colours as in make_v5_mechanism_20261110.py; English Qwen = the same blue, hatched (same model, other item language)
READOUTS = [  # key in the summary, legend label, colour, hatch
    ('qsp', 'Qwen3.8-27B, no thinking, Chinese items', '#8dbbef', None),
    ('qth', 'Qwen3.8-27B, thinking, Chinese items', '#2a78d6', None),
    ('qsp_en', 'Qwen3.8-27B, no thinking, English items', '#8dbbef', '////'),
    ('qth_en', 'Qwen3.8-27B, thinking, English items', '#2a78d6', '////'),
    ('g20', 'gpt-oss-20b, thinking', '#a61e4d', None),
    ('g120', 'gpt-oss-120b, thinking', '#e64980', None),
    ('m26', 'MiMo-V2.6-Distill-9B, default call', '#f08c00', None)]
WORDINGS = [('paper', 'main\n"You (the AI assistant)"'), ('pron', 'pronoun\n"You"'), ('role', 'role name\n"The AI assistant"')]
PANELS = [('去除', '(a) removal: own message ← User', (-1.06, 0.06), [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0]),
          ('植入', '(b) installation: User message ← own', (-0.2, 0.92), [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])]


def main():
    S = json.loads(SUMMARY.read_text(encoding='utf-8'))['by_readout']
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.75), gridspec_kw={'wspace': 0.2})
    w = 0.118
    missing = []
    for ax, (edit, title, ylim, yt) in zip(axes, PANELS):
        for j, (wd, _) in enumerate(WORDINGS):
            for i, (ro, _, col, hatch) in enumerate(READOUTS):
                s = S.get(ro, {}).get('wordings', {}).get(wd, {}).get(edit, {}).get('self')
                if s is None:
                    missing.append(f'{ro}/{wd}/{edit}')
                    continue
                m, (lo, hi) = s['mean'], s['ci_families']
                x = j + (i - (len(READOUTS) - 1) / 2) * w
                ax.bar(x, m, width=w * 0.92, color=col, edgecolor='white', lw=0.4, hatch=hatch, zorder=3)
                ax.errorbar(x, m, yerr=[[max(0.0, m - lo)], [max(0.0, hi - m)]], fmt='none', ecolor=C_INK, lw=0.7, capsize=1.2, zorder=4)
                print(f'{title[:3]} {wd:5s} {ro:6s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]')
        ax.axhline(0, color=C_ZERO, lw=0.7, zorder=2)
        ax.set_xticks(range(len(WORDINGS)))
        ax.set_xticklabels([lab for _, lab in WORDINGS], fontsize=7.2)
        ax.set_xlim(-0.55, len(WORDINGS) - 0.45)
        ax.set_ylim(*ylim)
        ax.set_yticks(yt)
        ax.set_title(title)
        ax.grid(axis='y', color=C_GRID, lw=0.6, zorder=0)
        ax.tick_params(axis='x', length=0)
        ax.spines['bottom'].set_visible(False)
    axes[0].set_ylabel('change in claim of the message\n(option naming the model)')
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=col, edgecolor='white', lw=0.4, hatch=hatch) for _, _, col, hatch in READOUTS]
    fig.legend(handles, [t for _, t, _, _ in READOUTS], loc='lower center', ncol=4, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.145), handlelength=1.3, handleheight=1.1, columnspacing=0.8, handletextpad=0.4)
    fig.savefig(OUT / 'self_wording.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'self_wording.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('missing:', ', '.join(missing) or 'none')
    print('written', OUT / 'self_wording.pdf')


if __name__ == '__main__':
    main()
