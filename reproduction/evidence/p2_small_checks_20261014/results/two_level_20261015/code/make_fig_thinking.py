"""Figure self_thinking (check N): answers to the self-reference question without thinking (template non-thinking mode,
single forward) and with thinking (template thinking mode, answer read after the model's own reasoning), for Assistant, User
and System messages carrying the same text. Stacked bars = mean probability of each answer over items (body x option order).
usage: make_fig_thinking.py N_SUMMARY OUT_DIR"""
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ANS = [('p_self', 'you (AI assistant)', '#2a78d6'), ('p_user', 'the user', '#eb6834'), ('p_system', 'the system', '#1baf7a')]
TRUE = {'assistant': 'p_self', 'user': 'p_user', 'system': 'p_system'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'axes.titlesize': 10.5,
                     'axes.titleweight': 'bold', 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5})


def main(n_json, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    N = json.load(open(n_json, encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 3.4), sharey=True, gridspec_kw={'wspace': 0.12})
    for ax, (role, title) in zip(axes, (('assistant', '(a) Assistant message'), ('user', '(b) User message'))):
        R = N[role]
        for x, mode in enumerate(('off', 'on')):
            bottom = 0.0
            for key, label, col in ANS:
                v = R[f'{mode}_{key}']
                ax.bar(x, v, bottom=bottom, width=0.62, color=col, edgecolor='white', linewidth=1.0,
                       label=label if (x == 0 and role == 'assistant') else None)
                if key == TRUE[role] or v >= 0.15:
                    ax.text(x, bottom + v / 2, f'{v:.2f}', ha='center', va='center', fontsize=9.5, color='white',
                            fontweight='bold' if key == TRUE[role] else 'normal')
                bottom += v
        ax.set_xticks([0, 1], ['without\nthinking', 'with\nthinking'])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 1.0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(title, loc='left')
    axes[0].set_ylabel('mean probability of the answer')
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, ['answer: ' + l for l in lab], loc='lower center', ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.97))
    fig.savefig(f'{out_dir}/self_thinking.pdf', bbox_inches='tight')
    fig.savefig(f'{out_dir}/self_thinking.png', dpi=150, bbox_inches='tight')
    print('saved')


if __name__ == '__main__':
    main(*sys.argv[1:])
