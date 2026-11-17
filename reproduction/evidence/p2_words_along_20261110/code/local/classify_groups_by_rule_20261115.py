"""Groups of the paper recomputed by rule (11-15, audit of the NMI version), so that the hand-written lists GROUPS and NAT_GROUPS of
build_final_tables_20261113.py can never drift from the criterion of the Methods again.

Rule. Layer index i is 0-based, i = the output of block i+1 (the geometry hooks every decoder block; the embeddings are never read).
A layer carries the self-index when the model's own message is singled out (R_A the smallest of the three ratios) in at least the
threshold share of the 75 held-out texts, on the message text or at the response onset. A model has a self-index when a layer with
i >= L/3 carries it (in block numbers: a block after block ceil(L/3)). Four-stream models use the mean over the four streams.
Self-reference is not recomputed here: it is the per-model attribution criterion of the paper (margin 0.10), i.e. the
'self-reference' lists of GROUPS and NAT_GROUPS. Natural order, the four templates that keep no System message after a User turn
(Qwen3.8-27B, gpt-oss-20b, R1-Distill-Llama-70B, Apertus-8B, marked with a dagger in Fig. 2): the main-probe level at the same
threshold, as in the paper.

Thresholds 70%, 80% (the paper's) and 90%. At 80% the result must equal GROUPS and NAT_GROUPS (assert).
Writes results/final_20261113/classify_groups_by_rule_20261115.json and .md. Read-only on all inputs.
usage (from projects/consciousness): python -I classify_groups_by_rule_20261115.py"""
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_final_tables_20261113 import GROUPS, NAT_GROUPS  # noqa: E402

EV = Path(__file__).resolve().parents[3]
OUT = EV / 'p2_words_along_20261110/results/final_20261113/classify_groups_by_rule_20261115'
W, M8 = EV / 'p2_crossmodel_wave3_20261019/runs', EV / 'p2_crossmodel_8models_20261018/models'
G8 = lambda m, mode='non_thinking', d='geo_attr': M8 / m / mode / d / '_run_summary.json'  # noqa: E731
STREAM = 'stream'
# main probe: the geometry of Table tab:self_crossmodel (the files of kl_vs_lastlayer_geometry_20261114.py)
MAIN = {'qwen38_27b': G8('qwen38_27b'), 'qwen38_flashnext': (STREAM, W / 'pro/flashnext_stream_20261020/flashnext_stream_geometry_non_20261020.json'),
        'glm46v_flash': G8('glm46v_flash'), 'glm47_flash': G8('glm47_flash'), 'gemma4_31b_en': G8('gemma4_31b'), 'qwen36_27b': G8('qwen36_27b'),
        'qwen36_35b_a3b': G8('qwen36_35b'), 'r1_distill_llama70b': G8('r1_llama70b', 'thinking', 'geo'),
        'gpt_oss_20b_en': W / 'pro/suite_think/gpt_oss_20b_mxfp4_en/summary.json', 'gpt_oss_120b_en': W / 'pro/suite_think/gpt_oss_120b_en/summary.json',
        'glm53_flash': (STREAM, W / 'server3/suite_think/glm53_flash_stream_geometry.json'), 'exaone4_32b_en': W / 'server3/suite/exaone4_32b_en/summary.json',
        'mimo_v26_distill_qwen9b': W / 'pro/suite/mimo_v26_distill_qwen9b/summary.json', 'phi4_en': W / 'server3/suite/phi4_en/summary.json',
        'granite42_30b_en': W / 'server3/suite/granite42_30b_en/summary.json', 'granite41_30b_en': W / 'server3/suite/granite41_30b_en/summary.json',
        'olmo2_13b_en': W / 'pro/suite/olmo2_13b_en/summary.json', 'olmo2_7b_en': W / 'pro/suite/olmo2_7b_en/summary.json',
        'apertus8b_en': W / 'pro/suite/apertus8b_instruct_2509_en/summary.json',
        'moonlight16b_a3b_instruct': W / 'server3/suite/moonlight16b_a3b_instruct/summary.json',
        'mimo7b_rl_0530': W / 'server3/suite/mimo7b_rl_0530/summary.json', 'mimo_vl7b_rl_2508': W / 'pro/suite/mimo_vl7b_rl_2508/summary.json',
        'ernie45_21b_a3b_thinking': W / 'server3/suite/ernie45_21b_a3b_thinking/summary.json',
        'seed_oss_36b': W / 'server3/suite/seed_oss_36b/summary.json', 'qwen35_27b': W / 'server3/suite/qwen35_27b/summary.json'}
# natural order: the separate-position geometry of the natural-order suite (DeepSeek-V4.1-Flash: four streams, mean over streams)
NAT = {t: W / f'server3/suite_nat3/{d}/summary.json' for t, d in (
    ('exaone4_32b_en', 'exaone4_32b_en'), ('gemma4_31b_en', 'gemma4_31b_en'), ('glm46v_flash', 'glm46v_flash'), ('granite41_30b_en', 'granite41_30b_en'),
    ('mimo7b_rl_0530', 'mimo7b_rl_0530'), ('moonlight16b_a3b_instruct', 'moonlight16b_a3b_instruct'), ('phi4_en', 'phi4_en'),
    ('ernie45_21b_a3b_thinking', 'ernie45_21b_a3b_thinking'), ('seed_oss_36b', 'seed_oss_36b'), ('dsv41_flash', 'dsv41_flash_1017'))}
NAT.update({t: W / f'pro/suite_nat3/{d}/summary.json' for t, d in (('olmo2_7b_en', 'olmo2_7b_en'), ('ministral3_14b_instruct_en', 'ministral3_14b_instruct_en'),
                                                                     ('ministral3_14b_reasoning_en', 'ministral3_14b_reasoning_en'))})
DAGGER = ['qwen38_27b', 'gpt_oss_20b_en', 'r1_distill_llama70b', 'apertus8b_en']
THRESHOLDS = (0.7, 0.8, 0.9)
GNAMES = ('self-reference', 'self-index only', 'neither')
EPS = 1e-9


def shares(src, order):
    if isinstance(src, tuple):
        s = json.loads(src[1].read_text(encoding='utf-8'))
        return {p: s[p]['mean_of_streams']['share_A_lowest_by_layer'] for p in ('content', 'onset')}
    s = json.loads(src.read_text(encoding='utf-8'))
    if order == 'main':
        return {p: s[p]['share_A_lowest'] for p in ('content', 'onset')}
    if 'separate_content' in s:
        return {p: s[f'separate_{p}']['share_A_lowest'] for p in ('content', 'onset')}
    return {p: s[f'separate_{p}_mean_of_streams']['share_A_lowest'] for p in ('content', 'onset')}


def classify_one(sh, thr):
    L = len(sh['content'])
    assert len(sh['onset']) == L
    beyond = {p: [i for i, x in enumerate(sh[p]) if x >= thr - EPS and i >= L / 3] for p in ('content', 'onset')}
    return {'L': L, 'L_over_3': L / 3, 'qualifying_layers_0based': beyond, 'index': bool(beyond['content'] or beyond['onset'])}


def regroup(base, new):
    """Group lists at a threshold: tags keep their order in the paper's lists, a tag that changes group goes to the end of its new
    group (the order build_final_tables would give it), so that the 80% lists are GROUPS / NAT_GROUPS themselves."""
    out = {g: [t for t in base[g] if new[t] == g] for g in GNAMES}
    for g in GNAMES:
        out[g] += [t for t0 in GNAMES if t0 != g for t in base[t0] if new[t] == g]
    return out


def classify(thr):
    SR = {'main': set(GROUPS['self-reference']), 'natural': set(NAT_GROUPS['self-reference'])}
    res = {'main': {}, 'natural': {}}
    for t, src in MAIN.items():
        c = classify_one(shares(src, 'main'), thr)
        res['main'][t] = {**c, 'group': 'self-reference' if t in SR['main'] else 'self-index only' if c['index'] else 'neither'}
    for t in [t for g in GNAMES for t in NAT_GROUPS[g]]:
        if t in DAGGER:
            m = res['main'][t]
            res['natural'][t] = {'dagger': True, 'index': m['index'], 'group': 'self-reference' if t in SR['natural'] else
                                 'self-index only' if m['index'] else 'neither'}
        else:
            c = classify_one(shares(NAT[t], 'natural'), thr)
            res['natural'][t] = {'dagger': False, **c, 'group': 'self-reference' if t in SR['natural'] else 'self-index only' if c['index'] else 'neither'}
    groups = {'main': regroup(GROUPS, {t: r['group'] for t, r in res['main'].items()}),
              'natural': regroup(NAT_GROUPS, {t: r['group'] for t, r in res['natural'].items()})}
    summ = {}
    for o, base in (('main', GROUPS), ('natural', NAT_GROUPS)):
        R = res[o]
        paper = {t: g for g in GNAMES for t in base[g]}
        summ[o] = {'with_index': [t for t in R if R[t]['index']], 'counts': {g: len(groups[o][g]) for g in GNAMES},
                   'selfref_without_index': [t for t in R if R[t]['group'] == 'self-reference' and not R[t]['index']],
                   'changed_from_paper': [{'tag': t, 'paper': paper[t], 'here': R[t]['group']} for t in R if R[t]['group'] != paper[t]]}
    n_idx, n_sr = len(summ['main']['with_index']), len(GROUPS['self-reference'])
    summ['main']['nesting_P_exact'] = None if summ['main']['selfref_without_index'] else comb(n_idx, n_sr) / comb(len(MAIN), n_sr)
    summ['main']['nesting_P_formula'] = f'C({n_idx},{n_sr})/C({len(MAIN)},{n_sr})'
    return {'threshold': thr, 'models': res, 'groups': groups, 'summary': summ}


def main():
    allres = {f'{int(round(t * 100))}': classify(t) for t in THRESHOLDS}
    r80 = allres['80']
    for o, base in (('main', GROUPS), ('natural', NAT_GROUPS)):
        assert r80['groups'][o] == base, (o, r80['groups'][o], base)   # the paper's lists = the rule at 80%
    for k, r in allres.items():
        for o in ('main', 'natural'):
            assert not r['summary'][o]['selfref_without_index'], (k, o)   # the nesting at every threshold
    OUT.with_suffix('.json').write_text(json.dumps({'method': __doc__.split('usage')[0].strip(), 'thresholds': allres},
                                                   ensure_ascii=False, indent=1), encoding='utf-8')
    L = ['# Groups by rule (11-15)', '', 'Rule: 0-based layer i (output of block i+1) qualifies when i >= L/3 and the share of held-out texts with',
         'R_A smallest is at least the threshold, on the message text or at the response onset. 80% is the paper\'s threshold.', '',
         '| threshold | order | self-reference | self-index only | neither | with index | nesting P (main) | changed from the paper |',
         '|---|---|---|---|---|---|---|---|']
    for k, r in allres.items():
        for o in ('main', 'natural'):
            s = r['summary'][o]
            ch = '; '.join(f"{c['tag']} {c['paper']} -> {c['here']}" for c in s['changed_from_paper']) or 'none'
            p = f"{s['nesting_P_formula']} = {s['nesting_P_exact']:.4f}" if o == 'main' else '-'
            L.append(f"| {k}% | {o} | {s['counts']['self-reference']} | {s['counts']['self-index only']} | {s['counts']['neither']} | "
                     f"{len(s['with_index'])} | {p} | {ch} |")
    OUT.with_suffix('.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
