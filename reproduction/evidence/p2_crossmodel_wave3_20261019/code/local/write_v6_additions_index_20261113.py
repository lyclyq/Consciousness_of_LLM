"""INDEX_additions.csv of evidence/paper_data_v6_additions_20261113/ (11-13): provisional elements of the final paper, one row per
evidence line, with the results, run code, aggregation code and item files in the package. Every referenced path must exist in the
package. The element names are provisional until the final tex fixes figure and table labels.
11-14: rows for the secret game under the second System prompt, the combined evidence figure and the ablation figure; the raw
per-body retention files (pulled 11-14) join the retention row. Items that paper_data_v5 already holds are named in the notes.
usage (from projects/consciousness): write_v6_additions_index_20261113.py"""
import csv
import os

O = 'evidence/paper_data_v6_additions_20261113'
WA = 'evidence/p2_words_along_20261110'
MS = 'evidence/p2_modeswitch_20261107'
H2 = MS + '/results/vast_h200_20261111'
BS = chr(92)


def lp(p):
    a = os.path.abspath(p)
    return a if os.name != 'nt' else BS * 2 + '?' + BS + a


def j(*p):
    return ';'.join(p)


ROWS = [
    ('Retention along the message, all models, main probe and natural order (provisional: Results, evidence line 3)', 'provisional',
     j(f'{WA}/results/stage1_20261111/extracts/b150', f'{WA}/results/stage2_20261113/extracts/b150/main/glm53_flash',
       f'{WA}/results/stage2_20261113/analysis_b150_20261113.json', f'{WA}/results/stage2_20261113/analysis_b150_20261113.md',
       f'{H2}/m/out/words/b150_main/glm53_flash_all', f'{WA}/results/raw_20261113'),
     j(f'{WA}/code/server3/words_along_20261110.py', f'{WA}/code/server3/dsv41_words_along_20261110.py',
       f'{WA}/code/server3/words_kl_combined_20261111.py', 'evidence/p2_persistence_20261106'),
     j(f'{WA}/code/server3/extract_gram_20261110.py', f'{WA}/code/local/analyze_words_along_20261110.py'),
     j(f'{WA}/items/words_v2.json', f'{WA}/items/content_index_coverage.json', f'{WA}/PROTOCOL_zh.md'),
     'Raw per-body files: results/raw_20261113/{pro,server3} (pulled 11-14, every file checked against the server sha256 list) and '
     'the H200 mirror for GLM-5.3. The Gram extracts reproduce every number.'),
    ('Retention, 200-body depth, mechanism readouts', 'provisional',
     j(f'{WA}/results/stage1_20261111/extracts/p3', f'{WA}/results/stage1_20261111/analysis_stage1_20261111.json',
       f'{WA}/results/stage1_20261111/analysis_stage1_20261111.md'),
     f'{WA}/code/server3/words_along_20261110.py',
     j(f'{WA}/code/server3/extract_gram_20261110.py', f'{WA}/code/local/analyze_words_along_20261110.py'),
     f'{WA}/items/words_v2.json', 'Qwen3.6-27B depth sealed (not in the package).'),
    ('Next-token KL of ordinary words against roles, all models, main probe and natural order (provisional: Results, evidence line 2)',
     'provisional',
     j(f'{WA}/results/kl_20261113_pro/kl', f'{WA}/results/kl_20261112/server3_dsv41', f'{WA}/results/kl_20261112/server3_r1/out',
       f'{H2}/m/out/kl/glm53_flash_all'),
     j(f'{WA}/code/pro', f'{WA}/code/server3/kl_words_20261111.py', f'{WA}/code/server3/dsv41_kl_words_nat_20261112.py',
       f'{WA}/code/server3/words_kl_combined_20261111.py'),
     f'{WA}/code/local/build_final_tables_20261113.py', f'{WA}/items/words_v2.json',
     'Sources per model: data source table in the archive README.'),
    ('Final tables and group decomposition of both ordinary-word readouts', 'provisional', f'{WA}/results/final_20261113', '',
     j(f'{WA}/code/local/build_final_tables_20261113.py', f'{WA}/code/local/group_decomposition_20261113.py'), '', ''),
    ('Combined evidence figure (main probe, natural order, both orders pooled) with 95% family-bootstrap intervals', 'provisional',
     j(f'{WA}/results/final_20261113/figs', f'{WA}/results/final_20261113/pooled_ci_20261114.json'), '',
     j(f'{WA}/code/local/fig_evidence_combined_20261114.py', f'{WA}/code/local/pooled_ci_20261114.py'), '',
     'Draft figure; fig_evidence_draft*.py and fig_evidence_pooled_20261114.py are its earlier layouts.'),
    ('DeepSeek-R1-Distill-Llama-70B, paper KL rerun on server3 (replaces the PRO values in the KL tables)', 'provisional',
     f'{WA}/results/kl_20261112/server3_r1/s3kl',
     j(f'{WA}/code/server3/r1_paperkl_s3_20261112.sh', f'{WA}/code/server3/r1_kl_words_s3_20261112.sh'), '', '', ''),
    ('Causal tests, EXAONE-4.0.1-32B (attribution 8 and 64 dims, random seeds, controls, KL swap, game smoke)', 'provisional',
     f'{MS}/results/pro_exaone32_20261112', f'{MS}/code/pro', '', '', 'Game: baseline does not separate the sources, edit not tested.'),
    ('Causal tests, GLM-5.3-Flash (attribution 8 and 64 dims, controls, KL swap, game smoke, main game n = 50)', 'provisional',
     j(f'{H2}/m/out/mech', f'{H2}/m/out/extra', f'{H2}/pull_20261112T2227/out/mech/fits', f'{H2}/m/run.log', f'{H2}/final_20261113'),
     f'{MS}/code/vast_h200_20261111', '', '', 'Run on a rented machine; paper geometry and KL of GLM-5.3 stay the server3 values.'),
    ('Causal tests, summary over seven readouts', 'provisional', f'{MS}/results/mech_summary_7readouts_20261113.md', '', '', '',
     'Written by hand from the per-readout summaries.'),
    ('Ablation figure over the seven readouts (attribution 8 -> 64 dims, KL, controls)', 'provisional', f'{MS}/results/figs', '',
     f'{MS}/code/local/fig_ablation_draft_20261114.py', '', 'Draft figure.'),
    ('Secret game under the second System prompt (11-14): GLM-5.3-Flash smoke + main game n = 50; MiMo-V2.6 and EXAONE smokes',
     'provisional', j(f'{MS}/results/vast_h200_20261114', f'{MS}/results/pro_game_v6_20261114'),
     j(f'{MS}/code/vast_h200_20261111/game_v6_glm53_vast_20261114.sh', f'{MS}/code/vast_h200_20261111/sync_h200_20261114.sh',
       f'{MS}/code/pro/game_v6_pro_20261114.sh', f'{MS}/code/pro/crossmodel_games_hc_20261111.py'), '', '',
     'Items: evidence/p2_crossmodel_wave3_20261019/items/games_secret6 (in paper_data_v5; zh 4d27d8a5, en 26553b13). Summaries are '
     'written by the runners (summary.json). MiMo and EXAONE do not pass the release rule (not released). The runs under the first '
     'System prompt are in the next row and in the GLM-5.3 and EXAONE rows; the MiMo smoke is in paper_data_v5.'),
    ('Secret game under the first System prompt, the four mechanism readouts (Qwen3.8-27B both modes, gpt-oss-20b, gpt-oss-120b)',
     'restored', j(*[f'evidence/p2_crossmodel_wave3_20261019/runs/{m}/games_swap/{n}_s5_{q}'
                     for m, n in (('pro', 'gpt_oss_120b'), ('pro', 'gpt_oss_20b'), ('server3', 'qwen38_27b'),
                                  ('server3', 'qwen38_27b_sp')) for q in ('smoke_main', 'smoke_control', 'main', 'control')
                     if not (n == 'gpt_oss_120b' and q in ('main', 'control'))]),
     '', '', '',
     'NOT_USED of 10-30 lifted 11-14 (user: both prompts reported). gpt-oss-120b: the eight-game check only, its game does not pass '
     'the release rule under this prompt, so the full game is not reported and not in the package. Run code, runners and items '
     '(items/games_secret5) are in paper_data_v5, as are the runs of these readouts under the second prompt.'),
    ('Restored from v5 PRUNED.csv (user 11-13: full reproducibility)', 'restored',
     j(f'{MS}/README_zh.md', f'{MS}/PREREG_zh.md', f'{MS}/SHA256SUMS_pro_20261108.txt', f'{MS}/results/pro/SHA256SUMS_pro_20261108_raw.txt',
       'evidence/p2_crossmodel_wave3_20261019/README_zh.md'),
     j(f'{MS}/code', 'evidence/p2_persistence_20261106', 'evidence/p2_selfspace_20261106'), '', '',
     'KL queue scripts, the persistence helper script, archive logs.'),
]

bad = [p for r in ROWS for fld in r[2:6] for p in fld.split(';') if p and not os.path.exists(lp(O + '/' + p))]
assert not bad, bad
with open(O + '/INDEX_additions.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['paper_element', 'status', 'results', 'run_code', 'aggregation_code', 'item_files', 'notes'])
    w.writerows(ROWS)
print('index rows', len(ROWS), 'all referenced paths exist in the package')
