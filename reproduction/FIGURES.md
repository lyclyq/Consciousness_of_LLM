# Where every figure and table of the paper comes from

Paths are relative to the root of this release. Run every command from that root with Python 3.12, numpy, scipy and matplotlib,
for example `uv run --no-project --with numpy --with scipy --with matplotlib python <command>`. Fig. 5 also needs xelatex.

The GitHub repository holds the code, the items, the statistics, the summary and identity of every run, and the aggregated
results. The Zenodo record holds the whole release: in addition, every per-item record, the Gram extracts of the share kept and
the fitted directions. Commands marked **(Zenodo)** read files that are only in the Zenodo record. Unzip the three zip files of
the record into the root of this release before running them.

Each command writes its output to the file that the paper used, so a run can be compared with the released file directly.

## Main figures

| Element | Data |
|---|---|
| Fig. 1, the probe and the two-level account | none (a schematic) |
| Fig. 2, three measures of the self-index | `evidence/p2_words_along_20261110/results/final_20261113/final_tables_20261113.json`, `pooled_ci_20261114.json`, `results/stage2_20261113/analysis_b150_20261113.json` |
| Fig. 3, recognition requires the self-index | Table tab:self_crossmodel (Supplementary Table 3) in `paper/source/self_en_body.tex`, checked against the run records by the two table checks below |
| Fig. 4, editing the index coordinates | `evidence/p2_crossmodel_wave3_20261019/stats/r05r06_20261031.json`, `v4_mechanism.json`, and the run summaries under `evidence/p2_modeswitch_20261107/results/` |
| Fig. 5, the secret game and the four tasks | `evidence/p2_modeswitch_20261107/results/game_two_prompts_20261114.json` and the two records of the game shown in panel b |

```
paper/scripts/make_overview_nmi_20261114.py paper/nmi/figures
evidence/p2_words_along_20261110/code/local/fig_evidence_combined_20261114.py evidence/p2_words_along_20261110/results/final_20261113/figs/fig_evidence_combined_draft
evidence/p2_crossmodel_wave3_20261019/code/local/figures/fig_nesting_nmi_20261114.py
evidence/p2_modeswitch_20261107/code/local/fig_ablation_draft_20261114.py evidence/p2_modeswitch_20261107/results/figs/fig_ablation_draft
evidence/p2_modeswitch_20261107/code/local/game_table_two_prompts_20261114.py
evidence/p2_modeswitch_20261107/code/local/fig_game_nmi_20261114.py
```

The table behind Fig. 3 is checked against the records by two scripts. The first covers the 18 models of the first waves and is
run from `evidence/p2_crossmodel_wave3_20261019`. The second covers the seven models added later **(Zenodo)**.
Its record, `evidence/p2_crossmodel_wave3_20261019/stats/check_newdev_tables_20261104.txt`, also holds the own-step row of the long
version, a table that is not part of this paper. A run from this release reports that row as not checked.

```
cd evidence/p2_crossmodel_wave3_20261019 && python code/local/check_crossmodel_hand_numbers_20261103.py
evidence/p2_crossmodel_wave3_20261019/code/local/check_newdev_tables_20261104.py evidence/p2_crossmodel_wave3_20261019 paper/source .
```

## Extended Data

| Element | Command | Data |
|---|---|---|
| Fig. 1, a role as such against the model's own role | `evidence/p2_words_along_20261110/code/local/fig_ed_decomposition_nmi_20261114.py` | `results/final_20261113/group_decomposition_20261113.json` |
| Figs. 2 and 3, a self-index at the response onset only, model size and benchmark scores | `evidence/p2_crossmodel_wave3_20261019/code/local/figures/fig_ed_onset_size_nmi_20261114.py` | the geometry run summaries named in the script, `stats/v6_benchmarks_20261104.csv`, Table tab:self_models in `paper/source/self_en_body.tex` |
| Fig. 4, where the coordinates are read | `evidence/p2_crossmodel_wave3_20261019/code/local/figures/fig_ed_windows_nmi_20261114.py` | `evidence/p2_crossmodel_wave3_20261019/stats/v4_mechanism.json` |
| Fig. 5, renderings and the distance ratio | `evidence/p2_crossmodel_wave3_20261019/code/local/figures/fig_ed_design_nmi_20261114.py evidence/p2_crossmodel_wave3_20261019/figures_nmi_20261114` | the token sequences recorded in the script |
| Fig. 6, how one offset is read or not read | `evidence/p2_crossmodel_wave3_20261019/code/local/figures/fig_ed_readout_nmi_20261115.py evidence/p2_crossmodel_wave3_20261019/figures_nmi_20261114` | none, a schematic |
| Tables 1 and 2 | `evidence/p2_modeswitch_20261107/code/local/make_ed_tables_nmi_20261114.py` | `results/figs/fig_ablation_draft.json`, `results/game_two_prompts_20261114.json` |
| Table 3, seven alternative readings | written by hand in the Extended Data of the paper | the numbers of the tables and figures above |

## Statistics quoted in the text

```
# share kept along the message, from the Gram extracts (Zenodo)
evidence/p2_words_along_20261110/code/local/analyze_words_along_20261110.py evidence/p2_words_along_20261110/results/stage2_20261113/extracts evidence/p2_words_along_20261110/results/stage2_20261113/analysis_b150_20261113 --words evidence/p2_words_along_20261110/items/words_v2.json --coverage evidence/p2_words_along_20261110/items/content_index_coverage.json
# the final tables of both measures with ordinary words (the Flash-Next path must be given as an absolute path)
evidence/p2_words_along_20261110/code/local/build_final_tables_20261113.py evidence/p2_words_along_20261110/results/final_20261113/final_tables_20261113 --flashnext <absolute path of evidence/p2_words_along_20261110/results/kl_20261113_pro/kl/flashnext_nt>
# group decomposition and rank correlations with the level (main probe and natural order)
evidence/p2_words_along_20261110/code/local/group_decomposition_20261113.py evidence/p2_words_along_20261110/results/final_20261113/final_tables_20261113.json evidence/p2_words_along_20261110/results/final_20261113/group_decomposition_20261113
# both orders together: intervals (Zenodo) and the rank test
evidence/p2_words_along_20261110/code/local/pooled_ci_20261114.py evidence/p2_words_along_20261110/results/final_20261113/pooled_ci_20261114.json
evidence/p2_words_along_20261110/code/local/pooled_rank_test_20261114.py
# divergence against the last-layer geometry (rho = 0.94, slope 1.57)
evidence/p2_words_along_20261110/code/local/kl_vs_lastlayer_geometry_20261114.py
# the random control over all seeds (at most 0.15, against at least 0.17 for the fitted removal)
evidence/p2_modeswitch_20261107/code/local/random_bound_all_seeds_20261115.py
# the groups of both orders recomputed by the rule of the Methods at 70%, 80% and 90% (asserts the groups used above at 80%)
evidence/p2_words_along_20261110/code/local/classify_groups_by_rule_20261115.py
# Supplementary Table 5: classification and statistics at three thresholds (run after the group decomposition and the rank test)
evidence/p2_words_along_20261110/code/local/threshold_grid_20261115.py
```

## The paper

`paper/scripts/make_supplementary_nmi_20261114.py` writes the Supplementary Information (`paper/nmi/supplementary_nmi.tex`) from
the tables of the long version of the paper and, for Supplementary Table 5, from
`evidence/p2_words_along_20261110/results/final_20261113/threshold_grid_20261115.json`. `paper/source/` holds only these tables and the template renderings, written by
`paper/scripts/make_release_tables_20261115.py`. `paper/scripts/make_arxiv_nmi_20261114.py` builds the arXiv version from the
LaTeX source of the paper, which is available from arXiv.
