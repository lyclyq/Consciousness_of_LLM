# Analysis

These scripts compute every table, figure and quoted number of the paper from the run outputs. They read the results package
(per-item results, run summaries, identity files) and, for two analyses, the feature package (per-layer arrays). They need
Python 3.10+ with numpy, scipy and matplotlib; Figure 4 also needs pdflatex and xelatex (ctex).

```
python analysis/make_all.py --results RESULTS --features FEATURES --out OUT [--item-cache CACHE] [--paper-source PAPER]
```

- `--results` and `--features` are the roots of the two packages (the folders that hold `evidence/`). Both are only read.
- Every output keeps the relative path of its archived counterpart in the packages, for example
  `OUT/evidence/p2_crossmodel_wave3_20261019/stats/v4_mechanism.json` or `OUT/paper/source/figures/self_mechanism.pdf`, so a run
  can be compared with the packages file by file. File names follow the archive, including its dates and analysis labels.
- A script that needs the output of an earlier one reads it from `OUT`; `make_all.py` runs them in the order listed below and
  writes the printout of each to `OUT/logs/`.
- `--item-cache` keeps a cache of the per-item records of each run (reading about 126,000 small files is slow on Windows).
- `--paper-source` (the paper's LaTeX folder) lets `hand_checks.py` read Table 5 and report whether each checked phrase is in the
  current text.
- On Windows, the packages are read through extended-length paths, since some run folders are deeper than 260 characters; keep
  `OUT` short.

Each script also runs alone with `--results RES --out OUT` (and `--features FEAT` where it needs features); `--help` prints
what it computes.

## Scripts, in the order of `make_all.py`

| Script | What it computes | Outputs (under `OUT`) |
|---|---|---|
| `common.py` | shared helpers: package paths, run lookup, per-item records, Wilson and bootstrap intervals, the item-paired change of the causal tests | – |
| `crossmodel_classification.py` | per-model tables of the cross-model comparison: body-level self-reference, thinking answer counts, layers at which the own message is singled out | `M8/stats/{attribution_non_thinking_body_level,thinking_attribution_counts,geometry_summary}.csv`; `W3/stats/` the same, `v4_*.csv`, `v5_olmo_*.csv` |
| `thresholds_orders.py` | the self-index criterion at 70/80/90% and over the last third and quarter; single-pass self-reference over two and three option orders | `W3/stats/r04_threshold_20261101.{json,md}`, `r14_orders_20261101.{json,md}`, `r14_tex_{en,zh}.tex` |
| `model_templates.py` | how each template marks the model's own message, next to its geometry and self-reference | `W3/stats/template_markers.{csv,md}` |
| `ownstep.py` | trust in the model's own steps (reasoning task), every model and mode; with `--reasoning-dir`, the Qwen3.8-27B reasoning summary | `W3/stats/v4_ownstep.csv` (`TL/summaries/M_reasoning_summary.json`) |
| `causal_tests.py` | the replacement in three models and four readouts: every layer, groups of positions, layer windows, 64 dimensions, content and arithmetic controls, the secret game, Gemma-4-31B in natural order | `W3/stats/v4_mechanism.{json}`, `v4_mechanism_tables.md`, `v4_mech_ext.json`, `v4_mech_ext_tables.md`; figures `self_mechanism`, `self_mechanism_where` |
| `causal_controls.py` | controls of the causal tests: fit and edit without the question, five random seeds, scaled random edits, exchange of the User and System coordinates (and the gpt-oss traces), the inverted replacement and the first-order sensitivity, wording-family intervals, the stricter tasks | `W3/stats/r05r06_20261031`, `r06f_20261031`, `r08_20261031`, `r08_traces_20261101`, `r03_20261101`, `r03b_layers_20261101`, `r09_cluster_20261101`, `p56full_20261101` (`.json` and `.md` each) |
| `hand_checks.py` | recomputes the numbers that the text states without a generated table (gap fractions, 64 dimensions, model facts, cross-model counts, reasoning-trace counts) and prints paper / recomputed / OK | `W3/stats/check_mechanism_hand_numbers_20261026.txt`, `check_crossmodel_hand_numbers_20261026.txt`, `trace_counts.txt` |
| `two_level_thinking.py` | Qwen3.8-27B self-reference with and without thinking (900 items) | `TL/summaries/N_thinking_final_900items.json` |
| `fig_crossmodel.py` | Figure 8 and its points | figure `self_crossmodel`; `W3/stats/self_crossmodel_points_v6.json` |
| `natural_order.py` | Figure 14 and its points | figure `self_natural`; `W3/stats/self_natural_points_v1.json` |
| `benchmarks.py` | rank correlations of the published GPQA scores with the two levels, main probe and pooled with natural order; the tables of Appendix D | `W3/stats/gpqa_vs_levels_20261029.json`, `gpqa_vs_levels_pooled_20261029.json`, `appendix_c_20261029/benchmarks_{en,zh}.tex` |
| `causal_tables.py` | LaTeX table bodies of the causal tests (and of the threshold table) | `W3/stats/v4_mech_tex_{en,zh}.tex`, `r05r06_tex_*`, `r03b_tex_*`, `r09_r04_tex_*`, `tran2_tex_*`, `v5_appendix_tex_en.tex` |
| `english_ablation.py` | the English ablation of Qwen3.8-27B (Appendix B): the causal tests with the English items next to the Chinese ones, the transfer of the fitted subspaces between the two languages, their overlap (recomputed when torch is installed), and the reasoning-trace counts | `W3/stats/q38_english_appendix_20261102.json`, `q38_english_tex_20261102_{en,zh}.tex`, `trace_counts_q38en_20261102.txt` |
| `fig_depth.py` | Figure 9 | figure `self_crossmodel_depth` |
| `fig_inducer.py` | Figure 12 | figure `self_inducer` |
| `fig_fd_layers.py` | Figure 15 | figure `self_fd_layers` |
| `two_level_geometry.py` (features) | Qwen3.8-27B native renderings: role geometry by position and layer, self-reference baselines, response-onset geometry | `C150/summaries/native_geometry150_summary.json`, `j150_numbers.txt`; `TL/summaries/J_native_geometry_default.json`, `J_onset_geometry.json` |
| `fig_geometry.py` | Figure 5 | figure `self_index_geometry` |
| `fig_selfref.py` | Figure 6 | figure `self_reference` |
| `fig_thinking.py` | Figure 7 | figure `self_thinking` |
| `fig_embedding.py` | Figure 13 | figure `self_embedding` |
| `eq1_tests.py` (features) | the test of Equation 1 and the direction stability for gpt-oss-20b; with `--defgeo`, the Qwen3.8-27B tests | `W3/runs/analysis/direction_stability_g_mxfp4.json`, `gptoss20b_mxfp4_eq1_{content,onset}.json` (`W3/runs/analysis/eq1/eq1_{restricted,perbody,residual_ra}.json`) |
| `message_geometry.py` (needs `--defgeo`) | Qwen3.8-27B message-content geometry and the numbers quoted from it | `TL/summaries/L_message_content_geometry.json`, `L_paper_numbers.json` |
| `appendix_tex.py` | the rendered items of Appendix A and the configuration table of Appendix B | `W3/stats/appendix_a_20261029/appendix_a_{en,zh}.tex`, `appendix_a_renderings.txt`; `appendix_b_20261029/appendix_b_{en,zh}.tex` |
| `fig_overview.py`, `fig_readout.py`, `fig_design.py` | Figures 1, 2, 3 (schematics, no data) | figures `self_overview`, `self_readout`, `self_design` |
| `fig_tasks.py` | Figure 4 (LaTeX/TikZ, from the items and two game replies) | `W3/code/local/figures/self_tasks_flow_{en,zh}.tex`; figures `self_tasks_flow_{en,zh}.pdf` |

`M8`, `W3`, `SC`, `TL` and `C150` stand for `evidence/p2_crossmodel_8models_20261018`, `evidence/p2_crossmodel_wave3_20261019`,
`evidence/p2_small_checks_20261014`, `SC/results/two_level_20261015` and `SC/results/compliant_150_20261015`. Figures are written
to `OUT/paper/source/figures/` as PDF (and PNG).

## Paper elements

Numbers follow the current version of the paper. A table body marked "block" is generated text that the paper source holds
between the comment lines `%<<label` and `%>>label`.

### Figures

| Figure | Label | Script | Output |
|---|---|---|---|
| 1 | fig:self_overview | `fig_overview.py` | `self_overview.pdf` |
| 2 | fig:self_readout | `fig_readout.py` | `self_readout.pdf` |
| 3 | fig:self_design | `fig_design.py` | `self_design.pdf` |
| 4 | fig:self_tasks | `fig_tasks.py` | `self_tasks_flow_en.pdf` (`_zh` for the Chinese paper) |
| 5 | fig:self_geometry | `two_level_geometry.py` (response onset, features) and the archived message-content summary; `fig_geometry.py` | `self_index_geometry.pdf` |
| 6 | fig:self_reference | `two_level_geometry.py`, `fig_selfref.py` | `self_reference.pdf` |
| 7 | fig:self_thinking | `two_level_thinking.py`, `fig_thinking.py` | `self_thinking.pdf` |
| 8 | fig:self_crossmodel | `crossmodel_classification.py`, `two_level_thinking.py`, `fig_crossmodel.py` | `self_crossmodel.pdf`, `self_crossmodel_points_v6.json` |
| 9 | fig:self_depth | `fig_depth.py` | `self_crossmodel_depth.pdf` |
| 10 | fig:self_mechanism | `causal_tests.py` | `self_mechanism.pdf` |
| 11 | fig:self_mechanism_where | `causal_tests.py` | `self_mechanism_where.pdf` |
| 12 | fig:self_inducer | `causal_tests.py`, `fig_inducer.py` | `self_inducer.pdf` |
| 13 | fig:self_embedding | `fig_embedding.py` (archived message-content summaries and role-word embeddings) | `self_embedding.pdf` |
| 14 | fig:self_natural | `natural_order.py` | `self_natural.pdf`, `self_natural_points_v1.json` |
| 15 | fig:self_fd | `fig_fd_layers.py` | `self_fd_layers.pdf` |

### Tables

| Table | Label | Script | Output |
|---|---|---|---|
| 1 | tab:self_related | literature, no data | – |
| 2 | tab:self_design | rows from `message_geometry.py` (archived `L_paper_numbers.json`), `two_level_geometry.py`, `eq1_tests.py` (archived `eq1_restricted.json`), `two_level_thinking.py`, `causal_tests.py`, and the sources of Table 5 | the JSON files named there |
| 3 | tab:self_models | `model_templates.py` (markers, geometry, self-reference); model facts from the identity files | `template_markers.csv` |
| 4 | tab:self_early | `message_geometry.py` (needs `--defgeo`; archived `L_paper_numbers.json`, entry `table`) | `L_paper_numbers.json` |
| 5 | tab:self_crossmodel | `crossmodel_classification.py`; cells checked by `hand_checks.py` | the stats CSVs, `check_crossmodel_hand_numbers_20261026.txt` |
| 6 | tab:self_alternatives | `causal_tests.py`, `causal_controls.py`, `thresholds_orders.py` | `v4_mechanism.json`, `v4_mech_ext.json`, `r05r06_*`, `r06f_*`, `r09_cluster_*`, `p56full_*`, `r04_threshold_*` |
| 7 | tab:self_mech_questions | `causal_tables.py` | block in `v4_mech_tex_{en,zh}.tex` |
| 8 | tab:self_mechanism | `causal_tables.py` | block in `v4_mech_tex_{en,zh}.tex` |
| 9 | tab:self_exchange | `causal_controls.py` (exchange), `causal_tables.py` | block in `tran2_tex_{en,zh}.tex` |
| 10 | tab:self_game | `causal_tables.py` | block in `v4_mech_tex_{en,zh}.tex` |
| 11 | tab:self_items | the item texts (`items/` of this repository), no analysis | – |
| 12 | tab:self_config | `appendix_tex.py` | `appendix_b_{en,zh}.tex` |
| 13 | tab:self_threshold | `thresholds_orders.py`, `causal_tables.py` | block in `r09_r04_tex_{en,zh}.tex` |
| 14 | tab:self_orders | `thresholds_orders.py` | block in `r14_tex_{en,zh}.tex` |
| 15 | tab:self_benchmarks | `benchmarks.py` | `benchmarks_{en,zh}.tex` |
| 16 | tab:self_benchmark_corr | `benchmarks.py` | `benchmarks_{en,zh}.tex` |
| 17 | tab:self_natural | read from the natural-order run summaries (`suite_nat3/*/summary.json`: `layers_A_lowest_80`, `attr`); no script | – |
| 18 | tab:self_natural2 | `crossmodel_classification.py` and the per-item records of the two-role natural-order runs | stats CSVs |
| 19 | tab:self_natural3 | read from the run summaries of DeepSeek-V4.1-Flash and Ministral-3-14B; no script | – |
| 20 | tab:self_mechanism_where | `causal_tables.py` | block in `v4_mech_tex_{en,zh}.tex` |
| 21 | tab:self_mechanism_windows | `causal_tables.py` | block in `v4_mech_tex_{en,zh}.tex` |
| 22 | tab:self_fd | `causal_controls.py` (fd_layers), `causal_tables.py` | block in `r03b_tex_{en,zh}.tex` |
| 23 | tab:self_seeds | `causal_controls.py` (without_question), `causal_tables.py` | block in `r05r06_tex_{en,zh}.tex` |
| 24 | tab:self_seeds_scaled | `causal_controls.py` (random_floor), `causal_tables.py` | block in `tran2_tex_{en,zh}.tex` |
| 25 | tab:self_exchange_full | `causal_controls.py` (exchange), `causal_tables.py` | block in `tran2_tex_{en,zh}.tex` |
| 26 | tab:self_rest | `causal_controls.py` (rest), `causal_tables.py` | block in `tran2_tex_{en,zh}.tex` |
| 27 | tab:self_cluster | `causal_controls.py` (families), `causal_tables.py` | block in `r09_r04_tex_{en,zh}.tex` |
| 28 | tab:self_question | `causal_controls.py` (without_question), `causal_tables.py` | block in `r05r06_tex_{en,zh}.tex` |
| 29 | tab:self_strict | `causal_controls.py` (strict), `causal_tables.py` | block in `tran2_tex_{en,zh}.tex` |
| 30 | tab:self_ablation_2x2 | `causal_tables.py` (ablation) | block in `v5_appendix_tex_en.tex` |
| 31 | tab:self_positions | `causal_tables.py` (ablation) | block in `v5_appendix_tex_en.tex` |
| 32 | tab:self_ownstep | `ownstep.py` | `v4_ownstep.csv` |

### Numbers in the text

| Element | Script | Source of the numbers |
|---|---|---|
| Abstract, Sec. 1, Sec. 2 (contributions), Conclusion | `message_geometry.py` (archived), `two_level_geometry.py`, `causal_tests.py`, `causal_controls.py` (exchange), `crossmodel_classification.py`, `benchmarks.py` | `L_paper_numbers.json`, `J_native_geometry_default.json` (`attr_baseline`), `v4_mechanism.json`, `r08_20261031.json`, Table 5 sources, `gpqa_vs_levels_20261029.json` |
| Definition 1, Sec. 5.4 (cross-model results) | `crossmodel_classification.py`, `fig_crossmodel.py`, `fig_depth.py`, `thresholds_orders.py`, `hand_checks.py` | stats CSVs, `self_crossmodel_points_v6.json`, `r04_threshold_20261101.json`, `check_crossmodel_hand_numbers_20261026.txt` |
| Sec. 4.1 (setup: model facts, renderings, letter mass, lens) | `two_level_geometry.py`, `hand_checks.py` | `native_geometry150_summary.json` (`lens_l63_maxabs`, `attr_letter_mass`), `check_mechanism_hand_numbers_20261026.txt` |
| Sec. 4.1.7 (cross-model setup), 4.1.8 (computation) | `crossmodel_classification.py` | stats CSVs |
| Sec. 4.2.2 (causal tests) | `causal_tests.py` | `v4_mechanism.json` |
| Sec. 5.1 (self-index on content and at the response onset) | `message_geometry.py` (archived), `two_level_geometry.py` | `L_paper_numbers.json`, `J_onset_geometry.json` |
| Sec. 5.2 (mirror test) | `two_level_geometry.py` | `native_geometry150_summary.json`, `j150_numbers.txt` |
| Sec. 5.3 (thinking) | `two_level_thinking.py` | `N_thinking_final_900items.json`; the item counts of the paragraph (98%/96%, 12 and 5 changed items, McNemar p, 8 weakly claimed items) have no script |
| Sec. 5.5 (natural order, templates without the main probe) | `natural_order.py` | `self_natural_points_v1.json`, natural-order run summaries |
| Sec. 6 intro and 6.1 (causal tests) | `causal_tests.py`, `causal_controls.py`, `hand_checks.py` | `v4_mechanism.json`, `v4_mech_ext.json`, the control JSON files, `check_mechanism_hand_numbers_20261026.txt`, `trace_counts.txt` |
| Sec. 6.1, the edited subspace holds the shared displacement | read from the run outputs `subspace_share/*.json` (`by_region.all`); no script | – |
| Sec. 6.2 (without the question) | `causal_controls.py`, `causal_tables.py` | `r05r06_20261031.json`, `p56full_20261101.json` |
| Sec. 6.3 (format ablation) | `causal_tables.py`, `model_templates.py`, `fig_inducer.py` | `v5_appendix_tex_en.tex`, `template_markers.csv` |
| Sec. 6.4 (token positions) | `causal_tables.py` | `v5_appendix_tex_en.tex` |
| Sec. 6.5 (token embeddings) | `message_geometry.py` (archived) | `L_paper_numbers.json`, `role_word_embeddings.json`, `runs/analysis/gptoss20b_embedding_roles.json` |
| Sec. 6.6 (test of Equation 1) | `eq1_tests.py` | gpt-oss: `direction_stability_g_mxfp4.json`, `gptoss20b_mxfp4_eq1_*.json`; Qwen: the archived `eq1_restricted.json`, `eq1_perbody.json`, `eq1_residual_ra.json` of `W3/runs/analysis/` (recomputed only with `--defgeo`); the message-level shares of the Qwen test are printed by `residual_ra` only |
| Sec. 7.1–7.3 | `crossmodel_classification.py`, `benchmarks.py`, `causal_tests.py`, `message_geometry.py` (archived) | as above |
| Appendix A (rendered items) | `appendix_tex.py` | `appendix_a_{en,zh}.tex` |
| Appendix B (configurations, statistics) | `appendix_tex.py` | `appendix_b_{en,zh}.tex`; versions from the identity files |
| Appendix C (threshold, option orders) | `thresholds_orders.py`, `causal_tables.py` | `r04_threshold_20261101.json`, `r14_orders_20261101.json` |
| Appendix D (published benchmark scores) | `benchmarks.py` | `gpqa_vs_levels*.json`, `v5_benchmarks_20261029.csv` (compiled from the model cards) |
| Appendix E (natural order) | `natural_order.py`, `crossmodel_classification.py` | natural-order run summaries and per-item records |
| Appendix F (further causal tests) | `causal_tests.py`, `causal_controls.py`, `causal_tables.py`, `fig_fd_layers.py` | the files of Tables 20–31 |
| Appendix G (trust in own records) | `ownstep.py` | `v4_ownstep.csv`; the Qwen3.8-27B reasoning summary `M_reasoning_summary.json` needs the per-problem records (`--reasoning-dir`) |

## Inputs not in the packages

- The per-layer features of the Qwen3.8-27B message-content run (check L): `message_geometry.py` and the Qwen part of
  `eq1_tests.py` run only with `--defgeo`. Their archived outputs (`L_message_content_geometry.json`, `L_paper_numbers.json`,
  `eq1_*.json`) are in the results package, and the figure scripts read those.
- The per-problem records of the Qwen3.8-27B reasoning run (check M): `ownstep.py --reasoning-dir`; the archived
  `M_reasoning_summary.json` is in the results package.
- Two parts of `hand_checks.py` concern earlier versions of an appendix (own-step group ranges over models without
  self-reference, the self-index subspace of a reasoning step); their inputs are not in the package and the parts are skipped.
