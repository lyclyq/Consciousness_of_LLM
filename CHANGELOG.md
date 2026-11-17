# Changelog

## 2.0.0

The complete reproduction material of the paper as restructured for version 7 of the preprint.

- `reproduction/`: the code as it was run (the analysis and figure scripts, and the run scripts of each machine), the items,
  the statistics, the summary and identity of every run, and the aggregated results. `reproduction/README.md` describes the
  layout, and `reproduction/FIGURES.md` maps every figure, table and quoted statistic to its script and its data.
- The whole release, with every per-item record, the Gram extracts of the share kept and the fitted directions of the causal
  edits, is in the Zenodo record https://doi.org/10.5281/zenodo.24235700 (data under CC BY 4.0).
- `selfindex/`, `analysis/` and `tools/` are unchanged from 1.1.0.

## 1.1.0

The code of the experiments, tables and figures added to the paper in this revision: the next-token prediction on the message
under test, MiMo-V2.6-Distill-Qwen-9B as the fifth readout of the causal tests, and the wording of the attribution question.

### Experiments (`selfindex/`)

| Addition | Paper |
|---|---|
| `next_token.py`: the next-token distributions on the body tokens of the message under test, rendered as the model's own, a User or a System message and followed by the reply request; KL(A‖U), KL(A‖S) and KL(U‖S) with the full distributions (`--measure full`, main and natural order) or with control and content tokens apart (`--measure split`, `--order main`, `natural` or `natural3`); family-bootstrap summaries | Section "Next-Token Prediction" (`sec:setup_nexttoken`), Section "The Self-Index in Next-Token Prediction" (`sec:res_nexttoken`), Appendix "Next-Token Prediction": Tables `tab:self_kl` (main probe) and `tab:self_kl_natural` (natural order), Figure `self_kl_scatter` (`fig:self_kl`) |
| `next_token_swap.py`: the same measurement with the eight self-index coordinates of the body replaced (removal, installation, random subspaces), with the fits of the causal tests (`fit_m.pt`) | Table `tab:self_kl_swap` |
| `dsv41/next_token.py`: the next-token measurement of DeepSeek-V4.1-Flash (natural order only) on its official model code | Table `tab:self_kl_natural` (templates that accept only the natural order) |
| `chat_order.py`: the two message orders of the measurement (`msgs_for`, `natural_system_ok`) | as `next_token.py` |
| `role_swap.py --wording main\|pron\|role`: the attribution question with the option that names the model worded by pronoun (你 / 我 / 系统, You / Me / The system) or by role name (AI 助手 / 用户 / 系统, The AI assistant / The user / The system), the stem and tail unchanged; the fits always use the main wording; `--fixed-date` for templates that write the current date (gpt-oss); `--show-render` | Appendix "How the Question Names the Model" (`app:mech_wording`, Section `sec:abl_question`): Table `tab:self_wording`, Figure `self_wording` |
| `items.attr_options`, `items.attr_question(..., wording=...)`; `render.template_kwargs` (template arguments with a fixed date) | as above |
| MiMo-V2.6-Distill-Qwen-9B as the fifth readout of the causal tests: the existing `role_swap.py` (`--make-fits`, `--region all\|message`, fit_q / fit_m, five random seeds, 64 dimensions, content and arithmetic questions) with the settings below | Tables `tab:self_mechanism`, `tab:self_mech_questions`, `tab:self_seeds`, `tab:self_cluster`, `tab:self_question`, Figure `self_mechanism` (five readouts) |

Settings of the fifth readout (the default call of its template; greedy generation, the answer read after the end of the reasoning,
as for the other readouts with thinking):

| Readout | Mode | Template arguments | Layers edited | Items per forward | New tokens at most |
|---|---|---|---|---|---|
| MiMo-V2.6-Distill-Qwen-9B, default call | `--mode thinking` | `{}` | 0-31 (every layer) | 16 | 3072 |

For the wording test the anchors are the runs with the fit on the message alone and the message edited only (`--fit-from fit_m.pt
--region message`), 8 dimensions at every layer; gpt-oss-20b and gpt-oss-120b render with `--fixed-date 2026-10-30`, the date on
which their anchor runs had rendered. In the secret game the unedited MiMo-V2.6-Distill-Qwen-9B does not separate the sources, so
the edit is not tested on that task.

### Analysis (`analysis/`)

| Script | Outputs | Paper |
|---|---|---|
| `causal_tests.py` (part 1b) | `v5_mechanism.json`, `v5_mechanism_tables.md`; Figure `self_mechanism` with the five readouts | Tables `tab:self_mechanism`, `tab:self_mech_questions`; Figure `self_mechanism` |
| `causal_controls.py` (`families_mimo`) | `r09_cluster_mimo_20261110.{json,md}` | Table `tab:self_cluster` (MiMo column) |
| `causal_tables.py` (`v5_mech_tex`, `v5_appendix_tex`) | `v5_mech_tex_{en,zh}.tex`; `v5_appendix_tex.json` (MiMo cells of `tab:self_seeds`, `tab:self_cluster`, `tab:self_question`; rows of `tab:self_kl`, `tab:self_kl_natural`, `tab:self_kl_swap`, `tab:self_wording`) | the tables named |
| `question_wording.py` | `wording_summary_all.{json,md}` (seven readouts, three wordings) | Table `tab:self_wording` |
| `fig_wording.py` | Figure `self_wording` | Figure `self_wording` (`fig:self_wording`) |
| `next_token.py` | `KL_INDEX_zh.md` (index of the next-token runs), `kl_ratio_distribution_20261110.md`, `kl_boundary_stats_20261110.md` | Section `sec:res_nexttoken` (the relation between the content ratios and the share of content layers that carry the index), Appendix "Next-Token Prediction" |
| `fig_kl_scatter.py` | Figure `self_kl_scatter` | Figure `fig:self_kl` |
| `fig_tasks.py` | Figure `self_tasks_flow`, new panels (e) the wording of the attribution question and (f) next-token prediction without a question | Figure `fig:self_tasks` |
| `fig_readout.py` | Figure `self_readout`, the third readout (next-token prediction on the message itself) | Figure `fig:self_readout` |

Table `tab:self_convergent` collects numbers from the outputs above and from the tables of version 1.0.0; it has no script.

### Items

`items/templates.json` gains the answer options of the wording test (`attribution_wordings`) and the layout of the next-token
measurement (`layouts.next_token`), which uses the existing reply request and natural-order request. The items are still held back
(see the README).
