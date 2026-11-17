# Data and code for "Where Is the 'Self' of a Large Language Model? From Self-Index to Self-Reference"

This release holds what is needed to check and reproduce every number, figure and table of the paper: the items, the rendered
inputs, the answer and generated text of every item, the per-text values, the aggregated results, the fitted directions of the
causal edits, and the code that produced all of them.

## Links

- Paper: arXiv:2610.16717, https://arxiv.org/abs/2610.16717
- Data (this release, Zenodo): https://doi.org/10.5281/zenodo.24235700
- Code: https://github.com/lyclyq/Consciousness_of_LLM

## What is where

| Part | Contents |
|---|---|
| `evidence/p2_crossmodel_8models_20261018/` | the first eight models of the comparison across models: geometry, attribution question, runs and summaries |
| `evidence/p2_crossmodel_wave3_20261019/` | the comparison across 25 models, natural order, next-token divergence, the causal edits of the Qwen3.8 and gpt-oss readouts, the secret game, statistics (`stats/`) and figure scripts |
| `evidence/p2_modeswitch_20261107/` | the edits evaluated by the next-token divergence, the causal tests of MiMo-V2.6, EXAONE-4.0.1 and GLM-5.3, the game under the second System prompt for these models |
| `evidence/p2_words_along_20261110/` | ordinary words: the next-token divergence and the share kept along the message, the group decomposition and the final tables |
| `evidence/p2_persistence_20261106/`, `evidence/p2_selfspace_20261106/` | the token-level measurements that the share kept builds on |
| `evidence/p2_small_checks_20261014/` | the measurements of Qwen3.8-27B in the main text |
| `evidence/p2_wording_swap_20261109/` | the attribution question asked in other wordings, with and without the edits |
| `release/p2_release_20261102/github/analysis/` | two analysis scripts of the code release of 4 November 2026, which wrote the tables of the seven models added then (`evidence/p2_crossmodel_wave3_20261019/stats/v6_newdev_*.csv`) |
| `paper/scripts/` | the scripts of Fig. 1, the Supplementary Information and the arXiv build |
| `paper/source/` | the tables of the long version of the paper that the scripts read (Fig. 3, Extended Data Fig. 3, the table checks, the Supplementary Information) |
| `paper/nmi/supplementary_nmi.tex` | the Supplementary Information, as written by `paper/scripts/make_supplementary_nmi_20261114.py` |
| `FIGURES.md` | for every figure, table and quoted statistic: the script and the data it reads |
| `MANIFEST_sha256.csv` | every file with its size and SHA-256 |

Inside an archive, `code/` holds the code as it was run (`local/` for analysis and figures, `pro/` and `server3/` for the runs on
the two machines), `items/` the inputs, `runs/` and `results/` the outputs of every run (one JSON record per item, a `summary.json`
and an `identity.json` with the model revision, the arguments and the checksums of the code of that run), and `stats/` the
aggregated tables. Each script states in its docstring what it computes, what it reads and what it writes.

## Reproducing

Run the scripts from the root of this release. The analysis and figure scripts read only files of this release, and `FIGURES.md`
gives the command of each. The GitHub repository holds the code, the items, the statistics, the summary and identity of every run
and the aggregated results. The Zenodo record (https://doi.org/10.5281/zenodo.24235700) holds the whole release, with every per-item record, the Gram extracts and the
fitted directions. The few commands that read these are marked in `FIGURES.md`. The run scripts
need the model weights, which are public on Hugging Face at the revisions given in Supplementary Table 1 of the paper, and a GPU.
Python 3.12, numpy, scipy and matplotlib suffice for the analysis.

## What is not included

- The intermediate model states, that is, the residual-stream vectors of every text and layer (about 22 GB). The released code
  regenerates them from the public model weights, and the released summaries and Gram matrices hold everything that the paper
  computes from them.
- The fit files of four further random seeds. They repeat the fitted directions of `fit_q.pt` and add random subspaces, which the
  runners draw again from the recorded seed.
- Internal working notes.
- The text of the long version of the paper. `paper/source/` holds its tables that the scripts read.

Strings that identify the machines or the local account (host names, network addresses, the user name in paths) were replaced
by placeholders such as `<host>`, `<server>` and `<home>`, also inside the two `.tgz` archives. The files where this happened are
listed in `SCRUBBED.csv`, with their checksums before and after. No value or result is affected.

## Licences

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-data`). The model outputs remain subject to the licences of the respective models.
