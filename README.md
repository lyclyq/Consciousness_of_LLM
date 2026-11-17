# Consciousness_of_LLM

Code for the paper **"Where Is the 'Self' of a Large Language Model? From Self-Index to Self-Reference"**
by Yuchen Li, Xintong Li, Zongqi Fan, Ken-Tye Yong and Nguyen H. Tran.

The paper measures a functional self in open-weight language models at two levels: a *self-index*, which marks the model's
own message in its representation, and *self-reference*, the model's recognition of a message as its own. The same text is
placed in an Assistant, a User or a System message of the model's chat template; the geometry of the three renderings, the
model's answer to "who wrote this message?", the model's prediction of the next token on the message itself, causal tests that
exchange the coordinates of the self-index between messages, and a comparison across twenty-five models follow from this design.

> **Status (version 2.0.0).** [`reproduction/`](reproduction/) holds everything behind the current version of the paper
> (arXiv:2610.16717, version 7): the code as it was run, the items, the statistics, the summary and identity of every run and
> the aggregated results. [`reproduction/FIGURES.md`](reproduction/FIGURES.md) names the script and the data of every figure,
> table and quoted number, and [`reproduction/README.md`](reproduction/README.md) describes the layout. The per-item records,
> the Gram extracts and the fitted directions of the causal edits are in the Zenodo record
> https://doi.org/10.5281/zenodo.24235700. `selfindex/`, `analysis/` and `tools/` are the packaged code of releases 1.0.0 and
> 1.1.0. They belong to the long version of the paper (versions 5 and 6 on arXiv), and the appendix, table and figure names in
> their documentation refer to that version. What changed is listed in [`CHANGELOG.md`](CHANGELOG.md).

## Layout

| Folder | Contents |
|---|---|
| `reproduction/` | the code as run, the items, the statistics and the run summaries of the current version of the paper; start with [`reproduction/README.md`](reproduction/README.md) |
| `selfindex/` | the experiments: the main probe (`main_probe.py`), thinking mode (`thinking.py`, `thinking_problems.py`), natural conversation order (`natural.py`), next-token prediction on the message (`next_token.py`, under the replacement of the self-index coordinates `next_token_swap.py`, message orders `chat_order.py`), the causal tests (`role_swap.py`, with the wording variants of the attribution question; `fits.py`, `games.py`, `subspace_share.py`, `readout_fd.py`), the format ablation (`format_ablation.py`), the experiments of Section 4 (`twolevel/`), the developer APIs (`api.py`), template renderings (`render.py`, `render_items.py`) and model loading (`load.py`) |
| `selfindex/patches/` | loading fixes that leave the measurement code unchanged: a pure-PyTorch FP8 kernel, simulated FP8, Qwen3.8-Flash-Next, Moonlight-16B-A3B-Instruct |
| `selfindex/dsv41/` | DeepSeek-V4.1-Flash on its official model code (MIT, in `official/`) with PyTorch replacements of the custom kernels, and their checks; natural order (`natural.py`), reasoning problems (`problems.py`) and next-token prediction (`next_token.py`) |
| `analysis/` | every table, figure and quoted number of the paper, computed from the run outputs; see [`analysis/README.md`](analysis/README.md) |
| `tools/` | generation of the secret-game items |

Each script describes what it computes, its inputs and its outputs in its docstring; `--help` prints the options.

## Requirements

- Experiments: Python 3.10+, PyTorch 2.5 or later with CUDA, Hugging Face Transformers 5.8.0 (5.19.0 for the models noted in
  Appendix B of version 6 of the preprint), numpy. The experiments stop if no GPU is available.
- Analysis: Python 3.10+ with numpy, scipy and matplotlib (pdflatex and xelatex for one figure).
- API runs read their keys from environment variables only (`ZAI_API_KEY`, `DASHSCOPE_API_KEY`, `DEEPSEEK_API_KEY`).

## Running

Run from the repository root, for example

```
python -m selfindex.main_probe --model MODEL_DIR --tmpl-kwargs '{}' --out OUT --lang en
python -m selfindex.patches.moonlight_compat MODEL_DIR selfindex.main_probe --model MODEL_DIR --tmpl-kwargs '{}' --out OUT --lang zh
python -m selfindex.next_token --model MODEL_DIR --tmpl-kwargs '{"enable_thinking": false}' --measure split --order main --out OUT
python -m selfindex.role_swap --make-fits FITS --mode thinking --model MODEL_DIR --on-kwargs '{}' --lang zh --band 0-31
python -m selfindex.role_swap --mode thinking --model MODEL_DIR --on-kwargs '{}' --lang zh --fit-from FITS/fit_m.pt --band 0-31 \
    --region message --wording role --batch 16 --conds base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8 --out OUT
python analysis/make_all.py --results RESULTS --features FEATURES --out OUT
```

The settings of each readout of the causal tests (layers edited, batch, template arguments, mode) are listed in
[`CHANGELOG.md`](CHANGELOG.md) for the fifth readout added in 1.1.0 and in Appendix B of version 6 of the preprint for the others.

The items are read from `items/` at the repository root, or from the folder named by `SELFINDEX_ITEMS`.

## License

MIT; see [`LICENSE`](LICENSE). The official DeepSeek-V4.1 code in `selfindex/dsv41/official/` keeps its own MIT license.
