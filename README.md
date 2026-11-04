# Consciousness_of_LLM

Code for the paper **"Where Is the 'Self' of a Large Language Model? From Self-Index to Self-Reference"**
by Yuchen Li, Xintong Li, Zongqi Fan, Ken-Tye Yong and Nguyen H. Tran.

The paper measures a functional self in open-weight language models at two levels: a *self-index*, which marks the model's
own message in its representation, and *self-reference*, the model's recognition of a message as its own. The same text is
placed in an Assistant, a User or a System message of the model's chat template; the geometry of the three renderings, the
model's answer to "who wrote this message?", causal tests that exchange the coordinates of the self-index between messages,
and a comparison across twenty-five models follow from this design.

> **Status (version 1.0.0).** This release contains the experiment code and the analysis scripts. The items (message bodies,
> questions in every option order, games and reasoning problems) and the configuration of every run will be added in a later
> release; until then the experiment code cannot be run end to end. The per-item results, fitted subspaces and per-layer
> features are available from the corresponding authors on request (see the paper).

## Layout

| Folder | Contents |
|---|---|
| `selfindex/` | the experiments: the main probe (`main_probe.py`), thinking mode (`thinking.py`, `thinking_problems.py`), natural conversation order (`natural.py`), the causal tests (`role_swap.py`, `fits.py`, `games.py`, `subspace_share.py`, `readout_fd.py`), the format ablation (`format_ablation.py`), the experiments of Section 4 (`twolevel/`), the developer APIs (`api.py`), template renderings (`render.py`, `render_items.py`) and model loading (`load.py`) |
| `selfindex/patches/` | loading fixes that leave the measurement code unchanged: a pure-PyTorch FP8 kernel, simulated FP8, Qwen3.8-Flash-Next, Moonlight-16B-A3B-Instruct |
| `selfindex/dsv41/` | DeepSeek-V4.1-Flash on its official model code (MIT, in `official/`) with PyTorch replacements of the custom kernels, and their checks |
| `analysis/` | every table, figure and quoted number of the paper, computed from the run outputs; see [`analysis/README.md`](analysis/README.md) |
| `tools/` | generation of the secret-game items |

Each script describes what it computes, its inputs and its outputs in its docstring; `--help` prints the options.

## Requirements

- Experiments: Python 3.10+, PyTorch 2.5 or later with CUDA, Hugging Face Transformers 5.8.0 (5.19.0 for the models noted in
  Appendix B of the paper), numpy. The experiments stop if no GPU is available.
- Analysis: Python 3.10+ with numpy, scipy and matplotlib (pdflatex and xelatex for one figure).
- API runs read their keys from environment variables only (`ZAI_API_KEY`, `DASHSCOPE_API_KEY`, `DEEPSEEK_API_KEY`).

## Running

Run from the repository root, for example

```
python -m selfindex.main_probe --model MODEL_DIR --tmpl-kwargs '{}' --out OUT --lang en
python -m selfindex.patches.moonlight_compat MODEL_DIR selfindex.main_probe --model MODEL_DIR --tmpl-kwargs '{}' --out OUT --lang zh
python analysis/make_all.py --results RESULTS --features FEATURES --out OUT
```

The items are read from `items/` at the repository root, or from the folder named by `SELFINDEX_ITEMS`.

## License

MIT; see [`LICENSE`](LICENSE). The official DeepSeek-V4.1 code in `selfindex/dsv41/official/` keeps its own MIT license.
