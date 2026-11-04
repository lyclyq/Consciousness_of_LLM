"""DeepSeek-V4.1-Flash-1017 without a Transformers implementation: the model code of its release with the custom kernels replaced by
PyTorch implementations of the same arithmetic, and the weights in the release format (FP8 E4M3 with E8M0 block scales, FP4 experts).

official/  the inference code and the prompt encoder of the release, unmodified, under the MIT License of the release
           (official/LICENSE, Copyright (c) 2023 DeepSeek). Source: Hugging Face repository deepseek-ai/DeepSeek-V4.1-Flash-1017,
           commit 1bae0e1088993b9b82de08f59de5be5a34e42170; the paper's runs used the ModelScope revision 9f9818127fdf of the same
           repository, whose files have the same sha256 (OFFICIAL_SHA256). official/inference/kernel.py (tilelang) is kept for
           reference; at run time it is replaced by kernel_torch.
kernel_torch   the PyTorch kernels (act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn).
runtime        checkpoint reading by memory map, model build, per-layer staging of the routed experts, the forward passes, and
               the rendering with DeepSeek's own encoder.
natural        the natural-order runner (geometry and self-reference): python -m selfindex.dsv41.natural
problems       the reasoning problems without thinking: python -m selfindex.dsv41.problems
check_kernels  checks of kernel_torch against float64 references written from the kernel source: python -m selfindex.dsv41.check_kernels
check_forward  checks of the runner's forward paths against the official forward on a small random model:
               python -m selfindex.dsv41.check_forward
"""
from pathlib import Path

OFFICIAL = Path(__file__).resolve().parent / 'official'
OFFICIAL_SHA256 = {
    'LICENSE': 'f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040',
    'encoding/encoding.py': '13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2',
    'encoding/README.md': 'a34a5ca98c65e5b5b2bedd693cc2e18e732fe50192723fb6937ae07219c23739',
    'inference/README.md': '2834402823199ee24e9a42bdf36a0fc6daf94448f444cb062c042a057a798f1c',
    'inference/config.json': '2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809',
    'inference/convert.py': '035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76',
    'inference/engram.py': '11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897',
    'inference/generate.py': '8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0',
    'inference/image_processor.py': '482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272',
    'inference/kernel.py': '1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455',
    'inference/model.py': 'e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d',
    'inference/requirements.txt': 'be82bcd5775ccf8361ddd22cac53ef7b3b83434697fc3c61c75870450fb9437e',
    'inference/run.sh': '91eddd7895f9fcd9bace53c21128c506e741e770f3c1a02464eff51f60401dfb',
    'inference/vision.py': '5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c',
}
