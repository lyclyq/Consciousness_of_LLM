"""Simulated FP8 inference on a BF16 model: the precision control for official FP8 checkpoints.

The official Qwen3.8-27B-FP8 checkpoint cannot be used as a control: transformers (5.8 and 5.19) merges each MLP's gate and up
projections and drops gate_proj.weight_scale_inv (reported UNEXPECTED), so the gate half runs with a wrong scale and the outputs
are garbage. Instead this module quantizes the same linear layers of the BF16 model the way the FP8 checkpoint stores them
(float8_e4m3fn weights with one scale per 128 x 128 block, scale = block max |w| / 448) and runs them W8A8 with dynamic per-token
activation quantization (blocks of 128 channels), using the fp8_fallback functions that match the official kernel.
Layer sets (regular expressions on module names), from the official checkpoints' scale_inv tensors:
  qwen3_5: linear_attn.in_proj_qkv / in_proj_z / out_proj, mlp.gate_proj / up_proj / down_proj, self_attn.q_proj / k_proj / v_proj / o_proj
"""
from __future__ import annotations
import re

import torch
import torch.nn as nn

from . import fp8_fallback as fb

PRESETS = {
    'qwen3_5': r'language_model\.layers\.\d+\.(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj))$',
}
BLOCK = 128


def quantize_weight(w: torch.Tensor):
    N, K = w.shape
    pn, pk = (-N) % BLOCK, (-K) % BLOCK
    wp = torch.nn.functional.pad(w.float(), (0, pk, 0, pn))
    blocks = wp.reshape((N + pn) // BLOCK, BLOCK, (K + pk) // BLOCK, BLOCK)
    s = blocks.abs().amax((1, 3)) / fb.FP8_MAX
    s = torch.where(s > 0, s, torch.ones_like(s))
    q = (blocks / s[:, None, :, None]).to(fb.FP8).reshape(N + pn, K + pk)[:N, :K].contiguous()
    return q, s


class SimFP8Linear(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        q, s = quantize_weight(lin.weight.data)
        self.register_buffer('qweight', q)
        self.register_buffer('scale', s)
        self.bias = lin.bias
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        qx, sx = fb.fp8_act_quant(x.contiguous(), BLOCK)
        y = fb.fp8_matmul(qx, self.qweight, sx, self.scale, [BLOCK, BLOCK], x.dtype)
        return y + self.bias if self.bias is not None else y


def apply(model: nn.Module, preset: str) -> int:
    """Replace the matching nn.Linear modules in place (each on its own device). Returns how many were replaced."""
    pat = re.compile(PRESETS[preset])
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and pat.search(n)]
    for n in names:
        parent_name, _, child = n.rpartition('.')
        parent = model.get_submodule(parent_name)
        lin = getattr(parent, child)
        with torch.no_grad():
            setattr(parent, child, SimFP8Linear(lin).to(lin.weight.device))
        del lin
    torch.cuda.empty_cache()
    return len(names)
