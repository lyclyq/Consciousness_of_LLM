"""What the Section 4 experiments share.

- Output writers in the format of the paper's runs: indented UTF-8 JSON, and .npz archives (deflate level 1, zip64) with a
  receipt <name>.receipt.json (path, bytes, sha256); both written through a .partial file, so that a file on disk is complete
  and an interrupted run resumes per item.
- The 150 bodies in the fixed run order (category, split, index) and the attribution question in its two option orders.
- Token positions: 20 endpoint-inclusive samples of a body, the body positions of a rendered text.
- The decoder-block feature collector of the message-content geometry and the full-vocabulary readout of the residual stream
  (final norm + LM head on a layer's state: at the last block this is the model's own logits, which is checked; at the middle
  blocks it is an uncalibrated diagnostic).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np

from .. import items as I
from ..load import checked_tokenizer, load_model

ROLES = I.ROLES                                      # ('assistant', 'user', 'system')
SAMPLES = 20                                         # body samples per message
LETTERS = ('A', 'B', 'C')
ORDERS = {'q1': I.orders()[0], 'q2': I.orders()[1]}  # option orders of the attribution question


def question(order):
    """The attribution question in option order 'q1' or 'q2' (Who wrote the message right before this question? A. ... B. ...
    C. ... Answer with the letter only.)."""
    return I.attr_question(ORDERS[order], 'zh')


def all_bodies():
    """The 150 bodies in the fixed run order."""
    rows = sorted(I.bodies('zh'), key=lambda r: (r['category'], r['split'], r['index']))
    assert len(rows) == 150 and len({r['id'] for r in rows}) == 150, len(rows)
    return rows


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def write_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    with tmp.open('wb') as stream:
        with zipfile.ZipFile(stream, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                             allowZip64=True) as archive:
            for name, value in arrays.items():
                with archive.open(name + '.npy', mode='w', force_zip64=True) as entry:
                    np.lib.format.write_array(entry, np.asanyarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    receipt = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha_file(path)}
    write_json(path.with_suffix('.receipt.json'), receipt)
    return receipt


def load(args, loaded=None):
    """The loaded checkpoint (selfindex.load.Loaded): `loaded` if given, else loaded from args.model."""
    return loaded or load_model(args.model, max_memory_gib=args.max_memory_gib)


def tokenizer(model_dir):
    """The tokenizer as load_model loads it, for the tokenizer-only checks."""
    from transformers import AutoTokenizer
    return checked_tokenizer(AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True), model_dir)


def model_parts(model):
    """Decoder blocks, final norm, LM head and input embedding of the text model of the Qwen3.8 checkpoint."""
    lm = model.model.language_model
    return lm.layers, lm.norm, model.get_output_embeddings(), lm.embed_tokens


def uniform_positions(length):
    """SAMPLES endpoint-inclusive indexes into a body of `length` tokens."""
    if length < SAMPLES:
        raise ValueError(f'body has only {length} tokens')
    result = np.rint(np.linspace(0, length - 1, SAMPLES)).astype(int).tolist()
    assert len(set(result)) == SAMPLES and result[0] == 0 and result[-1] == length - 1
    return result


def single_token_header(tok, word):
    """Token ids of '<|im_start|>' + word + newline when the word is a single token there, else None."""
    header = '<|im_start|>' + word + '\n'
    ids = tok.encode(header, add_special_tokens=False)
    wid = tok.encode(word, add_special_tokens=False)
    if len(ids) != 3 or len(wid) != 1 or ids[1] != wid[0]:
        return None
    return ids


def letter_ids(tok):
    out = {}
    for L in LETTERS:
        ids = tok.encode(L, add_special_tokens=False)
        assert len(ids) == 1, (L, ids)
        out[L] = ids[0]
    return out


def encode_with_body(tok, text, body):
    """Token ids of a rendered text and the (contiguous) positions of the tokens of `body` in it."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc['input_ids']), enc['offset_mapping']
    s = text.index(body)
    e = s + len(body)
    pos = [i for i, (a, b) in enumerate(offs) if b > s and a < e and b > a]
    assert pos == list(range(pos[0], pos[-1] + 1))
    assert tok.encode(text, add_special_tokens=False) == ids
    return ids, pos


class Collector:
    """Forward hooks on every decoder block, capturing at the positions set by reset(): the post-block residual, the FFN output
    before the residual add, and Q/K/V (linear-attention blocks: the in_proj_qkv output split into Q, K and V before the
    convolution and the recurrence; full-attention blocks: the inputs of q_norm and k_norm, before RoPE, and the v_proj output).
    Features are stored as the bits of the BF16 values (uint16, [positions, width]) under '<component>__L<block>'; the full
    residual of the last block is kept in .tail for the readout agreement check."""

    def __init__(self, torch, layers, layer_types):
        self.torch = torch
        self.last = len(layers) - 1
        self.positions = []
        self.features = {}
        self.tail = None
        self.handles = []
        for i, block in enumerate(layers):
            self.handles.append(block.register_forward_hook(self._out(i, 'residual')))
            self.handles.append(block.mlp.register_forward_hook(self._out(i, 'FFN')))
            if layer_types[i] == 'linear_attention':
                mixer = block.linear_attn
                kd, vd = int(mixer.key_dim), int(mixer.value_dim)
                if mixer.in_proj_qkv.out_features != 2 * kd + vd:
                    raise ValueError('linear qkv dims')
                self.handles.append(mixer.in_proj_qkv.register_forward_hook(self._linear(i, kd, vd)))
            elif layer_types[i] == 'full_attention':
                mixer = block.self_attn
                self.handles.append(mixer.q_norm.register_forward_pre_hook(self._pre(i, 'Q')))
                self.handles.append(mixer.k_norm.register_forward_pre_hook(self._pre(i, 'K')))
                self.handles.append(mixer.v_proj.register_forward_hook(self._out(i, 'V')))
            else:
                raise ValueError('unknown layer type')

    def reset(self, positions):
        self.positions, self.features, self.tail = list(positions), {}, None

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def capture(self, layer, component, value):
        if isinstance(value, tuple):
            value = value[0]
        if value.ndim < 3 or value.shape[0] != 1 or value.shape[1] <= max(self.positions):
            raise ValueError((component, layer, tuple(value.shape)))
        if value.dtype != self.torch.bfloat16:
            raise ValueError('feature not bf16')
        sel = value[0, self.positions].contiguous().reshape(len(self.positions), -1)
        self.features[f'{component}__L{layer:02d}'] = sel.view(self.torch.int16).cpu().numpy().view(np.uint16).copy()
        if component == 'residual' and layer == self.last:
            self.tail = value.detach()

    def _out(self, layer, component):
        def hook(module, inputs, output):
            self.capture(layer, component, output)
        return hook

    def _pre(self, layer, component):
        def hook(module, inputs):
            self.capture(layer, component, inputs[0])
        return hook

    def _linear(self, layer, kd, vd):
        def hook(module, inputs, output):
            q, k, v = output.split([kd, kd, vd], dim=-1)
            for name, value in (('Q', q), ('K', k), ('V', v)):
                self.capture(layer, name, value)
        return hook


def bits_to_float(torch, bits):
    return torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)


def js_rows(torch, logp, logq):
    """Jensen-Shannon divergence (nats) between rows of two log-probability tensors."""
    mix = torch.logaddexp(logp, logq) - math.log(2.)
    return (.5 * ((logp.exp() * (logp - mix)).sum(-1) + (logq.exp() * (logq - mix)).sum(-1))).clamp_min(0)


def lens_metrics(torch, norm, head, feats, natives, words, triplets, observed, n_layers):
    """Full-vocabulary readout at the body positions for every block. feats[word] holds the collector features (index 0 and 1
    = role word and newline, then the body samples); natives[word] = the model's own logits at the body samples, used for the
    last block. Per word, body position and block: entropy, log-probability of the observed next token, argmax id, JS
    divergence to the last block; per triplet and role pair (A-U, A-S, U-S): the JS divergence between the two roles."""
    C = len(words)
    nd, hd = next(norm.parameters()).device, next(head.parameters()).device
    final_logp = None
    entropy = np.zeros((C, SAMPLES, n_layers), np.float32)
    obs_logp = np.zeros_like(entropy)
    js_final = np.zeros_like(entropy)
    argmax = np.zeros((C, SAMPLES, n_layers), np.int64)
    pair_js = np.zeros((len(triplets), 3, SAMPLES, n_layers), np.float32)
    obs = torch.tensor(observed, device=hd)
    widx = {w: i for i, w in enumerate(words)}
    pairs = (('A', 'U'), ('A', 'S'), ('U', 'S'))
    with torch.no_grad():
        for layer in range(n_layers - 1, -1, -1):
            if layer == n_layers - 1:
                logits = torch.stack([natives[w].to(hd) for w in words]).float()
            else:
                states = torch.stack([bits_to_float(torch, feats[w][f'residual__L{layer:02d}'][2:]) for w in words])
                logits = head(norm(states.to(nd)).to(hd)).float()
            logp = torch.log_softmax(logits, dim=-1)
            del logits
            ent = -(logp.exp() * logp).sum(-1)
            entropy[:, :, layer] = ent.cpu().numpy()
            obs_logp[:, :, layer] = logp.gather(-1, obs.view(1, -1, 1).expand(C, -1, 1))[..., 0].cpu().numpy()
            argmax[:, :, layer] = logp.argmax(-1).cpu().numpy()
            if final_logp is None:
                final_logp = logp
            js_final[:, :, layer] = js_rows(torch, logp, final_logp).cpu().numpy()
            for t, trip in enumerate(triplets):
                for k, (x, y) in enumerate(pairs):
                    a, b = widx[trip['members'][x]], widx[trip['members'][y]]
                    pair_js[t, k, :, layer] = js_rows(torch, logp[a], logp[b]).cpu().numpy()
            if layer != n_layers - 1:
                del logp
    return {'entropy_nats': entropy, 'observed_next_logprob': obs_logp, 'argmax_id': argmax,
            'js_to_final_nats': js_final, 'triplet_pair_js_nats': pair_js}
