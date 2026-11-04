"""Response-onset geometry and self-reference (run label J): native template renderings in both template modes, every layer.

Input = the chat template's own output for [role message (body), user turn] + generation prompt (enable_thinking=False):
  mode 'default'        : template default; Assistant history carries the empty thinking block, User/System do not
  mode 'preserve_false' : preserve_thinking=False; no thinking block for any role (the roles differ only at position 1)
User turns: 'reply' = the reply request, 'attr' / 'attr2' = the attribution question in option orders q1 / q2.
Every sequence is checked to equal tok.apply_chat_template(...) for its mode.

Per forward: residual stream (decoder block outputs) at every layer at named positions
  header_newline, body_mean, 10 body points (the same body-token indexes for all roles), body_im_end, turn_newline (newline after
  '<|im_start|>user'), decision (last prompt token = response onset);
saved as float16. Decision-point full-vocabulary readout at every layer: final norm + LM head on that layer's state (last layer =
native logits, agreement recorded; middle layers are uncalibrated, diagnostic only); per layer the pairwise Jensen-Shannon
divergence between roles, entropy, top-5 ids, and the option-letter log-probabilities of the native logits.
Mode: teacher-forced single forward per sequence, use_cache=False, no generation.
Outputs: identity.json, features/<body>.npz (arrays <mode>__<turn>__<role>: layers x 15 positions x width, float16),
bodies/<body>.json, status.json.
usage: python -m selfindex.twolevel.native_geometry --model DIR --out DIR [--n-bodies N] [--max-memory-gib G]
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np

from .. import __version__
from .. import items as I
from . import base as B

ROLES = B.ROLES
TURNS = {'reply': I.reply_request('zh'), 'attr': B.question('q1'), 'attr2': B.question('q2')}
MODES = ('default', 'preserve_false')
N_POINTS = 10
POS_NAMES = ['header_newline', 'body_mean'] + [f'body_p{i}' for i in range(N_POINTS)] + ['body_im_end', 'turn_newline', 'decision']


def native_ids(tok, role, body, turn, mode):
    msgs = [{'role': role, 'content': body}, {'role': 'user', 'content': turn}]
    kw = dict(add_generation_prompt=True, enable_thinking=False)
    if mode == 'preserve_false':
        kw['preserve_thinking'] = False
    text = tok.apply_chat_template(msgs, tokenize=False, **kw)
    ids, body_pos = B.encode_with_body(tok, text, body)
    ref = tok.apply_chat_template(msgs, tokenize=True, **kw)
    ref = ref['input_ids'] if hasattr(ref, 'keys') else ref
    ref = list(ref[0]) if ref and isinstance(ref[0], list) else list(ref)
    assert ids == ref, (role, mode, 'text and token renderings disagree')
    return ids, body_pos


def positions(tok, ids, body_pos, ref_body_ids):
    toks = tok.convert_ids_to_tokens(ids)
    assert [ids[p] for p in body_pos] == ref_body_ids, 'body tokens differ across roles'
    end = body_pos[-1] + 1
    assert toks[end] == '<|im_end|>' and toks[end+2] == '<|im_start|>' and toks[end+3] == 'user' and toks[end+4] == 'Ċ', toks[end:end+5]
    assert toks[2] == 'Ċ'
    pts = [body_pos[int(round(x))] for x in np.linspace(0, len(body_pos)-1, N_POINTS)]
    return {'header_newline': 2, 'body': body_pos, 'points': pts, 'body_im_end': end, 'turn_newline': end+4, 'decision': len(ids)-1}


def run(args, loaded=None):
    out = Path(args.out)
    (out/'bodies').mkdir(parents=True, exist_ok=True)
    (out/'features').mkdir(parents=True, exist_ok=True)
    L_ = B.load(args, loaded)
    torch, tok, model = L_.torch, L_.tok, L_.model
    layers, norm, head, embed = B.model_parts(model)
    letters = B.letter_ids(tok)
    bodies = B.all_bodies()[:args.n_bodies]
    B.write_json(out/'identity.json', {'experiment': 'P2_native_geometry_J', 'selfindex_version': __version__,
        'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
        'modes': MODES, 'turns': TURNS, 'roles': ROLES, 'position_names': POS_NAMES, 'device_map': L_.placement,
        'bodies': [b['id'] for b in bodies],
        'labels': {'readout': 'teacher_forced_prompt_forward',
                   'lens': f'final norm + LM head per layer; L{len(layers) - 1} = native; middle layers diagnostic only',
                   'inputs': 'native chat-template renderings only'}})
    dev0 = embed.weight.device
    nd, hd = next(norm.parameters()).device, next(head.parameters()).device
    cap = {}

    def hook_for(layer):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            cap[layer] = h[0].detach()
            return None
        return hook

    def measure(row):
        """The record and the feature arrays of one body: MODES x TURNS x ROLES forwards."""
        rec = {'id': row['id'], 'split': row['split'], 'conditions': []}
        feats = {}
        for mode in MODES:
            for tname, turn in TURNS.items():
                seqs = {r: native_ids(tok, r, row['body'], turn, mode) for r in ROLES}
                ref_body = [seqs['assistant'][0][p] for p in seqs['assistant'][1]]
                lps, info = {}, {}
                for r in ROLES:
                    ids, body_pos = seqs[r]
                    P = positions(tok, ids, body_pos, ref_body)
                    cap.clear()
                    x = torch.tensor([ids], dtype=torch.long, device=dev0)
                    with torch.no_grad():
                        o = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=1)
                        native = torch.log_softmax(o.logits[0, -1].float(), -1)
                        arr = np.zeros((len(layers), len(POS_NAMES), cap[0].shape[-1]), np.float16)
                        lp_layers = []
                        for l in range(len(layers)):
                            h = cap[l]
                            sel = [P['header_newline']] + P['points'] + [P['body_im_end'], P['turn_newline'], P['decision']]
                            v = h[sel].float()
                            bm = h[P['body']].float().mean(0, keepdim=True)
                            arr[l] = torch.cat([v[:1], bm, v[1:]], 0).cpu().numpy().astype(np.float16)
                            lp_layers.append(torch.log_softmax(head(norm(h[P['decision']].to(nd)).to(hd)).float(), -1))
                        lp_layers = torch.stack(lp_layers)  # [layers, V]
                        agree = float((lp_layers[-1] - native.to(lp_layers.device)).abs().max())
                    feats[f'{mode}__{tname}__{r}'] = arr
                    lps[r] = lp_layers
                    ent = (-(lp_layers.exp() * lp_layers).sum(-1)).cpu().numpy()
                    top5 = lp_layers.topk(5, -1).indices.cpu().numpy().tolist()
                    # 'lens_l63_vs_native_maxabs' (key of the output format): last-layer readout vs native logits
                    info[r] = {'len': len(ids), 'positions': {k: (v if not isinstance(v, list) or k == 'points' else [v[0], v[-1]]) for k, v in P.items()},
                               'lens_l63_vs_native_maxabs': agree, 'entropy': ent.tolist(), 'top5': top5,
                               'native_letter_logprob': {Lt: float(native[i]) for Lt, i in letters.items()},
                               'native_top_id': int(native.argmax())}
                js = {}
                for a_, b_ in (('assistant', 'user'), ('assistant', 'system'), ('user', 'system')):
                    pa, pb = lps[a_].exp(), lps[b_].exp()
                    m = 0.5 * (pa + pb)
                    lm = m.clamp_min(1e-30).log()
                    js[f'{a_[0].upper()}{b_[0].upper()}'] = (0.5 * (pa * (lps[a_] - lm)).sum(-1) + 0.5 * (pb * (lps[b_] - lm)).sum(-1)).cpu().numpy().tolist()
                rec['conditions'].append({'mode': mode, 'turn': tname, 'roles': info, 'decision_js_by_layer': js})
                del lps
        return rec, feats

    handles = [layers[l].register_forward_hook(hook_for(l)) for l in range(len(layers))]
    t0 = time.time()
    try:
        for b_i, row in enumerate(bodies):
            done = out/'bodies'/(row['id']+'.json')
            if done.exists():
                continue
            rec, feats = measure(row)
            np.savez(out/'features'/(row['id']+'.npz'), **feats)
            B.write_json(done, rec)
            B.write_json(out/'status.json', {'state': 'running', 'bodies_done': b_i + 1, 'bodies_planned': len(bodies),
                                             'elapsed_s': time.time()-t0, 'heartbeat_epoch': time.time()})
    finally:
        for h in handles:
            h.remove()
    B.write_json(out/'status.json', {'state': 'complete', 'bodies_planned': len(bodies), 'elapsed_s': time.time()-t0})


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n-bodies', type=int, default=150, help='the first N bodies of the fixed order (default: all 150)')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        B.write_json(Path(args.out)/'status.json', {'state': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
