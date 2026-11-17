"""P2 check J: role geometry on native template renderings only (both template modes), all 64 layers.

Input = the chat template's own output for [role message (body), user turn] + generation prompt (enable_thinking=False):
  mode 'default'        : template default; Assistant history carries the empty thinking block, User/System do not
  mode 'preserve_false' : preserve_thinking=False; no thinking block for any role (the roles differ only at position 1)
User turns: 'reply' = '请根据上文回应。' (the check-D turn), 'attr' / 'attr2' = the attribution question of checks
E-I in option orders q1 / q2. --all-bodies runs the 150 bodies (default: the 30-body pilot set).
Nothing is forged: every sequence is asserted to equal tok.apply_chat_template(...) for its mode.

Per forward: residual stream (decoder block outputs) at all 64 layers at named positions
  header_newline, body_mean, 10 body points (same body-token indices for all roles), body_im_end,
  turn_newline (newline after '<|im_start|>user'), decision (last prompt token = response onset);
saved as float16. Decision-point full-vocabulary readout at every layer: final norm + LM head on that layer's state
(layer 63 = native logits, agreement recorded; middle layers are uncalibrated, diagnostic only); per layer the
pairwise Jensen-Shannon divergence between roles, entropy, top-5 ids, and option-letter log-probabilities.
Mode: teacher-forced single forward per sequence, use_cache=False, no generation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_null_pilot as base  # noqa: E402
from p2_attribution_patch import question, letter_ids  # noqa: E402
from p2_attribution_default import encode_with_body  # noqa: E402
from p2_body_sets import all_bodies  # noqa: E402

ROLES = ('assistant', 'user', 'system')
TURNS = {'reply': '请根据上文回应。', 'attr': question('q1'), 'attr2': question('q2')}
MODES = ('default', 'preserve_false')
N_POINTS = 10
POS_NAMES = ['header_newline', 'body_mean'] + [f'body_p{i}' for i in range(N_POINTS)] + ['body_im_end', 'turn_newline', 'decision']


def native_ids(tok, role, body, turn, mode):
    msgs = [{'role': role, 'content': body}, {'role': 'user', 'content': turn}]
    kw = dict(add_generation_prompt=True, enable_thinking=False)
    if mode == 'preserve_false':
        kw['preserve_thinking'] = False
    text = tok.apply_chat_template(msgs, tokenize=False, **kw)
    ids, body_pos = encode_with_body(tok, text, body)
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


def run(args):
    out = Path(args.out)
    (out/'bodies').mkdir(parents=True, exist_ok=True)
    (out/'features').mkdir(parents=True, exist_ok=True)
    torch, tok, model, layers, norm, head, embed, placement, dtypes = base.load_model(args)
    letters = letter_ids(tok)
    bodies = all_bodies(args.bodies) if args.all_bodies else base.select_bodies(args.bodies)
    base.atomic(out/'identity.json', {'experiment': 'P2_native_geometry_J', 'script_sha256': base.sha_file(__file__),
        'host': __import__('socket').gethostname(), 'torch': torch.__version__,
        'transformers': __import__('transformers').__version__,
        'modes': MODES, 'turns': TURNS, 'roles': ROLES, 'position_names': POS_NAMES, 'device_map': placement,
        'bodies': [b['id'] for b in bodies],
        'labels': {'readout': 'teacher_forced_prompt_forward', 'lens': 'final norm + LM head per layer; L63 = native; middle layers diagnostic only',
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
    handles = [layers[l].register_forward_hook(hook_for(l)) for l in range(len(layers))]
    t0 = time.time()
    for b_i, row in enumerate(bodies):
        done = out/'bodies'/(row['id']+'.json')
        if done.exists():
            continue
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
                        lp_layers = torch.stack(lp_layers)  # [64, V]
                        agree = float((lp_layers[-1] - native.to(lp_layers.device)).abs().max())
                    feats[f'{mode}__{tname}__{r}'] = arr
                    lps[r] = lp_layers
                    ent = (-(lp_layers.exp() * lp_layers).sum(-1)).cpu().numpy()
                    top5 = lp_layers.topk(5, -1).indices.cpu().numpy().tolist()
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
        np.savez(out/'features'/(row['id']+'.npz'), **feats)
        base.atomic(done, rec)
        base.atomic(out/'status.json', {'state': 'running', 'bodies_done': b_i + 1, 'bodies_planned': len(bodies),
                                        'elapsed_s': time.time()-t0, 'heartbeat_epoch': time.time()})
    for h in handles:
        h.remove()
    base.atomic(out/'status.json', {'state': 'complete', 'bodies_planned': len(bodies), 'elapsed_s': time.time()-t0})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--bodies', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--all-bodies', action='store_true')
    p.add_argument('--max-memory-gib', type=float, default=38)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        base.atomic(Path(args.out)/'status.json', {'state': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
