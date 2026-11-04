"""Format ablation and token positions: the geometry of the main probe with the Assistant-only header tokens of the own message
masked out of attention or with explicit position ids (template-native renderings only).

The geometry phase of selfindex.main_probe (--geometry-only, one body at a time): every body under the three roles,
[ROLE: body][user: reply request], one teacher-forced forward per rendering, use_cache=False; the residual of every decoder layer
is kept as the mean over the body tokens (identical ids across roles) and at the response onset. The Assistant-only header tokens
are the tokens after the role word that the User rendering of the same body does not have (minus the tokens both headers share at
their start and end): the empty think block of a Qwen3.8 history message in the default rendering, gpt-oss <|channel|>final. The
format ablation proper of a template that has a switch is a rendering option (--tmpl-kwargs, e.g. Qwen3.8 preserve_thinking=false,
which renders the history message without the empty think block); without a switch:
  --mask-extra    the Assistant-only header tokens of the own message get attention mask 0, so no later position reads them; the
                  tokens and the position ids stay as rendered (with --mask-pos the position ids follow the mask, cumulative sum
                  minus one, as in generation, so the position offset of the masked tokens disappears too; for standard-attention
                  models: linear-attention layers ignore the mask at batch 1)
  --pos-mode compress   the Assistant-only header tokens stay visible but take the position of the token before them and every
                  later token moves back by their number, so the body and everything after it have the positions of the User
                  rendering
  --pos-mode gap  every token of the own message's rendering from the body on moves forward by --pos-gap (a rendering without such
                  tokens given the offset that the default rendering has; --pos-gap 0 reproduces the rendered positions)
Position ids are passed explicitly (Qwen3.8 / qwen3_5 models: four identical rows, text and the three rotary rows); only the
full-attention layers of a hybrid model use positions. The User and System renderings are never changed.
Outputs in --out (those of the main probe's geometry): identity.json, G/<body>.json (per role the numbers of body tokens and tokens,
the masked positions and tokens; with --pos-mode also the mode and the position of the body start) and G/<body>.npz (float16
arrays content_<role>, onset_<role>: layers x hidden), status.json, summary.json (main_probe.summarize: distance ratios per layer,
the share of held-out bodies in which the own message is singled out, direction consistency).
usage: python -m selfindex.format_ablation --model DIR --tmpl-kwargs JSON --lang zh|en --out DIR [--mask-extra [--mask-pos]]
         [--pos-mode none|compress|gap [--pos-gap N]] [--body-ids FILE]
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from . import __version__
from . import items as I
from .io import atomic
from .load import find_layers, load_model
from .main_probe import summarize
from .render import Renderer, common_body
from .role_swap import compressed, selected_bodies, unpaired

ROLES = I.ROLES


def positions(geo, mode, gap):
    """Explicit position ids of the Assistant rendering (geo: {role: (ids, body span)}) for --pos-mode compress or gap."""
    a, u = (geo['assistant'][0], geo['assistant'][1][0]), (geo['user'][0], geo['user'][1][0])
    if mode == 'gap':
        return [i if i < a[1] else i + gap for i in range(len(a[0]))]
    return compressed(a, u)


def run(args, loaded=None):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert not args.mask_pos or args.mask_extra, '--mask-pos needs --mask-extra'
    kw = json.loads(args.tmpl_kwargs)
    L_ = loaded or load_model(args.model, max_memory_gib=args.max_memory_gib, cpu_offload_gib=args.cpu_offload_gib)
    torch, tok, model, cfg = L_.torch, L_.tok, L_.model, L_.cfg
    lname, layers = find_layers(model)
    L = len(layers)
    R = Renderer(tok, kw)
    lang = args.lang
    bodies = selected_bodies(args)
    q_reply = I.reply_request(lang)
    ident = {'experiment': 'P2_crossmodel_suite', 'selfindex_version': __version__, 'model': args.model, 'model_class': L_.model_class,
             'architectures': cfg.architectures, 'layers_module': lname, 'n_layers': L, 'tmpl_kwargs': kw, 'device_map': L_.placement,
             'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
             'cpu_offload_gib': args.cpu_offload_gib, 'geometry_only': True, 'tokenizer_rebuilt_from': getattr(tok, '_p2_rebuilt_from', None),
             'fp8_backend': L_.fp8_backend, 'sim_fp8': L_.sim_fp8, 'disk_offload_dir': None, 'g_batch': 1, 'n_bodies': len(bodies),
             'phases': 'G', 'lang': lang, 'attr_question_example': I.attr_question(I.orders()[0], lang), 'reply_question': q_reply,
             'mask_extra': args.mask_extra, 'mask_pos': args.mask_pos, 'pos_mode': args.pos_mode, 'pos_gap': args.pos_gap,
             'body_ids': args.body_ids}
    atomic(out / 'identity.json', ident)
    print(json.dumps({k: ident[k] for k in ('model_class', 'layers_module', 'n_layers')}), flush=True)
    dev0 = model.get_input_embeddings().weight.device
    st = {'cap': None}

    def hook(i):
        def h(mod, inp, outp):
            if st['cap'] is not None:
                x = outp[0] if isinstance(outp, tuple) else outp
                st['cap'][i] = x[0].detach()
            return None
        return h

    def capture(ids, zero=None, pos=None):
        st['cap'] = {}
        x = torch.tensor([ids], dtype=torch.long, device=dev0)
        with torch.no_grad():
            am = torch.ones_like(x)
            if zero:  # these positions are not read by any position
                am[0, zero] = 0
            if pos is not None:  # explicit position ids (qwen3_5 takes four identical rows)
                pt = torch.tensor([pos], dtype=torch.long, device=dev0)
                if getattr(cfg, 'model_type', '') in ('qwen3_5', 'qwen3_5_moe'):
                    pt = pt.view(1, 1, -1).expand(4, 1, -1)
                model(input_ids=x, attention_mask=am, position_ids=pt, use_cache=False, return_dict=True, logits_to_keep=1)
            elif zero and args.mask_pos:  # the position ids that generation derives from the mask
                model(input_ids=x, attention_mask=am, position_ids=(am.cumsum(-1) - 1).clamp(min=0), use_cache=False,
                      return_dict=True, logits_to_keep=1)
            else:
                model(input_ids=x, attention_mask=am, use_cache=False, return_dict=True, logits_to_keep=1)
        cap = st['cap']
        st['cap'] = None
        return cap

    handles = [layers[i].register_forward_hook(hook(i)) for i in range(L)]
    t0 = time.time()
    gdir = out / 'G'
    gdir.mkdir(exist_ok=True)
    try:
        for bi, b in enumerate(bodies):
            done = gdir / f'{b["id"]}.json'
            if done.exists():
                continue
            geo = {r: R.render([{'role': r, 'content': b['body']}, {'role': 'user', 'content': q_reply}], b['body']) for r in ROLES}
            spans = common_body(geo)
            arrs, rec = {}, {'id': b['id'], 'split': b['split'], 'category': b['category'], 'roles': {}}
            for r in ROLES:
                ids = geo[r][0]
                own = r == 'assistant'
                zero = unpaired((ids, geo[r][1][0]), (geo['user'][0], geo['user'][1][0])) if (args.mask_extra and own) else None
                pid = positions(geo, args.pos_mode, args.pos_gap) if (args.pos_mode != 'none' and own) else None
                cap = capture(ids, zero, pid)
                pos = spans[r]
                arrs[f'content_{r}'] = torch.stack([cap[i][pos].float().mean(0).to(dev0) for i in range(L)]).half().cpu().numpy()
                arrs[f'onset_{r}'] = torch.stack([cap[i][len(ids) - 1].float().to(dev0) for i in range(L)]).half().cpu().numpy()
                rec['roles'][r] = {'n_body_tokens': len(pos), 'n_tokens': len(ids), 'masked': zero,
                                   'masked_tokens': tok.convert_ids_to_tokens([ids[i] for i in zero]) if zero else None}
                if args.pos_mode != 'none':
                    rec['roles'][r].update(pos_mode=args.pos_mode if own else 'none',
                                           body_start_position=pid[geo['assistant'][1][0]] if pid is not None else geo[r][1][0])
            np.savez_compressed(gdir / f'{b["id"]}.npz', **arrs)
            atomic(done, rec)
            if bi % 10 == 0:
                print(f'G {bi + 1}/{len(bodies)} {time.time() - t0:.0f}s', flush=True)
    finally:
        for h in handles:
            h.remove()
    atomic(out / 'status.json', {'state': 'complete', 'elapsed_s': time.time() - t0})
    summarize(out, L)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', required=True)
    p.add_argument('--tmpl-kwargs', default='{}', help='chat-template keyword arguments of the rendering (JSON)')
    p.add_argument('--out', required=True)
    p.add_argument('--lang', choices=('zh', 'en'), default='zh')
    p.add_argument('--mask-extra', action='store_true', help='mask the Assistant-only header tokens of the own message out of attention')
    p.add_argument('--mask-pos', action='store_true', help='with --mask-extra: position ids follow the mask (as in generation)')
    p.add_argument('--pos-mode', choices=('none', 'compress', 'gap'), default='none', help='explicit position ids for the own message')
    p.add_argument('--pos-gap', type=int, default=4, help='--pos-mode gap: offset added from the body on')
    p.add_argument('--body-ids', default='', help='JSON list of body ids (or objects with an id): only these bodies, in this order')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    p.add_argument('--cpu-offload-gib', type=float, default=0, help='allow weights beyond the GPU budget to stay in CPU memory')
    p.add_argument('--summarize-only', type=int, default=0, help='rewrite summary.json from the outputs in --out for this many layers')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.summarize_only:
        summarize(Path(args.out), args.summarize_only)
        return
    try:
        run(args)
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        atomic(Path(args.out) / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
