"""Derive crossmodel_suite_mask3.py from crossmodel_suite_mask2.py (cdbd0b17) by exact, once-only text replacements (user 10-28: "qwen
27b 的位置重算还没做，也得排除位置的影响"). --pos-mode for the Assistant rendering in phase G (the User and System renderings keep their
positions), position ids passed explicitly (Qwen3.8 / qwen3_5: four identical rows, as in the paper's p2_default_geometry.py):
  compress  the Assistant-only header tokens (Qwen3.8 default: the empty think block) share the position of the token before them and
            every later token moves back by their number, so the body and everything after it have the positions of the User rendering
            while the tokens stay visible
  gap       (a rendering without such tokens, e.g. preserve_thinking=false) every token from the body on moves forward by --pos-gap,
            the offset that the default rendering has, without the tokens
For Qwen3.8 only the 16 full-attention layers use positions; the 48 linear-attention layers have no positional encoding.
10-28 fix (cdd0691a -> this version): the position ids are held in pid. The G loop reuses the name pos for the body span after the
forward, so cdd0691a recorded body_start_position from the span list (a wrong value; IndexError when the body is shorter than the body
start, as in the thinking rendering); its forwards and arrays are unaffected.
usage: make_suite_mask3.py SRC OUT"""
import sys

src = open(sys.argv[1], encoding='utf-8').read().replace('\r\n', '\n')
REP = [
    ('"""MASK VARIANT 2 (make_suite_mask2.py;', '"""MASK VARIANT 3 (make_suite_mask3.py; --pos-mode compress|gap for the Assistant rendering) of the\nMASK VARIANT 2 (make_suite_mask2.py;'),
    ("    def fwd(ids, mode=None, keep=1, zero=None, **extra):\n", "    def fwd(ids, mode=None, keep=1, zero=None, pos=None, **extra):\n"),
    ("            if zero and args.mask_pos:  # the position ids that generation derives from the mask\n",
     "            if pos is not None:  # --pos-mode: explicit position ids (qwen3_5 takes four identical rows)\n"
     "                pt = torch.tensor([pos], dtype=torch.long, device=dev0)\n"
     "                if getattr(cfg, 'model_type', '') in ('qwen3_5', 'qwen3_5_moe'):\n"
     "                    pt = pt.view(1, 1, -1).expand(4, 1, -1)\n"
     "                o = model(input_ids=x, attention_mask=am, position_ids=pt, use_cache=False, return_dict=True, logits_to_keep=keep)\n"
     "            elif zero and args.mask_pos:  # the position ids that generation derives from the mask\n"),
    ("                zero = extra_header(geo) if (args.mask_extra and r == 'assistant') else None\n"
     "                _, cap = fwd(ids, 'capture', zero=zero)\n",
     "                zero = extra_header(geo) if (args.mask_extra and r == 'assistant') else None\n"
     "                pid = positions(geo, args.pos_mode, args.pos_gap) if (args.pos_mode != 'none' and r == 'assistant') else None\n"
     "                _, cap = fwd(ids, 'capture', zero=zero, pos=pid)\n"),
    ("                                       'masked_tokens': tok.convert_ids_to_tokens([ids[i] for i in zero]) if zero else None}\n",
     "                                       'masked_tokens': tok.convert_ids_to_tokens([ids[i] for i in zero]) if zero else None,\n"
     "                                       'pos_mode': args.pos_mode if r == 'assistant' else 'none',\n"
     "                                       'body_start_position': (pid[pos_body(geo)] if pid is not None else geo[r][1][0])}\n"),
    ("    p.add_argument('--mask-pos', action='store_true', help='with --mask-extra: position ids follow the mask (as in generation)')\n",
     "    p.add_argument('--mask-pos', action='store_true', help='with --mask-extra: position ids follow the mask (as in generation)')\n"
     "    p.add_argument('--pos-mode', choices=('none', 'compress', 'gap'), default='none', help='explicit position ids for the Assistant rendering')\n"
     "    p.add_argument('--pos-gap', type=int, default=4, help='--pos-mode gap: offset added from the body on')\n"),
]
for a, b in REP:
    assert src.count(a) == 1, (src.count(a), a[:80])
    src = src.replace(a, b)
HELPER = '''

def pos_body(geo):
    return geo['assistant'][1][0]


def positions(geo, mode, gap):
    """Explicit position ids of the Assistant rendering. compress: the Assistant-only header tokens take the position of the token
    before them and later tokens move back by their number (body positions equal the User rendering's); gap: every token from the
    body on moves forward by gap."""
    n = len(geo['assistant'][0])
    if mode == 'gap':
        b0 = pos_body(geo)
        return [i if i < b0 else i + gap for i in range(n)]
    z = extra_header(geo)
    assert z and z == list(range(z[0], z[-1] + 1)), 'compress needs one contiguous run of Assistant-only header tokens'
    k = len(z)
    p = [i if i < z[0] else (z[0] - 1 if i <= z[-1] else i - k) for i in range(n)]
    assert p[pos_body(geo)] == geo['user'][1][0], 'compressed body start differs from the User rendering'
    return p
'''
k = src.index('\n\ndef common_body(spans_ids):')
src = src[:k] + HELPER + src[k:]
open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write(src)
print('written', sys.argv[2], len(src))
