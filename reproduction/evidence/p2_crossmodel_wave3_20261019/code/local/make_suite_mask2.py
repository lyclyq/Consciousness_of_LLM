"""Derive crossmodel_suite_mask2.py from crossmodel_suite_mask.py (c8a29196) by exact, once-only text replacements (user 10-28):
--mask-pos gives a masked forward the position ids that generation derives from the attention mask (cumulative sum minus one), so the
masked tokens are not counted and the position offset they cause disappears, as in the masked attribution runs
(crossmodel_role_swap_mask.py) and as with Qwen3.8's template switch. Without --mask-pos the masked forward keeps the rendered
positions (the 10-28 gpt-oss geometry). For standard-attention models (gpt-oss); not for Qwen3.8 (linear attention ignores the mask).
usage: make_suite_mask2.py SRC OUT"""
import sys

src = open(sys.argv[1], encoding='utf-8').read().replace('\r\n', '\n')
REP = [
    ('"""MASK VARIANT (make_suite_mask.py)', '"""MASK VARIANT 2 (make_suite_mask2.py; --mask-pos: position ids follow the mask) of the\nMASK VARIANT (make_suite_mask.py)'),
    ("            o = model(input_ids=x, attention_mask=am, use_cache=False, return_dict=True, logits_to_keep=keep)\n",
     "            if zero and args.mask_pos:  # the position ids that generation derives from the mask\n"
     "                o = model(input_ids=x, attention_mask=am, position_ids=(am.cumsum(-1) - 1).clamp(min=0), use_cache=False,\n"
     "                          return_dict=True, logits_to_keep=keep)\n"
     "            else:\n"
     "                o = model(input_ids=x, attention_mask=am, use_cache=False, return_dict=True, logits_to_keep=keep)\n"),
    ("    p.add_argument('--mask-extra', action='store_true', help='phase G: mask the Assistant-only header tokens out of attention')\n",
     "    p.add_argument('--mask-extra', action='store_true', help='phase G: mask the Assistant-only header tokens out of attention')\n"
     "    p.add_argument('--mask-pos', action='store_true', help='with --mask-extra: position ids follow the mask (as in generation)')\n"),
]
for a, b in REP:
    assert src.count(a) == 1, (src.count(a), a[:80])
    src = src.replace(a, b)
assert 'args' in src.split('def fwd(')[0][-2000:] or 'args' in src  # fwd is defined inside run(args)
open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write(src)
print('written', sys.argv[2], len(src))
