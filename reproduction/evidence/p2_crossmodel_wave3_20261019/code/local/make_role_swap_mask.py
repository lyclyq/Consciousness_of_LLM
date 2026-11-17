"""Derive crossmodel_role_swap_mask.py from crossmodel_role_swap.py (f2362716) by exact, once-only text replacements (user 10-28: the
inducer test for gpt-oss, whose harmony template has no switch that removes <|channel|>final from history Assistant messages; the
template switch preserve_thinking=false plays this role for Qwen3.8). --mask-extra: the header tokens that only the Assistant rendering
of X has (after the role word, not shared with the User header; gpt-oss <|channel|>final) get attention mask 0 wherever X is an
Assistant message, in the generation (whose position ids transformers derives from the attention mask, so the masked tokens are not
counted and the position offset they cause disappears as well), in a single-pass forward and in a donor pass, both given the same
position ids (cumulative sum of the mask minus one). Not for Qwen3.8: its linear-attention layers ignore the mask for one sequence.
usage: make_role_swap_mask.py SRC OUT"""
import sys

src = open(sys.argv[1], encoding='utf-8').read().replace('\r\n', '\n')
REP = [
    ('"""', '"""MASK VARIANT (make_role_swap_mask.py) of crossmodel_role_swap.py f2362716: --mask-extra masks the Assistant-only header\n'
     'tokens of X out of attention (position ids follow the mask). ', 1),
    ("    def in_region(i, r):\n",
     "    def unpaired(r, d):\n"
     "        \"\"\"positions of the header tokens of rendering r (X under the Assistant role) that the rendering d (X under the User role)\n"
     "        does not have: after the role word, minus the tokens both headers share at their start and end\"\"\"\n"
     "        ri, rs, di, ds = r[0], r[1], d[0], d[1]\n"
     "        p = 0\n"
     "        while p < min(rs, ds) and ri[p] == di[p]:\n"
     "            p += 1\n"
     "        ah, uh = ri[p + 1:rs], di[p + 1:ds]\n"
     "        i = 0\n"
     "        while i < min(len(ah), len(uh)) and ah[i] == uh[i]:\n"
     "            i += 1\n"
     "        j = 0\n"
     "        while j < min(len(ah), len(uh)) - i and ah[-1 - j] == uh[-1 - j]:\n"
     "            j += 1\n"
     "        return list(range(p + 1 + i, p + 1 + len(ah) - j))\n\n"
     "    def in_region(i, r):\n", 1),
    ("        x = torch.tensor([d[0]], device=dev0)\n        with torch.no_grad():\n"
     "            model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)\n",
     "        x = torch.tensor([d[0]], device=dev0)\n"
     "        amd = torch.ones_like(x)\n"
     "        if a.mask_extra and drole == 'assistant':\n"
     "            amd[0, unpaired(d, render('user', b, oi))] = 0\n"
     "        with torch.no_grad():\n"
     "            model(input_ids=x, attention_mask=amd, position_ids=(amd.cumsum(-1) - 1).clamp(min=0), use_cache=False, logits_to_keep=1)\n", 1),
    ("            am = torch.tensor([[0] * (Lmax - len(q)) + [1] * len(q) for q in seqs], dtype=torch.long, device=dev0)\n",
     "            am = torch.tensor([[0] * (Lmax - len(q)) + [1] * len(q) for q in seqs], dtype=torch.long, device=dev0)\n"
     "            masked = [unpaired(r, render('user', b, oi)) for (b, oi), r in zip(chunk, rend)] \\\n"
     "                if (a.mask_extra and rrole == 'assistant') else [[] for _ in chunk]\n"
     "            for row, (q, mp) in enumerate(zip(seqs, masked)):\n"
     "                if a.mask_extra and rrole == 'assistant':\n"
     "                    assert mp, 'no Assistant-only header token to mask'\n"
     "                am[row, [Lmax - len(q) + i for i in mp]] = 0\n", 1),
    ("                    lp = torch.log_softmax(model(input_ids=x, attention_mask=am, use_cache=False, logits_to_keep=1).logits[:, -1].float(), -1)\n",
     "                    lp = torch.log_softmax(model(input_ids=x, attention_mask=am, position_ids=(am.cumsum(-1) - 1).clamp(min=0), use_cache=False,\n"
     "                                                 logits_to_keep=1).logits[:, -1].float(), -1)\n", 1),
    ("                rec = {'cond': name, 'id': b['id'], 'recipient': rrole, 'donor': drole, 'basis': which, 'k': k, 'order': list(ORDERS[oi])}\n",
     "                rec = {'cond': name, 'id': b['id'], 'recipient': rrole, 'donor': drole, 'basis': which, 'k': k, 'order': list(ORDERS[oi]),\n"
     "                       'mask_extra': a.mask_extra, 'masked_positions': masked[j]}\n", 1),
    ("'max_new': a.max_new, 'torch': torch.__version__,", "'max_new': a.max_new, 'mask_extra': a.mask_extra, 'torch': torch.__version__,", 1),
    ("    p.add_argument('--check-align', action='store_true', help='tokenizer only: alignment coverage for every held-out body')\n",
     "    p.add_argument('--check-align', action='store_true', help='tokenizer only: alignment coverage for every held-out body')\n"
     "    p.add_argument('--mask-extra', action='store_true', help='mask the Assistant-only header tokens of X out of attention')\n", 1),
]
for a, b, n in REP:
    c = src.count(a)
    assert c >= n and (a == '"""' or c == 1), (c, a[:80])
    src = src.replace(a, b, n)
open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write(src)
print('written', sys.argv[2], len(src))
