"""Derive crossmodel_role_swap_pos.py from crossmodel_role_swap.py (f2362716) by exact, once-only text replacements (user 10-28:
"直出归属和所有的几何压缩行了"): --pos-compress, single pass only. Where X is an Assistant message, the header tokens that only the
Assistant rendering has (Qwen3.8 default: the empty think block) share the position of the token before them and every later token
moves back by their number, so the body and everything after it have the positions of the User rendering while the tokens stay
visible; position ids are passed explicitly (qwen3_5: four identical rows). The geometry counterpart is crossmodel_suite_mask3.py
--pos-mode compress. Single pass runs one item per forward (no padding).
usage: make_role_swap_pos.py SRC OUT"""
import sys

src = open(sys.argv[1], encoding='utf-8').read().replace('\r\n', '\n')
REP = [
    ('"""', '"""POSITION VARIANT (make_role_swap_pos.py) of crossmodel_role_swap.py f2362716: --pos-compress (single pass) gives the\n'
     'Assistant-only header tokens of X the position of the token before them; later positions move back. ', 1),
    ("    def in_region(i, r):\n",
     "    def compressed(r, d):\n"
     "        \"\"\"position ids of rendering r (X under the Assistant role) with its Assistant-only header tokens (not in the User rendering\n"
     "        d) at the position of the token before them and every later token moved back by their number\"\"\"\n"
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
     "        z0, k = p + 1 + i, len(ah) - j - i\n"
     "        assert k > 0, 'no Assistant-only header token'\n"
     "        pos = [t if t < z0 else (z0 - 1 if t < z0 + k else t - k) for t in range(len(ri))]\n"
     "        assert pos[rs] == ds, 'compressed body start differs from the User rendering'\n"
     "        return pos\n\n"
     "    def in_region(i, r):\n", 1),
    ("                    lp = torch.log_softmax(model(input_ids=x, attention_mask=am, use_cache=False, logits_to_keep=1).logits[:, -1].float(), -1)\n",
     "                    if a.pos_compress and rrole == 'assistant':\n"
     "                        assert len(chunk) == 1 and Bk is None, 'pos-compress: unedited conditions, one item per forward'\n"
     "                        (b_, oi_), r_ = chunk[0], rend[0]\n"
     "                        pt = torch.tensor([compressed(r_, render('user', b_, oi_))], dtype=torch.long, device=dev0)\n"
     "                        if getattr(model.config, 'model_type', '') in ('qwen3_5', 'qwen3_5_moe'):\n"
     "                            pt = pt.view(1, 1, -1).expand(4, 1, -1)\n"
     "                        lp = torch.log_softmax(model(input_ids=x, attention_mask=am, position_ids=pt, use_cache=False,\n"
     "                                                     logits_to_keep=1).logits[:, -1].float(), -1)\n"
     "                    else:\n"
     "                        lp = torch.log_softmax(model(input_ids=x, attention_mask=am, use_cache=False, logits_to_keep=1).logits[:, -1].float(), -1)\n", 1),
    ("'max_new': a.max_new, 'torch': torch.__version__,", "'max_new': a.max_new, 'pos_compress': a.pos_compress, 'torch': torch.__version__,", 1),
    ("    p.add_argument('--check-align', action='store_true', help='tokenizer only: alignment coverage for every held-out body')\n",
     "    p.add_argument('--check-align', action='store_true', help='tokenizer only: alignment coverage for every held-out body')\n"
     "    p.add_argument('--pos-compress', action='store_true', help='single pass: compress the positions of the Assistant-only header tokens')\n", 1),
]
for a, b, n in REP:
    c = src.count(a)
    assert c >= 1 and (a == '"""' or c == 1), (c, a[:80])
    src = src.replace(a, b, n)
open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write(src)
print('written', sys.argv[2], len(src))
