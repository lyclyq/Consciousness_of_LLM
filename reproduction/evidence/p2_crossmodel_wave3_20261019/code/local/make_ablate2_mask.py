"""Derive crossmodel_thinking_ablate2_mask.py from crossmodel_thinking_ablate2.py (c9ef2cbe, the fit of gpt-oss-120b) by exact,
once-only text replacements (user 10-28: the ablation of the own-message format tokens in gpt-oss needs its own fit, as Qwen3.8's
template switch was fitted on the switched rendering). --mask-extra: in the fit forwards, the header tokens that only the Assistant
rendering has (gpt-oss <|channel|>final) get attention mask 0 and the position ids follow the mask (cumulative sum minus one), the
condition of crossmodel_role_swap_mask.py and crossmodel_suite_mask2.py --mask-pos. The fit rows (body and later tokens) are unchanged.
For standard-attention models only.
usage: make_ablate2_mask.py SRC OUT"""
import sys

src = open(sys.argv[1], encoding='utf-8').read().replace('\r\n', '\n')
REP = [
    ('"""', '"""MASK VARIANT (make_ablate2_mask.py) of crossmodel_thinking_ablate2.py c9ef2cbe: --mask-extra masks the Assistant-only\n'
     'header tokens out of attention in the fit forwards (position ids follow the mask).\n', 1),
    ("                        ids, m0, s0, e, ee = render(role, b, oi, fit=True)\n"
     "                        state['cap'] = {l: None for l in fit_layers}\n"
     "                        x = torch.tensor([ids], device=dev0)\n"
     "                        model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)\n",
     "                        ids, m0, s0, e, ee = render(role, b, oi, fit=True)\n"
     "                        state['cap'] = {l: None for l in fit_layers}\n"
     "                        x = torch.tensor([ids], device=dev0)\n"
     "                        amf = torch.ones_like(x)\n"
     "                        if a.mask_extra and role == 'assistant':\n"
     "                            ui, _, us, _, _ = render('user', b, oi, fit=True)\n"
     "                            amf[0, unpaired(ids, s0, ui, us)] = 0\n"
     "                        model(input_ids=x, attention_mask=amf, position_ids=(amf.cumsum(-1) - 1).clamp(min=0), use_cache=False,\n"
     "                              logits_to_keep=1)\n", 1),
    ("    def render(role, b, oi, fit=False):\n",
     "    def unpaired(ri, rs, di, ds):\n"
     "        \"\"\"positions of the header tokens of the Assistant rendering (ri, body start rs) that the User rendering (di, ds) does\n"
     "        not have: after the role word, minus the tokens both headers share at their start and end\"\"\"\n"
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
     "        z = list(range(p + 1 + i, p + 1 + len(ah) - j))\n"
     "        assert z, 'no Assistant-only header token to mask'\n"
     "        return z\n\n"
     "    def render(role, b, oi, fit=False):\n", 1),
    ("    p.add_argument('--a-mean', action='store_true', help='also fit the Assistant mean (conditions VA / RA: move toward the own message)')\n",
     "    p.add_argument('--a-mean', action='store_true', help='also fit the Assistant mean (conditions VA / RA: move toward the own message)')\n"
     "    p.add_argument('--mask-extra', action='store_true', help='fit forwards: mask the Assistant-only header tokens (positions follow)')\n", 1),
]
for a, b, n in REP:
    c = src.count(a)
    assert c >= 1 and (a == '"""' or c == 1), (c, a[:80])
    src = src.replace(a, b, n)
open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write(src)
print('written', sys.argv[2], len(src))
