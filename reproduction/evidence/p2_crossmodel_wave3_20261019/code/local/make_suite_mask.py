"""Derive crossmodel_suite_mask.py from crossmodel_suite.py 516f117e (the version of the gpt-oss MXFP4 geometry on PRO) by exact,
once-only text replacements (user 10-28: the inducer test for gpt-oss-20b and gpt-oss-120b, whose harmony template has no switch
that removes <|channel|>final from history Assistant messages; calibrated first on Qwen3.8-27B, where preserve_thinking=false gives
the template-native rendering without the empty think block). --mask-extra: in phase G, the header tokens that only the Assistant
rendering has (after the role word, not shared with the User header: Qwen3.8 <think>\\n\\n</think>\\n\\n, gpt-oss <|channel|>final)
get attention mask 0, so no later position reads them; the tokens and the position ids stay as rendered. Nothing else changes.
usage: make_suite_mask.py SRC OUT"""
import sys

src = open(sys.argv[1], encoding='utf-8').read().replace('\r\n', '\n')
REP = [
    ('"""Cross-model suite', '"""MASK VARIANT (make_suite_mask.py) of crossmodel_suite.py 516f117e: --mask-extra masks, in phase G, the Assistant-only\n'
     'header tokens (no counterpart in the User header) out of attention; tokens and position ids unchanged.\nCross-model suite'),
    ("    def fwd(ids, mode=None, keep=1, **extra):\n", "    def fwd(ids, mode=None, keep=1, zero=None, **extra):\n"),
    ("            o = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=keep)\n",
     "            am = torch.ones_like(x)\n"
     "            if zero:  # --mask-extra: these positions are not read by any position\n"
     "                am[0, zero] = 0\n"
     "            o = model(input_ids=x, attention_mask=am, use_cache=False, return_dict=True, logits_to_keep=keep)\n"),
    ("                _, cap = fwd(ids, 'capture')\n",
     "                zero = extra_header(geo) if (args.mask_extra and r == 'assistant') else None\n"
     "                _, cap = fwd(ids, 'capture', zero=zero)\n"),
    ("                    rec['roles'][r] = {'n_body_tokens': len(pos), 'n_tokens': len(ids)}\n",
     "                    rec['roles'][r] = {'n_body_tokens': len(pos), 'n_tokens': len(ids), 'masked': zero,\n"
     "                                       'masked_tokens': tok.convert_ids_to_tokens([ids[i] for i in zero]) if zero else None}\n"),
    ("    p.add_argument('--geometry-only', action='store_true', help='phase G without the attribution forwards (thinking-mode renderings)')\n",
     "    p.add_argument('--geometry-only', action='store_true', help='phase G without the attribution forwards (thinking-mode renderings)')\n"
     "    p.add_argument('--mask-extra', action='store_true', help='phase G: mask the Assistant-only header tokens out of attention')\n"),
]
for a, b in REP:
    assert src.count(a) == 1, (src.count(a), a[:80])
    src = src.replace(a, b)
HELPER = '''

def extra_header(geo):
    """Positions of the header tokens that only the Assistant rendering has: after the role word (the first token at which the
    Assistant and User renderings differ) and before the body, minus the tokens that both headers share at their start and end."""
    (ai, asp), (ui, usp) = geo['assistant'], geo['user']
    a0, u0 = asp[0], usp[0]
    p = 0
    while p < min(a0, u0) and ai[p] == ui[p]:
        p += 1
    ah, uh = ai[p + 1:a0], ui[p + 1:u0]
    i = 0
    while i < min(len(ah), len(uh)) and ah[i] == uh[i]:
        i += 1
    j = 0
    while j < min(len(ah), len(uh)) - i and ah[-1 - j] == uh[-1 - j]:
        j += 1
    return list(range(p + 1 + i, p + 1 + len(ah) - j))
'''
k = src.index('\n\ndef common_body(spans_ids):')
src = src[:k] + HELPER + src[k:]
assert "args.mask_extra" in src and src.count('def extra_header') == 1
open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write(src)
print('written', sys.argv[2], len(src))
