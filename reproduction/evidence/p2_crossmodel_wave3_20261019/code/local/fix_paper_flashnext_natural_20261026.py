"""User 10-26: one basis for every number. Section 4.5 quoted the natural-order single pass of Qwen3.8-Flash-Next as P("you") 0.13
(own) vs 0.09 (User), the mean over all three option orders; every other non-thinking number of the paper uses the first two orders
(check_crossmodel_hand_numbers_20261026.py). Over the first two orders both are 0.07 (the three-order excess comes from the third
order alone, where "you" is the last option). The sentence's claim, no self-reference in natural order, is unchanged.
The script recomputes the pair from the per-body records and stops unless each replaced phrase occurs exactly once.
usage: fix_paper_flashnext_natural_20261026.py ARCHIVE8 PAPER_SOURCE_DIR
  (ARCHIVE8 = evidence/p2_crossmodel_8models_20261018)"""
import glob
import json
import sys
from pathlib import Path

A8, D = Path(sys.argv[1]), Path(sys.argv[2])
recs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(str(A8 / 'models/qwen38_flashnext/non_thinking/natural/role150_*.json')))]
assert len(recs) == 150, len(recs)
p = {r: sum(sum(x['roles'][r]['by_order'][o]['self'] for o in (0, 1)) / 2 for x in recs) / len(recs) for r in ('assistant', 'user')}
own, usr = f"{p['assistant']:.2f}", f"{p['user']:.2f}"
print('first two orders: own', round(p['assistant'], 4), 'User', round(p['user'], 4))
assert own == usr == '0.07', (own, usr)

REP = {
    'self_en_body.tex': [("(probability of ``you'' 0.13 for its own messages and 0.09 for User messages)",
                          f"(probability of ``you'' {own} for its own messages and {usr} for User messages)")],
    'self_zh_body.tex': [('自己的消息为 0.13，User 消息为 0.09', f'自己的消息与 User 消息都为 {own}')],
}
for name, reps in REP.items():
    f = D / name
    t = f.read_text(encoding='utf-8')
    for a, b in reps:
        assert t.count(a) == 1, (name, a[:60], t.count(a))
        t = t.replace(a, b)
    f.write_text(t, encoding='utf-8')
    print(name, len(reps), 'replaced')
