"""Trust in the model's own steps (the reasoning task, 120 problems), every model and mode in the wave-3 archive, for appendix B of
paper v4. Single pass (suite M): mean adoption index of wrong steps in Assistant history minus that of the same steps from the user,
with a 95% bootstrap interval over problems (as tab:self_ownstep), plus the accuracy-relevant correct-step means. Thinking (thinking M):
share of finished wrong-step items whose answer adopts the error, own minus user, item-paired by problem (both finished), 95%
bootstrap interval over problems; and the share of finished correct-step items answered correctly, per source.
Output: WAVE3_ARCHIVE/stats/v4_ownstep.csv
With ARCHIVE8 (the eight-model archive), its single-pass runs (models/<model>/non_thinking/reason) are included under the same rule.
usage: ownstep_v4.py WAVE3_ARCHIVE [ARCHIVE8]"""
import csv
import glob
import json
import random
import sys
from pathlib import Path

A = Path(sys.argv[1])
R = A / 'runs'
SUITE = [(d, d + '/M') for d in sorted(glob.glob(str(R / 'server3/suite/*_M')))]
if len(sys.argv) > 2:
    SUITE += [(str(Path(d).parent.parent) + '_8models', d) for d in sorted(glob.glob(str(Path(sys.argv[2]) / 'models' / '*' / 'non_thinking' / 'reason')))]
THINK = sorted(glob.glob(str(R / '*/thinking_m/*')) + glob.glob(str(R / 'api/*_thinkm')))


def boot(d, reps=4000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(reps))
    return m[int(0.025 * reps)], m[int(0.975 * reps) - 1]


rows = []
for d, md in SUITE:
    probs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(md + '/m*.json'))]
    if not probs:
        continue
    g = lambda p, s, r: next(x['adoption_index'] for x in p['results'] if x['step'] == s and x['role'] == r)
    diff = [g(p, 'wrong', 'assistant') - g(p, 'wrong', 'user') for p in probs]
    lo, hi = boot(diff)
    rows.append(dict(model=Path(d).name, mode='single_pass', n=len(probs), own=round(sum(g(p, 'wrong', 'assistant') for p in probs) / len(probs), 4),
                     user=round(sum(g(p, 'wrong', 'user') for p in probs) / len(probs), 4), own_minus_user=round(sum(diff) / len(diff), 4),
                     ci_lo=round(lo, 4), ci_hi=round(hi, 4), correct_own='', correct_user='', unfinished=''))
for d in THINK:
    if 'smoke' in Path(d).name:
        continue
    its = [json.load(open(f, encoding='utf-8')) for f in glob.glob(d + '/items/*.json')]
    if not its:
        continue
    by = {(x['id'], x['step'], x['role']): x for x in its}
    ids = sorted({x['id'] for x in its})
    adopt = lambda x: float(x.get('class') == 'adopted')
    pairs = [(by[(i, 'wrong', 'assistant')], by[(i, 'wrong', 'user')]) for i in ids
             if (i, 'wrong', 'assistant') in by and (i, 'wrong', 'user') in by
             and by[(i, 'wrong', 'assistant')].get('finished') and by[(i, 'wrong', 'user')].get('finished')]
    diff = [adopt(a) - adopt(u) for a, u in pairs]
    lo, hi = boot(diff) if diff else (float('nan'),) * 2
    fin = lambda s, r: [x for x in its if x['step'] == s and x['role'] == r and x.get('finished')]
    share = lambda xs, c: round(sum(x.get('class') == c for x in xs) / max(1, len(xs)), 4)
    rows.append(dict(model=Path(d).name, mode='thinking', n=len(pairs), own=share(fin('wrong', 'assistant'), 'adopted'),
                     user=share(fin('wrong', 'user'), 'adopted'), own_minus_user=round(sum(diff) / max(1, len(diff)), 4),
                     ci_lo=round(lo, 4), ci_hi=round(hi, 4), correct_own=share(fin('correct', 'assistant'), 'correct'),
                     correct_user=share(fin('correct', 'user'), 'correct'),
                     unfinished=sum(not x.get('finished') for x in its)))
with open(A / 'stats' / 'v4_ownstep.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['model']:28s} {r['mode']:11s} n={r['n']:3d} own={r['own']:.3f} user={r['user']:.3f} diff={r['own_minus_user']:+.3f} "
          f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] corr={r['correct_own']}/{r['correct_user']} unf={r['unfinished']}")
