"""Qwen3.8-27B, self-reference with and without thinking (check N; same items, template non-thinking vs thinking mode), from the
per-item records in N_thinking_items_900.tgz of the results package (extracted into OUT/_work).

Per role: mean P(self), P(user), P(system) without thinking (single forward) and with thinking (answer position after the
model's own reasoning); accuracy = probability of the true author (Assistant -> you, User -> the user, System -> the system).
Paired change with thinking (on - off) and 95% bootstrap interval over bodies (10,000 resamples, seed 0). For Assistant items:
Spearman correlation between the no-thinking P(self) and the gain, i.e. whether thinking helps most where single-pass
self-reference is weakest. Items present in several record folders count once (the first in sorted path order).
Output: summaries/N_thinking_final_900items.json of the two-level archive.

usage: two_level_thinking.py --results RES --out OUT
"""
import glob
import json
import os
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TL, Roots, arguments, need  # noqa: E402

TRUE = {'assistant': 'p_self', 'user': 'p_user', 'system': 'p_system'}


def boot(by_body, seed=0, n=10000):
    keys = sorted(by_body)
    vals = np.array([np.mean(by_body[k]) for k in keys])
    rng = np.random.default_rng(seed)
    m = vals[rng.integers(0, len(vals), (n, len(vals)))].mean(1)
    return float(vals.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main_(out_json, *patterns):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    items = {}
    for f in files:
        r = json.load(open(f, encoding='utf-8'))
        items.setdefault((r['id'], r['role'], r['order']), r)
    res = {'n_items': len(items)}
    print(f'items {len(items)}')
    for role in ('assistant', 'user', 'system'):
        rs = [r for (b, ro, o), r in items.items() if ro == role]
        ok = [r for r in rs if 'p_self' in r['on']]
        unfinished = 1 - len(ok) / max(1, len(rs))
        out = {'n': len(rs), 'n_with_answer': len(ok), 'unfinished_share': unfinished,
               'thinking_tokens_median': float(np.median([r['on'].get('thinking_tokens', np.nan) for r in ok])) if ok else None}
        for k in ('p_self', 'p_user', 'p_system'):
            out[f'off_{k}'] = float(np.mean([r['off'][k] for r in ok]))
            out[f'on_{k}'] = float(np.mean([r['on'][k] for r in ok]))
        t = TRUE[role]
        gain = defaultdict(list)
        for r in ok:
            gain[r['id']].append(r['on'][t] - r['off'][t])
        out['accuracy_off'] = out[f'off_{t}']
        out['accuracy_on'] = out[f'on_{t}']
        out['accuracy_gain'], out['accuracy_gain_ci'] = boot(gain)
        out['answer_share_on'] = {L: float(np.mean([r['on'].get('answer_letter') == L for r in ok])) for L in 'ABC'}
        if role == 'assistant' and ok:
            off = np.array([r['off']['p_self'] for r in ok]); on = np.array([r['on']['p_self'] for r in ok])
            out['rho_off_vs_gain'] = spearman(off, on - off)
            out['share_claimed_off'] = float((off > 0.5).mean()); out['share_claimed_on'] = float((on > 0.5).mean())
            lo = off < np.median(off)
            out['gain_lower_half_off'] = float((on - off)[lo].mean()); out['gain_upper_half_off'] = float((on - off)[~lo].mean())
        res[role] = out
        print(f"{role:9s} n={len(rs)} answered={len(ok)} unfinished={unfinished:.3f} median think tokens={out['thinking_tokens_median']}")
        print(f"   off: self {out['off_p_self']:.3f} user {out['off_p_user']:.3f} system {out['off_p_system']:.3f} | "
              f"on: self {out['on_p_self']:.3f} user {out['on_p_user']:.3f} system {out['on_p_system']:.3f}")
        print(f"   accuracy (P of the true author) off {out['accuracy_off']:.3f} -> on {out['accuracy_on']:.3f}; gain {out['accuracy_gain']:+.3f} {out['accuracy_gain_ci']}")
        if role == 'assistant' and ok:
            print(f"   claimed (P>0.5) off {out['share_claimed_off']:.3f} on {out['share_claimed_on']:.3f}; rho(off P(self), gain) {out['rho_off_vs_gain']:+.3f}; "
                  f"gain lower half {out['gain_lower_half_off']:+.3f} upper half {out['gain_upper_half_off']:+.3f}")
    json.dump(res, open(out_json, 'w'), indent=1)


def main(argv=None):
    args = arguments(__doc__).parse_args(argv)
    r = Roots(args)
    tgz = r.res(TL + '/N_thinking_items_900.tgz')
    msg = need([tgz], 'two_level_thinking')
    if msg:
        print(msg)
        return
    dest = r.out('_work/N_thinking')
    with tarfile.open(tgz) as T:  # items/*.json of each record folder of the archive
        for mem in T.getmembers():
            if mem.isfile() and re.match(r'^[^/]+/items/[^/]+\.json$', mem.name):
                p = dest / mem.name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(T.extractfile(mem).read())
    main_(str(r.out(TL + '/summaries') / 'N_thinking_final_900items.json'), str(dest / '*' / 'items' / '*.json'))


if __name__ == '__main__':
    main()
