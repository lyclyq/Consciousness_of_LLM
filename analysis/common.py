"""Shared helpers of the analysis scripts.

Paths. Every script reads the results package (--results, the folder that holds evidence/ and paper/) and, where it needs
per-layer feature arrays, the feature package (--features, same layout). It writes into an output folder (--out) under the
same relative paths as the archived outputs, e.g. OUT/evidence/p2_crossmodel_wave3_20261019/stats/v4_mechanism.json or
OUT/paper/source/figures/self_mechanism.pdf, so that a run can be compared file by file with the package. Outputs that a later
script reads (for example v4_mechanism.json) are read from the output folder, never from the package.

Run folders. The runs of an archive sit in runs/<group>/<family>/<run>; the group only records where a run was made. A run is
addressed here by its path below the group (for example 'thinking_swap/gpt_oss_20b_all') and found under whichever group holds
it (run_path). The next-token archive (MS) and the wording archive (WS) keep their runs in results/<group>/<family>/<run>
instead; result_path finds them the same way.

Statistics shared by the causal tests (items, wilson, boot, val, summarize): per-item records of a run; Wilson intervals for
shares; percentile bootstrap over items for means; the item-paired difference from the unedited message of the same role.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

# the printed tables contain arrows and Chinese labels; print them as UTF-8 whatever the console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# archives inside the results and feature packages
W3 = 'evidence/p2_crossmodel_wave3_20261019'      # cross-model comparison (main runs, wave 3) and the causal tests
M8 = 'evidence/p2_crossmodel_8models_20261018'    # cross-model comparison, first eight models
SC = 'evidence/p2_small_checks_20261014'          # Qwen3.8-27B: the two-level analyses
TL = SC + '/results/two_level_20261015'          # message geometry (L), thinking (N), reasoning (M) summaries
C150 = SC + '/results/compliant_150_20261015'    # native renderings, 150 bodies (J)
MS = 'evidence/p2_modeswitch_20261107'            # next-token (KL) runs; the replacement runs of MiMo-V2.6-Distill-Qwen-9B
WS = 'evidence/p2_wording_swap_20261109'          # wording variants of the attribution question in the causal tests
FIGURES = 'paper/source/figures'


def longpath(p):
    """Absolute form of p; on Windows the extended-length form, so that run folders deeper than 260 characters can be read."""
    s = os.path.abspath(str(p))
    if os.name == 'nt' and not s.startswith('\\\\?\\'):
        s = '\\\\?\\UNC\\' + s[2:] if s.startswith('\\\\') else '\\\\?\\' + s
    return Path(s)


def shown(p):
    """p without the extended-length prefix (for messages and for paths recorded in an output)."""
    s = str(p)
    if s.startswith('\\\\?\\UNC\\'):
        return '\\\\' + s[8:]
    return s[4:] if s.startswith('\\\\?\\') else s


def arguments(doc, features=False, item_cache=False, results=True):
    """Command line shared by the scripts: --results (and --features) to read, --out to write."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=results, help='root of the results package (the folder that holds evidence/)')
    if features:
        ap.add_argument('--features', help='root of the feature package (per-layer arrays, same layout as the results package)')
    ap.add_argument('--out', required=True, help='output folder; outputs keep the relative paths of the archived ones')
    if item_cache:
        ap.add_argument('--item-cache', help='optional folder for a cache of the per-item records of each run (see items)')
    return ap


class Roots:
    """RES (results package), OUT (output folder), FEAT (feature package or None) from parsed arguments."""

    def __init__(self, args):
        global ITEM_CACHE
        self.RES = longpath(args.results) if args.results else None
        self.OUT = Path(os.path.abspath(args.out))  # a plain path: figures and LaTeX runs write here (keep it short on Windows)
        self.FEAT = longpath(args.features) if getattr(args, 'features', None) else None
        if getattr(args, 'item_cache', None):
            ITEM_CACHE = os.path.abspath(args.item_cache)

    def res(self, rel):
        return self.RES / rel

    def out(self, rel, mkdir=True):
        p = self.OUT / rel
        if mkdir:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def feat(self, rel):
        if self.FEAT is None:
            return None
        return self.FEAT / rel


def prefer(r, rel):
    """OUT/rel if an earlier script of this run wrote it, else the archived copy in the results package (for summaries that
    need the feature package to be recomputed)."""
    p = r.OUT / rel
    return p if p.exists() else r.RES / rel


def need(paths, what):
    """None if every path exists, else a message naming the inputs that are not in the packages."""
    miss = [shown(p) for p in paths if p is None or not Path(p).exists()]
    return None if not miss else f'{what}: needs inputs not in the package: ' + '; '.join(miss)


def run_path(archive, rel):
    """runs/<group>/<rel> of an archive, whichever group holds it; a path that does not exist if no group does."""
    hits = sorted(glob.glob(str(Path(archive) / 'runs' / '*' / rel)))
    if len(hits) > 1:
        raise RuntimeError(f'{rel} is found under more than one group: {hits}')
    return Path(hits[0]) if hits else Path(archive) / 'runs' / '_not_in_package' / rel


def run_glob(archive, pattern):
    """Sorted paths matching runs/<group>/<pattern> over all groups (as strings, like glob.glob)."""
    return sorted(glob.glob(str(Path(archive) / 'runs' / '*' / pattern)))


def result_path(archive, rel):
    """results/<group>/<rel> of the next-token and wording archives (MS, WS), whichever group holds it; these archives keep their
    runs under results/<group>/ instead of runs/<group>/. A path that does not exist if no group does."""
    hits = sorted(glob.glob(str(Path(archive) / 'results' / '*' / rel)))
    if len(hits) > 1:
        raise RuntimeError(f'{rel} is found under more than one group: {hits}')
    return Path(hits[0]) if hits else Path(archive) / 'results' / '_not_in_package' / rel


# ---------------------------------------------------------------------------------------------------------------------------
# per-item records and the statistics of the causal tests

ITEM_CACHE = None


def items(d, cache=True):
    """Per-item records of a run (items/*.json, sorted). Reading tens of thousands of item files is slow on Windows; with
    --item-cache the records of each run (without string fields over 2,000 characters, i.e. the generated text, which no
    statistic based on these records reads) are cached there, keyed by the run path and refreshed when the number or the
    newest time of its item files changes. cache=False always reads the item files (for statistics of the generated text)."""
    fs = sorted(glob.glob(os.path.join(d, 'items', '*.json')))
    c = ITEM_CACHE if cache else None
    if not c:
        return [json.load(open(f, encoding='utf-8')) for f in fs]
    sig = [len(fs), max((os.path.getmtime(f) for f in fs), default=0.0)]
    p = os.path.join(c, hashlib.sha1(os.path.abspath(d).encode()).hexdigest() + '.json')
    if os.path.exists(p):
        cc = json.load(open(p, encoding='utf-8'))
        if cc['sig'] == sig:
            return cc['recs']
    recs = [{k: v for k, v in json.load(open(f, encoding='utf-8')).items() if not (isinstance(v, str) and len(v) > 2000)} for f in fs]
    os.makedirs(c, exist_ok=True)
    json.dump({'path': os.path.abspath(d), 'sig': sig, 'recs': recs}, open(p, 'w', encoding='utf-8'))
    return recs


BASE = {'assistant': 'base_A', 'user': 'base_U', 'system': 'base_S'}


def wilson(k, n):
    if n == 0:
        return (float('nan'),) * 2
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def boot(v, reps=4000, seed=0):
    rng = random.Random(seed)
    m = [sum(rng.choice(v) for _ in v) / len(v) for _ in range(reps)]
    m.sort()
    return m[int(0.025 * reps)], m[int(0.975 * reps) - 1]


def val(r):
    return r['p']['self'] if 'p' in r else float(r.get('on_answer') == 'self')


def summarize(recs, key):
    """Per condition: n, the statistic with a 95% interval (single pass: mean P("you"), bootstrap over items; thinking: share
    of finished items answered "you", Wilson; content questions: share answered correctly) and, for edited conditions, the
    item-paired difference from the unedited condition of the same recipient (pairs: body x option order; with thinking, items
    finished in both) with a 95% bootstrap interval."""
    out = {}
    for c in sorted({r['cond'] for r in recs}):
        rs = [r for r in recs if r['cond'] == c]
        if 'p' in rs[0]:
            v = [r['p']['self'] for r in rs]
            m = sum(v) / len(v)
            lo, hi = boot(v)
            s = {'n': len(rs), 'stat': m, 'ci': [lo, hi], 'p_user': sum(r['p']['user'] for r in rs) / len(rs),
                 'p_system': sum(r['p']['system'] for r in rs) / len(rs)}
        else:
            fin = [r for r in rs if r.get('finished', True)]
            k = sum(r.get('on_answer') == 'self' for r in fin)
            s = {'n': len(rs), 'finished': len(fin), 'stat': k / max(1, len(fin)), 'ci': list(wilson(k, len(fin))),
                 'answers': {a: sum(r.get('on_answer') == a for r in fin) for a in ('self', 'user', 'system', None)}}
            if rs[0].get('content'):
                s = {'n': len(rs), 'finished': len(fin), 'accuracy': sum(r['content']['correct'] for r in fin) / max(1, len(fin))}
        out[c] = s
    for c, s in out.items():
        rs = [r for r in recs if r['cond'] == c]
        b = BASE.get(rs[0]['recipient'])
        if b is None or b == c or b not in out or 'accuracy' in s:
            continue
        bb = {(r['id'], tuple(r['order'])): r for r in recs if r['cond'] == b}
        d = [val(r) - val(bb[(r['id'], tuple(r['order']))]) for r in rs if (r['id'], tuple(r['order'])) in bb
             and r.get('finished', True) and bb[(r['id'], tuple(r['order']))].get('finished', True)]
        if d:
            s['diff'] = sum(d) / len(d)
            s['diff_ci'] = list(boot(d, seed=1))
            s['n_pairs'] = len(d)
    return out


def write_text(path, text):
    """Write a text output (UTF-8), creating its folder."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding='utf-8')
