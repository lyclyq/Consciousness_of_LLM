"""Scan of the paper data packages for strings that identify the authors, their machines or their location (11-14, user: "这个肯定
得去"; the submission is double-blind and the release is public). Read-only: nothing is changed. Every non-binary file of the
given package folders is read as text (UTF-8, errors ignored) and searched for the patterns below. Output: a JSON report with, per
pattern, the number of files and of matches, split by top-level archive and by file extension, and up to 12 example lines.
Binary formats (.npz, .npy, .pt, .safetensors, .png, .pdf, .whl, .gz, .zip, .tar) are listed by count only, they are not searched.
usage (from projects/consciousness): scan_identity_20261114.py OUT_JSON PACKAGE_DIR [PACKAGE_DIR ...]"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 11-15: the patterns themselves (names, addresses, user names) are not written in this script: they are read from a local
# file that is not part of the release, so the released copy of this script holds none of them.
PATTERNS = json.loads(Path('release/_local/identity_patterns.json').read_text(encoding='utf-8'))['patterns']   # {name: regex}
RX = {k: re.compile(v, re.I if k in ('secrets', 'location', 'email') else 0) for k, v in PATTERNS.items()}
BIN = {'.npz', '.npy', '.pt', '.safetensors', '.png', '.pdf', '.whl', '.gz', '.zip', '.tar', '.jpg', '.bin'}


def lp(p):
    a = os.path.abspath(p)
    return a if a.startswith('\\\\?\\') or os.name != 'nt' else '\\\\?\\' + a


def scan(path):
    try:
        with open(lp(path), 'rb') as f:
            t = f.read().decode('utf-8', 'ignore')
    except OSError as e:
        return path, {'_error': str(e)}
    out = {}
    for k, rx in RX.items():
        ms = list(rx.finditer(t))
        if ms:
            m = ms[0]
            a, b = t.rfind('\n', 0, m.start()) + 1, t.find('\n', m.end())
            out[k] = (len(ms), t[a:b if b >= 0 else len(t)][:200])
    return path, out


def main():
    out, roots = sys.argv[1], sys.argv[2:]
    files, binary = [], Counter()
    for r in roots:
        for d, _, fs in os.walk(lp(r)):
            for f in fs:
                p = os.path.join(d, f)[4:] if os.name == 'nt' else os.path.join(d, f)
                ext = os.path.splitext(f)[1].lower()
                if ext in BIN:
                    binary[ext] += 1
                else:
                    files.append(p)
    print('text files', len(files), 'binary', dict(binary), flush=True)
    rep = {k: {'files': 0, 'matches': 0, 'by_archive': Counter(), 'by_ext': Counter(), 'examples': []} for k in PATTERNS}
    hit_files = defaultdict(list)
    base = os.path.abspath('.')
    with ThreadPoolExecutor(8) as ex:
        for i, (p, res) in enumerate(ex.map(scan, files, chunksize=64)):
            rel = os.path.relpath(p, base).replace('\\', '/')
            for k, v in res.items():
                if k == '_error':
                    continue
                n, line = v
                r = rep[k]
                r['files'] += 1
                r['matches'] += n
                parts = rel.split('/')
                r['by_archive']['/'.join(parts[1:4])] += 1
                r['by_ext'][os.path.splitext(rel)[1].lower() or '(none)'] += 1
                if len(r['examples']) < 12:
                    r['examples'].append([rel[-110:], line.strip()[:160]])
                hit_files[k].append(rel)
            if i % 20000 == 0:
                print('scanned', i, flush=True)
    for r in rep.values():
        r['by_archive'] = dict(r['by_archive'].most_common(12))
        r['by_ext'] = dict(r['by_ext'].most_common(8))
    Path(out).write_text(json.dumps({'roots': roots, 'text_files': len(files), 'binary_files': dict(binary), 'patterns': PATTERNS,
                                     'report': rep}, ensure_ascii=False, indent=1), encoding='utf-8')
    Path(out).with_suffix('.files.json').write_text(json.dumps(hit_files, ensure_ascii=False), encoding='utf-8')
    for k, r in rep.items():
        print(f"{k:14s} files {r['files']:6d}  matches {r['matches']:7d}  ext {r['by_ext']}")


if __name__ == '__main__':
    main()
