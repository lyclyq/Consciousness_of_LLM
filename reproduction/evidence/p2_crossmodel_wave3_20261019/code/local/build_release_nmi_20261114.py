"""Public release of the NMI paper's data and code (11-14; user: upload GitHub and Zenodo after the arXiv version, intermediate
model states not deposited). Source: the two paper-data packages (evidence/paper_data_v5 and evidence/paper_data_v6_additions_20261113,
the latter taking precedence), read only. Paths inside the release are the project paths, so that every script runs unchanged
from the release root.
Excluded:
  - intermediate model states (.npz except the Gram extracts of the share kept under */extracts/*): per-text geometry states,
    per-text state differences of the share kept, other state arrays,
  - the fit files of the four further random seeds (fit_seed1..4.pt): they repeat the fitted directions of fit_q.pt and add
    random subspaces that the runners draw again from the recorded seed,
  - internal working logs (README_zh.md, PREREG_zh.md, KL_INDEX_zh.md and other *_zh.md notes of the archives).
  - the figures of the long version of the paper (paper/source/figures/, 11-15): no released script reads them.
Every text file is scrubbed of strings that identify the machines or the local account (SCRUB, read from a local file that is not released). The scrubbed files are
listed with both checksums in SCRUBBED.csv. Results are not affected: the strings occur in paths, host names and run scripts.
11-15: the text members of .tgz archives are scrubbed too (scrub_tgz), and the release adds the files of EXTRA (the table-only
copies of the long version that five released scripts parse) and the stage-1 extracts at the stage-2 paths (collect).
Outputs, under release/p2_nmi_20261115/:
  tree/      the full release tree (text records, Gram extracts, fitted directions), with MANIFEST_sha256.csv
  github/    the part for the GitHub repository: code, items, statistics and aggregated results (no per-item records)
  zenodo/    README.md, LICENSE files, MANIFEST_sha256.csv and three zip files: records (all text files), share-kept Gram
             extracts, fitted directions
usage (from projects/consciousness): build_release_nmi_20261114.py [--dry-run]"""
import copy
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

PK = [Path('evidence/paper_data_v5'), Path('evidence/paper_data_v6_additions_20261113')]
# 11-15: files written for the release, at their project paths: the table-only copies of the long version that five released
# scripts parse (paper/source/self_{en,zh}_{body,appendix}.tex, written by paper/scripts/make_release_tables_20261115.py)
EXTRA = Path('release/p2_nmi_extra_20261115')
WA_EXT = ('evidence/p2_words_along_20261110/results/stage1_20261111/extracts/b150/',
          'evidence/p2_words_along_20261110/results/stage2_20261113/extracts/b150/')
OUT = Path('release/p2_nmi_20261115')   # 11-14: a fresh folder (the 11-14 folder was locked by Windows and could not be replaced)
DOCS = Path('release/p2_nmi_docs_20261114')
TEXT = {'.py', '.sh', '.md', '.txt', '.json', '.jsonl', '.csv', '.tsv', '.tex', '.log', '.yaml', '.yml', ''}
INTERNAL = re.compile(r'(^|/)(README_zh|PREREG_zh|KL_INDEX_zh)\.md$|_zh\.md$')
# 11-15: the strings that identify the machines and the local account (home paths, user names, server addresses and ports,
# host names) are not written in this script: they are read from a local file that is not part of the release, so the
# released copy of this script holds none of them. The public GitHub address of the paper is not scrubbed.
LOCAL_SCRUB = Path('release/_local/scrub_patterns.json')   # {'scrub': [[pattern, replacement], ...], 'left': pattern}
_LOCAL = json.loads(LOCAL_SCRUB.read_text(encoding='utf-8'))
SCRUB = [(re.compile(p), r) for p, r in _LOCAL['scrub']]   # (pattern, replacement), applied in this order
LEFT = re.compile(_LOCAL['left'])   # any of these left after scrubbing stops the build
GITHUB_DROP = re.compile(r'(^|/)(runs|bodies|items|G|extracts|raw_20261113)(/|$)')
GITHUB_KEEP_ITEMS = re.compile(r'^evidence/[^/]+/items/')   # the archives' item files stay in GitHub
# the summaries and identities of every run stay in GitHub too (small, and enough for the statistics that read run summaries);
# the per-item records under items/, bodies/ and G/ are in the Zenodo records only
# 11-15: also the stream geometry files of the two four-stream models (read by kl_vs_lastlayer_geometry_20261114.py)
GITHUB_KEEP_SUMMARY = re.compile(r'(^|/)(_run_summary|summary[^/]*|identity[^/]*|[^/]*stream_geometry[^/]*)\.json$')
GITHUB_PER_ITEM = re.compile(r'(^|/)(bodies|items|G|extracts|raw_20261113)(/|$)')
# 11-15: the two records of the game shown in Fig. 5b (read by fig_game_nmi_20261114.py)
GITHUB_KEEP_FILES = {f'evidence/p2_crossmodel_wave3_20261019/runs/pro/games_swap/gpt_oss_20b_s6_main/items/{c}__secret6_en_000.json'
                     for c in ('base_A', 'base_U')}


def github_part(paths):
    return [x for x in paths if not GITHUB_DROP.search(x[0]) or GITHUB_KEEP_ITEMS.match(x[0]) or x[0] in GITHUB_KEEP_FILES
            or (GITHUB_KEEP_SUMMARY.search(x[0]) and not GITHUB_PER_ITEM.search(x[0]))]


def scrub_text(t):
    for rx, rep in SCRUB:
        t = rx.sub(rep, t)
    return t


def scrub_tgz(b):
    """11-15: the text members of a .tgz are scrubbed like any text file (the scan of 11-14 read only plain files). Returns
    (bytes, changed, leftover): the archive is re-packed only when a member changed, with the members in their order, their
    headers kept apart from the size, and a gzip header without a time stamp."""
    src = tarfile.open(fileobj=io.BytesIO(b), mode='r:gz')
    members, changed, left = [], False, False
    for m in src.getmembers():
        data = src.extractfile(m).read() if m.isfile() else None
        if data is not None and os.path.splitext(m.name)[1].lower() in TEXT:
            t = data.decode('utf-8', 'surrogateescape')
            t2 = scrub_text(t)
            if t2 != t:
                changed, data = True, t2.encode('utf-8', 'surrogateescape')
            left = left or bool(LEFT.search(t2))
        if LEFT.search(m.name) or LEFT.search(m.uname or '') or LEFT.search(m.gname or ''):
            changed, m = True, copy.copy(m)
            m.name, m.uname, m.gname = scrub_text(m.name), '', ''
        members.append((m, data))
    if not changed:
        return b, False, left
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode='wb', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w', format=src.format) as dst:
            for m, data in members:
                if data is None:
                    dst.addfile(m)
                else:
                    m = copy.copy(m)
                    m.size = len(data)
                    dst.addfile(m, io.BytesIO(data))
    return out.getvalue(), True, left


def lp(p):
    a = os.path.abspath(p)
    return a if a.startswith('\\\\?\\') or os.name != 'nt' else '\\\\?\\' + a


def sha(b):
    return hashlib.sha256(b).hexdigest()


def collect():
    files = {}
    for pk in PK:
        with open(pk / 'MANIFEST.csv', encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                files[r['project_path']] = (pk / r['package_path'], int(r['bytes']), r['sha256'])
    for d, _, fs in os.walk(EXTRA):
        for f in fs:
            src = Path(d) / f
            b = src.read_bytes()
            files[src.relative_to(EXTRA).as_posix()] = (src, len(b), sha(b))
    # 11-15: the stage-2 analysis of the share kept (analyze_words_along_20261110.py, pooled_ci_20261114.py) read the 25 units of
    # stage 1 from its own folder, where they are byte-identical copies (148 files, left out of the package as copies). The release
    # puts them back at the stage-2 paths, so that both scripts run on the layout they ran on.
    s1, s2 = WA_EXT
    for q, v in list(files.items()):
        if q.startswith(s1):
            files.setdefault(s2 + q[len(s1):], v)
    return files


def kind(path):
    ext = os.path.splitext(path)[1].lower()
    if INTERNAL.search(path):
        return 'internal'
    if path.startswith('generated/') or ext == '.part':   # v5's generated stand-in for a helper (not needed since v6); partial downloads
        return 'dropped'
    if path.startswith('paper/source/figures/'):   # 11-15: the figures of the long version (read by no released script, the long version is not released)
        return 'dropped'
    if ext == '.npz':
        return 'gram' if '/extracts/' in path else 'state'
    if ext == '.pt':
        return 'seedfit' if re.search(r'fit_seed\d\.pt$', path) else 'fit'
    if ext in TEXT:
        return 'text'
    return 'other'


def main():
    dry = '--dry-run' in sys.argv
    files = collect()
    groups = {}
    for p, (src, n, h) in files.items():
        groups.setdefault(kind(p), []).append((p, src, n, h))
    for k, v in sorted(groups.items()):
        print(f'{k:9s} {len(v):7d} files {sum(x[2] for x in v) / 1e9:7.2f} GB')
    keep = {k: v for k, v in groups.items() if k in ('text', 'gram', 'fit', 'other')}
    gh = github_part(keep['text'] + keep.get('other', []))
    print('github', len(gh), 'files', round(sum(x[2] for x in gh) / 1e6, 1), 'MB')
    if dry:
        return
    if '--github-only' in sys.argv:   # rebuild only the GitHub part from an existing tree, into github/ (removed first)
        tree, ghd = OUT / 'tree', OUT / 'github'
        if ghd.exists():   # remove the contents, not the folder (Windows keeps a folder locked while a shell sits in it)
            for c in ghd.iterdir():
                shutil.rmtree(lp(c)) if c.is_dir() else c.unlink()
        for p, src, n, h in gh:
            os.makedirs(lp((ghd / p).parent), exist_ok=True)
            shutil.copyfile(lp(tree / p), lp(ghd / p))
        for f in sorted(DOCS.iterdir()):
            shutil.copyfile(f, ghd / f.name)
        print('github part rebuilt', len(gh), 'files')
        return
    tree = OUT / 'tree'
    if OUT.exists():
        shutil.rmtree(lp(OUT))
    man, scrubbed, left = [], [], []
    for k in ('text', 'other', 'gram', 'fit'):
        for p, src, n, h in sorted(keep.get(k, [])):
            dst = tree / p
            os.makedirs(lp(dst.parent), exist_ok=True)
            b = open(lp(src), 'rb').read()
            assert sha(b) == h, ('package copy differs from its manifest', p)
            if k == 'text':
                t = b.decode('utf-8', 'surrogateescape')
                t2 = scrub_text(t)
                if t2 != t:
                    b2 = t2.encode('utf-8', 'surrogateescape')
                    scrubbed.append((p, h, sha(b2)))
                    b = b2
                if LEFT.search(t2):   # checked here, on the bytes that are written, instead of reading the tree again
                    left.append(p)
            elif p.endswith(('.tgz', '.tar.gz')):
                b2, changed, lft = scrub_tgz(b)
                if changed:
                    scrubbed.append((p, h, sha(b2)))
                    b = b2
                if lft:
                    left.append(p)
            open(lp(dst), 'wb').write(b)
            man.append((p, len(b), sha(b), k))
    with open(tree / 'MANIFEST_sha256.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'bytes', 'sha256', 'kind'])
        w.writerows(man)
    with open(OUT / 'SCRUBBED.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'sha256_in_package', 'sha256_released'])
        w.writerows(scrubbed)
    # leftover identifying strings: none may remain in the released text
    assert not left, left[:5]
    print('tree written', len(man), 'files, no identifying string left', flush=True)
    # GitHub part
    ghd = OUT / 'github'
    for p, src, n, h in gh:
        os.makedirs(lp((ghd / p).parent), exist_ok=True)
        shutil.copyfile(lp(tree / p), lp(ghd / p))
    # Zenodo zips
    zd = OUT / 'zenodo'
    zd.mkdir(parents=True)
    for name, ks in (('p2_records.zip', ('text', 'other')), ('p2_share_kept_gram.zip', ('gram',)), ('p2_fitted_directions.zip', ('fit',))):
        with zipfile.ZipFile(lp(zd / name), 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for p, n, h, k in man:
                if k in ks:
                    zf.write(lp(tree / p), p)
        print(name, round(os.path.getsize(lp(zd / name)) / 1e9, 3), 'GB')
    shutil.copyfile(tree / 'MANIFEST_sha256.csv', zd / 'MANIFEST_sha256.csv')
    shutil.copyfile(OUT / 'SCRUBBED.csv', zd / 'SCRUBBED.csv')
    # the reader-facing documents (README, FIGURES, licences), written by hand in release/p2_nmi_docs_20261114
    for d in (tree, ghd, zd):
        for f in sorted(DOCS.iterdir()):
            shutil.copyfile(f, d / f.name)
    print('written', len(man), 'files,', len(scrubbed), 'scrubbed')


if __name__ == '__main__':
    main()
