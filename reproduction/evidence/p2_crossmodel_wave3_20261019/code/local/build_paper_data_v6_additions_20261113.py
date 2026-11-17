r"""Additions to the paper-data package for the final version (11-13): everything of the 11-11 .. 11-13 round that
evidence/paper_data_v5/ (11-10) does not hold, in one new folder evidence/paper_data_v6_additions_20261113/, with the path rule of
v4/v5 (package path = project path). The full v6 is v5 overlaid with this folder; it is built once the final tex fixes the
element labels. v5 and the archives are only read.

User decision of 11-13 ("必须写到能复现，并且给全部的代码和原始数据"): the retention and ordinary-word work is no longer withheld.
So the three archives that v5 never referenced (NEVER in build_paper_data_v5_20261110.py) come in whole, and the files v5 listed in
PRUNED.csv (helper script, archive logs, KL queue scripts, checksum lists) are restored.

Included (ROOTS), minus EXCLUDE: folders _sealed_*, _not_used, tmp, _smoke*, folders marked NOT_USED.txt, *.GO markers, and
  - the earlier H200 pulls (duplicates of the continuous mirror m/; only the fits come from pull_20261112T2227, the mirror skips them),
  - results/kl_20261111/pro (the 11-11 mirror of 16 settings, contained in results/kl_20261113_pro),
  - results/stage2_20261113/extracts/b150 except the GLM-5.3 unit (the rest is a copy of stage 1).
A file whose project path is in v5's MANIFEST with the same sha256 is skipped; with another sha256 it is copied and marked
updated_since_v5 (archive notes that kept growing).
Outputs in the folder: MANIFEST.csv (package_path, project_path, bytes, sha256 of the copy, status), INDEX_additions.csv
(provisional elements: evidence line -> results, run code, aggregation code, items), SKIPPED.csv (excluded folders and why).
README_zh.md is written by hand.
11-14 extension (user: "按照之前的规范 整理到 paper 需要的合并目录下 包括 code 也是"), same rules, more roots:
  - results/vast_h200_20261114 (GLM-5.3 secret game under the second System prompt, rented 4 x H200 NVL, smoke + main n = 50),
  - results/pro_game_v6_20261114 (MiMo-V2.6 and EXAONE smokes under the second System prompt on PRO, not released),
  - results/figs (ablation figure draft),
  - the secret game of the four mechanism readouts under the first System prompt (wave3 runs/*/games_swap/*_s5_*, 16 folders):
    the user decided 11-14 to report both prompts, so their NOT_USED markers of 10-30 were lifted (kept as
    NOT_USED_20261030_lifted_20261114.txt); Gemma's first-prompt game stays out (its NOT_USED is the 10-29 mechanism-model rule);
    of gpt-oss-120b only the eight-game check comes in: its game does not pass the release rule under the first prompt, and
    the two full-game folders carry a new NOT_USED.txt (user 11-14: one rule for every game, raw data out of the package too);
  the raw per-body retention files pulled on 11-14 (WA/results/raw_20261113) come in with the WA root. The cancelled H200 side
  environment (code/vast_h200_20261111/proenv_20261114) carries NOT_USED.txt. The build is incremental: a file whose source has the
  sha256 the package MANIFEST already records for it, and whose copy exists, is not copied again (status kept); package copies that
  no candidate file produces any more are listed (--dry-run) and removed. The 11-13 version is in tmp/retired_20261114_paperdata_builders.
usage (from projects/consciousness):  build_paper_data_v6_additions_20261113.py [--dry-run] [--workers N]"""
import csv
import hashlib
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = 'evidence/paper_data_v6_additions_20261113'
V5 = 'evidence/paper_data_v5'
WA = 'evidence/p2_words_along_20261110'
MS = 'evidence/p2_modeswitch_20261107'
H2 = MS + '/results/vast_h200_20261111'
ROOTS = [WA, 'evidence/p2_persistence_20261106', 'evidence/p2_selfspace_20261106',
         MS + '/code', MS + '/README_zh.md', MS + '/PREREG_zh.md', MS + '/KL_INDEX_zh.md', MS + '/SHA256SUMS_pro_20261108.txt',
         MS + '/results/pro/SHA256SUMS_pro_20261108_raw.txt', MS + '/results/pro_exaone32_20261112',
         MS + '/results/mech_summary_7readouts_20261113.md',
         H2 + '/README_zh.md', H2 + '/m', H2 + '/final_20261113', H2 + '/pull_20261112T2227/out/mech/fits',
         'evidence/p2_crossmodel_wave3_20261019/README_zh.md',
         MS + '/results/vast_h200_20261114', MS + '/results/pro_game_v6_20261114', MS + '/results/figs',
         'evidence/p2_crossmodel_wave3_20261019/figures_nmi_20261114',
         'evidence/p2_crossmodel_wave3_20261019/code/local/figures/fig_nesting_nmi_20261114.py',
         # 11-14 late: Extended Data, the two-prompt game table, the release and package builders, the scripts of the paper
         'evidence/p2_crossmodel_wave3_20261019/code/local/figures/fig_ed_onset_size_nmi_20261114.py',
         'evidence/p2_crossmodel_wave3_20261019/code/local/figures/fig_ed_windows_nmi_20261114.py',
         'evidence/p2_crossmodel_wave3_20261019/code/local/figures/fig_ed_design_nmi_20261114.py',
         'evidence/p2_crossmodel_wave3_20261019/code/local/figures/fig_ed_readout_nmi_20261115.py',   # 11-15: Extended Data Fig. 6
         'evidence/p2_crossmodel_wave3_20261019/code/local/build_release_nmi_20261114.py',
         'evidence/p2_crossmodel_wave3_20261019/code/local/scan_identity_20261114.py',
         'evidence/p2_crossmodel_wave3_20261019/code/local/build_paper_data_v6_additions_20261113.py',
         'evidence/p2_crossmodel_wave3_20261019/code/local/write_v6_additions_index_20261113.py',
         MS + '/results/game_two_prompts_20261114.json', MS + '/results/game_two_prompts_20261114.md',
         MS + '/results/random_bound_all_seeds_20261115.json', MS + '/results/random_bound_all_seeds_20261115.md',   # 11-15: the random control over all seeds
         # 11-15: classify_groups_by_rule_20261115.py, threshold_grid_20261115.py and their results (results/final_20261113) come in
         # with the WA root above
         'paper/scripts/make_overview_nmi_20261114.py', 'paper/scripts/make_supplementary_nmi_20261114.py',
         'paper/scripts/make_arxiv_nmi_20261114.py', 'paper/scripts/count_words_nmi_20261114.py',
         'paper/scripts/make_release_tables_20261115.py',   # 11-15: the table-only copies of the long version for the release
         'evidence/p2_crossmodel_wave3_20261019/code/local/check_crossmodel_hand_numbers_20261103.py',   # 11-15: named in FIGURES.md (Fig. 3)
         'evidence/p2_crossmodel_wave3_20261019/code/local/check_newdev_tables_20261104.py',   # 11-15: own-step check made conditional (in v5 unpatched)
         'evidence/p2_crossmodel_wave3_20261019/stats/check_crossmodel_hand_numbers_20261103.txt']   # its recorded output
# the secret game of the four mechanism readouts under the first System prompt (NOT_USED of 10-30 lifted 11-14: both prompts reported)
ROOTS += [f'evidence/p2_crossmodel_wave3_20261019/runs/{m}/games_swap/{n}_s5_{q}'
          for m, n in (('pro', 'gpt_oss_120b'), ('pro', 'gpt_oss_20b'), ('server3', 'qwen38_27b'), ('server3', 'qwen38_27b_sp'))
          for q in ('smoke_main', 'smoke_control', 'main', 'control')]
SKIP_DIRS = {MS + '/results/figs/fig5_build': 'LaTeX build folder of the Fig. 5 draft',
             WA + '/results/kl_20261111/pro': 'contained in results/kl_20261113_pro (the full PRO mirror of 11-13)',
             WA + '/results/stage2_20261113/extracts/b150/natural': 'copy of stage 1',
             H2 + '/final_20261113/files/ref_full.tar': 'upload bundle, its content is in m/ and the code folder',
             H2 + '/final_20261113/files/up_code.tar': 'upload bundle, its content is in the code folder'}
BAD_PARTS = ('_not_used', 'tmp', '__pycache__')


def lp(p):
    a = os.path.abspath(p)
    return a if a.startswith('\\\\?\\') or os.name != 'nt' else '\\\\?\\' + a


def excluded(rel, marked):
    parts = rel.split('/')
    for i, s in enumerate(parts[:-1]):
        if s.startswith('_sealed_') or s.startswith('_smoke') or s in BAD_PARTS:
            return f'folder {s}'
        if '/'.join(parts[:i + 1]) in marked:
            return 'folder marked NOT_USED.txt'
    if rel.endswith('.GO'):
        return '.GO marker'
    for d, why in SKIP_DIRS.items():
        if rel == d or rel.startswith(d + '/'):
            return why
    st2 = WA + '/results/stage2_20261113/extracts/b150/main/'
    if rel.startswith(st2) and not rel.startswith(st2 + 'glm53_flash/'):
        return 'copy of stage 1'
    return None


def walk(root):
    if os.path.isfile(lp(root)):
        yield root
        return
    for d, _, fs in os.walk(lp(root)):
        rel_d = os.path.relpath(d, lp('.')).replace('\\', '/')
        for f in fs:
            yield rel_d + '/' + f


def sha_copy(src, dst):
    h = hashlib.sha256()
    if dst:
        os.makedirs(os.path.dirname(lp(dst)), exist_ok=True)
    with open(lp(src), 'rb') as fi:
        fo = open(lp(dst), 'wb') if dst else None
        while True:
            b = fi.read(1 << 22)
            if not b:
                break
            h.update(b)
            if fo:
                fo.write(b)
        if fo:
            fo.close()
            shutil.copystat(lp(src), lp(dst))
    return h.hexdigest()


def main():
    dry = '--dry-run' in sys.argv
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 6
    assert os.path.isdir(V5) and os.path.isdir(WA), 'run from projects/consciousness'
    v5 = {}
    with open(V5 + '/MANIFEST.csv', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            v5[r['project_path']] = r['sha256']
    files, skipped = [], {}
    for root in ROOTS:
        assert os.path.exists(lp(root)), root
        allf = list(walk(root))
        marked = {os.path.dirname(x) for x in allf if os.path.basename(x) == 'NOT_USED.txt'}
        for rel in allf:
            why = excluded(rel, marked)
            if why:
                key = (rel.rsplit('/', 1)[0], why)
                skipped[key] = skipped.get(key, 0) + 1
            else:
                files.append(rel)
    files = sorted(set(files))
    print('candidate files', len(files), 'skipped folders', len(skipped), flush=True)
    prev = {}
    if os.path.exists(OUT + '/MANIFEST.csv'):
        with open(OUT + '/MANIFEST.csv', encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                prev[r['project_path']] = r['sha256']
    copied = set()

    def one(rel):
        size = os.path.getsize(lp(rel))
        h = sha_copy(rel, None)
        if rel in v5 and h == v5[rel]:
            return rel, size, h, 'in_v5_same'
        st = 'updated_since_v5' if rel in v5 else 'new'
        if prev.get(rel) == h and os.path.isfile(lp(OUT + '/' + rel)):
            return rel, size, h, st
        if not dry:
            h2 = sha_copy(rel, OUT + '/' + rel)
            assert h2 == h, ('source changed while copying', rel)
            copied.add(rel)
        return rel, size, h, st

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, files))
    keep = [r for r in res if r[3] != 'in_v5_same']
    tot = sum(r[1] for r in keep)
    want = {r[0] for r in keep}
    have = set()
    for d, _, fs in os.walk(lp(OUT + '/evidence')):
        rel_d = os.path.relpath(d, lp(OUT)).replace('\\', '/')
        have |= {rel_d + '/' + f for f in fs}
    stale = sorted(have - want)
    changed = [r[0] for r in keep if prev.get(r[0]) != r[2]]
    print(f"new {sum(r[3] == 'new' for r in res)}, updated {sum(r[3] == 'updated_since_v5' for r in res)}, "
          f"already in v5 {sum(r[3] == 'in_v5_same' for r in res)}; {tot / 1e6:.1f} MB; "
          f"to copy (not in the package or changed) {len(changed)}, {sum(r[1] for r in keep if r[0] in changed) / 1e6:.1f} MB; "
          f"stale package copies {len(stale)}", flush=True)
    for s in stale[:40]:
        print('  stale', s)
    if dry:
        tops = {}
        for c in changed:
            k = '/'.join(c.split('/')[:5])
            tops[k] = tops.get(k, 0) + 1
        for k, n in sorted(tops.items()):
            print(f'  to copy {n:6d}  {k}')
        return
    for s in stale:
        os.remove(lp(OUT + '/' + s))
    # verify every copy made in this run against the bytes just written
    bad = [r[0] for r in keep if r[0] in copied and sha_copy(OUT + '/' + r[0], None) != r[2]]
    assert not bad, bad[:5]
    with open(OUT + '/MANIFEST.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['package_path', 'project_path', 'bytes', 'sha256', 'status'])
        for rel, size, h, st in keep:
            w.writerow([rel, rel, size, h, st])
    with open(OUT + '/SKIPPED.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['folder', 'files', 'reason'])
        for (d, why), n in sorted(skipped.items()):
            w.writerow([d, n, why])
    print('written', len(keep), 'files, all copies verified', flush=True)


if __name__ == '__main__':
    main()
