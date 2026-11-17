"""User 10-26: one basis for gpt-oss-20b. The paper measures gpt-oss with its native MXFP4 weights (PRO, suite_think/gpt_oss_20b_mxfp4_en),
but the direction cosines of Section 5.3 came from the server3 run that dequantised the weights to BF16 (suite_think/gpt_oss_20b_en).
Both scripts were re-run on PRO on the MXFP4 G-phase states (same crossmodel_suite_516f.py; direction_stability_g.py ab2214b6,
eq1_test_offline.py 99d83940) and pulled to runs/pro/analysis_mxfp4/. This script
  - seals the old local files in _sealed_20261026/superseded_20261026/ (the BF16 direction file; the two Eq. 1 files, whose content the
    MXFP4 re-run reproduces value for value, so they were computed on the MXFP4 states already, but with no record of the input);
  - writes runs/analysis/direction_stability_g.json = the Qwen3.8-27B entry unchanged + the MXFP4 gpt-oss entry (BF16 entry dropped),
    and the MXFP4 Eq. 1 files under their old names, with a provenance note beside them;
  - marks the BF16 run folder NOT_USED and records everything in SEALED_MANIFEST.csv.
usage: replace_gptoss_mxfp4_20261026.py   (run from evidence/p2_crossmodel_wave3_20261019)"""
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

A = Path('runs/analysis')
NEW = Path('runs/pro/analysis_mxfp4')
SEAL = Path('_sealed_20261026')
SUP = SEAL / 'superseded_20261026' / 'runs' / 'analysis'
BF16_KEY, MX_KEY, QWEN_KEY = 'suite_think/gpt_oss_20b_en', 'suite_think/gpt_oss_20b_mxfp4_en', 'suite/qwen38_27b_bf16_s3'
REASON_DIR = '口径不一致：gpt-oss-20b 方向余弦取自 server3 的 BF16 解压运行，论文的 gpt-oss 一律用原生 MXFP4（PRO）；已用 MXFP4 重算替换'
REASON_EQ1 = '输入未记录：式 1 检验旧文件；MXFP4 重算逐项相同，已替换为记录了输入的版本'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def eq1_values(p):
    d = json.load(open(p, encoding='utf-8'))
    return {k: v for k, v in d.items()}


old_dir = json.load(open(A / 'direction_stability_g.json', encoding='utf-8'))
new_dir = json.load(open(NEW / 'direction_stability_g_mxfp4.json', encoding='utf-8'))
assert set(old_dir) == {QWEN_KEY, BF16_KEY} and set(new_dir) == {MX_KEY}, (list(old_dir), list(new_dir))
for kind in ('content', 'onset'):
    assert eq1_values(A / f'gptoss20b_eq1_{kind}.json') == eq1_values(NEW / f'gptoss20b_mxfp4_eq1_{kind}.json'), kind

SUP.mkdir(parents=True, exist_ok=True)
moves = []
for name, reason in (('direction_stability_g.json', REASON_DIR), ('gptoss20b_eq1_content.json', REASON_EQ1),
                     ('gptoss20b_eq1_onset.json', REASON_EQ1)):
    src, dst = A / name, SUP / name
    assert not dst.exists(), dst
    h, n = sha(src), src.stat().st_size
    shutil.move(src, dst)
    moves.append([src.as_posix(), dst.as_posix(), f'{reason}（sha256 {h[:16]}）', 1, n])

merged = {QWEN_KEY: old_dir[QWEN_KEY], MX_KEY: new_dir[MX_KEY]}
json.dump(merged, open(A / 'direction_stability_g.json', 'w', encoding='utf-8'), indent=1)
for kind in ('content', 'onset'):
    shutil.copyfile(NEW / f'gptoss20b_mxfp4_eq1_{kind}.json', A / f'gptoss20b_eq1_{kind}.json')

c, o = new_dir[MX_KEY]['content'], new_dir[MX_KEY]['onset']
oc = old_dir[BF16_KEY]
on = json.load(open(A / 'gptoss20b_eq1_onset.json', encoding='utf-8'))
note = f"""Provenance of direction_stability_g.json and gptoss20b_eq1_{{content,onset}}.json (2026-10-26)

gpt-oss-20b, native MXFP4 weights, thinking rendering: PRO /root/p2_scale_20261015/crossmodel/suite_think/gpt_oss_20b_mxfp4_en/G
(G-phase per-body means of crossmodel_suite_516f.py; 150 bodies, npz kept on PRO only; run folder runs/pro/suite_think/gpt_oss_20b_mxfp4_en).
Commands on PRO (cwd /root/p2_scale_20261015/crossmodel):
  python direction_stability_g.py analysis_mxfp4/direction_stability_g_mxfp4.json suite_think/gpt_oss_20b_mxfp4_en
  python eq1_test_offline.py suite_think/gpt_oss_20b_mxfp4_en/G analysis_mxfp4/gptoss20b_mxfp4_eq1_content.json content
  python eq1_test_offline.py suite_think/gpt_oss_20b_mxfp4_en/G analysis_mxfp4/gptoss20b_mxfp4_eq1_onset.json onset
Scripts: code/server3/direction_stability_g.py (sha256 {sha('code/server3/direction_stability_g.py')}),
         code/local/eq1_test_offline.py (sha256 {sha('code/local/eq1_test_offline.py')}).
Raw outputs as pulled: runs/pro/analysis_mxfp4/.

direction_stability_g.json
  {QWEN_KEY}: Qwen3.8-27B (BF16, its native precision), unchanged from the 10-23 file.
  {MX_KEY}: gpt-oss-20b MXFP4. Median cosine over held-out bodies, content {c['min_median'][0]:.4f} (L{c['min_median'][1]}) to {c['max_median'][0]:.4f} (L{c['max_median'][1]}),
    onset {o['min_median'][0]:.4f} (L{o['min_median'][1]}) to {o['max_median'][0]:.4f} (L{o['max_median'][1]}); positive in every held-out body at every layer.
  The BF16 entry {BF16_KEY} (content {oc['content']['min_median'][0]:.4f}-{oc['content']['max_median'][0]:.4f}, onset {oc['onset']['min_median'][0]:.4f}-{oc['onset']['max_median'][0]:.4f})
    is sealed with the old file in {SUP.as_posix()}/direction_stability_g.json.
gptoss20b_eq1_{{content,onset}}.json
  The MXFP4 outputs, renamed. The old files (input not recorded) hold the same values; sealed beside the old direction file.
  Onset: the lowest share over held-out bodies at any layer is {min(on['share_min_over_bodies_by_layer']):.4f} (L{on['share_min_over_bodies_by_layer'].index(min(on['share_min_over_bodies_by_layer']))}).
"""
(A / 'PROVENANCE_gptoss_mxfp4_20261026.txt').write_text(note, encoding='utf-8')

bf = Path('runs/server3/suite_think/gpt_oss_20b_en')
(bf / 'NOT_USED.txt').write_text(
    'Not used by the paper (10-26). gpt-oss-20b with its MXFP4 expert weights dequantised to BF16 on server3. The paper measures gpt-oss with '
    'the native MXFP4 weights: runs/pro/suite_think/gpt_oss_20b_mxfp4_en. The direction cosines once taken from this run were recomputed '
    'on the MXFP4 states (runs/analysis/PROVENANCE_gptoss_mxfp4_20261026.txt).\n', encoding='utf-8')

with open(SEAL / 'SEALED_MANIFEST.csv', 'a', encoding='utf-8', newline='') as fh:
    w = csv.writer(fh)
    for m in moves:
        w.writerow(m)
for m in moves:
    print('sealed', m[0], '->', m[1])
print('wrote', A / 'direction_stability_g.json', list(merged))
print(note)
