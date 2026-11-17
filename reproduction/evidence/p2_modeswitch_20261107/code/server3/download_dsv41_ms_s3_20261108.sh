#!/bin/bash
# Re-download DeepSeek-V4.1-Flash-1017 to server3 for its natural3 KL (user 11-08: "能补的…自然 KL 都补", Table tab:self_natural3;
# deletions approved 11-08 "那就删除选1": Qwen3.5-122B-FP8, Qwen3.6-35B, GLM-4.7, Granite-4.2, Qwen3.5-27B, Qwen3.8-Flash-Next).
# Same source and method as download_dsv41_ms_s3.sh (ModelScope, 10-19), with these changes:
#   - pinned to the paper's files: the current ModelScope listing must equal the 10-19 listing dsv41_ms_files.tsv (4517bfc5,
#     the files of the paper's runs; its LFS sha256 equalled Hugging Face revision 9f9818127fdf) in paths, sizes and LFS sha256,
#     otherwise nothing is downloaded;
#   - every LFS file is checked against the 10-19 sha256; the small non-LFS files are compared afterwards with the copies of the
#     same release kept in tpl_probe/ (logged, nothing deleted);
#   - fresh markers (dsv41_ok_20261108; the 10-19 markers in dsv41_ok are left as they are);
#   - starts with >= 500 GiB free (the weights are 475 GiB).
# Up to 4 files at a time, curl resume, a file failing the check is renamed *.bad_N (not deleted).
# Stop: touch STOP_DL_DSV41_1108. Log: download_dsv41_flash_1017_20261108.log (DONE at the end). Runs in tmux.
set -u
D=/root/data; C=$D/p2_small_checks_20261014/crossmodel; PY=/root/recursive_state_moe_server3_20260711/venv/bin/python
R=deepseek-ai/DeepSeek-V4.1-Flash-1017; OUT=$D/hf_models/$R; OK=$C/dsv41_ok_20261108; OLD=$C/dsv41_ms_files.tsv
NEW=$C/dsv41_ms_files_20261108.tsv; LOG=$C/download_dsv41_flash_1017_20261108.log; STOP=$C/STOP_DL_DSV41_1108
TP=$D/tpl_probe/$R
cd $C || exit 1
[ -e $STOP ] && exit 0
[ "$(sha256sum $OLD | cut -c1-8)" = 4517bfc5 ] || { echo "$(date '+%F %T') 10-19 list changed, stop" >> $LOG; exit 1; }
free=$(df -BG $D | tail -1 | awk '{print $4}' | tr -d G)
[ "$free" -ge 500 ] || { echo "$(date '+%F %T') only $free GiB free, need 500" >> $LOG; exit 1; }
echo "$(date '+%F %T') start $R (ModelScope), $free GiB free" >> $LOG
curl -s -m 120 "https://modelscope.cn/api/v1/models/$R/repo/files?Recursive=true" | $PY -c "
import json, sys
for f in json.load(sys.stdin)['Data']['Files']:
    if f['Type'] == 'blob' and not f['Path'].startswith('inference/examples/'):
        print(f['Path'], f['Size'], f['Sha256'] if f['IsLFS'] else '-', sep='\t')" > $NEW
[ -s $NEW ] || { echo "$(date '+%F %T') file list failed" >> $LOG; exit 1; }
if ! cmp -s <(sort $OLD) <(sort $NEW); then
  echo "$(date '+%F %T') listing differs from 10-19 (not the paper's files): nothing downloaded" >> $LOG
  diff <(sort $OLD) <(sort $NEW) >> $LOG; exit 1
fi
echo "$(date '+%F %T') listing identical to 10-19 ($(wc -l < $NEW) files)" >> $LOG
mkdir -p $OK $OUT
one() {  # path size sha256
  local p=$1 n=$2 h=$3 f=$OUT/$1 m=$OK/${1//\//__}.ok
  [ -e "$m" ] && return 0
  mkdir -p "$(dirname "$f")"
  for i in $(seq 1 30); do
    [ -e $STOP ] && return 1
    curl -s -L -C - --retry 5 --retry-delay 10 --speed-limit 10000 --speed-time 120 -o "$f" "https://modelscope.cn/models/$R/resolve/master/$p"
    if [ "$(stat -c %s "$f" 2>/dev/null)" = "$n" ]; then
      if [ "$h" = "-" ] || [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$h" ]; then
        touch "$m"; echo "$(date '+%F %T') ok $p" >> $LOG; return 0
      fi
      echo "$(date '+%F %T') sha256 mismatch $p, kept as $f.bad_$i" >> $LOG; mv "$f" "$f.bad_$i"
    fi
    sleep 30
  done
  echo "$(date '+%F %T') FAILED $p" >> $LOG; return 1
}
export -f one; export R OUT OK LOG STOP
xargs -P 4 -n 3 bash -c 'one "$@"' _ < $OLD
if [ "$(ls $OK | wc -l)" = "$(wc -l < $OLD)" ]; then
  while IFS=$'\t' read -r p n h; do
    [ "$h" = "-" ] || continue
    if [ -e "$TP/$p" ]; then
      cmp -s "$OUT/$p" "$TP/$p" && echo "$(date '+%F %T') same as tpl_probe: $p" >> $LOG || echo "$(date '+%F %T') DIFFERS from tpl_probe: $p" >> $LOG
    else echo "$(date '+%F %T') no tpl_probe copy: $p" >> $LOG; fi
  done < $OLD
  echo "$(date '+%F %T') DONE $(du -sh $OUT | cut -f1)" >> $LOG
else
  echo "$(date '+%F %T') incomplete: $(ls $OK | wc -l) of $(wc -l < $OLD) files" >> $LOG
fi
