#!/bin/bash
# Final step of the DeepSeek-V4.1-Flash-1017 re-download on server3 (11-10). The v3 run (download_dsv41_ms_s3_v3_20261109.sh,
# dabaa994) downloaded and checked all 80 listed files (11-09 20:37:56 CST: model-00048 ok), but its final count used
# `ls $OK | wc -l`, which leaves out names starting with a dot, so the marker .gitattributes.ok was not counted ("incomplete: 79
# of 80 files") and DONE was not written; the KL queue run_dsv41_kl_s3_20261108.sh waits for DONE in the same log.
# This script makes v3's final step with every file checked by name: each of the 80 files of the 10-19 listing (4517bfc5) must
# have its marker (made by v3 or the earlier runs only after the size and, for LFS files, the 10-19 sha256 matched) and its
# listed size on disk; the small files are compared with the tpl_probe copies as v3 does; then DONE goes to the same log.
# It downloads, moves and deletes nothing. Runs once in tmux (server3: no signals).
set -u
D=/root/data; C=$D/p2_small_checks_20261014/crossmodel
R=deepseek-ai/DeepSeek-V4.1-Flash-1017; OUT=$D/hf_models/$R; OK=$C/dsv41_ok_20261108; OLD=$C/dsv41_ms_files.tsv
LOG=$C/download_dsv41_flash_1017_20261108.log; TP=$D/tpl_probe/$R
cd $C || exit 1
log() { echo "$(date '+%F %T') [fin] $*" >> $LOG; }
[ "$(sha256sum $OLD | cut -c1-8)" = 4517bfc5 ] || { log "10-19 list changed, stop"; exit 1; }
if pgrep -f 'resolve/master/' > /dev/null; then log "a transfer is still running, stop"; exit 1; fi
bad=0; n=0
while IFS=$'\t' read -r p s h; do
  n=$((n + 1))
  [ -e "$OK/${p//\//__}.ok" ] || { log "no marker: $p"; bad=1; }
  [ "$(stat -c %s "$OUT/$p" 2>/dev/null || echo -1)" = "$s" ] || { log "size not as listed: $p"; bad=1; }
done < $OLD
if [ $n != 80 ] || [ $bad != 0 ]; then log "not complete ($n listed): no DONE"; exit 1; fi
log "all 80 listed files have their marker and listed size ($(ls -A $OK | wc -l) markers; v3 counted 79 because ls leaves out .gitattributes.ok)"
while IFS=$'\t' read -r p s h; do
  [ "$h" = "-" ] || continue
  if [ -e "$TP/$p" ]; then
    cmp -s "$OUT/$p" "$TP/$p" && log "same as tpl_probe: $p" || log "DIFFERS from tpl_probe: $p"
  else log "no tpl_probe copy: $p"; fi
done < $OLD
echo "$(date '+%F %T') DONE $(du -sh $OUT | cut -f1)" >> $LOG
