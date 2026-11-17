#!/bin/bash
# Continue the DeepSeek-V4.1-Flash-1017 re-download on server3 with 8 files at a time (user 11-09: "下载并发加上").
# The 11-08 run (download_dsv41_ms_s3_20261108.sh, 4 at a time, tmux dsv41dl) is stopped through its own stop file
# STOP_DL_DSV41_1108, touched before this starts: it starts no new file, and the files it is downloading finish or stop
# by themselves (no signals).
# Same pinning as 11-08: the current ModelScope listing must equal the 10-19 listing dsv41_ms_files.tsv (4517bfc5), and
# every LFS file is checked against its 10-19 sha256. Same markers (dsv41_ok_20261108) and the same log
# download_dsv41_flash_1017_20261108.log, with DONE at the end, which the KL queue run_dsv41_kl_s3_20261108.sh waits for.
# Files the 11-08 run is still downloading when this starts (a running curl or a partial file) go to the end of the queue and
# are touched only after that run has exited (then resumed with curl -C -). Until those finish, up to 12 connections are open.
# A file failing the check is renamed *.bad_N (not deleted). Stop: touch STOP_DL_DSV41_1109. Runs in tmux dsv41dl8.
set -u
D=/root/data; C=$D/p2_small_checks_20261014/crossmodel; PY=/root/recursive_state_moe_server3_20260711/venv/bin/python
R=deepseek-ai/DeepSeek-V4.1-Flash-1017; OUT=$D/hf_models/$R; OK=$C/dsv41_ok_20261108; OLD=$C/dsv41_ms_files.tsv
NEW=$C/dsv41_ms_files_20261109.tsv; LOG=$C/download_dsv41_flash_1017_20261108.log; STOP=$C/STOP_DL_DSV41_1109
Q=$C/dsv41_queue_p8_20261109.tsv; TP=$D/tpl_probe/$R
cd $C || exit 1
[ -e $STOP ] && exit 0
log() { echo "$(date '+%F %T') [p8] $*" >> $LOG; }
[ -e $C/STOP_DL_DSV41_1108 ] || { log "the 11-08 run has no stop flag: not starting"; exit 1; }
[ "$(sha256sum $OLD | cut -c1-8)" = 4517bfc5 ] || { log "10-19 list changed, stop"; exit 1; }
curl -s -m 120 "https://modelscope.cn/api/v1/models/$R/repo/files?Recursive=true" | $PY -c "
import json, sys
for f in json.load(sys.stdin)['Data']['Files']:
    if f['Type'] == 'blob' and not f['Path'].startswith('inference/examples/'):
        print(f['Path'], f['Size'], f['Sha256'] if f['IsLFS'] else '-', sep='\t')" > $NEW
[ -s $NEW ] || { log "file list failed"; exit 1; }
if ! cmp -s <(sort $OLD) <(sort $NEW); then
  log "listing differs from 10-19 (not the paper's files): nothing downloaded"
  diff <(sort $OLD) <(sort $NEW) >> $LOG; exit 1
fi
sleep 10  # let a file the 11-08 run may have just begun show up in ps or on disk
running=$(ps -eo args | grep '[r]esolve/master/' | sed 's#.*/resolve/master/##')
: > $Q; busy=''; need=0; nb=0; nf=0
while IFS=$'\t' read -r p n h; do
  [ -e "$OK/${p//\//__}.ok" ] && continue
  have=$(stat -c %s "$OUT/$p" 2>/dev/null || echo 0); need=$((need + n - have))
  if [ -e "$OUT/$p" ] || grep -qxF "$p" <<< "$running"; then busy+="$p	$n	$h	1"$'\n'; nb=$((nb + 1))
  else printf '%s\t%s\t%s\t0\n' "$p" "$n" "$h" >> $Q; nf=$((nf + 1)); fi
done < $OLD
printf '%s' "$busy" >> $Q
free=$(df -B1 $D | tail -1 | awk '{print $4}')
[ "$free" -ge $((need + 21474836480)) ] || { log "free $((free >> 30)) GiB, need $((need >> 30)) + 20: not starting"; exit 1; }
log "listing identical to 10-19; $nf files to start, $nb left by the 11-08 run (last, after it exits); $((need >> 30)) GiB to go, $((free >> 30)) GiB free"
one2() {  # path size sha256 busy
  local p=$1 n=$2 h=$3 b=$4 f=$OUT/$1 m=$OK/${1//\//__}.ok
  if [ "$b" = 1 ]; then
    while pgrep -f '[d]ownload_dsv41_ms_s3_20261108[.]sh' > /dev/null; do [ -e $STOP ] && return 1; sleep 60; done
  fi
  [ -e "$m" ] && return 0
  mkdir -p "$(dirname "$f")"
  for i in $(seq 1 30); do
    [ -e $STOP ] && return 1
    curl -s -L -C - --retry 5 --retry-delay 10 --speed-limit 10000 --speed-time 120 -o "$f" "https://modelscope.cn/models/$R/resolve/master/$p"
    if [ "$(stat -c %s "$f" 2>/dev/null)" = "$n" ]; then
      if [ "$h" = "-" ] || [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$h" ]; then
        touch "$m"; echo "$(date '+%F %T') [p8] ok $p" >> $LOG; return 0
      fi
      echo "$(date '+%F %T') [p8] sha256 mismatch $p, kept as $f.bad_p8_$i" >> $LOG; mv "$f" "$f.bad_p8_$i"
    fi
    sleep 30
  done
  echo "$(date '+%F %T') [p8] FAILED $p" >> $LOG; return 1
}
export -f one2; export R OUT OK LOG STOP
xargs -P 8 -n 4 bash -c 'one2 "$@"' _ < $Q
if [ "$(ls $OK | wc -l)" = "$(wc -l < $OLD)" ]; then
  while IFS=$'\t' read -r p n h; do
    [ "$h" = "-" ] || continue
    if [ -e "$TP/$p" ]; then
      cmp -s "$OUT/$p" "$TP/$p" && log "same as tpl_probe: $p" || log "DIFFERS from tpl_probe: $p"
    else log "no tpl_probe copy: $p"; fi
  done < $OLD
  echo "$(date '+%F %T') DONE $(du -sh $OUT | cut -f1)" >> $LOG
else
  log "incomplete: $(ls $OK | wc -l) of $(wc -l < $OLD) files"
fi
