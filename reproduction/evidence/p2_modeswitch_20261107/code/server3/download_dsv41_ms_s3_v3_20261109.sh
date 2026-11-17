#!/bin/bash
# DeepSeek-V4.1-Flash-1017 re-download on server3, third version (11-09), replacing
#   download_dsv41_ms_s3_20261108.sh (4 at a time) and download_dsv41_ms_s3_p8_20261109.sh (8 at a time).
# Why: curl 7.81 with --retry and -C - restarts a failed transfer from the offset the command started at, so every stall
# threw away what that curl had written (seen 11-09: model-00028 went 10 MiB -> 0 -> 21 MiB in a minute). Here curl runs
# without --retry: a stalled connection (< 10 kB/s for 120 s) ends curl, and the next attempt resumes from the current size.
# Server3's total bandwidth stays at 6-7 MiB/s with 4 or 11 connections, so this keeps at most 8 transfers running in all
# (the earlier runs' included) instead of adding more.
# Both earlier runs are stopped through their stop files (touched before this starts): they start no new file and their
# running transfers finish or stop by themselves (no signals). Files they are writing when this starts go to the end of the
# queue and are touched only after both have exited.
# Same pinning: the current ModelScope listing must equal the 10-19 listing dsv41_ms_files.tsv (4517bfc5), and every LFS file
# is checked against its 10-19 sha256. Same markers (dsv41_ok_20261108) and the same log
# download_dsv41_flash_1017_20261108.log, with DONE at the end, which the KL queue run_dsv41_kl_s3_20261108.sh waits for.
# A file failing the check is renamed *.bad_v3_N (not deleted). Stop: touch STOP_DL_DSV41_V3. Runs in tmux dsv41dl3.
set -u
D=/root/data; C=$D/p2_small_checks_20261014/crossmodel; PY=/root/recursive_state_moe_server3_20260711/venv/bin/python
R=deepseek-ai/DeepSeek-V4.1-Flash-1017; OUT=$D/hf_models/$R; OK=$C/dsv41_ok_20261108; OLD=$C/dsv41_ms_files.tsv
NEW=$C/dsv41_ms_files_20261109_v3.tsv; LOG=$C/download_dsv41_flash_1017_20261108.log; STOP=$C/STOP_DL_DSV41_V3
Q=$C/dsv41_queue_v3_20261109.tsv; TP=$D/tpl_probe/$R
cd $C || exit 1
[ -e $STOP ] && exit 0
log() { echo "$(date '+%F %T') [v3] $*" >> $LOG; }
for s in STOP_DL_DSV41_1108 STOP_DL_DSV41_1109; do [ -e $C/$s ] || { log "$s missing: an earlier run is not stopped, not starting"; exit 1; }; done
[ "$(sha256sum $OLD | cut -c1-8)" = 4517bfc5 ] || { log "10-19 list changed, stop"; exit 1; }
# the listing request can be cut off while the link is full (11-09: 3,612 bytes after 63 s), so retry until it parses
for t in $(seq 1 30); do
  [ -e $STOP ] && exit 0
  curl -s -m 300 "https://modelscope.cn/api/v1/models/$R/repo/files?Recursive=true" | $PY -c "
import json, sys
for f in json.load(sys.stdin)['Data']['Files']:
    if f['Type'] == 'blob' and not f['Path'].startswith('inference/examples/'):
        print(f['Path'], f['Size'], f['Sha256'] if f['IsLFS'] else '-', sep='\t')" > $NEW 2> /dev/null && [ -s $NEW ] && break
  : > $NEW; sleep 60
done
[ -s $NEW ] || { log "file list failed 30 times"; exit 1; }
if ! cmp -s <(sort $OLD) <(sort $NEW); then
  log "listing differs from 10-19 (not the paper's files): nothing downloaded"
  diff <(sort $OLD) <(sort $NEW) >> $LOG; exit 1
fi
sleep 10
running=$(ps -eo args | grep '[r]esolve/master/' | sed 's#.*/resolve/master/##')
: > $Q; busy=''; need=0; nb=0; nf=0
while IFS=$'\t' read -r p n h; do
  [ -e "$OK/${p//\//__}.ok" ] && continue
  have=$(stat -c %s "$OUT/$p" 2>/dev/null || echo 0); need=$((need + n - have))
  if [ -e "$OUT/$p" ] || grep -qxF "$p" <<< "$running"; then busy+="$p $n $h 1"$'\n'; nb=$((nb + 1))
  else echo "$p $n $h 0" >> $Q; nf=$((nf + 1)); fi
done < $OLD
printf '%s' "$busy" >> $Q
free=$(df -B1 $D | tail -1 | awk '{print $4}')
[ "$free" -ge $((need + 21474836480)) ] || { log "free $((free >> 30)) GiB, need $((need >> 30)) + 20: not starting"; exit 1; }
log "listing identical to 10-19; $nf files not started, $nb being written by the earlier runs (last, after they exit); $((need >> 30)) GiB to go, $((free >> 30)) GiB free"
one3() {  # path size sha256 busy
  local p=$1 n=$2 h=$3 b=$4 f=$OUT/$1 m=$OK/${1//\//__}.ok s
  if [ "$b" = 1 ]; then
    while pgrep -f 'download_dsv41_ms_s3_(20261108|p8_20261109)[.]sh' > /dev/null; do [ -e $STOP ] && return 1; sleep 60; done
  fi
  [ -e "$m" ] && return 0
  mkdir -p "$(dirname "$f")"
  for i in $(seq 1 1000); do
    [ -e $STOP ] && return 1
    while [ "$(pgrep -fc 'resolve/master/')" -ge 8 ]; do [ -e $STOP ] && return 1; sleep $((15 + RANDOM % 30)); done
    curl -s -L -C - --speed-limit 10000 --speed-time 120 -o "$f" "https://modelscope.cn/models/$R/resolve/master/$p"
    s=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$s" = "$n" ]; then
      if [ "$h" = "-" ] || [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$h" ]; then
        touch "$m"; echo "$(date '+%F %T') [v3] ok $p" >> $LOG; return 0
      fi
      echo "$(date '+%F %T') [v3] sha256 mismatch $p, kept as $f.bad_v3_$i" >> $LOG; mv "$f" "$f.bad_v3_$i"
    elif [ "$s" -gt "$n" ]; then
      echo "$(date '+%F %T') [v3] larger than listed $p, kept as $f.bad_v3_$i" >> $LOG; mv "$f" "$f.bad_v3_$i"
    fi
    sleep 10
  done
  echo "$(date '+%F %T') [v3] FAILED $p" >> $LOG; return 1
}
export -f one3; export R OUT OK LOG STOP
xargs -P 8 -n 4 bash -c 'one3 "$@"' _ < $Q
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
