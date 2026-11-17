#!/bin/bash
# Pull of the raw per-body retention outputs (words_along runs) from PRO or server3 into the local archive (11-13, user: full raw
# data are released with the paper). Read-only on the server: per unit (one run folder) a sha256 list is made remotely, the files
# come as one tar stream, and every local file is checked against the list. A unit whose check passed gets <unit>.verified and is
# skipped on a rerun, so the pull is resumable. Nothing on the server is changed or stopped.
# Local target: results/raw_20261113/<machine>/<set>/<tag>/. Log: results/raw_20261113/pull_<machine>.log.
# usage: bash pull_raw_words_along_20261113.sh pro|server3
set -u
M=$1
case $M in
  pro) PORT=<port>; R=/root/p2_scale_20261015/crossmodel/words_along; SETS="b150_main b150_natural p3_main";;
  server3) PORT=<port>; R=/root/data/p2_small_checks_20261014/crossmodel/words_along; SETS="b150_main b150_natural";;
  *) echo "pro or server3"; exit 1;;
esac
L=<home>/Documents/WSL/Ubuntu-22.04/consciousness_ver_7_20261014_100530/projects/consciousness/evidence/p2_words_along_20261110/results/raw_20261113/$M
K=(-o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -o ConnectTimeout=120 -o BatchMode=yes -o ServerAliveInterval=15 -p $PORT root@<server>)
mkdir -p "$L"; cd "$L" || exit 1
LOG=../pull_$M.log
log() { echo "$(date -u '+%F %T') $*" | tee -a $LOG; }
check() {  # $1 = list file; prints the number of files that are missing or differ
  awk '{print substr($0, 67)}' "$1" | tr '\n' '\0' | xargs -0 sha256sum 2>/dev/null | sed -E 's#^([0-9a-f]{64}) [ *]#\1  #' > _loc_$M.txt
  # the local list is read in BEGIN: with NR==FNR an empty local list would make awk take the remote list for the first file
  # and report 0 differences (this happened in the first run of 11-13, one false .verified, removed)
  awk -v L=_loc_$M.txt -v MISS=_miss_$M.txt 'BEGIN {while ((getline x < L) > 0) l[substr(x,67)]=substr(x,1,64)}
    {if (l[substr($0,67)] != substr($0,1,64)) {b++; print substr($0,67) > MISS}} END {print b+0}' "$1"
}
rssh() {  # ssh with retries: the host drops connections now and then
  local i
  for i in 1 2 3 4 5 6; do timeout "$1" ssh -n "${K[@]}" "$2" && return 0; sleep 30; done
  return 1
}
log "start $M"
for s in $SETS; do
  units=$(rssh 120 "cd $R && ls -d $s/*/ 2>/dev/null" | sed 's#/$##')
  [ -n "$units" ] || { log "listing of $s failed"; continue; }
  for u in $units; do
    v=$(echo $u | tr '/' '_').verified
    [ -e "$v" ] && continue
    lst=$(echo $u | tr '/' '_').sha256
    rssh 1800 "cd $R && find $u -type f -print0 | sort -z | xargs -0 sha256sum" > "$lst" 2>/dev/null
    n=$(wc -l < "$lst"); [ "$n" -gt 0 ] || { log "$u: remote list failed"; continue; }
    for try in 1 2 3 4 5 6; do
      rm -f _miss_$M.txt; bad=$(check "$lst")
      [ "$bad" = 0 ] && break
      timeout 3600 ssh "${K[@]}" "cd $R && tar cf - -T -" < _miss_$M.txt 2>/dev/null | tar xf - 2>/dev/null
      sleep 5
    done
    rm -f _miss_$M.txt; bad=$(check "$lst")
    if [ "$bad" = 0 ]; then date -u '+%F %T' > "$v"; log "$u: $n files verified ($(du -sh $u | cut -f1))"; else log "$u: $bad of $n files missing or different"; fi
  done
done
rm -f _loc_$M.txt _miss_$M.txt
log "end $M: $(ls *.verified 2>/dev/null | wc -l) units verified, $(du -sh . | cut -f1)"
