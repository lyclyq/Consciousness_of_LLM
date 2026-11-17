#!/bin/bash
# Final check of the rented 3xH200 (vast) before it is returned, 11-13 after game300 ended (07:08:30 UTC).
# 1. sha256 of every file under /root/glm53 on the H200 (remote_all_sha256.txt).
# 2. sha256 of every local copy in the earlier pulls and the continuous mirror m/.
# 3. Each remote file must have a local copy with the same sha (FINAL_MAP.txt names where it lives).
# 4. Files with no matching local copy are pulled into final_20261113/files/ and checked again.
# Nothing is changed on the H200. usage: bash final_verify_h200_20261113.sh
set -u
B=<home>/Documents/WSL/Ubuntu-22.04/consciousness_ver_7_20261014_100530/projects/consciousness/evidence/p2_modeswitch_20261107/results/vast_h200_20261111
K=(-o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -o ConnectTimeout=120 -o BatchMode=yes -o ServerAliveInterval=15 -p 29378 root@38.80.173.43)
F=final_20261113
cd "$B" || exit 1
mkdir -p $F/files
log() { echo "$(date -u '+%F %T') $*" | tee -a $F/final.log; }
log "remote listing"
timeout 900 ssh -n "${K[@]}" 'cd /root/glm53 && find . -type f -print0 | sort -z | xargs -0 sha256sum' 2>/dev/null | sed 's#  \./#  #' > $F/remote_all_sha256.txt
n=$(wc -l < $F/remote_all_sha256.txt); [ "$n" -gt 0 ] || { log "remote listing failed"; exit 1; }
log "remote files: $n"
local_index() {
  find m pull_20261112T2227 pull_20261112T0840 pull_20261112T0605 pull_20261112T0445 pull_20261112T0030 $F/files -type f \
    ! -name MANIFEST_sha256.txt ! -name sync.log ! -name remote_SHA256SUMS.txt -print0 | xargs -0 sha256sum | sed -E 's#^([0-9a-f]{64}) [ *]#\1  #' \
    | awk '{h=$1; p=substr($0, 67); split(p, a, "/"); pre=a[1]; if (pre == "final_20261113") pre = a[1] "/" a[2]; rel=substr(p, length(pre) + 2); print h "\t" rel "\t" p}' > $F/_local_index.tsv
}
match() {
  awk -F'\t' 'NR==FNR {k=$1 "\t" $2; if (!(k in s)) s[k]=$3; next}
    {h=substr($0,1,64); f=substr($0,67); k=h "\t" f; if (k in s) print h "  " f "  ->  " s[k] > "'$F'/FINAL_MAP.txt"; else print f > "'$F'/_missing.txt"}' \
    $F/_local_index.tsv $F/remote_all_sha256.txt
}
log "local index"
local_index; : > $F/FINAL_MAP.txt; : > $F/_missing.txt; match
miss=$(wc -l < $F/_missing.txt); log "matched $(wc -l < $F/FINAL_MAP.txt), missing or different $miss"
if [ "$miss" -gt 0 ]; then
  log "pulling $miss files into $F/files"
  (cd $F/files && timeout 3600 ssh "${K[@]}" 'cd /root/glm53 && tar cf - -T -' < ../_missing.txt 2>/dev/null | tar xf - 2>/dev/null)
  local_index; : > $F/FINAL_MAP.txt; : > $F/_missing.txt; match
  miss=$(wc -l < $F/_missing.txt); log "after pull: matched $(wc -l < $F/FINAL_MAP.txt), still missing or different $miss"
fi
[ "$miss" = 0 ] && log "ALL $n REMOTE FILES HAVE A VERIFIED LOCAL COPY" || log "NOT COMPLETE: see $F/_missing.txt"
