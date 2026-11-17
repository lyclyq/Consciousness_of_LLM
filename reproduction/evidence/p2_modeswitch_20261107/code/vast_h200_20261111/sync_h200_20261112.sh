#!/bin/bash
# Incremental mirror of the rented H200's GLM-5.3 outputs (11-12, user: "每次 h200 有新结果完成一波就拉一波同步，确保本地永远跟进最新的").
# Every INTERVAL seconds: list the files under run.log and out/ that changed since the previous sync (find -newer on the remote
# side, with a 120 s overlap). Fits are excluded (pulled once in pull_20261112T0030; they never change). Then tar exactly those
# files, extract them into results/vast_h200_20261111/m/, and verify each against the remote sha256. Mismatches are files still
# being written: the next round picks them up again. A cumulative manifest (m/MANIFEST_sha256.txt, latest sha per path) and a log
# (m/sync.log) record every round. Local only; nothing is written on the H200 except a temporary reference file in /tmp.
# usage: bash sync_h200_20261112.sh [INTERVAL_S] [ROUNDS]
set -u
INTERVAL=${1:-600}; ROUNDS=${2:-200}
L=<home>/Documents/WSL/Ubuntu-22.04/consciousness_ver_7_20261014_100530/projects/consciousness/evidence/p2_modeswitch_20261107/results/vast_h200_20261111/m
mkdir -p "$L"; cd "$L" || exit 1
K=(-o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -o ConnectTimeout=120 -o BatchMode=yes -o ServerAliveInterval=15 -p 29378 root@38.80.173.43)
[ -e .last_epoch ] || echo 0 > .last_epoch
touch MANIFEST_sha256.txt
for ((r = 1; r <= ROUNDS; r++)); do
  last=$(cat .last_epoch); now=$(date +%s); since=$((last > 120 ? last - 120 : 0))
  timeout 300 ssh -n "${K[@]}" "cd /root/glm53 && touch -d @$since /tmp/.sync_ref && find run.log out -path out/mech/fits -prune -o -type f -newer /tmp/.sync_ref -print0 | xargs -0 -r sha256sum" \
    2>/dev/null > _round_sha.txt
  rc=$?
  n=$(wc -l < _round_sha.txt)
  if [ $rc -ne 0 ] && [ $n = 0 ]; then echo "$(date -u '+%F %T') round $r: listing failed (ssh)" >> sync.log; sleep $INTERVAL; continue; fi
  ok=0; bad=0
  if [ $n -gt 0 ]; then
    cut -c67- _round_sha.txt > _round_list.txt
    timeout 1800 ssh "${K[@]}" 'cd /root/glm53 && tar cf - -T -' < _round_list.txt 2>/dev/null | tar xf - 2>/dev/null
    while IFS= read -r line; do
      h=${line:0:64}; f=${line:66}
      if [ -f "$f" ] && [ "$(sha256sum "$f" | cut -c1-64)" = "$h" ]; then
        ok=$((ok + 1)); grep -v -F "  $f" MANIFEST_sha256.txt > _m.tmp; echo "$h  $f" >> _m.tmp; mv _m.tmp MANIFEST_sha256.txt
      else bad=$((bad + 1)); fi
    done < _round_sha.txt
  fi
  [ $bad = 0 ] && echo $now > .last_epoch
  echo "$(date -u '+%F %T') round $r: changed $n, verified $ok, still changing $bad, manifest $(wc -l < MANIFEST_sha256.txt), $(grep -a -E '^\[(mech|part1|extra)\]' run.log 2>/dev/null | tail -1 | cut -c1-120)" >> sync.log
  rm -f _round_sha.txt _round_list.txt
  [ $r -lt $ROUNDS ] && sleep $INTERVAL
done
