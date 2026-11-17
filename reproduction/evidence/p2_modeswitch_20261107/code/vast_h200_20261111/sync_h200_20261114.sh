#!/bin/bash
# One incremental mirror round of the second rented 4 x H200 NVL machine (11-14, GLM-5.3 secret game v6 only). User 11-14: "h200 10
# 分钟一看 注意出来的结果也要同时同步", "这样最后就不需要额外时间拉去了". Called every 10 minutes by the session's check.
# Same procedure as sync_h200_20261113.sh, one round per call: list the files under run.log, setup.log, test_hc_edit.log and out/
# changed since the previous round (find -newer on the remote side, 120 s overlap), tar exactly those, extract into
# results/vast_h200_20261114/m/, verify each against the remote sha256. Mismatches are files still being written: the next round
# picks them up again. Cumulative manifest m/MANIFEST_sha256.txt (latest sha per path), log m/sync.log. Fits are not pulled (the
# fit_q.pt on the machine is the local file 07b80490, uploaded and checked by the run). Local only; nothing is written on the
# machine except a temporary reference file in /tmp.
# usage: bash sync_h200_20261114.sh
set -u
L=<home>/Documents/WSL/Ubuntu-22.04/consciousness_ver_7_20261014_100530/projects/consciousness/evidence/p2_modeswitch_20261107/results/vast_h200_20261114/m
mkdir -p "$L"; cd "$L" || exit 1
K=(-o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -o ConnectTimeout=60 -o BatchMode=yes -o ServerAliveInterval=15 -p <port> root@<server>)
[ -e .last_epoch ] || echo 0 > .last_epoch
touch MANIFEST_sha256.txt
last=$(cat .last_epoch); now=$(date +%s); since=$((last > 120 ? last - 120 : 0))
timeout 300 ssh -n "${K[@]}" "cd /root/glm53 && touch -d @$since /tmp/.sync_ref && find run.log setup.log test_hc_edit.log out -path out/mech -prune -o -type f -newer /tmp/.sync_ref -print0 | xargs -0 -r sha256sum" \
  2>/dev/null > _round_sha.txt
rc=$?
n=$(wc -l < _round_sha.txt)
if [ $rc -ne 0 ] && [ $n = 0 ]; then echo "$(date -u '+%F %T') listing failed (ssh rc $rc)" | tee -a sync.log; rm -f _round_sha.txt; exit 1; fi
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
echo "$(date -u '+%F %T') changed $n, verified $ok, still changing $bad, manifest $(wc -l < MANIFEST_sha256.txt), $(grep -a '^\[v6\]' run.log 2>/dev/null | tail -1 | cut -c1-140)" | tee -a sync.log
rm -f _round_sha.txt _round_list.txt
