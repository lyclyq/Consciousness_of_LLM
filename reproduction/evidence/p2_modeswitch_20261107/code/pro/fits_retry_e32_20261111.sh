#!/bin/bash
# EXAONE-4.0.1-32B mechanism battery, step 1 (fits) re-run with a memory log (11-11). The first run (14:33 UTC) was killed (exit 137)
# right after loading: the PRO container's memory limit (94 GiB, cgroup v1) was reached twice (failcnt 2). The command is the
# battery's own (mech_exaone_pro_20261111.sh, step 1, same lock); every 5 s the cgroup usage, rss, cache, shmem, mapped_file and
# the fit process's RSS go to modeswitch/mech_e32/fits_mem.log. Fits on success go to r05r06_fits/e32, where the battery finds them.
# usage (tmux klw, window E32FIT, in the crossmodel directory): bash fits_retry_e32_20261111.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=crossmodel_role_swap_r05_hc_20261111.py; LOCK=$C/persistence/gpu.lock; W=$C/modeswitch/mech_e32
MODEL=/root/hf_models_p2nat/LGAI-EXAONE/EXAONE-4.0.1-32B; KW='{"enable_thinking": true}'; FD=r05r06_fits/e32
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
st() { echo "[mech e32] $(date '+%F %T') $*" | tee -a $W/status.log >> $C/modeswitch/queue.log; }
[ -s $FD/fit_seed4.pt ] && { st "fits already written"; exit 0; }
[ -e $W/fits.log ] && mv $W/fits.log $W/fits_try1_20261111.log
CG=/sys/fs/cgroup/memory
( while :; do
    p=$(pgrep -f "^$PY $S --make-fits" | head -1); r=0; [ -n "$p" ] && r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null)
    echo "$(date '+%T') usage_gib $(( $(cat $CG/memory.usage_in_bytes) >> 30 )) $(awk '$1~/^total_(rss|cache|shmem|mapped_file|dirty)$/{printf "%s %d ", $1, $2/1073741824}' $CG/memory.stat) proc_rss_gib $(( ${r:-0} >> 20 )) failcnt $(cat $CG/memory.failcnt)"
    sleep 5; done ) >> $W/fits_mem.log 2>&1 &
SP=$!
st "fits retry start (memory log fits_mem.log)"
( flock -x 9; CUDA_VISIBLE_DEVICES=0 $PY $S --make-fits $FD --mode thinking --model $MODEL --on-kwargs "$KW" --lang en \
    --bodies bodies_en_150_list.json --splits $B --band 0-63 --max-memory-gib 90 ) 9>$LOCK > $W/fits.log 2>&1 < /dev/null
rc=$?
kill $SP 2>/dev/null
st "fits retry exit $rc | $(tail -1 $W/fits.log | cut -c1-200) | peak $(sort -k3 -n $W/fits_mem.log | tail -1)"
