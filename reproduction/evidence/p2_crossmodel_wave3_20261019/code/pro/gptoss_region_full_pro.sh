#!/bin/bash
# PRO, user 10-23: gpt-oss-20b, region-restricted per-position replacement, full runs (all 75 held-out bodies x 2 orders, all 24
# layers, rank 8), after the region smokes (own <- User: body 7/8 user, end of the message 8/8 user, question 0/8; random
# unchanged; User <- own 0/8 in each region). Regions body / end / question; base_A, base_U, swU8_A, rswU8_A, swA8_U. Watchdog as
# gptoss_swap_full_pro.sh. Waits until gptoss_swap_full_pro.sh has exited. Skip with STOP_GPTOSS_REGIONFULL.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-20b
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
WD_MIN=15
REG="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8"
mkdir "$C/gptoss_regionfull.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss_k8_status.log; }
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
PROG='^[A-Za-z0-9_]+ [0-9]+/[0-9]+ [0-9]+s|^[0-9]+ items [0-9]+s|waiting for '
wd() {  # wd NAME LOG -- command...: run under the watchdog, at most two attempts
  local name=$1 log=$2; shift 2
  local t pid start ok rc
  for t in 1 2; do
    "$@" > "$log.attempt$t" 2>&1 &
    pid=$!; start=$(date +%s); ok=0
    while kill -0 $pid 2>/dev/null; do
      grep -qE "$PROG" "$log.attempt$t" && ok=1
      if [ $ok = 0 ] && [ $(( $(date +%s) - start )) -gt $(( WD_MIN * 60 )) ]; then
        st "$name watchdog: no progress after $WD_MIN min, stopping pid $pid (attempt $t)"
        kill -TERM $pid; sleep 30; kill -9 $pid 2>/dev/null
        break
      fi
      sleep 30
    done
    wait $pid; rc=$?
    cp "$log.attempt$t" "$log"
    if [ $ok = 1 ] || [ $rc = 0 ]; then return $rc; fi
    st "$name attempt $t ended without progress (exit $rc)"
  done
  return 1
}
until ! pgrep -f '[g]ptoss_swap_full_pro.sh' > /dev/null; do [ -e STOP_GPTOSS_REGIONFULL ] && exit 0; sleep 60; done
SW="--mode thinking --model $M --lang en --bodies bodies_en_150_list.json --splits $B --batch 4 --max-new 3072 --max-memory-gib 30 --save-gen --fit-from thinking_ablate4/gpt_oss_20b_smoke/fit.pt --band 0-23 --conds $REG"
for reg in body end question; do
  [ -e STOP_GPTOSS_REGIONFULL ] && exit 0
  touch thinking_swap/gpt_oss_20b_region_${reg}.GO
  st "H region $reg full start"
  wd H_$reg gptoss_region_${reg}_full.log $PY crossmodel_role_swap.py $SW --region $reg --out thinking_swap/gpt_oss_20b_region_${reg}
  st "H region $reg full exit $?"
done
