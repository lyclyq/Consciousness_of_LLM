#!/bin/bash
# PRO, user 10-23: gpt-oss-20b, per-position replacement (crossmodel_role_swap.py 2b032831), rank 8, thinking:
#   G1-G3 region smokes (4 held-out bodies x 2 orders, all layers): the replacement only in the body / the end of the message /
#         the question text; base_A, base_U, swU8_A with its random control, swA8_U
#   F1    full run, all layers, all 75 held-out bodies x 2 orders, the ten conditions of the smoke (approved after the smoke:
#         own <- User 8/8 user, own <- System 7/8 user, User <- own 4/8 you, random unchanged, identity = base)
#   F2    its content control (the last-number question; own and User messages)
# Each step under the watchdog of gptoss_retry_pro.sh (15 min without a progress line: stop and start once more). Waits until
# gptoss_retry_pro.sh has exited (one gpt-oss job at a time). Skip with STOP_GPTOSS_SWAPFULL.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-20b
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
WD_MIN=15
ALL="base_A:assistant:-:-:0,base_U:user:-:-:0,base_S:system:-:-:0,idA8_A:assistant:assistant:V:8,swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8,rswA8_U:user:assistant:R:8,swA8_S:system:assistant:V:8"
REG="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8"
CON="base_A:assistant:-:-:0,swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,base_U:user:-:-:0,swA8_U:user:assistant:V:8"
mkdir "$C/gptoss_swapfull.lock" 2>/dev/null || exit 0
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
until ! pgrep -f '[g]ptoss_retry_pro.sh' > /dev/null; do [ -e STOP_GPTOSS_SWAPFULL ] && exit 0; sleep 60; done
SW="--mode thinking --model $M --lang en --bodies bodies_en_150_list.json --splits $B --batch 4 --max-new 3072 --max-memory-gib 30 --save-gen --fit-from thinking_ablate4/gpt_oss_20b_smoke/fit.pt --band 0-23"
for reg in body end question; do
  [ -e STOP_GPTOSS_SWAPFULL ] && exit 0
  st "G region $reg smoke start"
  wd G_$reg gptoss_region_${reg}_smoke.log $PY crossmodel_role_swap.py $SW --limit-bodies 4 --conds $REG --region $reg --out thinking_swap/gpt_oss_20b_region_${reg}_smoke
  st "G region $reg exit $?"
done
[ -e STOP_GPTOSS_SWAPFULL ] && exit 0
touch thinking_swap/gpt_oss_20b_all.GO thinking_swap/gpt_oss_20b_all_content.GO
st "F1 swap full all layers start"
wd F1 gptoss_swap_full.log $PY crossmodel_role_swap.py $SW --conds $ALL --out thinking_swap/gpt_oss_20b_all
st "F1 exit $?"
[ -e STOP_GPTOSS_SWAPFULL ] && exit 0
st "F2 swap content control start"
wd F2 gptoss_swap_content.log $PY crossmodel_role_swap.py $SW --question content --conds $CON --out thinking_swap/gpt_oss_20b_all_content
st "F2 exit $?"
