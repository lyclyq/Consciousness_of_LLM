#!/bin/bash
# PRO, user 10-24: specificity controls for the two replacements kept in the main text (all layers; the last third, L16-23),
# gpt-oss-20b, thinking, full (75 held-out bodies, one question each): the content question for the last third (all layers already
# run) and an unrelated two-digit multiplication after the edited message, all layers and last third (crossmodel_role_swap.py
# --question arith). Conditions as the all-layer content control. Watchdog as gptoss_swap_full_pro.sh. Waits until
# gptoss_thirds_full_pro.sh has exited. Skip with STOP_GPTOSS_CTRL.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-20b
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
WD_MIN=15
CON="base_A:assistant:-:-:0,swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,base_U:user:-:-:0,swA8_U:user:assistant:V:8"
mkdir "$C/gptoss_ctrl.lock" 2>/dev/null || exit 0
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
until ! pgrep -f '[g]ptoss_thirds_full_pro.sh' > /dev/null; do [ -e STOP_GPTOSS_CTRL ] && exit 0; sleep 60; done
SW="--mode thinking --model $M --lang en --bodies bodies_en_150_list.json --splits $B --batch 4 --max-new 3072 --max-memory-gib 30 --save-gen --fit-from thinking_ablate4/gpt_oss_20b_smoke/fit.pt --conds $CON"
for spec in content_late:content:16-23:third_late_content arith_all:arith:0-23:all_arith arith_late:arith:16-23:third_late_arith; do
  [ -e STOP_GPTOSS_CTRL ] && exit 0
  IFS=: read name q band out <<< "$spec"
  touch thinking_swap/gpt_oss_20b_${out}.GO
  st "K control $name start"
  wd K_$name gptoss_ctrl_${name}.log $PY crossmodel_role_swap.py $SW --question $q --band $band --out thinking_swap/gpt_oss_20b_${out}
  st "K control $name exit $?"
done
