#!/bin/bash
# PRO, 10-24 (user: the depth-band replacement): gpt-oss-20b, per-position replacement (crossmodel_role_swap.py 2b032831), rank 8,
# thinking, prompt only, in one third of the depth at a time, by the paper's thirds: L0-7, L8-15, L16-23. Smokes (full: all 75 held-out bodies
# x 2 orders): base_A, base_U, own <- User (swU8_A) with its random control, User <- own (swA8_U). Watchdog as gptoss_swap_full_pro.sh.
# Full runs (75 held-out bodies x 2 orders) after the smokes. Skip with STOP_GPTOSS_THIRDSFULL.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-20b
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
WD_MIN=15
REG="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8"
mkdir "$C/gptoss_thirds_full.lock" 2>/dev/null || exit 0
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
SW="--mode thinking --model $M --lang en --bodies bodies_en_150_list.json --splits $B --batch 4 --max-new 3072 --max-memory-gib 30 --save-gen --fit-from thinking_ablate4/gpt_oss_20b_smoke/fit.pt --conds $REG"
for spec in early:0-7 middle:8-15 late:16-23; do
  [ -e STOP_GPTOSS_THIRDSFULL ] && exit 0
  name=${spec%%:*}; band=${spec#*:}
  touch thinking_swap/gpt_oss_20b_third_${name}.GO
  st "T third $name ($band) full start"
  wd T_$name gptoss_third_${name}_full.log $PY crossmodel_role_swap.py $SW --band $band --out thinking_swap/gpt_oss_20b_third_${name}
  st "T third $name exit $?"
done
