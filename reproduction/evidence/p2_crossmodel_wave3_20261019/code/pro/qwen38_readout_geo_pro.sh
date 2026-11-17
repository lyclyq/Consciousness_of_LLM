#!/bin/bash
# PRO, user 10-28 ("no think 看读出行了"): response-onset (readout position) geometry of Qwen3.8-27B with preserve_thinking=false,
# brought forward from inducer_geometry_pro.sh: the no-think body geometry lost the own-message separation after layer ~15, while
# with thinking the model still claims its own messages, so the question is whether the readout position still separates them.
# crossmodel_suite_mask.py (c8a29196), phase G, geometry only, non-thinking rendering, 150 bodies x 3 roles: per-layer body mean
# ("content") and the last prompt token ("onset"). Two versions, same script and machine: switch (preserve_thinking=false), then
# default. Outputs go to the folders that inducer_geometry_pro.sh uses (it skips finished bodies there).
#   0. pause the full runs (qwen38_nothink_full_pro.sh) between steps with STOP_Q38NF; wait until session q38nf has exited
#   1. switch, then default      2. resume the full runs
# Runs in tmux session q38ro.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; Q=/root/hf_models/Qwen/Qwen3.8-27B
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/q38ro.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> qwen38_readout_geo_status.log; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
       PYTHONUNBUFFERED=1
touch STOP_Q38NF
st "0 STOP_Q38NF set; waiting for q38nf to exit"
while tmux has-session -t q38nf 2>/dev/null; do sleep 30; done
st "0 q38nf exited: $(tail -1 qwen38_nothink_full_status.log)"
for v in switch default; do
  K='{"enable_thinking": false}'
  [ $v = switch ] && K='{"enable_thinking": false, "preserve_thinking": false}'
  $PY crossmodel_suite_mask.py --model $Q --tmpl-kwargs "$K" --bodies $B --problems problems.json --out suite_mask/qwen38_27b_$v \
    --phases G --geometry-only --max-memory-gib 88 --lang zh > inducer_qwen38_$v.log 2>&1
  st "1 qwen38 $v exit $? ($(ls suite_mask/qwen38_27b_$v/G/*.json 2>/dev/null | wc -l) bodies)"
done
rm -f STOP_Q38NF; rmdir q38nf.lock
tmux new-session -d -s q38nf "bash $C/qwen38_nothink_full_pro.sh"
st "2 full runs resumed"
st "done"
