#!/bin/bash
# PRO, user 10-22 (plan B: run alongside the Qwen3.6-27B top-up, 2 concurrent jobs, the PRO limit): the thinking interventions
# of Qwen3.8-27B for gpt-oss-20b (native MXFP4, English, 24 layers), same designs and rules.
#   1. fourth design smoke (crossmodel_thinking_ablate2.py --suite smoke --no-gen-edit --edit-end before_gen --band 0-23
#      --main-layer 15, 4 held-out bodies)
#   2. first design layer sweep smoke (crossmodel_thinking_ablate.py --suite layersweep, layers 6,9,12,15,18,21 = the relative
#      depths of Qwen's 15,23,31,39,47,55; fit reused from the first-design smoke thinking_ablate/gpt_oss_20b_smoke)
#   3. fourth design full + content control (each waits for its .GO file, written after the smoke has been read)
#   4. first design at the chosen layer (gptoss_v1_layer.txt, chosen after the sweep by the rule fixed before it: the largest
#      drop of "you" against the same-layer random control, with that control finished >= 90% and User "you" <= 0.25),
#      held-out bodies except the 4 of the sweep, + content control
# Skip with STOP_GPTOSS_INT (checked between steps); a step whose .SKIP file exists exits at once.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-20b
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/gptoss_int.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss_int_status.log; }
stop() { [ -e STOP_GPTOSS_INT ] && { st "stopped by STOP_GPTOSS_INT"; exit 0; }; }
# the environment of gptoss20b_think900_pro.sh (not offline: the MXFP4 path loads the hub kernel kernels-community/gpt-oss-triton-kernels)
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
A2="--model $M --lang en --main-layer 15 --band 0-23 --no-gen-edit --edit-end before_gen --bodies bodies_en_150_list.json --splits $B --batch 16 --max-new 3072 --max-memory-gib 30 --save-gen"
A1="--model $M --lang en --main-layer 15 --bodies bodies_en_150_list.json --splits $B --batch 16 --max-new 3072 --max-memory-gib 30 --save-gen"
st "1 fourth-design smoke start"
$PY crossmodel_thinking_ablate2.py --suite smoke $A2 --out thinking_ablate4/gpt_oss_20b_smoke --limit-bodies 4 > gptoss_abl4_smoke.log 2>&1
st "1 exit $?"; stop
st "2 first-design layer sweep smoke start"
$PY crossmodel_thinking_ablate.py --suite layersweep --sweep-layers 6,9,12,15,18,21 --fit-from thinking_ablate/gpt_oss_20b_smoke/fit.pt $A1 \
  --out thinking_ablate/gpt_oss_20b_v1sweep_smoke --limit-bodies 4 > gptoss_v1sweep_smoke.log 2>&1
st "2 exit $?"; stop
st "3 fourth-design full (waits for thinking_ablate4/gpt_oss_20b.GO)"
$PY crossmodel_thinking_ablate2.py --suite full --layers band $A2 --out thinking_ablate4/gpt_oss_20b > gptoss_abl4_full.log 2>&1
st "3 full exit $?"; stop
$PY crossmodel_thinking_ablate2.py --suite content --layers band $A2 --fit-from thinking_ablate4/gpt_oss_20b/fit.pt \
  --out thinking_ablate4/gpt_oss_20b_content > gptoss_abl4_content.log 2>&1
st "3 content exit $?"; stop
st "4 waiting for gptoss_v1_layer.txt"
until [ -s gptoss_v1_layer.txt ]; do stop; sleep 120; done
L=$(tr -dc '0-9' < gptoss_v1_layer.txt)
st "4 first design at layer $L, full (waits for thinking_ablate/gpt_oss_20b_L$L.GO)"
$PY crossmodel_thinking_ablate.py --suite single --layer $L --exclude-first 4 --fit-from thinking_ablate/gpt_oss_20b_smoke/fit.pt $A1 \
  --out thinking_ablate/gpt_oss_20b_L$L > gptoss_v1L_full.log 2>&1
st "4 full exit $?"; stop
$PY crossmodel_thinking_ablate.py --suite single --layer $L --exclude-first 4 --question content --fit-from thinking_ablate/gpt_oss_20b_smoke/fit.pt $A1 \
  --out thinking_ablate/gpt_oss_20b_L${L}_content > gptoss_v1L_content.log 2>&1
st "4 content exit $?"
