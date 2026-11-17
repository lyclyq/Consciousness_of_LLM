#!/bin/bash
# gpt-oss-20b precision control on PRO (user 10-19): the official MXFP4 weights run natively (transformers 5.19 overlay,
# Triton 3.6), against the server3 run that dequantizes to BF16. Same items and settings as server3: G (geometry only,
# English, 150 bodies) and the 180-item thinking sample (thinking-only template). Runs beside the Qwen3.8-27B review
# experiment (2 jobs at most on PRO). Waits for download_gptoss_pro.log DONE; skip with STOP_GPTOSS.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-20b
mkdir "$C/gptoss_mxfp4_pro.lock" 2>/dev/null || exit 0
# online through the mirror: the native MXFP4 path fetches kernels-community/gpt-oss-triton-kernels on first use (10-19 first
# attempt failed in offline mode)
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss_mxfp4_status.log; }
until grep -q DONE download_gptoss_pro.log 2>/dev/null; do [ -e STOP_GPTOSS ] && exit 0; sleep 120; done
[ -e STOP_GPTOSS ] && exit 0
st "G start"
$PY crossmodel_suite.py --model $M --tmpl-kwargs '{}' --bodies /root/selfindex_three_actual_20261011_v2/role_bodies.jsonl --problems problems.json --out suite_think/gpt_oss_20b_mxfp4_en --phases G --geometry-only --max-memory-gib 30 --lang en --bodies-en bodies_en_150.json > gptoss_mxfp4_G.log 2>&1
st "G exit $?"
[ -e STOP_GPTOSS ] && exit 0
st "think180 start"
$PY crossmodel_thinking_b.py --model $M --off-kwargs '{}' --on-kwargs '{}' --skip-off --bodies-zh pilot_bodies_en_list.json --lang en --out thinking/gpt_oss_20b_mxfp4_en --batch 16 --max-new 3072 --max-memory-gib 30 > gptoss_mxfp4_think180.log 2>&1
st "think180 exit $?"
