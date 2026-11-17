#!/bin/bash
# gpt-oss-20b tier 2 on PRO in native MXFP4 (user 10-19: every gpt-oss run on one path; the server3 BF16 runs are the
# precision check). Waits for the 180-item MXFP4 run (gptoss_mxfp4_pro.sh) to end, then runs the 900-item sample
# (150 English bodies x 3 roles x 2 orders) into the same folder, reusing its 180 items. Skip with STOP_GPTOSS900.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-20b
mkdir "$C/gptoss20b_think900_pro.lock" 2>/dev/null || exit 0
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss_mxfp4_status.log; }
until tail -1 gptoss_mxfp4_status.log | grep -q 'think180 exit'; do [ -e STOP_GPTOSS900 ] && exit 0; sleep 300; done
[ -e STOP_GPTOSS900 ] && exit 0
st "think900 start"
$PY crossmodel_thinking_b.py --model $M --off-kwargs '{}' --on-kwargs '{}' --skip-off --bodies-zh bodies_en_150_list.json --lang en --out thinking/gpt_oss_20b_mxfp4_en --batch 16 --max-new 3072 --max-memory-gib 30 > gptoss_mxfp4_think900.log 2>&1
st "think900 exit $?"
