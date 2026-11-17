#!/bin/bash
# gpt-oss-120b tier 2 on PRO in native MXFP4 (10-17 tier rule: index and self-reference both present; user 10-19: the gpt-oss
# models get the 900-item version). Same settings and path as the 20b 900-item run and the 120b 180-item run (max 90 GiB):
# 150 English bodies x 3 roles x 2 orders into thinking/gpt_oss_120b_en, reusing its 180 items. Skip with STOP_GPTOSS120_900.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-120b
mkdir "$C/gptoss120b_think900_pro.lock" 2>/dev/null || exit 0
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss120b_status.log; }
[ -e STOP_GPTOSS120_900 ] && exit 0
st "think900 start"
$PY crossmodel_thinking_b.py --model $M --off-kwargs '{}' --on-kwargs '{}' --skip-off --bodies-zh bodies_en_150_list.json --lang en --out thinking/gpt_oss_120b_en --batch 16 --max-new 3072 --max-memory-gib 90 > gptoss120b_think900.log 2>&1
st "think900 exit $?"
