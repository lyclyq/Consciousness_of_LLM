#!/bin/bash
# Two-role natural-order checks on PRO (user 10-20: one model per developer in the natural order). The Qwen and gpt-oss
# templates cannot keep a System message in place after a user turn, so only Assistant and User are compared, and only the
# self-reference behaviour is measured (no distance ratios without the third role).
#   1. Qwen3.8-27B, single pass: crossmodel_natural.py (b9244d3a), exactly as for Qwen3.8-Flash-Next in v2 (150 bodies, 3 orders)
#   2. gpt-oss-20b, thinking (its only mode), native MXFP4: crossmodel_thinking_b.py --natural (7f62ca41; without --natural it is
#      b2e46749), same settings as its 900-item run (150 English bodies x 2 roles x 2 orders, batch 16, up to 3072 new tokens)
# One run at a time. Skip with STOP_NATURAL2.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/natural2_pro.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> natural2_status.log; }
[ -e STOP_NATURAL2 ] && exit 0
st "qwen38_27b natural start"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $PY crossmodel_natural.py --model /root/hf_models/Qwen/Qwen3.8-27B --tmpl-kwargs '{"enable_thinking": false}' --bodies $B \
  --out natural/qwen38_27b --max-memory-gib 86 > natural_qwen38_27b.log 2>&1
st "qwen38_27b natural exit $?"
[ -e STOP_NATURAL2 ] && exit 0
st "gptoss20b natural start"
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY crossmodel_thinking_b.py --model /root/hf_models/openai/gpt-oss-20b --off-kwargs '{}' --on-kwargs '{}' --skip-off --natural \
  --bodies-zh bodies_en_150_list.json --lang en --out thinking_natural/gpt_oss_20b_mxfp4_en --batch 16 --max-new 3072 \
  --max-memory-gib 30 > natural_gptoss20b.log 2>&1
st "gptoss20b natural exit $?"
