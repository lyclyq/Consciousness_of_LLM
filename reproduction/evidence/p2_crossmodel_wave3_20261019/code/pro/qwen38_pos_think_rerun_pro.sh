#!/bin/bash
# PRO, 10-28: the thinking-rendering position cells of Qwen3.8-27B again with crossmodel_suite_mask3.py 6e479c4b (cdd0691a recorded
# body_start_position from the body span and stopped with an IndexError in the thinking rendering, whose body starts after the
# template's system message). New output folders. Waits until qwen38_pos_then_fill_pro.sh (session qposf) has finished.
# Runs in tmux session qpost.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
Q=/root/hf_models/Qwen/Qwen3.8-27B
mkdir "$C/qpost.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> qwen38_pos_status.log; }
until [ "$(tail -1 qwen38_pos_status.log | awk '{print $3}')" = done ] && ! tmux has-session -t qposf 2>/dev/null; do sleep 120; done
for v in think_compress think_switch_gap4; do
  K='{}'; M=compress; G=0
  [ $v = think_switch_gap4 ] && { K='{"preserve_thinking": false}'; M=gap; G=4; }
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1 \
    $PY crossmodel_suite_mask3.py --model $Q --tmpl-kwargs "$K" --bodies $B --problems problems.json --out suite_mask/qwen38_27b_${v}_v2 \
    --phases G --geometry-only --max-memory-gib 88 --lang zh --pos-mode $M --pos-gap $G > qwen38_pos_${v}_v2.log 2>&1
  st "${v}_v2 exit $? ($(ls suite_mask/qwen38_27b_${v}_v2/G/*.json 2>/dev/null | wc -l) bodies): $(grep -h -E 'content:|onset:' qwen38_pos_${v}_v2.log | cut -c1-110 | tr '\n' ' ')"
done
st "rerun done"
