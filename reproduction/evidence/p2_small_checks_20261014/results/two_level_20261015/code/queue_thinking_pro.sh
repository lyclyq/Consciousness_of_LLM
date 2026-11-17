#!/bin/bash
# PRO 6000 (one run at a time): check N shards 0/4 then 1/4. STOP_N file stops before the next shard.
W1=/root/p2_scale_20261015
mkdir "$W1/queue_thinking.lock" 2>/dev/null || exit 0
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $W1
for S in 0 1; do
  [ -e $W1/STOP_N ] && exit 0
  /root/consciousness_a6000pro/.venv/bin/python code/p2_thinking_attribution.py --model /root/hf_models/Qwen/Qwen3.8-27B --bodies /root/selfindex_three_actual_20261011_v2/role_bodies.jsonl --out $W1/thinking_n --shard $S/4 --max-new-tokens 4096 --max-memory-gib 88 > $W1/thinking_n_$S.log 2>&1
  echo EXIT $? >> $W1/thinking_n_$S.log
done
