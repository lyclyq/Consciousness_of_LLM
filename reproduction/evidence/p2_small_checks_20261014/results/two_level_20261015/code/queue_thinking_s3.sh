#!/bin/bash
# server3, one GPU pair: wait for check M's shard on this pair, then check N shard (2/4 on GPUs 0,1; 3/4 on GPUs 2,3).
SLOT=$1
W=/root/data/p2_small_checks_20261014
mkdir "$W/queue_thinking_s$SLOT.lock" 2>/dev/null || exit 0
if [ "$SLOT" = "01" ]; then GPUS=0,1; MK=0; NS=2; else GPUS=2,3; MK=1; NS=3; fi
until grep -qs EXIT $W/reasoning_m_shard$MK.log; do sleep 60; done
[ -e $W/STOP_N ] && exit 0
export CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $W
/root/recursive_state_moe_server3_20260711/venv/bin/python code/p2_thinking_attribution.py --model /root/data/hf_models/Qwen/Qwen3.8-27B --bodies /root/selfindex_three_actual_20261011_v2/role_bodies.jsonl --out $W/thinking_n --shard $NS/4 --max-new-tokens 4096 > $W/thinking_n_$NS.log 2>&1
echo EXIT $? >> $W/thinking_n_$NS.log
