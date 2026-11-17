#!/bin/bash
# usage: run_reasoning_m.sh GPUS SHARD   (check M; one run per GPU pair)
W=/root/data/p2_small_checks_20261014
K=${2%/*}
mkdir "$W/reasoning_m_shard$K.lock" 2>/dev/null || exit 0
export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $W
/root/recursive_state_moe_server3_20260711/venv/bin/python code/p2_reasoning_ownership.py --model /root/data/hf_models/Qwen/Qwen3.8-27B --problems $W/reasoning_m/problems.json --out $W/reasoning_m --shard $2 > $W/reasoning_m_shard$K.log 2>&1
echo EXIT $? >> $W/reasoning_m_shard$K.log
