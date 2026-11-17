#!/bin/bash
# usage: run_default_geometry.sh GPUS SHARD   (check L, template-default geometry; one run per GPU pair)
W=/root/data/p2_small_checks_20261014
K=${2%/*}
mkdir "$W/defgeo150_shard$K.lock" 2>/dev/null || exit 0
export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $W
/root/recursive_state_moe_server3_20260711/venv/bin/python code/p2_default_geometry.py --model /root/data/hf_models/Qwen/Qwen3.8-27B --bodies /root/selfindex_three_actual_20261011_v2/role_bodies.jsonl --out $W/defgeo150 --shard $2 > $W/defgeo150_shard$K.log 2>&1
echo EXIT $? >> $W/defgeo150_shard$K.log
