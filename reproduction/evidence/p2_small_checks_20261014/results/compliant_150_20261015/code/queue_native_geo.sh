#!/bin/bash
# wait for check I to end (no I process left, neither status.json still "running"), then run check J on GPUs 0,1
# (one run per GPU pair, as before).
W=/root/data/p2_small_checks_20261014
mkdir "$W/queue_native_geo.lock" 2>/dev/null || exit 0
while pgrep -f p2_attribution_native.py > /dev/null || grep -qs '"state": "running"' $W/attrnat_s0/status.json $W/attrnat_s1/status.json; do sleep 60; done
[ -e "$W/STOP_QUEUE_NATIVE_GEO" ] && exit 0
mkdir "$W/attrgeo.lock" 2>/dev/null || exit 0
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $W
/root/recursive_state_moe_server3_20260711/venv/bin/python code/p2_native_geometry.py --model /root/data/hf_models/Qwen/Qwen3.8-27B --bodies /root/selfindex_three_actual_20261011_v2/role_bodies.jsonl --out $W/attrgeo > $W/attrgeo.log 2>&1
echo EXIT $? >> $W/attrgeo.log
