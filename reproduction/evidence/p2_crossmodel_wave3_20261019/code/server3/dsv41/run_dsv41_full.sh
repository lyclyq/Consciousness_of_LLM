#!/bin/bash
# DeepSeek-V4.1-Flash-1017 natural order, all 150 bodies: one process per GPU, each on every 4th body (every sequence is still
# computed alone, so the split does not change any result), then --finalize assembles and summarizes. Resumable: finished
# sequences stay on disk. Exits non-zero if any shard fails.
cd /root/data/p2_small_checks_20261014/crossmodel || exit 1
PY=/root/data/venv_t210/bin/python
MD=/root/data/hf_models/deepseek-ai/DeepSeek-V4.1-Flash-1017
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
OUT=suite_nat3/dsv41_flash_1017
pids=()
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k $PY dsv41/dsv41_natural3.py --model-dir $MD --bodies $B --out $OUT --wave-seqs 64 --shard $k --n-shards 4 \
    > nat3_dsv41_shard$k.log 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { echo "$(date '+%F %T') a shard failed"; exit 1; }
$PY dsv41/dsv41_natural3.py --model-dir $MD --bodies $B --out $OUT --finalize
