#!/bin/bash
# Flash-Next non-thinking, main probe, KL of the ordinary words: second attempt (11-12). The first one (fn_run_20261112.sh, 10:38 UTC)
# was killed while loading the weights (10:54, no Python error), at the PRO container's memory limit (94 GiB; max usage equals the
# limit). This attempt keeps everything the same except the placement: 90 GiB on the GPU and 46 GiB CPU offload, instead of 84 + 52.
# Placement does not change the computation: offloaded layers run on the GPU after the transfer. The strict check against the
# paper's ctrl run (relative difference <= 0.02) still gates the run. Memory is logged every 5 s to fn_run2_mem.log.
# usage (tmux klw, window FNRUN2, in the crossmodel directory): bash fn_run2_20261112.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; WJ=$C/words_v2.json; MC=$C/modeswitch/ctrl
O=$C/words_along/kl; STOP=$C/words_along/STOP_KLW; NT='{"enable_thinking": false}'
tag=flashnext_nt; FN=/root/hf_models_words/Qwen/Qwen3.8-Flash-Next-FP8; MEM=${MEM:-90}; OFF=${OFF:-46}
log() { echo "[klw fnrun2] $(date '+%F %T') $*" | tee -a $O/queue.log; }
free_gib() { df --output=avail -BG /root | tail -1 | tr -dc 0-9; }
[ -e $O/$tag/DONE ] && { log "already DONE"; exit 0; }
[ -e $O/dlo_$tag.ok ] || { log "download not accepted: not run"; exit 1; }
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { log "GPU busy: not starting"; exit 1; }
CG=/sys/fs/cgroup/memory
( while :; do echo "$(date '+%T') usage_gib $(( $(cat $CG/memory.usage_in_bytes) >> 30 )) $(awk '$1~/^total_(rss|cache|shmem|mapped_file)$/{printf "%s %d ", $1, $2/1073741824}' $CG/memory.stat) failcnt $(cat $CG/memory.failcnt)"; sleep 5; done ) >> $O/fn_run2_mem.log 2>&1 &
SP=$!
log "start $tag (main), placement $MEM GiB GPU + $OFF GiB CPU"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TF19 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  $PY kl_words_orders_20261111.py --model $FN --tmpl-kwargs "$NT" --bodies $B1 --words $WJ --orders main --lang zh \
  --max-memory-gib $MEM --cpu-offload-gib $OFF --stop-file $STOP --out-main $O/$tag --check-ctrl $MC/$tag >> $O/orders_$tag.log 2>&1 < /dev/null
rc=$?
kill $SP 2>/dev/null
log "ended $tag (exit $rc): $(grep -a -E 'summary \(content|CHECK FAILED|Error|stop' $O/orders_$tag.log | tail -3 | tr '\n' ' ' | cut -c1-500) | peak $(sort -k3 -n $O/fn_run2_mem.log | tail -1)"
if [ -e $O/$tag/DONE ] && [ -d $FN ] && [ ! -L $FN ]; then
  rm -rf /root/hf_models_words/Qwen/Qwen3.8-Flash-Next-FP8; log "weights of $tag deleted ($(free_gib) GiB free)"
fi
