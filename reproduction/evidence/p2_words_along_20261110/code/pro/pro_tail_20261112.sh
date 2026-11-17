#!/bin/bash
# The PRO cells left after R1 moved to server3 (11-12, user: "准了"). The lanes DL and RUN of kl_words_orders_pro_20261111b.sh were
# stopped with STOP_KLW (05:2x UTC). R1's download had failed five times near its end, and Flash-Next waited for 183 GiB free disk
# that R1's partial 128 GiB blocked. Steps:
#   0. wait until both old lanes have exited, then remove STOP_KLW (this script and the KL runs use it as their stop flag again)
#   1. delete PRO's partial R1 download (R1 runs on server3, r1_kl_words_s3_20261112.sh). Literal path, only if no process has a
#      file of it open or mapped.
#   2. Flash-Next non-thinking, main probe, row flashnext_nt of the lane: commit 236dfdf2. The download is file by file with
#      retries (hf_complete_20261112.py, because the mirror's 503s on single files break snapshot_download). Then the lane's own
#      command line: PYTHONPATH tf_next, 84 GiB, CPU offload 52 GiB, offline, zh, non-thinking, check against
#      modeswitch/ctrl/flashnext_nt. Weights deleted once DONE.
#   3. Ministral-3-14B-Reasoning, natural3 (ministral_r_fix_20261112.sh: the two files still missing, chat_template.jinja and
#      processor_config.json, retried. The cell runs only when they are complete. Weights deleted once DONE).
# One GPU job at a time. Stop: touch words_along/STOP_KLW. usage (tmux klw, window TAIL, in the crossmodel directory): bash pro_tail_20261112.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; WJ=$C/words_v2.json; MC=$C/modeswitch/ctrl
O=$C/words_along/kl; STOP=$C/words_along/STOP_KLW; NT='{"enable_thinking": false}'
log() { echo "[klw tail] $(date '+%F %T') $*" | tee -a $O/queue.log; }
free_gib() { df --output=avail -BG /root | tail -1 | tr -dc 0-9; }
in_use() { local p; for p in /proc/[0-9]*; do grep -q -F "$1/" $p/maps 2>/dev/null && return 0; ls -l $p/fd 2>/dev/null | grep -q -F "$1/" && return 0; done; return 1; }
# 0
while pgrep -f "kl_words_orders_pro_20261111b.sh" > /dev/null; do sleep 20; done
rm -f $STOP
log "old lanes exited, STOP_KLW removed; tail starts ($(free_gib) GiB free)"
# 1
R1=/root/hf_models_p2nat/deepseek-ai/DeepSeek-R1-Distill-Llama-70B
if [ -d $R1 ] && [ ! -L $R1 ] && ! in_use $R1; then
  rm -rf /root/hf_models_p2nat/deepseek-ai/DeepSeek-R1-Distill-Llama-70B
  log "partial R1 download deleted (R1 runs on server3; $(free_gib) GiB free)"
fi
# 2
tag=flashnext_nt; FN=/root/hf_models_words/Qwen/Qwen3.8-Flash-Next-FP8
if [ ! -e $O/$tag/DONE ]; then
  [ -e $STOP ] && { log "stop flag"; exit 2; }
  if [ ! -e $O/dlo_$tag.ok ]; then
    log "download Qwen/Qwen3.8-Flash-Next-FP8@236dfdf2 file by file (~173 GiB, $(free_gib) GiB free)"
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY hf_complete_20261112.py \
      Qwen/Qwen3.8-Flash-Next-FP8 236dfdf285828023ca3bcd3f37366c58a3469b13 $FN --tries 30 --sleep 30 >> $O/dlo_$tag.log 2>&1 < /dev/null \
      && touch $O/dlo_$tag.ok
    log "Flash-Next download: $(grep -a '^complete:' $O/dlo_$tag.log | tail -1 | cut -c1-200) ($(du -sh $FN 2>/dev/null | cut -f1))"
  fi
  if [ -e $O/dlo_$tag.ok ]; then
    [ -e $STOP ] && { log "stop flag"; exit 2; }
    [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || log "warning: a compute process is on the GPU"
    log "start $tag (main)"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TF19 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
      $PY kl_words_orders_20261111.py --model $FN --tmpl-kwargs "$NT" --bodies $B1 --words $WJ --orders main --lang zh \
      --max-memory-gib 84 --cpu-offload-gib 52 --stop-file $STOP --out-main $O/$tag --check-ctrl $MC/$tag >> $O/orders_$tag.log 2>&1 < /dev/null
    log "ended $tag: $(grep -a -E 'summary \(content|CHECK FAILED|Error|stop' $O/orders_$tag.log | tail -3 | tr '\n' ' ' | cut -c1-500)"
    if [ -e $O/$tag/DONE ] && [ -d $FN ] && [ ! -L $FN ]; then
      rm -rf /root/hf_models_words/Qwen/Qwen3.8-Flash-Next-FP8; log "weights of $tag deleted ($(free_gib) GiB free)"
    fi
  fi
fi
# 3
[ -e $STOP ] && { log "stop flag"; exit 2; }
bash ministral_r_fix_20261112.sh
log "tail done"
