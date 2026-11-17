#!/bin/bash
# DeepSeek-R1-Distill-Llama-70B on PRO, in the environment of its paper KL run (11-12). The user's rule: R1 runs on server3 first;
# if that result is clean, the PRO run is not needed, otherwise PRO runs it too ("反之则让它继续"). Server3 finished 06:05 UTC
# (r1_kl_words_s3_20261112.sh). Its A-S and U-S agree with PRO's paper run within 1% on the 75-body means. A-U, the smallest KL
# and the denominator of the words / A-U ratio, is 6.7% higher on the main probe and 11.8% higher in the natural order, a
# systematic shift. So PRO measures the cell as well, and the paper takes PRO's numbers.
# Starts after pro_tail_20261112.sh ("tail done"). Disk: Flash-Next (173 GiB) has to be deleted before R1 (132 GiB) fits.
# Download file by file (hf_complete_20261112.py) at commit b1c0b44b, skipping .gitattributes, *.md and figures/*, which no
# loader reads (the mirror answers 503 on single small files). Then the lane's own command line (row r1_distill_llama70b of
# kl_words_orders_pro_20261111b.sh): zh, '{}', 84 GiB, CPU offload 60 GiB, offline, main + natural, checks against
# modeswitch/ctrl/r1_distill_llama70b and modeswitch/r1_distill_llama70b (kind full). Weights deleted once both are DONE.
# Stop: touch words_along/STOP_KLW. usage (tmux klw, window R1PRO, in the crossmodel directory): bash pro_r1_20261112.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python
B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; WJ=$C/words_v2.json; MS=$C/modeswitch; MC=$MS/ctrl
O=$C/words_along/kl; STOP=$C/words_along/STOP_KLW; tag=r1_distill_llama70b
M=/root/hf_models_p2nat/deepseek-ai/DeepSeek-R1-Distill-Llama-70B; REV=b1c0b44b4369b597ad119a196caf79a9c40e141e
log() { echo "[klw r1pro] $(date '+%F %T') $*" | tee -a $O/queue.log; }
free_gib() { df --output=avail -BG /root | tail -1 | tr -dc 0-9; }
[ -e $O/$tag/DONE ] && [ -e $O/natural/$tag/DONE ] && { log "already DONE"; exit 0; }
until grep -a -q '^\[klw tail\] .* tail done' $O/queue.log; do [ -e $STOP ] && { log "stop flag"; exit 2; }; sleep 60; done
log "tail done: R1 starts ($(free_gib) GiB free)"
until [ $(free_gib) -ge 140 ]; do [ -e $STOP ] && { log "stop flag"; exit 2; }; log "waiting for 140 GiB free ($(free_gib))"; sleep 600; done
if [ ! -e $O/dlo_$tag.fixed.ok ]; then
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY hf_complete_20261112.py \
    deepseek-ai/DeepSeek-R1-Distill-Llama-70B $REV $M --ignore '.gitattributes' '*.md' 'figures/*' --tries 30 --sleep 30 \
    >> $O/dlo_$tag.fixed.log 2>&1 < /dev/null && touch $O/dlo_$tag.fixed.ok
  log "R1 download: $(grep -a '^complete:' $O/dlo_$tag.fixed.log | tail -1 | cut -c1-200) ($(du -sh $M 2>/dev/null | cut -f1))"
fi
[ -e $O/dlo_$tag.fixed.ok ] || { log "R1 download incomplete: not run"; exit 1; }
[ -e $STOP ] && { log "stop flag"; exit 2; }
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || log "warning: a compute process is on the GPU"
todo=""
[ -e $O/$tag/DONE ] || todo="$todo,main"
[ -e $O/natural/$tag/DONE ] || todo="$todo,natural"
todo=${todo#,}
a=(--model $M --tmpl-kwargs '{}' --bodies $B1 --words $WJ --orders $todo --lang zh --max-memory-gib 84 --cpu-offload-gib 60 --stop-file $STOP)
case $todo in *main*) a+=(--out-main $O/$tag --check-ctrl $MC/$tag) ;; esac
case $todo in *natural*) a+=(--out-natural $O/natural/$tag --check-natural $MS/$tag --check-natural-kind full) ;; esac
mkdir -p $O/natural
log "start $tag ($todo)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  $PY kl_words_orders_20261111.py "${a[@]}" >> $O/orders_$tag.log 2>&1 < /dev/null
log "ended $tag: $(grep -a -E 'summary \(content|CHECK FAILED|Error|stop' $O/orders_$tag.log | tail -3 | tr '\n' ' ' | cut -c1-600)"
if [ -e $O/$tag/DONE ] && [ -e $O/natural/$tag/DONE ] && [ -d $M ] && [ ! -L $M ]; then
  rm -rf /root/hf_models_p2nat/deepseek-ai/DeepSeek-R1-Distill-Llama-70B; log "weights of $tag deleted ($(free_gib) GiB free)"
fi
