#!/bin/bash
# P2 ordinary words in next-token prediction, GLM-5.3-Flash on server3 (11-11; user: align with the words-along models, GLM-5.3
# "只做每类第一对": the three roles and the first pair of each category). Its paper KL environment and command (the ctrl run of
# modeswitch_glm53_s3_20261108.sh: venv_t26, P2_FP8_FORCE_FALLBACK=1, GPUs 0-3 at 40 GiB, 300 GiB CPU offload, template kwargs
# {}), checked against that run by kl_words_20261111.py (server3 copy, --pairs first). Starts after the words-along stage 1 of
# GLM-5.3-Flash (words_along/laneG53.done) and 5 idle minutes on every GPU. About 19 forwards per body at ~35 s, ~14 h.
# Stop: touch words_along/STOP_KLW; no signals.
# usage (tmux session words, window KLG53, in the crossmodel directory): bash kl_words_glm53_s3_20261111.sh
set -u
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C || exit 1
PY=/root/data/venv_t26/bin/python
O=$C/words_along/kl; STOP=$C/words_along/STOP_KLW
mkdir -p $O
log() { echo "[klw glm53] $(date '+%F %T') $*" | tee -a $O/queue_s3.log; }
log "waiting for the words-along stage 1 of GLM-5.3-Flash"
until [ -e $C/words_along/laneG53.done ]; do [ -e $STOP ] && exit 2; sleep 120; done
idle=0
while [ $idle -lt 5 ]; do [ -e $STOP ] && exit 2
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; then idle=$((idle + 1)); else idle=0; fi; sleep 60; done
[ -e $O/glm53_flash/DONE ] && { log "already DONE"; exit 0; }
log "GPUs idle for 5 minutes: GLM-5.3-Flash starts"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 P2_FP8_FORCE_FALLBACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY kl_words_20261111.py --model /root/data/hf_models/zai-org/GLM-5.3-Flash --tmpl-kwargs '{}' \
  --bodies /root/selfindex_three_actual_20261011_v2/role_bodies.jsonl --words $C/words_v2.json --pairs first --out $O/glm53_flash \
  --check-ctrl $C/modeswitch/ctrl/glm53_flash --lang zh --max-memory-gib 40 --cpu-offload-gib 300 --stop-file $STOP \
  >> $O/glm53_flash.log 2>&1 < /dev/null
if [ -e $O/glm53_flash/DONE ]; then log "DONE: $(grep 'summary (content)' $O/glm53_flash.log | tail -1)"
else log "NOT DONE, see $O/glm53_flash.log: $(grep -a -E 'CHECK FAILED|Error|stop' $O/glm53_flash.log | tail -1)"; fi
