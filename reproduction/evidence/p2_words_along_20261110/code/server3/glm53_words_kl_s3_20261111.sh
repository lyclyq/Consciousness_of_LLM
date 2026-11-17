#!/bin/bash
# GLM-5.3-Flash: next-token KL of the ordinary words and their retention, all 32 pairs, from the same forwards (11-11, user: option A,
# then "你要不管一些需要重跑的吧 ... 可复现性很好": the paper's GLM-5.3 numbers stay from server3, so the words run here in the
# paper environment). words_kl_combined_20261111.py in the environment and placement of the paper's GLM-5.3 runs (venv_t26,
# P2_FP8_FORCE_FALLBACK=1, GPUs 0-3 at 40 GiB, 300 GiB CPU offload, template kwargs {}), checked on the first 2 held-out bodies
# against the paper's ctrl run (KL, relative difference <= 0.02) and against the stage-1 words-along run (roles and the first pair
# of each category, cosine >= 0.999, relative error <= 0.03). The mechanism battery runs elsewhere (rented 3 x H200, where the KL
# ratios reproduce the paper's: 1.82 vs 1.83, 1.13 vs 1.13). About 70 forwards per body at ~32 s, ~45 h.
# Starts once no compute process has been on any GPU for 5 minutes. Stop: touch words_along/STOP_G53A; no signals.
# usage (tmux session words, window G53A, in the crossmodel directory): bash glm53_words_kl_s3_20261111.sh
set -u
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C || exit 1
PY=/root/data/venv_t26/bin/python; W=$C/words_along; STOP=$W/STOP_G53A
log() { echo "[g53A] $(date '+%F %T') $*" | tee -a $W/queue_s3.log; }
idle=0
while [ $idle -lt 5 ]; do [ -e $STOP ] && exit 2
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; then idle=$((idle + 1)); else idle=0; fi; sleep 60; done
log "start (combined $(sha256sum words_kl_combined_20261111.py | cut -c1-8))"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 P2_FP8_FORCE_FALLBACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY words_kl_combined_20261111.py --model /root/data/hf_models/zai-org/GLM-5.3-Flash --tmpl-kwargs '{}' \
  --bodies /root/selfindex_three_actual_20261011_v2/role_bodies.jsonl --words $C/words_v2.json --lang zh \
  --check-ctrl $C/modeswitch/ctrl/glm53_flash --gate-pilot $W/b150_main/glm53_flash \
  --out-words $W/b150_main/glm53_flash_all --out-kl $W/kl/glm53_flash_all --max-memory-gib 40 --cpu-offload-gib 300 --stop-file $STOP \
  >> $W/glm53_flash_all.log 2>&1 < /dev/null
if [ -e $W/kl/glm53_flash_all/DONE ]; then log "DONE: $(grep -a 'summary (content' $W/glm53_flash_all.log | tail -1 | cut -c1-400)"
else log "NOT DONE, see words_along/glm53_flash_all.log: $(grep -a -E '^gate:|GATE FAILED|Error|stop' $W/glm53_flash_all.log | tail -2 | tr '\n' ' ' | cut -c1-500)"; fi
