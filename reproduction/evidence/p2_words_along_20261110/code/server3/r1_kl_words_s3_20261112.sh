#!/bin/bash
# DeepSeek-R1-Distill-Llama-70B: KL of the ordinary words, main probe and natural order, on server3 (11-12, user: "准了").
# On PRO the model (140 GB) needs 60 GiB of CPU offload on its one 96 GB GPU (~6 s per forward, ~16 h for ~10,000 forwards).
# Here it fits on the four GPUs without offload. The weights are the same commit as on PRO (b1c0b44b, server3's own copy at
# hf_models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B).
# The code is a self-contained copy of PRO's (crossmodel_suite 516f117e and the other deps, same hashes as on PRO, in this
# directory), with kl_words_orders_20261112b.py, which is 7772e4d6 plus --check-info-only. The references are the paper's runs
# on PRO (modeswitch/ctrl/r1_distill_llama70b for main, modeswitch/r1_distill_llama70b kind full for natural). The check is
# computed and stored for every body but never stops the run: the user's rule of 11-12 says an environment difference is not
# a reason to re-run.
# Environment: the server3 venv with transformers 5.8.0 as on PRO (torch 2.5.1). Template kwargs '{}', Chinese, as the PRO row.
# Starts once no compute process has been on any GPU for 5 minutes. Stop: touch STOP_R1KL in this directory; never kill anything.
# usage (tmux words, window R1KL, in crossmodel/r1kl_20261112): bash r1_kl_words_s3_20261112.sh
set -u
D=/root/data/p2_small_checks_20261014/crossmodel/r1kl_20261112; cd $D || exit 1
PY=/root/recursive_state_moe_server3_20260711/venv/bin/python
M=/root/data/hf_models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
STOP=$D/STOP_R1KL; tag=r1_distill_llama70b
log() { echo "[r1kl s3] $(date '+%F %T') $*" | tee -a $D/queue_r1kl.log; }
[ -e $D/out/$tag/DONE ] && [ -e $D/out/natural/$tag/DONE ] && { log "already DONE"; exit 0; }
log "start (script $(sha256sum kl_words_orders_20261112b.py | cut -c1-8), suite $(sha256sum crossmodel_suite.py | cut -c1-8), weights commit $(cat $M/.cache/huggingface/download/config.json.metadata 2>/dev/null | head -1 | cut -c1-8))"
idle=0
while [ $idle -lt 5 ]; do [ -e $STOP ] && { log "stop flag"; exit 2; }
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; then idle=$((idle + 1)); else idle=0; fi; sleep 60; done
log "GPUs idle for 5 minutes"
mkdir -p $D/out/natural
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  $PY kl_words_orders_20261112b.py --model $M --tmpl-kwargs '{}' --bodies $B --words $D/words_v2.json --orders main,natural --lang zh \
  --max-memory-gib 44 --cpu-offload-gib 0 --stop-file $STOP --check-info-only \
  --out-main $D/out/$tag --check-ctrl $D/modeswitch/ctrl/$tag \
  --out-natural $D/out/natural/$tag --check-natural $D/modeswitch/$tag --check-natural-kind full \
  >> $D/orders_$tag.log 2>&1 < /dev/null
log "ended: $(grep -a -E '^check|summary \(content|Error|stop' $D/orders_$tag.log | tail -4 | tr '\n' ' ' | cut -c1-700)"
