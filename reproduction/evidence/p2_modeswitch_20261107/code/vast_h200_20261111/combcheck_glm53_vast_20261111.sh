#!/bin/bash
# Checks of words_kl_combined_20261111.py on the first 2 held-out bodies (KL against the paper's ctrl run, retention against stage 1),
# before the mechanism battery starts; writes /root/glm53/combined_check.done on a pass (11-11).
set -u
G=/root/glm53; cd $G/up_code || exit 1
PY=/root/venv_t26/bin/python; M=/dev/shm/hf_models/zai-org/GLM-5.3-Flash; R=$G/ref; O=$G/out; STOP=$G/STOP
MEM=${MEM:-100}; OFF=${OFF:-0}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 P2_FP8_FORCE_FALLBACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3
log() { echo "[combcheck] $(date '+%F %T') $*" | tee -a $G/run.log; }
until [ -e $G/weights.ok ] && grep -q "venv OK" $G/setup.log; do [ -e $STOP ] && exit 2; sleep 20; done
mkdir -p $O/kl $O/words/b150_main
log "start (MEM $MEM, OFF $OFF)"
$PY words_kl_combined_20261111.py --model $M --tmpl-kwargs '{}' --bodies $R/role_bodies.jsonl --words $R/words_v2.json --lang zh \
  --check-ctrl $R/modeswitch/ctrl/glm53_flash --gate-pilot $R/words_along/b150_main/glm53_flash --out-words $O/words/b150_main/glm53_flash_all \
  --out-kl $O/kl/glm53_flash_all --max-memory-gib $MEM --cpu-offload-gib $OFF --stop-file $STOP --check-only >> $O/combined.log 2>&1 < /dev/null
if grep -q 'CHECK ONLY: checks passed' $O/combined.log; then touch $G/combined_check.done; log "PASS: $(grep -a '^gate:' $O/combined.log | tail -1 | cut -c1-700)"
else log "FAILED: $(grep -a -E '^gate:|GATE FAILED|Error|Traceback' $O/combined.log | tail -3 | tr '\n' ' ' | cut -c1-700)"; fi
