#!/bin/bash
# GLM-5.3-Flash on the rented vast.ai machine (11-11, user: "我们抓紧对 glm5.3 按照协议做吧"), part 1: next-token KL of the ordinary
# words and their retention along the message, with the server3 scripts of these measurements (kl_words_20261111.py f297e130,
# words_along_20261110.py 583d3fff, crossmodel_suite.py ad284bb6, fp8_fallback.py b2d91c80) in an environment built from server3's
# venv_t26 freeze (torch 2.6.0+cu124, transformers 5.19.0), P2_FP8_FORCE_FALLBACK=1 as there, weights at server3's commit 46a050be.
# Different hardware (4 x H100 PCIe 80 GB): each measurement is checked against its server3 run before it measures (KL: role KLs of
# 2 bodies against the paper's ctrl run, relative difference <= 0.02; retention: words_along's own gate against the stage-1 run,
# cos >= 0.999 and relative error <= 0.03).
#   0. waits for the weights and the environment
#   1. hc_edit unit tests (CPU)
#   2. load test: placement, memory, one forward
#   3. KL words, roles + all 32 pairs -> out/kl/glm53_flash_all
#   4. retention, roles + all 32 pairs -> out/words/b150_main/glm53_flash_all (the first 8 pairs repeat stage 1: a cross-environment check)
# 11-11 order change: part 1 starts after the mechanism battery (out/mech/DONE).
# Stop: touch /root/glm53/STOP. usage (tmux glm53, window run1): bash run_glm53_vast_20261111.sh
set -u
G=/root/glm53; cd $G/up_code || exit 1
PY=/root/venv_t26/bin/python; M=/dev/shm/hf_models/zai-org/GLM-5.3-Flash; R=$G/ref; O=$G/out; STOP=$G/STOP
MEM=${MEM:-72}; OFF=${OFF:-40}
mkdir -p $O/kl $O/words/b150_main
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 P2_FP8_FORCE_FALLBACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3
log() { echo "[run1] $(date '+%F %T') $*" | tee -a $G/run.log; }
until [ -e $G/weights.ok ] && grep -q "venv OK" $G/setup.log; do [ -e $STOP ] && exit 2; sleep 30; done
log "weights and environment ready: $(grep 'venv OK' $G/setup.log | tail -1)"
$PY test_hc_edit_20261111.py > $G/test_hc_edit.log 2>&1; log "hc_edit tests: $(tail -1 $G/test_hc_edit.log)"
until [ -e $O/mech/DONE ]; do [ -e $STOP ] && exit 2; sleep 60; done
log "mechanism battery done: part 1 starts"
# 11-11 v3: KL and retention from the same forwards (words_kl_combined_20261111.py). Its checks on the first 2 bodies ran before the
# mechanism battery (window combcheck, combined_check.done); the run resumes from those records.
if [ ! -e $O/kl/glm53_flash_all/DONE ] || [ ! -e $O/words/b150_main/glm53_flash_all/DONE ]; then
  log "KL + retention (combined) start (MEM $MEM, OFF $OFF)"
  $PY words_kl_combined_20261111.py --model $M --tmpl-kwargs '{}' --bodies $R/role_bodies.jsonl --words $R/words_v2.json --lang zh     --check-ctrl $R/modeswitch/ctrl/glm53_flash --gate-pilot $R/words_along/b150_main/glm53_flash --out-words $O/words/b150_main/glm53_flash_all     --out-kl $O/kl/glm53_flash_all --max-memory-gib $MEM --cpu-offload-gib $OFF --stop-file $STOP >> $O/combined.log 2>&1 < /dev/null
fi
log "KL + retention: $(grep -a -E '^gate:|GATE FAILED|summary \(content|finished|Error' $O/combined.log | tail -3 | tr '\n' ' ' | cut -c1-600)"
log "part 1 done"
