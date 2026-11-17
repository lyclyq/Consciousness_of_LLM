#!/bin/bash
# GLM-5.3-Flash re-run in the rented H200 environment, part 1 (11-11, user: "连同几何一块重跑 ... 这样总干净了吧"; "不跑游戏"). The states
# in this environment differ from server3 (4 x RTX 4090) beyond the protocol's tolerance (first held-out body: last-token cosine
# 0.995 at L16 down to 0.95 at L44, role KL 5-16%; the same numbers with and without CPU offload, so the difference is hardware
# numerics, MoE routing), so every GLM-5.3 measurement of the paper is repeated here, in one environment:
#   0. the paper's KL ctrl run (mode_switch_ctrl_20261107.py, window klctrl, before the mechanism battery)
#   1. the mechanism battery (mech_glm53_vast_20261111.sh)                         -> waits for out/mech/DONE
#   2. geometry, the paper's as-run glm53f_G: crossmodel_suite.py (ad284bb6) --phases G --geometry-only --g-batch 1, thinking rendering
#      ('{}'), 150 bodies; then stream_geometry.py (81c736ce) for the four streams and their mean -> out/paper_rerun/suite_think/glm53_flash
#   3. KL of the ordinary words and their retention (words_kl_combined_20261111.py), checked against step 0's ctrl run (same
#      environment); the comparison with server3's stage 1 is recorded (--gate-info-only)
# Placement as the other steps (MEM GiB per GPU, OFF GiB CPU). Stop: touch /root/glm53/STOP.
# 11-11 v5: geometry dropped (paper keeps server3); KL words + retention run here after the mechanism (user: "还不如排上").
# usage (tmux glm53, window part1): bash part1_glm53_vast_20261111.sh
set -u
G=/root/glm53; cd $G/up_code || exit 1
PY=/root/venv_t26/bin/python; M=/dev/shm/hf_models/zai-org/GLM-5.3-Flash; R=$G/ref; O=$G/out; P=$O/paper_rerun; STOP=$G/STOP
MEM=${MEM:-125}; OFF=${OFF:-20}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 P2_FP8_FORCE_FALLBACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2
log() { echo "[part1] $(date '+%F %T') $*" | tee -a $G/run.log; }
until [ -e $O/mech/DONE ]; do [ -e $STOP ] && exit 2; sleep 60; done
log "mechanism battery done: part 1 starts (MEM $MEM, OFF $OFF)"
# 2 geometry: not re-run (user 11-11: the paper keeps server3's GLM-5.3 geometry; the KL ratios reproduce here)
log "geometry not re-run (paper keeps server3 values)"
# 3 KL words + retention
if [ ! -e $O/kl/glm53_flash_all/DONE ] || [ ! -e $O/words/b150_main/glm53_flash_all/DONE ]; then
  log "KL + retention (combined) start"
  mkdir -p $O/kl $O/words/b150_main
  $PY words_kl_combined_20261111.py --model $M --tmpl-kwargs '{}' --bodies $R/role_bodies.jsonl --words $R/words_v2.json --lang zh \
    --check-ctrl $P/modeswitch/ctrl/glm53_flash --gate-pilot $R/words_along/b150_main/glm53_flash --gate-info-only \
    --out-words $O/words/b150_main/glm53_flash_all --out-kl $O/kl/glm53_flash_all --max-memory-gib $MEM --cpu-offload-gib $OFF \
    --stop-file $STOP >> $O/combined_rerun.log 2>&1 < /dev/null
fi
log "KL + retention: $(grep -a -E '^gate:|GATE FAILED|summary \(content|finished|Error' $O/combined_rerun.log | tail -3 | tr '\n' ' ' | cut -c1-600)"
log "part 1 done"
