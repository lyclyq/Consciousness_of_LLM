#!/bin/bash
# P2 mode switch, GLM-5.3-Flash on server3 (user 11-08: "能补的主索引的 KL 和自然 KL 都补"; server3 released by the user on 11-08:
# "4090 server 3 你可以用了"). GLM-5.3-Flash is in the paper's main comparison with a thinking mode only. It is not on the
# natural-order list (Zhipu's smallest model, GLM-4.6V-Flash, is), so its protocol result is the main order. The full script
# computes the natural order anyway; it is recorded, not part of the protocol table (as for the main6 models on PRO).
# Scripts identical to PRO: mode_switch_20261107.py (171e7c53), mode_switch_ctrl_20261107.py (d1e71212), render_dump_20261108.py
# (1a12b467), persistence_readout_20261106.py (fab9c7b2, already here). Loader: server3's crossmodel_suite.py (ad284bb6), the one
# of the paper's GLM-5.3-Flash runs. Environment of those runs (suite_think/glm53_flash, phase G): venv_t26 (torch 2.6,
# transformers 5.19), FP8 release weights, P2_FP8_FORCE_FALLBACK=1, one process over the four GPUs with 40 GiB each and 300 GiB
# offloaded to host memory, template default call {} (the thinking rendering, the template's only mode).
# No layer gate: its layer outputs are [streams, hidden], and the gate of mode_switch_20261107.py only covers rank-3 outputs.
# Steps: render dump + dry run; ctrl (main order: full and content KL; the first body is checked in the log); full (main and
# natural order, full KL). Weights are kept: they are the paper's reference copy on server3.
# Stop: touch modeswitch/STOP_MS (checked before each step and before each body; rerun resumes). No signals. Runs in tmux.
# usage: bash modeswitch_glm53_s3_20261108.sh
set -u
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C || exit 1
PY=/root/data/venv_t26/bin/python
MD=/root/data/hf_models/zai-org/GLM-5.3-Flash; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
P=$C/modeswitch; STOP=$P/STOP_MS; TAG=glm53_flash; KW='{}'
F=mode_switch_20261107.py; X=mode_switch_ctrl_20261107.py
mkdir -p $P/ctrl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 P2_FP8_FORCE_FALLBACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3
log() { echo "[glm53 ms] $(date '+%F %T') $*" | tee -a $P/queue.log; }
log "start (scripts $(sha256sum $F | cut -c1-8) $(sha256sum $X | cut -c1-8) suite $(sha256sum crossmodel_suite.py | cut -c1-8))"
$PY render_dump_20261108.py --model $MD --tmpl-kwargs "$KW" --bodies $B1 > $P/$TAG.render.txt 2>&1 < /dev/null
$PY $F --model $MD --tmpl-kwargs "$KW" --bodies $B1 --out $P/dry_$TAG --dry-run > $P/$TAG.dry.log 2>&1 < /dev/null
grep -q 'DRY RUN OK' $P/$TAG.dry.log || { log "render check FAILED, see $P/$TAG.dry.log"; exit 1; }
log "render check OK: $(grep -o '"natural_system_ok": [a-z]*' $P/$TAG.dry.log | head -1)"

task() {  # script outdir [extra]
  local s=$1 out=$2/$TAG lg=$2/$TAG.log; shift 2
  [ -e $STOP ] && { log "stop flag: not starting $out"; exit 2; }
  [ -e $out/DONE ] && { log "$out already DONE"; return 0; }
  nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: stopping before $out"; exit 3; }
  log "start $out"
  $PY $s --model $MD --tmpl-kwargs "$KW" --bodies $B1 --out $out --stop-file $STOP --max-memory-gib 40 --cpu-offload-gib 300 \
    "$@" >> $lg 2>&1 < /dev/null
  if [ -e $out/DONE ]; then log "DONE $out: $(grep '^summary:' $lg | tail -1 | cut -c1-500)"
  elif [ -e $STOP ]; then log "stopped in $out (stop flag)"; exit 2
  else log "FAILED $out, see $lg"; exit 1; fi
}
task $X $P/ctrl
task $F $P --min-free-gib 4
log "glm53 queue done"
