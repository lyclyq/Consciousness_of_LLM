#!/bin/bash
# Flash-Next, main probe only, both modes (user 11-07: "主探针跑一下呗 两个模式都跑"; delete the weights afterwards).
# mode_switch_ctrl_20261107.py (full KL, content KL, control-token mass, p_close, lp_next); four residual streams, so no gate;
# CPU offload as the paper's Flash-Next runs (--cpu-offload-gib 52, --max-memory-gib 84, transformers overlay tf_next).
# Exclusive flock on persistence/gpu.lock. Stop: touch modeswitch/STOP_MS; no signals.  usage: bash modeswitch_fn_pro_20261107.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
M=/root/hf_models/Qwen/Qwen3.8-Flash-Next-FP8; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=mode_switch_ctrl_20261107.py; P=$C/modeswitch/ctrl; STOP=$C/modeswitch/STOP_MS; LOCK=$C/persistence/gpu.lock
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[modeswitch-fn] $(date '+%F %T') $*" | tee -a $C/modeswitch/queue.log; }
task() {  # tag kw
  local tag=$1 kw=$2 out=$P/$1 lg=$P/$1.log
  [ -e $STOP ] && { log "stop flag: not starting $tag"; exit 2; }
  [ -e $out/DONE ] && { log "$tag already DONE"; return 0; }
  (
    flock -x 9
    log "start ctrl/$tag"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TF19 $PY $S --model $M --tmpl-kwargs "$kw" --bodies $B1 --out $out --stop-file $STOP \
      --max-memory-gib 84 --cpu-offload-gib 52 >> $lg 2>&1
  ) 9>$LOCK
  if [ -e $out/DONE ]; then log "DONE ctrl/$tag: $(grep '^summary:' $lg | tail -1 | cut -c1-600)"; else log "FAILED or stopped ctrl/$tag, see $lg"; fi
}
task flashnext_nt '{"enable_thinking": false}'
task flashnext_think '{}'
log "fn queue done"
