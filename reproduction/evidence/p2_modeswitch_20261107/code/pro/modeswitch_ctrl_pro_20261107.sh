#!/bin/bash
# Follow-up (user 11-07: is the mode-switch effect just the prediction of the end token?): mode_switch_ctrl_20261107.py on the
# default-format settings of the main table, main order. Exclusive flock on persistence/gpu.lock (PRO: at most two jobs).
# Stop: touch modeswitch/STOP_MS; no signals.  usage: bash modeswitch_ctrl_pro_20261107.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
H=/root/hf_models; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=mode_switch_ctrl_20261107.py; P=$C/modeswitch/ctrl; STOP=$C/modeswitch/STOP_MS; LOCK=$C/persistence/gpu.lock
mkdir -p $P
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[modeswitch-ctrl] $(date '+%F %T') $*" | tee -a $C/modeswitch/queue.log; }
task() {  # tag model kw lang pythonpath net max_mem [extra]
  local tag=$1 m=$H/$2 kw=$3 lang=$4 pp=$5 net=$6 mem=$7; shift 7
  local out=$P/$tag lg=$P/$tag.log a
  [ -e $STOP ] && { log "stop flag: not starting $tag"; exit 2; }
  [ -e $out/DONE ] && { log "$tag already DONE"; return 0; }
  a=(--model $m --tmpl-kwargs "$kw" --bodies $B1 --out $out --stop-file $STOP --max-memory-gib $mem)
  [ $lang = en ] && a+=(--lang en --bodies-en bodies_en_150.json)
  a+=("$@")
  (
    flock -x 9
    log "start ctrl/$tag"
    if [ $net = 1 ]; then env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $S "${a[@]}" >> $lg 2>&1
    else CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $S "${a[@]}" >> $lg 2>&1; fi
  ) 9>$LOCK
  if [ -e $out/DONE ]; then log "DONE ctrl/$tag: $(grep '^summary:' $lg | tail -1 | cut -c1-600)"; else log "FAILED or stopped ctrl/$tag, see $lg"; fi
}
NT='{"enable_thinking": false}'
task qwen38_nt Qwen/Qwen3.8-27B "$NT" zh '' 0 90
task qwen38_think Qwen/Qwen3.8-27B '{}' zh '' 0 90
task gpt_oss_20b_en openai/gpt-oss-20b '{}' en $TF19 1 44 --fixed-date 2026-10-19
task mimo_v26_nt XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B "$NT" zh '' 0 44
task apertus8b_en swiss-ai/Apertus-8B-Instruct-2509 '{}' en '' 0 44 --fixed-date 2026-11-03
task mimo_vl7b XiaomiMiMo/MiMo-VL-7B-RL-2508 '{}' zh '' 0 44
log "ctrl queue done"
