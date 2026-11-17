#!/bin/bash
# P2 mode switch, protocol completion on PRO (user 11-08: run what the protocol covers; mechanism group + the natural-order list
# of the appendix; models in neither are not run). After the Flash-Next weights were deleted, PRO got gpt-oss-120b and OLMo-2-7B.
#   gpt-oss-120b: mechanism group -> main probe reported (the full script also computes the natural order; not reported).
#   OLMo-2-7B (Ai2's model in the natural-order appendix): main + natural.
#   Both: control-token split (mode_switch_ctrl_20261107.py, main order), as for every other model.
# Settings as persistence lane R (persist_pro_20261107.sh): gpt-oss online through hf-mirror for the MXFP4 kernel, fixed date
# 2026-10-19, max 90 GiB; OLMo offline, 44 GiB. Gate against the G runs suite_think/gpt_oss_120b_en and suite/olmo2_7b_en.
# Exclusive flock on persistence/gpu.lock (one job at a time). Stop: touch modeswitch/STOP_MS; no signals.
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
H=/root/hf_models; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
P=$C/modeswitch; STOP=$P/STOP_MS; LOCK=$C/persistence/gpu.lock
mkdir -p $P/ctrl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[modeswitch-1108] $(date '+%F %T') $*" | tee -a $P/queue.log; }
task() {  # script outdir tag model lang gate pythonpath net max_mem [extra]
  local s=$1 od=$2 tag=$3 m=$H/$4 lang=$5 gate=$6 pp=$7 net=$8 mem=$9; shift 9
  local out=$od/$tag lg=$od/$tag.log a
  [ -e $STOP ] && { log "stop flag: not starting $out"; exit 2; }
  [ -e $out/DONE ] && { log "$out already DONE"; return 0; }
  nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: queue stops before $out"; exit 3; }
  a=(--model $m --tmpl-kwargs '{}' --bodies $B1 --out $out --stop-file $STOP --max-memory-gib $mem)
  [ $lang = en ] && a+=(--lang en --bodies-en bodies_en_150.json)
  [ -n "$gate" ] && a+=(--gate-g $gate --min-free-gib 4)
  a+=("$@")
  (
    flock -x 9
    log "start $out"
    if [ $net = 1 ]; then env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $s "${a[@]}" >> $lg 2>&1
    else CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $s "${a[@]}" >> $lg 2>&1; fi
  ) 9>$LOCK
  if [ -e $out/DONE ]; then log "DONE $out: $(grep '^summary:' $lg | tail -1 | cut -c1-600)"
  elif [ -e $STOP ]; then log "stopped in $out (stop flag)"; exit 2
  else log "FAILED $out, see $lg; queue continues"; fi
}
F=mode_switch_20261107.py; X=mode_switch_ctrl_20261107.py
task $F $P olmo2_7b_en allenai/OLMo-2-1124-7B-Instruct en suite/olmo2_7b_en '' 0 44
task $X $P/ctrl olmo2_7b_en allenai/OLMo-2-1124-7B-Instruct en '' '' 0 44
task $F $P gpt_oss_120b_en openai/gpt-oss-120b en suite_think/gpt_oss_120b_en $TF19 1 90 --fixed-date 2026-10-19
task $X $P/ctrl gpt_oss_120b_en openai/gpt-oss-120b en '' $TF19 1 90 --fixed-date 2026-10-19
log "1108 queue done"
