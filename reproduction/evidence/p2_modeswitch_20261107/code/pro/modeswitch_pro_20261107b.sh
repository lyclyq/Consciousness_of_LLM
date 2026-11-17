#!/bin/bash
# P2 mode-switch test on PRO (11-07, evidence/p2_modeswitch_20261107/PREREG_zh.md). One RTX PRO 6000. Every task takes the
# exclusive flock on persistence/gpu.lock, so it runs alone and never beside the persistence lane R (PRO policy: at most two).
# v2 (11-07): Flash-Next removed at the user's request (125B, too large to be useful here). Resumes from per-body files.
# Stop: touch modeswitch/STOP_MS (checked before every task and every body); no signals.  usage: bash modeswitch_pro_20261107b.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
H=/root/hf_models; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=mode_switch_20261107.py; P=$C/modeswitch; STOP=$P/STOP_MS; LOCK=$C/persistence/gpu.lock
mkdir -p $P
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[modeswitch] $(date '+%F %T') $*" | tee -a $P/queue.log; }

task() {  # tag model kw lang gate pythonpath net max_mem cpu_offload [extra script args]
  local tag=$1 m=$H/$2 kw=$3 lang=$4 gate=$5 pp=$6 net=$7 mem=$8 off=$9; shift 9
  local out=$P/$tag lg=$P/$tag.log a
  [ -e $STOP ] && { log "stop flag: not starting $tag"; exit 2; }
  [ -e $out/DONE ] && { log "$tag already DONE"; return 0; }
  nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: queue stops before $tag"; exit 3; }
  a=(--model $m --tmpl-kwargs "$kw" --bodies $B1 --out $out --stop-file $STOP --min-free-gib 4 --max-memory-gib $mem --cpu-offload-gib $off)
  [ $lang = en ] && a+=(--lang en --bodies-en bodies_en_150.json)
  [ -n "$gate" ] && a+=(--gate-g $gate)
  a+=("$@")
  log "waiting for the exclusive lock: $tag"
  (
    flock -x 9
    log "start $tag"
    if [ $net = 1 ]; then env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $S "${a[@]}" >> $lg 2>&1
    else CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $S "${a[@]}" >> $lg 2>&1; fi
  ) 9>$LOCK
  if [ -e $out/DONE ]; then log "DONE $tag: $(grep '^summary:' $lg | tail -1 | cut -c1-400)"
  elif [ -e $STOP ]; then log "stopped in $tag (stop flag)"; exit 2
  else log "FAILED $tag, see $lg; queue continues"; fi
}

NT='{"enable_thinking": false}'
task mimo_v26_nt XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B "$NT" zh suite/mimo_v26_distill_qwen9b '' 0 44 0
task mimo_v26_default XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B '{}' zh suite_think/mimo_v26_distill_qwen9b '' 0 44 0
task mimo_vl7b XiaomiMiMo/MiMo-VL-7B-RL-2508 '{}' zh suite/mimo_vl7b_rl_2508 '' 0 44 0
task apertus8b_en swiss-ai/Apertus-8B-Instruct-2509 '{}' en suite/apertus8b_instruct_2509_en '' 0 44 0 --fixed-date 2026-11-03
task gpt_oss_20b_en openai/gpt-oss-20b '{}' en suite_think/gpt_oss_20b_mxfp4_en $TF19 1 44 0 --fixed-date 2026-10-19
task qwen38_nt Qwen/Qwen3.8-27B "$NT" zh suite_mask/qwen38_27b_default '' 0 90 0
task qwen38_nt_switch Qwen/Qwen3.8-27B '{"enable_thinking": false, "preserve_thinking": false}' zh suite_mask/qwen38_27b_switch '' 0 90 0
task qwen38_think Qwen/Qwen3.8-27B '{}' zh suite_mask/qwen38_27b_think_default '' 0 90 0
task qwen38_think_switch Qwen/Qwen3.8-27B '{"preserve_thinking": false}' zh suite_mask/qwen38_27b_think_switch '' 0 90 0
log "queue done"
