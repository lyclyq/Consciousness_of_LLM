#!/bin/bash
# P2 persistence pilot, server3 queue (11-06, PROTOCOL_zh.md). Lanes (one task per GPU pair, never more than one run per GPU):
#   A   GPUs 0,1        B   GPUs 2,3        R1  GPUs 0-3, after both lanes have finished phase 1
# Phase 1 = main order (roles at every length + words), gate against phase G on every body (strict); phase 2 = natural order
# (roles only). Every task resumes from its per-body files. Stop: touch persistence/STOP_PERSIST (checked before every body and
# every task); no signals are ever sent. If nvidia-smi fails (driver drop) the lane stops and writes persistence/DRIVER_DOWN.
# usage (in the crossmodel directory, tmux session persistQ, one window per lane): bash persist_s3_20261106.sh A|B|R1
set -u
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C
PY=/root/recursive_state_moe_server3_20260711/venv/bin/python
H=/root/data/hf_models; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=persistence_readout_20261106.py; P=$C/persistence; STOP=$P/STOP_PERSIST
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REQ_ZH='请写一段简短的描述。'; REQ_EN='Please write a short description.'
LANE=${1:?lane A, B or R1}

log() { echo "[queue $LANE] $(date '+%F %T') $*" | tee -a $P/queue_s3.log; }

task() {  # gpus order tag model kw lang gate [wrapper]
  local gpus=$1 order=$2 tag=$3 m=$H/$4 kw=$5 lang=$6 gate=$7 wrap=${8:-}
  local out=$P/$order/$tag lg=$P/${order}_${tag}.log
  [ -e $STOP ] && { log "stop flag: not starting $order/$tag"; return 2; }
  [ -e $out/DONE ] && { log "$order/$tag already DONE"; return 0; }
  nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails (driver down?): lane stops before $order/$tag"; touch $P/DRIVER_DOWN; return 3; }
  local a=(--model $m --tmpl-kwargs "$kw" --bodies $B --out $out --stop-file $STOP --order $order --min-free-gib 32 --max-memory-gib 44)
  [ $lang = en ] && a+=(--lang en --bodies-en bodies_en_150.json)
  if [ $order = natural ]; then if [ $lang = en ]; then a+=(--request "$REQ_EN"); else a+=(--request "$REQ_ZH"); fi; fi
  [ $order = main ] && [ -n "$gate" ] && a+=(--gate-g $gate --gate-strict)
  log "start $order/$tag on GPUs $gpus"
  if [ -n "$wrap" ]; then CUDA_VISIBLE_DEVICES=$gpus $PY $wrap $m $S "${a[@]}" >> $lg 2>&1
  else CUDA_VISIBLE_DEVICES=$gpus $PY $S "${a[@]}" >> $lg 2>&1; fi
  local rc=$?
  [ -e $out/DONE ] && { log "DONE $order/$tag"; return 0; }
  [ -e $STOP ] && { log "stopped in $order/$tag (stop flag)"; return 2; }
  grep -q "CUDA unavailable" $lg && { log "CUDA unavailable in $order/$tag: lane stops"; touch $P/DRIVER_DOWN; return 3; }
  grep -q "GiB free <" $lg && { log "disk below the limit in $order/$tag: lane stops"; touch $P/DISK_LOW; return 2; }
  log "FAILED $order/$tag (rc $rc), see $lg; lane continues"; return 1
}

t() { task "$@"; local rc=$?; [ $rc -ge 2 ] && exit $rc; return 0; }
waitfor() { while :; do local ok=1; for f in "$@"; do [ -e "$f" ] || ok=0; done; [ $ok = 1 ] && return 0; [ -e $STOP ] && exit 2; sleep 60; done; }

Q38='{"enable_thinking": false}'; Q38T='{"enable_thinking": true}'; NT='{"enable_thinking": false}'
case $LANE in
  A)
    while tmux has-session -t persistA 2>/dev/null; do [ -e $STOP ] && exit 2; sleep 60; done   # the first Qwen3.8 run
    t 0,1 main qwen38_27b Qwen/Qwen3.8-27B "$Q38" zh suite/qwen38_27b_bf16_s3
    t 0,1 main exaone4_32b_en LGAI-EXAONE/EXAONE-4.0.1-32B "$NT" en suite/exaone4_32b_en
    t 0,1 main granite41_30b_en ibm-granite/granite-4.1-30b '{}' en suite/granite41_30b_en
    t 0,1 main seed_oss_36b ByteDance-Seed/Seed-OSS-36B-Instruct '{"thinking_budget": 0}' zh suite/seed_oss_36b
    t 0,1 main qwen38_27b_think Qwen/Qwen3.8-27B "$Q38T" zh suite_think/qwen38_27b
    touch $P/laneA1.done; log "phase 1 done"
    waitfor $P/r1.done
    t 0,1 natural qwen38_27b Qwen/Qwen3.8-27B "$Q38" zh ''
    t 0,1 natural exaone4_32b_en LGAI-EXAONE/EXAONE-4.0.1-32B "$NT" en ''
    t 0,1 natural granite41_30b_en ibm-granite/granite-4.1-30b '{}' en ''
    t 0,1 natural seed_oss_36b ByteDance-Seed/Seed-OSS-36B-Instruct '{"thinking_budget": 0}' zh ''
    t 0,1 natural qwen38_27b_think Qwen/Qwen3.8-27B "$Q38T" zh ''
    touch $P/laneA2.done; log "phase 2 done" ;;
  B)
    t 2,3 main gemma4_31b_en google/gemma-4-31B-it '{}' en suite/gemma4_31b_en
    t 2,3 main glm46v_flash zai-org/GLM-4.6V-Flash "$NT" zh suite/glm46v_flash
    t 2,3 main phi4_en microsoft/phi-4 '{}' en suite/phi4_en
    t 2,3 main ernie45_21b_a3b_thinking baidu/ERNIE-4.5-21B-A3B-Thinking '{}' zh suite/ernie45_21b_a3b_thinking
    t 2,3 main mimo7b_rl_0530 XiaomiMiMo/MiMo-7B-RL-0530 "$NT" zh suite/mimo7b_rl_0530
    t 2,3 main moonlight16b_a3b_instruct moonshotai/Moonlight-16B-A3B-Instruct '{}' zh suite/moonlight16b_a3b_instruct moonlight_compat_20261103.py
    touch $P/laneB1.done; log "phase 1 done"
    waitfor $P/r1.done
    t 2,3 natural gemma4_31b_en google/gemma-4-31B-it '{}' en ''
    t 2,3 natural glm46v_flash zai-org/GLM-4.6V-Flash "$NT" zh ''
    t 2,3 natural phi4_en microsoft/phi-4 '{}' en ''
    t 2,3 natural ernie45_21b_a3b_thinking baidu/ERNIE-4.5-21B-A3B-Thinking '{}' zh ''
    t 2,3 natural mimo7b_rl_0530 XiaomiMiMo/MiMo-7B-RL-0530 "$NT" zh ''
    t 2,3 natural moonlight16b_a3b_instruct moonshotai/Moonlight-16B-A3B-Instruct '{}' zh '' moonlight_compat_20261103.py
    touch $P/laneB2.done; log "phase 2 done" ;;
  R1)
    waitfor $P/laneA1.done $P/laneB1.done
    t 0,1,2,3 main r1_distill_llama70b deepseek-ai/DeepSeek-R1-Distill-Llama-70B '{}' zh suite/r1_distill_llama70b_v2
    t 0,1,2,3 natural r1_distill_llama70b deepseek-ai/DeepSeek-R1-Distill-Llama-70B '{}' zh ''
    touch $P/r1.done; log "R1 done" ;;
esac
