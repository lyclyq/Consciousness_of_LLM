#!/bin/bash
# P2 persistence, server3 phase-3 queue (11-07, PREREG_phase3_zh.md): 400 new bodies, --lean, no gate. Starts after phases 1-2
# (laneA2.done, laneB2.done, r1.done). Lanes A3 = GPUs 0,1 and B3 = GPUs 2,3 (one task per GPU pair); R13 = GPUs 0-3 after both.
# Qwen3.8 runs its phase 3 on PRO. Disk guard 32 GiB (the other project's watchdog warns at 30): without more free disk the
# lanes stop cleanly and resume when restarted. Stop: touch persistence/STOP_PERSIST; no signals.
# usage: bash persist_s3_p3_20261107.sh A3|B3|R13
set -u
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C
PY=/root/recursive_state_moe_server3_20260711/venv/bin/python
H=/root/data/hf_models; B3=$C/phase3_bodies_zh.jsonl; E3=$C/phase3_bodies_en.json
S=persistence_readout_20261106.py; P=$C/persistence; STOP=$P/STOP_PERSIST
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REQ_ZH='请写一段简短的描述。'; REQ_EN='Please write a short description.'
LANE=${1:?lane A3, B3 or R13}
log() { echo "[queue $LANE] $(date '+%F %T') $*" | tee -a $P/queue_s3.log; }
task() {  # gpus order tag model kw lang [wrapper]
  local gpus=$1 order=$2 tag=$3 m=$H/$4 kw=$5 lang=$6 wrap=${7:-}
  local out=$P/p3$order/$tag lg=$P/p3${order}_${tag}.log
  [ -e $STOP ] && { log "stop flag: not starting p3$order/$tag"; return 2; }
  [ -e $out/DONE ] && { log "p3$order/$tag already DONE"; return 0; }
  nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: lane stops before p3$order/$tag"; touch $P/DRIVER_DOWN; return 3; }
  local a=(--model $m --tmpl-kwargs "$kw" --bodies $B3 --out $out --stop-file $STOP --order $order --min-free-gib 32 --max-memory-gib 44 --lean)
  [ $lang = en ] && a+=(--lang en --bodies-en $E3)
  if [ $order = natural ]; then if [ $lang = en ]; then a+=(--request "$REQ_EN"); else a+=(--request "$REQ_ZH"); fi; fi
  log "start p3$order/$tag on GPUs $gpus"
  if [ -n "$wrap" ]; then CUDA_VISIBLE_DEVICES=$gpus $PY $wrap $m $S "${a[@]}" >> $lg 2>&1
  else CUDA_VISIBLE_DEVICES=$gpus $PY $S "${a[@]}" >> $lg 2>&1; fi
  [ -e $out/DONE ] && { log "DONE p3$order/$tag"; return 0; }
  [ -e $STOP ] && { log "stopped in p3$order/$tag (stop flag)"; return 2; }
  grep -q "CUDA unavailable" $lg && { log "CUDA unavailable in p3$order/$tag: lane stops"; touch $P/DRIVER_DOWN; return 3; }
  grep -q "GiB free <" $lg && { log "disk below the limit in p3$order/$tag: lane stops"; touch $P/DISK_LOW; return 2; }
  log "FAILED p3$order/$tag, see $lg; lane continues"; return 1
}
t() { task "$@"; local rc=$?; [ $rc -ge 2 ] && exit $rc; return 0; }
waitfor() { while :; do local ok=1; for f in "$@"; do [ -e "$f" ] || ok=0; done; [ $ok = 1 ] && return 0; [ -e $STOP ] && exit 2; sleep 60; done; }
NT='{"enable_thinking": false}'
waitfor $P/laneA2.done $P/laneB2.done $P/r1.done
case $LANE in
  A3)
    for o in main natural; do t 0,1 $o exaone4_32b_en LGAI-EXAONE/EXAONE-4.0.1-32B "$NT" en; done
    for o in main natural; do t 0,1 $o granite41_30b_en ibm-granite/granite-4.1-30b '{}' en; done
    for o in main natural; do t 0,1 $o seed_oss_36b ByteDance-Seed/Seed-OSS-36B-Instruct '{"thinking_budget": 0}' zh; done
    touch $P/laneA3.done; log "phase 3 done" ;;
  B3)
    for o in main natural; do t 2,3 $o gemma4_31b_en google/gemma-4-31B-it '{}' en; done
    for o in main natural; do t 2,3 $o glm46v_flash zai-org/GLM-4.6V-Flash "$NT" zh; done
    for o in main natural; do t 2,3 $o phi4_en microsoft/phi-4 '{}' en; done
    for o in main natural; do t 2,3 $o ernie45_21b_a3b_thinking baidu/ERNIE-4.5-21B-A3B-Thinking '{}' zh; done
    for o in main natural; do t 2,3 $o mimo7b_rl_0530 XiaomiMiMo/MiMo-7B-RL-0530 "$NT" zh; done
    for o in main natural; do t 2,3 $o moonlight16b_a3b_instruct moonshotai/Moonlight-16B-A3B-Instruct '{}' zh moonlight_compat_20261103.py; done
    touch $P/laneB3.done; log "phase 3 done" ;;
  R13)
    waitfor $P/laneA3.done $P/laneB3.done
    for o in main natural; do t 0,1,2,3 $o r1_distill_llama70b deepseek-ai/DeepSeek-R1-Distill-Llama-70B '{}' zh; done
    touch $P/r13.done; log "R1 phase 3 done" ;;
esac
