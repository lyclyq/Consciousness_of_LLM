#!/bin/bash
# P2 persistence pilot, PRO queue (11-06, PROTOCOL_zh.md). One RTX PRO 6000; at most two runs at a time (lanes P1 and P2,
# 44 GiB each); P3 runs alone (90 GiB) after both. Phase 1 = main order with the strict gate against phase G; phase 2 = natural
# order (roles only). gpt-oss runs in the transformers 5.19 overlay (native MXFP4, as in the paper); the others in the venv's 5.8.
# P3 waits for the user's disk cleanup and download_persist_pro_20261106.sh before gpt-oss-120b and OLMo-2-7B.
# Stop: touch persistence/STOP_PERSIST; no signals are sent.  usage: bash persist_pro_20261106.sh P1|P2|P3
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
H=/root/hf_models; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=persistence_readout_20261106.py; P=$C/persistence; STOP=$P/STOP_PERSIST
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REQ_ZH='请写一段简短的描述。'; REQ_EN='Please write a short description.'
LANE=${1:?lane P1, P2 or P3}

log() { echo "[queue $LANE] $(date '+%F %T') $*" | tee -a $P/queue_pro.log; }

task() {  # mem order tag model kw lang gate pythonpath
  local mem=$1 order=$2 tag=$3 m=$H/$4 kw=$5 lang=$6 gate=$7 pp=${8:-}
  local out=$P/$order/$tag lg=$P/${order}_${tag}.log
  [ -e $STOP ] && { log "stop flag: not starting $order/$tag"; return 2; }
  [ -e $out/DONE ] && { log "$order/$tag already DONE"; return 0; }
  nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: lane stops before $order/$tag"; touch $P/DRIVER_DOWN; return 3; }
  local a=(--model $m --tmpl-kwargs "$kw" --bodies $B --out $out --stop-file $STOP --order $order --min-free-gib 5 --max-memory-gib $mem)
  [ $lang = en ] && a+=(--lang en --bodies-en bodies_en_150.json)
  if [ $order = natural ]; then if [ $lang = en ]; then a+=(--request "$REQ_EN"); else a+=(--request "$REQ_ZH"); fi; fi
  [ $order = main ] && [ -n "$gate" ] && a+=(--gate-g $gate --gate-strict)
  log "start $order/$tag (max $mem GiB)"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $S "${a[@]}" >> $lg 2>&1
  local rc=$?
  [ -e $out/DONE ] && { log "DONE $order/$tag"; return 0; }
  [ -e $STOP ] && { log "stopped in $order/$tag (stop flag)"; return 2; }
  grep -q "CUDA unavailable" $lg && { log "CUDA unavailable in $order/$tag: lane stops"; touch $P/DRIVER_DOWN; return 3; }
  grep -q "GiB free <" $lg && { log "disk below the limit in $order/$tag: lane stops"; touch $P/DISK_LOW; return 2; }
  log "FAILED $order/$tag (rc $rc), see $lg; lane continues"; return 1
}

t() { task "$@"; local rc=$?; [ $rc -ge 2 ] && exit $rc; return 0; }
waitfor() { while :; do local ok=1; for f in "$@"; do [ -e "$f" ] || ok=0; done; [ $ok = 1 ] && return 0; [ -e $STOP ] && exit 2; sleep 60; done; }

NT='{"enable_thinking": false}'
case $LANE in
  P1)
    t 44 main gpt_oss_20b_en openai/gpt-oss-20b '{}' en suite_think/gpt_oss_20b_mxfp4_en $TF19
    t 44 main mimo_v26_distill_qwen9b XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B "$NT" zh suite/mimo_v26_distill_qwen9b
    t 44 natural gpt_oss_20b_en openai/gpt-oss-20b '{}' en '' $TF19
    t 44 natural mimo_v26_distill_qwen9b XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B "$NT" zh ''
    touch $P/P1.done; log "lane done" ;;
  P2)
    t 44 main apertus8b_en swiss-ai/Apertus-8B-Instruct-2509 '{}' en suite/apertus8b_instruct_2509_en
    t 44 main mimo_vl7b_rl_2508 XiaomiMiMo/MiMo-VL-7B-RL-2508 '{}' zh suite/mimo_vl7b_rl_2508
    t 44 natural apertus8b_en swiss-ai/Apertus-8B-Instruct-2509 '{}' en ''
    t 44 natural mimo_vl7b_rl_2508 XiaomiMiMo/MiMo-VL-7B-RL-2508 '{}' zh ''
    touch $P/P2.done; log "lane done" ;;
  P3)
    waitfor $P/P1.done $P/P2.done
    t 90 main qwen38_27b_think Qwen/Qwen3.8-27B '{"enable_thinking": true}' zh suite_mask/qwen38_27b_think_default
    t 90 natural qwen38_27b_think Qwen/Qwen3.8-27B '{"enable_thinking": true}' zh ''
    waitfor $P/download_gptoss120b.done
    t 90 main gpt_oss_120b_en openai/gpt-oss-120b '{}' en suite_think/gpt_oss_120b_en $TF19
    t 90 natural gpt_oss_120b_en openai/gpt-oss-120b '{}' en '' $TF19
    waitfor $P/download_olmo2_7b.done
    t 44 main olmo2_7b_en allenai/OLMo-2-1124-7B-Instruct '{}' en suite/olmo2_7b_en
    t 44 natural olmo2_7b_en allenai/OLMo-2-1124-7B-Instruct '{}' en ''
    touch $P/P3.done; log "lane done" ;;
esac
