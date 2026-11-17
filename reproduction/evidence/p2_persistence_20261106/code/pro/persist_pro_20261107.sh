#!/bin/bash
# P2 persistence, PRO queue v4 (11-07, PREREG_phase3_zh.md). One RTX PRO 6000. GPU sharing through flock on persistence/gpu.lock:
# lanes Q1 and Q2 (44 GiB each) take a shared lock, so at most these two run together; every other task (Q3, R) takes an
# exclusive one and runs alone (PRO policy: at most two runs at a time).
# All lanes start after the old lane P3 (v3) has finished Qwen3.8 thinking (natural/qwen38_27b_think/DONE). P3 then waits for a
# marker that is no longer written (download_gptoss120b.done) and holds no GPU; gpt-oss-120b and OLMo-2-7B run here instead, after
# download_persist_pro_20261107.sh (markers dl_*.ok). Phase 3 = 400 new bodies (items phase3_bodies_*), --lean, no gate.
# Stop: touch persistence/STOP_PERSIST; no signals are sent.  usage: bash persist_pro_20261107.sh Q1|Q2|Q3|R
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
H=/root/hf_models; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
B3=$C/phase3_bodies_zh.jsonl; E3=$C/phase3_bodies_en.json
S=persistence_readout_20261106.py; P=$C/persistence; STOP=$P/STOP_PERSIST; LOCK=$P/gpu.lock
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REQ_ZH='请写一段简短的描述。'; REQ_EN='Please write a short description.'
LANE=${1:?lane Q1, Q2, Q3 or R}

log() { echo "[queue $LANE] $(date '+%F %T') $*" | tee -a $P/queue_pro.log; }

task() {  # lock(s|x) mem phase(1|3) order tag model kw lang gate pythonpath net [extra script args]
  local lk=$1 mem=$2 ph=$3 order=$4 tag=$5 m=$H/$6 kw=$7 lang=$8 gate=$9 pp=${10} net=${11}; shift 11
  local dir=$order; [ $ph = 3 ] && dir=p3$order
  local out=$P/$dir/$tag lg=$P/${dir}_${tag}.log
  [ -e $STOP ] && { log "stop flag: not starting $dir/$tag"; return 2; }
  [ -e $out/DONE ] && { log "$dir/$tag already DONE"; return 0; }
  nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: lane stops before $dir/$tag"; touch $P/DRIVER_DOWN; return 3; }
  local bodies=$B1 a
  [ $ph = 3 ] && bodies=$B3
  a=(--model $m --tmpl-kwargs "$kw" --bodies $bodies --out $out --stop-file $STOP --order $order --min-free-gib 8 --max-memory-gib $mem)
  if [ $lang = en ]; then [ $ph = 3 ] && a+=(--lang en --bodies-en $E3) || a+=(--lang en --bodies-en bodies_en_150.json); fi
  if [ $order = natural ]; then if [ $lang = en ]; then a+=(--request "$REQ_EN"); else a+=(--request "$REQ_ZH"); fi; fi
  [ $ph = 1 ] && [ $order = main ] && [ -n "$gate" ] && a+=(--gate-g $gate --gate-strict)
  [ $ph = 3 ] && a+=(--lean)
  a+=("$@")
  local fl=-s; [ $lk = x ] && fl=-x
  log "waiting for the $lk lock: $dir/$tag"
  (
    flock $fl 9
    log "start $dir/$tag (max $mem GiB, lock $lk, net $net)"
    if [ $net = 1 ]; then env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $S "${a[@]}" >> $lg 2>&1
    else CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $S "${a[@]}" >> $lg 2>&1; fi
  ) 9>$LOCK
  [ -e $out/DONE ] && { log "DONE $dir/$tag"; return 0; }
  [ -e $STOP ] && { log "stopped in $dir/$tag (stop flag)"; return 2; }
  grep -q "CUDA unavailable" $lg && { log "CUDA unavailable in $dir/$tag: lane stops"; touch $P/DRIVER_DOWN; return 3; }
  grep -q "GiB free <" $lg && { log "disk below the limit in $dir/$tag: lane stops"; touch $P/DISK_LOW; return 2; }
  log "FAILED $dir/$tag, see $lg; lane continues"; return 1
}
t() { task "$@"; local rc=$?; [ $rc -ge 2 ] && exit $rc; return 0; }
waitfor() { while :; do local ok=1; for f in "$@"; do [ -e "$f" ] || ok=0; done; [ $ok = 1 ] && return 0; [ -e $STOP ] && exit 2; sleep 60; done; }

NT='{"enable_thinking": false}'; Q38='{"enable_thinking": false}'; Q38T='{"enable_thinking": true}'
D19='--fixed-date 2026-10-19'; D03='--fixed-date 2026-11-03'
waitfor $P/natural/qwen38_27b_think/DONE
case $LANE in
  Q1)
    t s 44 3 main gpt_oss_20b_en openai/gpt-oss-20b '{}' en '' $TF19 1 $D19
    t s 44 3 natural gpt_oss_20b_en openai/gpt-oss-20b '{}' en '' $TF19 1 $D19
    t s 44 3 main mimo_v26_distill_qwen9b XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B "$NT" zh '' '' 0
    t s 44 3 natural mimo_v26_distill_qwen9b XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B "$NT" zh '' '' 0
    log "lane done" ;;
  Q2)
    t s 44 3 main apertus8b_en swiss-ai/Apertus-8B-Instruct-2509 '{}' en '' '' 0 $D03
    t s 44 3 natural apertus8b_en swiss-ai/Apertus-8B-Instruct-2509 '{}' en '' '' 0 $D03
    t s 44 3 main mimo_vl7b_rl_2508 XiaomiMiMo/MiMo-VL-7B-RL-2508 '{}' zh '' '' 0
    t s 44 3 natural mimo_vl7b_rl_2508 XiaomiMiMo/MiMo-VL-7B-RL-2508 '{}' zh '' '' 0
    log "lane done" ;;
  Q3)
    t x 90 3 main qwen38_27b Qwen/Qwen3.8-27B "$Q38" zh '' '' 0
    t x 90 3 natural qwen38_27b Qwen/Qwen3.8-27B "$Q38" zh '' '' 0
    t x 90 3 main qwen38_27b_think Qwen/Qwen3.8-27B "$Q38T" zh '' '' 0
    t x 90 3 natural qwen38_27b_think Qwen/Qwen3.8-27B "$Q38T" zh '' '' 0
    log "lane done" ;;
  R)
    waitfor $P/dl_gptoss120b.ok
    t x 90 1 main gpt_oss_120b_en openai/gpt-oss-120b '{}' en suite_think/gpt_oss_120b_en $TF19 1 $D19
    t x 90 1 natural gpt_oss_120b_en openai/gpt-oss-120b '{}' en '' $TF19 1 $D19
    t x 90 3 main gpt_oss_120b_en openai/gpt-oss-120b '{}' en '' $TF19 1 $D19
    t x 90 3 natural gpt_oss_120b_en openai/gpt-oss-120b '{}' en '' $TF19 1 $D19
    waitfor $P/dl_olmo2_7b.ok
    t x 44 1 main olmo2_7b_en allenai/OLMo-2-1124-7B-Instruct '{}' en suite/olmo2_7b_en '' 0
    t x 44 1 natural olmo2_7b_en allenai/OLMo-2-1124-7B-Instruct '{}' en '' '' 0
    t x 44 3 main olmo2_7b_en allenai/OLMo-2-1124-7B-Instruct '{}' en '' '' 0
    t x 44 3 natural olmo2_7b_en allenai/OLMo-2-1124-7B-Instruct '{}' en '' '' 0
    log "lane done" ;;
esac
