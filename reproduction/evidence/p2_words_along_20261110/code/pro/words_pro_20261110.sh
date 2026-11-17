#!/bin/bash
# P2 ordinary words along the message, PRO queue (11-10, ../../PROTOCOL_zh.md).
# One RTX PRO 6000, at most two tasks at a time: lanes S1 and S2 take the shared flock on persistence/gpu.lock (44 GiB each),
# lane X takes the exclusive one (90 GiB) and runs alone (the pilot's lock file, so other PRO jobs obey the same rule).
# Natural order only for the models of the paper's natural-order appendix (app:natural) and the natural-only models (user 11-10).
#   S1  weights on PRO: MiMo-V2.6 (b150 main 28 new; p3 main 32), gpt-oss-20b (b150 main 28 new, natural 32; p3 main 32)
#   S2  small downloads, one at a time: Apertus-8B and OLMo-2-7B (main 28 new, natural 32), MiMo-VL-7B (main 28 new),
#       OLMo-2-13B (untested: main roles + 32), Ministral-3-14B Instruct and Reasoning (natural only: roles + 32, gate on
#       suite_nat3 N)
#   X   gpt-oss-120b (b150 main 28 new; p3 32) first; then Qwen3.8-Flash-Next (Stage 1 critical path, main roles + 32, FP8 +
#       CPU offload, ~7 s per forward): 172.8 GiB, so PRO's gpt-oss-120b weights are deleted first (user approval 11-10, once
#       both of its cells are DONE with 75 + 200 bodies and extracted; the queue then writes words_along/GO_DEL_GPTOSS120B;
#       lane S2 pauses its downloads while Flash-Next holds the disk); then Qwen3.8-27B
#       thinking (b150 main 28 new; p3 32) and non-thinking (p3 32), then Qwen3.6-27B (download; b150 main roles + 32 and p3
#       roles + 32, gate on the paper's G of crossmodel/qwen36_27b); Flash-Next at the end if the go came later.
# Tested settings gate against their pilot run (persistence/<order>/<tag>, persistence/p3main/<tag>). Downloads: pinned commits,
# hf-mirror, no xet, into /root/hf_models_words; deleted (literal paths) once every run of the model is DONE and extracted. The
# weights already on PRO are not deleted. Stop: touch words_along/STOP_WORDS; no signals.
# usage (tmux session words, one window per lane): bash words_pro_20261110.sh S1|S2|X
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
H=/root/hf_models; D=/root/hf_models_words; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
B3=$C/phase3_bodies_zh.jsonl; E3=$C/phase3_bodies_en.json
S=words_along_20261110.py; X=extract_gram_20261110.py; WJ=$C/words_v2.json
P=$C/persistence; O=$C/words_along; E=$O/extracts; STOP=$O/STOP_WORDS; LOCK=$P/gpu.lock
mkdir -p $O $E $D
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
REQ_ZH='请写一段简短的描述。'; REQ_EN='Please write a short description.'
LANE=${1:?lane S1, S2 or X}
log() { echo "[words $LANE] $(date '+%F %T') $*" | tee -a $O/queue_pro.log; }
free_gib() { df --output=avail -BG /root | tail -1 | tr -dc 0-9; }

extract() {  # rundir set order tag source
  [ -e $E/$2/$3/$4/$5.json ] && return 0
  mkdir -p $E/$2/$3/$4
  $PY $X $1 $E/$2/$3/$4/$5 >> $O/extract.log 2>&1 && log "extracted $2/$3/$4/$5" || { log "extraction FAILED $1"; return 1; }
}

task() {  # lock(s|x) mem set(b150|p3) order tag model kw lang mode(new|all|roles) pythonpath net gate-args... -- [extra]
  local lk=$1 mem=$2 st=$3 order=$4 tag=$5 m=$6 kw=$7 lang=$8 mode=$9 pp=${10} net=${11}; shift 11
  local g=() x=() a
  while [ $# -gt 0 ] && [ "$1" != -- ]; do g+=("$1"); shift; done; [ $# -gt 0 ] && shift; x=("$@")
  local out=$O/${st}_$order/$tag lg=$O/${st}_${order}_$tag.log
  [ -e $STOP ] && { log "stop flag: not starting $st/$order/$tag"; return 2; }
  if [ ! -e $out/DONE ]; then
    nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: lane stops before $st/$order/$tag"; touch $O/DRIVER_DOWN; return 3; }
    local bodies=$B1 ben=bodies_en_150.json
    [ $st = p3 ] && { bodies=$B3; ben=$E3; }
    a=(--model $m --tmpl-kwargs "$kw" --bodies $bodies --words $WJ --out $out --order $order --stop-file $STOP --min-free-gib 8 --max-memory-gib $mem)
    [ $lang = en ] && a+=(--lang en --bodies-en $ben)
    if [ $order = natural ]; then if [ $lang = en ]; then a+=(--request "$REQ_EN"); else a+=(--request "$REQ_ZH"); fi; fi
    case $mode in roles) a+=(--roles --pairs all) ;; new) a+=(--pairs new) ;; all) a+=(--pairs all) ;; esac
    a+=("${g[@]}" "${x[@]}")
    local fl=-s; [ $lk = x ] && fl=-x
    log "waiting for the $lk lock: $st/$order/$tag"
    (
      flock $fl 9
      log "start $st/$order/$tag (max $mem GiB, lock $lk, net $net)"
      if [ $net = 1 ]; then env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $S "${a[@]}" >> $lg 2>&1 < /dev/null
      else HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $S "${a[@]}" >> $lg 2>&1 < /dev/null; fi
    ) 9>$LOCK
  fi
  if [ -e $out/DONE ]; then
    log "DONE $st/$order/$tag"
    extract $out $st $order $tag words || return 1
    if [ $mode != roles ]; then
      local pd=$P/$order/$tag; [ $st = p3 ] && pd=$P/p3$order/$tag
      [ -e $pd/DONE ] && extract $pd $st $order $tag pilot
    fi
    return 0
  fi
  [ -e $STOP ] && { log "stopped in $st/$order/$tag (stop flag)"; return 2; }
  grep -q "GATE FAILED" $lg && { log "GATE FAILED in $st/$order/$tag, see $lg; lane continues"; return 1; }
  grep -q "CUDA unavailable" $lg && { log "CUDA unavailable in $st/$order/$tag: lane stops"; touch $O/DRIVER_DOWN; return 3; }
  grep -q "GiB free <" $lg && { log "disk below the limit in $st/$order/$tag: lane stops"; touch $O/DISK_LOW; return 2; }
  log "FAILED $st/$order/$tag, see $lg; lane continues"; return 1
}
t() { task "$@"; local rc=$?; [ $rc -ge 2 ] && exit $rc; return 0; }

dl() {  # tag repo commit gib [ignore patterns as python list] [margin GiB, default 30]
  local tag=$1 repo=$2 rev=$3 gb=$4 ign=${5:-[]} mg=${6:-30} dir=$D/$2
  [ -e $O/dl_$tag.ok ] && return 0
  # Flash-Next reserves the disk from the moment lane X reaches it until its weights are deleted again
  while [ $tag != qwen38_flashnext ] && [ -e $O/FN_RESERVE ]; do [ -e $STOP ] && exit 2; log "disk reserved for Flash-Next: $repo waits"; sleep 300; done
  until [ $(free_gib) -ge $(( gb + mg )) ]; do [ -e $STOP ] && exit 2; log "waiting for $(( gb + mg )) GiB free ($(free_gib)) for $repo"; sleep 300; done
  log "download $repo@${rev:0:12} ($gb GiB, $(free_gib) GiB free)"
  for i in 1 2 3 4 5 6; do
    [ -e $STOP ] && exit 2
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=$TF19 $PY -c "from huggingface_hub import snapshot_download; p = snapshot_download('$repo', revision='$rev', local_dir='$dir', max_workers=8, ignore_patterns=$ign); print('snapshot', p)" >> $O/dl_$tag.log 2>&1 && {
      log "downloaded $repo $(du -sh $dir | cut -f1)"; touch $O/dl_$tag.ok; return 0; }
    log "$repo attempt $i failed"; sleep 120
  done
  touch $O/dl_$tag.failed; log "download FAILED $repo"; return 1
}

del_weights() {  # literal paths of this round's downloads only
  case $1 in
    apertus8b_en) rm -rf /root/hf_models_words/swiss-ai/Apertus-8B-Instruct-2509 ;;
    mimo_vl7b_rl_2508) rm -rf /root/hf_models_words/XiaomiMiMo/MiMo-VL-7B-RL-2508 ;;
    olmo2_7b_en) rm -rf /root/hf_models_words/allenai/OLMo-2-1124-7B-Instruct ;;
    olmo2_13b_en) rm -rf /root/hf_models_words/allenai/OLMo-2-1124-13B-Instruct ;;
    ministral3_14b_instruct_en) rm -rf /root/hf_models_words/mistralai/Ministral-3-14B-Instruct-2512-BF16 ;;
    ministral3_14b_reasoning_en) rm -rf /root/hf_models_words/mistralai/Ministral-3-14B-Reasoning-2512 ;;
    qwen36_27b) rm -rf /root/hf_models_words/Qwen/Qwen3.6-27B ;;
    qwen38_flashnext) rm -rf /root/hf_models_words/Qwen/Qwen3.8-Flash-Next-FP8 ;;
    *) log "del_weights: unknown tag $1, nothing deleted" ;;
  esac
}

finish() {  # tag dir1 [dir2 ...]: delete the weights when every listed run is DONE and extracted
  local tag=$1 ok=1 d; shift
  for d in "$@"; do [ -e $O/$d/DONE ] || ok=0; done
  if [ $ok = 1 ]; then del_weights $tag; touch $O/run_$tag.done; log "finished $tag; weights deleted ($(free_gib) GiB free)"
  else log "$tag not complete: weights kept"; fi
}

dry() {  # tag model kw lang [extra]: tokenizer-only render check
  local tag=$1 m=$2 kw=$3 lang=$4; shift 4; local e=()
  [ $lang = en ] && e=(--lang en --bodies-en bodies_en_150.json)
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=${DPP:-} $PY $S --model $m --tmpl-kwargs "$kw" --bodies $B1 --words $WJ --out $O/dry_$tag "${e[@]}" "$@" --dry-run > $O/$tag.dry.log 2>&1 < /dev/null
  grep -q 'DRY RUN OK' $O/$tag.dry.log || { log "dry run FAILED for $tag (weights kept), see $O/$tag.dry.log"; return 1; }
}

verified_120b() {  # both gpt-oss-120b cells DONE, every body on disk (75 and 200 body records) and extracted
  local a b
  a=$(ls $O/b150_main/gpt_oss_120b_en/*.json 2>/dev/null | grep -vc identity.json)
  b=$(ls $O/p3_main/gpt_oss_120b_en/*.json 2>/dev/null | grep -vc identity.json)
  [ -e $O/b150_main/gpt_oss_120b_en/DONE ] && [ -e $O/p3_main/gpt_oss_120b_en/DONE ] && [ "$a" = 75 ] && [ "$b" = 200 ] &&     [ -e $E/b150/main/gpt_oss_120b_en/words.json ] && [ -e $E/p3/main/gpt_oss_120b_en/words.json ]
}

flashnext() {  # Stage 1, main order only (not in the natural-order appendix). The user approved (11-10) deleting PRO's
               # gpt-oss-120b weights once both of its cells are finished and verified; the queue checks that and writes the flag.
  [ -e $O/run_qwen38_flashnext.done ] && return 0
  until verified_120b; do
    [ "${1:-}" = wait ] || { log "Flash-Next deferred: gpt-oss-120b cells not verified yet"; return 0; }
    [ -e $STOP ] && exit 2; log "Flash-Next waits for verified gpt-oss-120b cells"; sleep 300
  done
  [ -e $O/GO_DEL_GPTOSS120B ] || { echo "user approval 11-10; gpt-oss-120b cells DONE, 75 + 200 bodies, extracted ($(date '+%F %T'))" > $O/GO_DEL_GPTOSS120B
    log "GO_DEL_GPTOSS120B written: user approval 11-10, both gpt-oss-120b cells verified"; }
  touch $O/FN_RESERVE
  if [ -d /root/hf_models/openai/gpt-oss-120b ]; then
    rm -rf /root/hf_models/openai/gpt-oss-120b; log "deleted PRO's gpt-oss-120b weights for Flash-Next (user go), $(free_gib) GiB free"; fi
  dl qwen38_flashnext Qwen/Qwen3.8-Flash-Next-FP8 236dfdf285828023ca3bcd3f37366c58a3469b13 173 '[]' 10 && \
    DPP=$TF19 dry qwen38_flashnext $D/Qwen/Qwen3.8-Flash-Next-FP8 "$NT" zh --roles && \
    t x 84 b150 main qwen38_flashnext $D/Qwen/Qwen3.8-Flash-Next-FP8 "$NT" zh roles $TF19 1 --gate-g suite_think/qwen38_flashnext --gate-tmpl-kwargs '{}' -- --cpu-offload-gib 52
  finish qwen38_flashnext b150_main/qwen38_flashnext
  [ -e $O/run_qwen38_flashnext.done ] && rm -f $O/FN_RESERVE
}

NT='{"enable_thinking": false}'; Q38T='{"enable_thinking": true}'
D19='--fixed-date 2026-10-19'; D03='--fixed-date 2026-11-03'
case $LANE in
  S1)   # natural order only where the paper's natural-order appendix has the model (gpt-oss-20b yes, MiMo-V2.6 no)
    t s 44 b150 main mimo_v26_distill_qwen9b $H/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B "$NT" zh new '' 0 --gate-pilot $P/main/mimo_v26_distill_qwen9b
    for o in main natural; do
      t s 44 b150 $o gpt_oss_20b_en $H/openai/gpt-oss-20b '{}' en $([ $o = main ] && echo new || echo all) $TF19 1 --gate-pilot $P/$o/gpt_oss_20b_en -- $D19
    done
    t s 44 p3 main mimo_v26_distill_qwen9b $H/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B "$NT" zh all '' 0 --gate-pilot $P/p3main/mimo_v26_distill_qwen9b
    t s 44 p3 main gpt_oss_20b_en $H/openai/gpt-oss-20b '{}' en all $TF19 1 --gate-pilot $P/p3main/gpt_oss_20b_en -- $D19
    log "lane S1 done" ;;
  S2)
    dl apertus8b_en swiss-ai/Apertus-8B-Instruct-2509 b946d40447b2b597999b9c86d44bee0b452c919f 15 && dry apertus8b_en $D/swiss-ai/Apertus-8B-Instruct-2509 '{}' en --pairs new $D03 && \
      for o in main natural; do t s 44 b150 $o apertus8b_en $D/swiss-ai/Apertus-8B-Instruct-2509 '{}' en $([ $o = main ] && echo new || echo all) '' 0 --gate-pilot $P/$o/apertus8b_en -- $D03; done
    finish apertus8b_en b150_main/apertus8b_en b150_natural/apertus8b_en
    dl mimo_vl7b_rl_2508 XiaomiMiMo/MiMo-VL-7B-RL-2508 ec4ebdc29b83b94aa7f62670c78376250dad8dbc 16 && dry mimo_vl7b_rl_2508 $D/XiaomiMiMo/MiMo-VL-7B-RL-2508 '{}' zh --pairs new && \
      t s 44 b150 main mimo_vl7b_rl_2508 $D/XiaomiMiMo/MiMo-VL-7B-RL-2508 '{}' zh new '' 0 --gate-pilot $P/main/mimo_vl7b_rl_2508
    finish mimo_vl7b_rl_2508 b150_main/mimo_vl7b_rl_2508
    dl olmo2_7b_en allenai/OLMo-2-1124-7B-Instruct 470b1fba1ae01581f270116362ee4aa1b97f4c84 14 && dry olmo2_7b_en $D/allenai/OLMo-2-1124-7B-Instruct '{}' en --pairs new && \
      for o in main natural; do t s 44 b150 $o olmo2_7b_en $D/allenai/OLMo-2-1124-7B-Instruct '{}' en $([ $o = main ] && echo new || echo all) '' 0 --gate-pilot $P/$o/olmo2_7b_en; done
    finish olmo2_7b_en b150_main/olmo2_7b_en b150_natural/olmo2_7b_en
    dl olmo2_13b_en allenai/OLMo-2-1124-13B-Instruct 3a5c85baefbb1896a54d56fe2e76c0395627ddf4 26 && dry olmo2_13b_en $D/allenai/OLMo-2-1124-13B-Instruct '{}' en --roles && \
      t s 44 b150 main olmo2_13b_en $D/allenai/OLMo-2-1124-13B-Instruct '{}' en roles '' 0 --gate-g suite/olmo2_13b_en
    finish olmo2_13b_en b150_main/olmo2_13b_en
    for mi in "ministral3_14b_instruct_en mistralai/Ministral-3-14B-Instruct-2512-BF16 3cea74c1ebaf5ce5f5a2553de470e2ceab825142" \
              "ministral3_14b_reasoning_en mistralai/Ministral-3-14B-Reasoning-2512 51f9210f3cd20f3452a80d5819d15dc61cc50630"; do
      set -- $mi
      dl $1 $2 $3 26 "['consolidated*']" && DPP=$TF19 dry $1 $D/$2 '{}' en --roles --order natural --request "$REQ_EN" && \
        t s 44 b150 natural $1 $D/$2 '{}' en roles $TF19 0 --gate-g suite_nat3/$1 --gate-sub N
      finish $1 b150_natural/$1
    done
    log "lane S2 done" ;;
  X)   # gpt-oss-120b first, so that Flash-Next (the critical path) can follow as soon as the user allows deleting its weights;
       # Qwen3.8-27B thinking and gpt-oss-120b are not in the natural-order appendix: main order (and p3) only
    t x 90 b150 main gpt_oss_120b_en $H/openai/gpt-oss-120b '{}' en new $TF19 1 --gate-pilot $P/main/gpt_oss_120b_en -- $D19
    t x 90 p3 main gpt_oss_120b_en $H/openai/gpt-oss-120b '{}' en all $TF19 1 --gate-pilot $P/p3main/gpt_oss_120b_en -- $D19
    flashnext
    t x 90 b150 main qwen38_27b_think $H/Qwen/Qwen3.8-27B "$Q38T" zh new '' 0 --gate-pilot $P/main/qwen38_27b_think
    t x 90 p3 main qwen38_27b $H/Qwen/Qwen3.8-27B "$NT" zh all '' 0 --gate-pilot $P/p3main/qwen38_27b
    t x 90 p3 main qwen38_27b_think $H/Qwen/Qwen3.8-27B "$Q38T" zh all '' 0 --gate-pilot $P/p3main/qwen38_27b_think
    dl qwen36_27b Qwen/Qwen3.6-27B 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 52 && dry qwen36_27b $D/Qwen/Qwen3.6-27B "$NT" zh --roles && {
      t x 90 b150 main qwen36_27b $D/Qwen/Qwen3.6-27B "$NT" zh roles '' 0 --gate-g qwen36_27b
      t x 90 p3 main qwen36_27b $D/Qwen/Qwen3.6-27B "$NT" zh roles '' 0 --gate-g qwen36_27b -- --gate-bodies $B1
    }
    finish qwen36_27b b150_main/qwen36_27b p3_main/qwen36_27b
    flashnext wait
    log "lane X done" ;;
esac
