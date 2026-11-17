#!/bin/bash
# P2 ordinary words along the message, server3 queue (11-10, ../../PROTOCOL_zh.md).
# Stage 1 order (user 11-10, second decision): the other models first, GLM-5.3-Flash last. Lanes U (GPUs 0,1) and B (GPUs 2,3)
# start at once; lane A (Qwen3.8-27B) on GPUs 0,1 after lane U; lane R (R1-Distill-Llama-70B) on GPUs 0-3 after lanes A and B;
# lane D (DeepSeek-V4.1-Flash, natural order, words_dsv41_s3_20261110.sh: 2-body smoke with the gate against the paper's
# natural3 features, then 4 shards only if the gate passes) after lane R; lane G53 (GLM-5.3-Flash, GPUs 0-3, main order,
# roles + the first pair of each category, ~64 s per forward, ~25 h) after lanes U, B, A, R, D and 5 idle minutes on every GPU.
# Lane C (11-10, added after the launch because the downloads run at ~7 MB/s): on GPUs 2,3 once lane B is done, while lane U
# still downloads, Qwen3.8-27B main and natural (lane A, which waits for lane U, then finds them DONE) and the DeepSeek-V4.1-Flash
# smoke with its gate on GPU 2 alone (words_dsv41_s3_20261110.sh smoke); lane D then runs the 4 shards only if that gate passed.
# Never more than one task per GPU.
# Stage 2 (later, only with the flag file words_along/GO_GLM53_STAGE2): lane G53B, GLM-5.3-Flash's other 24 pairs, same
# environment, gate against its stage-1 run (role states and two of its word pairs); stage-1 cells are not re-run.
# Tested settings (11-06 pilot run on this machine): 28 new pairs in the main order, 32 pairs in the natural order, gate against
# the pilot run. Untested models: roles + 32 pairs, main order, gate against the paper's G features; downloaded one at a time
# into hf_models_words (pinned commit, hf-mirror, no xet), deleted (literal path) once every run of the model is DONE and
# extracted. server3's own models are never deleted. Environment: the pilot's (venv torch 2.5.1, transformers 5.8).
# Stop: touch words_along/STOP_WORDS (checked before every body and task); no signals are ever sent.
# usage (in the crossmodel directory, tmux session words, one window per lane): bash words_s3_20261110.sh U|B|C|A|R|D|G53|G53B
set -u
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C || exit 1
PY=/root/recursive_state_moe_server3_20260711/venv/bin/python
H=/root/data/hf_models; D=/root/data/hf_models_words; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=words_along_20261110.py; X=extract_gram_20261110.py; WJ=$C/words_v2.json
P=$C/persistence; O=$C/words_along; E=$O/extracts; STOP=$O/STOP_WORDS
mkdir -p $O $E $D
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
REQ_ZH='请写一段简短的描述。'; REQ_EN='Please write a short description.'
LANE=${1:?lane U, B, C, A, R, D, G53 or G53B}
log() { echo "[words $LANE] $(date '+%F %T') $*" | tee -a $O/queue_s3.log; }
free_gib() { df --output=avail -BG /root/data | tail -1 | tr -dc 0-9; }

extract() {  # rundir set order tag source
  [ -e $E/$2/$3/$4/$5.json ] && return 0
  mkdir -p $E/$2/$3/$4
  $PY $X $1 $E/$2/$3/$4/$5 >> $O/extract.log 2>&1 && log "extracted $2/$3/$4/$5" || { log "extraction FAILED $1"; return 1; }
}

task() {  # gpus order tag model kw lang mode(new|all|roles|roles_first|rest) gate-args... -- [extra script args]
  local gpus=$1 order=$2 tag=$3 m=$4 kw=$5 lang=$6 mode=$7; shift 7
  local g=() x=() a src=words
  while [ $# -gt 0 ] && [ "$1" != -- ]; do g+=("$1"); shift; done; [ $# -gt 0 ] && shift; x=("$@")
  local rt=$tag; [ $mode = rest ] && { rt=${tag}_stage2; src=words_stage2; }
  local out=$O/b150_$order/$rt lg=$O/b150_${order}_$rt.log
  [ -e $STOP ] && { log "stop flag: not starting $order/$tag"; return 2; }
  if [ ! -e $out/DONE ]; then
    nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: lane stops before $order/$tag"; touch $O/DRIVER_DOWN; return 3; }
    a=(--model $m --tmpl-kwargs "$kw" --bodies $B1 --words $WJ --out $out --order $order --stop-file $STOP --min-free-gib 4 --max-memory-gib 44)
    [ $lang = en ] && a+=(--lang en --bodies-en bodies_en_150.json)
    if [ $order = natural ]; then if [ $lang = en ]; then a+=(--request "$REQ_EN"); else a+=(--request "$REQ_ZH"); fi; fi
    case $mode in roles) a+=(--roles --pairs all) ;; roles_first) a+=(--roles --pairs first) ;; rest) a+=(--pairs rest) ;;
                  new) a+=(--pairs new) ;; all) a+=(--pairs all) ;; esac
    a+=("${g[@]}" "${x[@]}")
    log "start $order/$tag on GPUs $gpus"
    if [ $tag = moonlight16b_a3b_instruct ]; then CUDA_VISIBLE_DEVICES=$gpus $PY moonlight_compat_20261103.py $m $S "${a[@]}" >> $lg 2>&1 < /dev/null
    else CUDA_VISIBLE_DEVICES=$gpus $PY $S "${a[@]}" >> $lg 2>&1 < /dev/null; fi
  fi
  if [ -e $out/DONE ]; then
    log "DONE $order/$tag"
    extract $out b150 $order $tag $src || return 1
    case $mode in new|all) extract $P/$order/$tag b150 $order $tag pilot ;; esac
    return 0
  fi
  [ -e $STOP ] && { log "stopped in $order/$tag (stop flag)"; return 2; }
  grep -q "GATE FAILED" $lg && { log "GATE FAILED in $order/$tag, see $lg; lane continues"; return 1; }
  grep -q "CUDA unavailable" $lg && { log "CUDA unavailable in $order/$tag: lane stops"; touch $O/DRIVER_DOWN; return 3; }
  grep -q "GiB free <" $lg && { log "disk below the limit in $order/$tag: lane stops"; touch $O/DISK_LOW; return 2; }
  log "FAILED $order/$tag, see $lg; lane continues"; return 1
}
t() { task "$@"; local rc=$?; [ $rc -ge 2 ] && exit $rc; return 0; }
waitfor() { while :; do local ok=1; for f in "$@"; do [ -e "$f" ] || ok=0; done; [ $ok = 1 ] && return 0; [ -e $STOP ] && exit 2; sleep 60; done; }

dl() {  # tag repo commit gib
  local tag=$1 repo=$2 rev=$3 gb=$4 dir=$D/$2
  [ -e $O/dl_$tag.ok ] && return 0
  until [ $(free_gib) -ge $(( gb + 6 )) ]; do [ -e $STOP ] && exit 2; log "waiting for $(( gb + 6 )) GiB free ($(free_gib)) for $repo"; sleep 300; done
  log "download $repo@${rev:0:12} ($gb GiB, $(free_gib) GiB free)"
  for i in 1 2 3 4 5 6; do
    [ -e $STOP ] && exit 2
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY -c "from huggingface_hub import snapshot_download; p = snapshot_download('$repo', revision='$rev', local_dir='$dir', max_workers=8); print('snapshot', p)" >> $O/dl_$tag.log 2>&1 && {
      log "downloaded $repo $(du -sh $dir | cut -f1)"; touch $O/dl_$tag.ok; return 0; }
    log "$repo attempt $i failed"; sleep 120
  done
  touch $O/dl_$tag.failed; log "download FAILED $repo"; return 1
}

del_weights() {  # literal paths of this round's downloads only
  case $1 in
    qwen36_35b_a3b) rm -rf /root/data/hf_models_words/Qwen/Qwen3.6-35B-A3B ;;
    qwen35_27b) rm -rf /root/data/hf_models_words/Qwen/Qwen3.5-27B ;;
    glm47_flash) rm -rf /root/data/hf_models_words/zai-org/GLM-4.7-Flash ;;
    granite42_30b_en) rm -rf /root/data/hf_models_words/ibm-granite/granite-4.2-30b ;;
    *) log "del_weights: unknown tag $1, nothing deleted" ;;
  esac
}

untested() {  # tag repo commit gib lang kw   (main order, roles + 32 pairs, gate against suite/<tag>/G)
  local tag=$1 repo=$2 rev=$3 gb=$4 lang=$5 kw=$6 m=$D/$2 e=()
  [ -e $O/run_$tag.done ] && return 0
  dl $tag $repo $rev $gb || { log "skip $tag: download failed"; return 0; }
  [ $lang = en ] && e=(--lang en --bodies-en bodies_en_150.json)
  $PY $S --model $m --tmpl-kwargs "$kw" --bodies $B1 --words $WJ --out $O/dry_$tag --roles --pairs all "${e[@]}" --dry-run > $O/$tag.dry.log 2>&1 < /dev/null
  grep -q 'DRY RUN OK' $O/$tag.dry.log || { log "dry run FAILED for $tag (weights kept), see $O/$tag.dry.log"; return 0; }
  t 0,1 main $tag $m "$kw" $lang roles --gate-g suite/$tag
  if [ -e $O/b150_main/$tag/DONE ] && [ -e $E/b150/main/$tag/words.json ]; then
    del_weights $tag; touch $O/run_$tag.done; log "finished $tag; weights deleted ($(free_gib) GiB free)"
  else log "$tag not complete: weights kept"; fi
}

NT='{"enable_thinking": false}'
case $LANE in
  D)   # DeepSeek-V4.1-Flash, natural order, roles + 32 pairs: own reader on the official runner; smoke + gate first
    waitfor $O/laneR.done $C/words_dsv41_s3_20261110.sh
    bash $C/words_dsv41_s3_20261110.sh; log "lane D finished (rc $?)"
    touch $O/laneD.done ;;
  G53)   # stage 1, last: main order, roles + the first pair of each category (paper environment of GLM-5.3-Flash)
    waitfor $O/laneU.done $O/laneB.done $O/laneA.done $O/laneR.done $O/laneD.done
    idle=0
    while [ $idle -lt 5 ]; do [ -e $STOP ] && exit 2
      if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; then idle=$((idle + 1)); else idle=0; fi; sleep 60; done
    log "GPUs idle for 5 minutes: GLM-5.3 stage 1 starts"
    PY=/root/data/venv_t26/bin/python; export P2_FP8_FORCE_FALLBACK=1
    t 0,1,2,3 main glm53_flash $H/zai-org/GLM-5.3-Flash '{}' zh roles_first --gate-g suite_think/glm53_flash -- --max-memory-gib 40 --cpu-offload-gib 300
    touch $O/laneG53.done; log "lane G53 (stage 1) done" ;;
  U)   # largest first, while the outputs of the other lanes are still small
    untested qwen36_35b_a3b Qwen/Qwen3.6-35B-A3B 995ad96eacd98c81ed38be0c5b274b04031597b0 67 zh "$NT"
    untested glm47_flash zai-org/GLM-4.7-Flash 7dd20894a642a0aa287e9827cb1a1f7f91386b67 59 zh "$NT"
    untested granite42_30b_en ibm-granite/granite-4.2-30b 9e668ce1c538387ef24d3644e9b0606647762636 55 en "$NT"
    untested qwen35_27b Qwen/Qwen3.5-27B fc05daec18b0a78c049392ed2e771dde82bdf654 52 zh "$NT"
    touch $O/laneU.done; log "lane U done" ;;
  B)
    for o in main natural; do
      t 2,3 $o gemma4_31b_en $H/google/gemma-4-31B-it '{}' en new --gate-pilot $P/$o/gemma4_31b_en
      t 2,3 $o glm46v_flash $H/zai-org/GLM-4.6V-Flash "$NT" zh new --gate-pilot $P/$o/glm46v_flash
      t 2,3 $o phi4_en $H/microsoft/phi-4 '{}' en new --gate-pilot $P/$o/phi4_en
      t 2,3 $o ernie45_21b_a3b_thinking $H/baidu/ERNIE-4.5-21B-A3B-Thinking '{}' zh new --gate-pilot $P/$o/ernie45_21b_a3b_thinking
      t 2,3 $o mimo7b_rl_0530 $H/XiaomiMiMo/MiMo-7B-RL-0530 "$NT" zh new --gate-pilot $P/$o/mimo7b_rl_0530
      t 2,3 $o moonlight16b_a3b_instruct $H/moonshotai/Moonlight-16B-A3B-Instruct '{}' zh new --gate-pilot $P/$o/moonlight16b_a3b_instruct
      t 2,3 $o exaone4_32b_en $H/LGAI-EXAONE/EXAONE-4.0.1-32B "$NT" en new --gate-pilot $P/$o/exaone4_32b_en
      t 2,3 $o granite41_30b_en $H/ibm-granite/granite-4.1-30b '{}' en new --gate-pilot $P/$o/granite41_30b_en
      t 2,3 $o seed_oss_36b $H/ByteDance-Seed/Seed-OSS-36B-Instruct '{"thinking_budget": 0}' zh new --gate-pilot $P/$o/seed_oss_36b
    done
    touch $O/laneB.done; log "lane B done" ;;
  C)   # GPUs 2,3 after lane B, while lane U holds GPUs 0,1 for its downloads and runs (>= 233 GB at ~7 MB/s)
    waitfor $O/laneB.done
    if [ -e $O/laneU.done ]; then log "lane U already done: Qwen3.8-27B is left to lane A"
    else for o in main natural; do t 2,3 $o qwen38_27b $H/Qwen/Qwen3.8-27B "$NT" zh new --gate-pilot $P/$o/qwen38_27b; done; fi
    SMOKE_GPU=2 bash $C/words_dsv41_s3_20261110.sh smoke; log "DeepSeek-V4.1-Flash smoke finished (rc $?)"
    touch $O/laneC.done; log "lane C done" ;;
  A)
    waitfor $O/laneU.done
    for o in main natural; do
      t 0,1 $o qwen38_27b $H/Qwen/Qwen3.8-27B "$NT" zh new --gate-pilot $P/$o/qwen38_27b
    done
    touch $O/laneA.done; log "lane A done" ;;
  R)
    waitfor $O/laneA.done $O/laneB.done
    for o in main natural; do
      t 0,1,2,3 $o r1_distill_llama70b $H/deepseek-ai/DeepSeek-R1-Distill-Llama-70B '{}' zh new --gate-pilot $P/$o/r1_distill_llama70b
    done
    touch $O/laneR.done; log "lane R done" ;;
  G53B)   # stage 2, only after the user's go: touch words_along/GO_GLM53_STAGE2 (the other 24 pairs; gate on the stage-1 run)
    waitfor $O/GO_GLM53_STAGE2 $O/laneR.done $O/b150_main/glm53_flash/DONE
    PY=/root/data/venv_t26/bin/python; export P2_FP8_FORCE_FALLBACK=1
    t 0,1,2,3 main glm53_flash $H/zai-org/GLM-5.3-Flash '{}' zh rest --gate-pilot $O/b150_main/glm53_flash -- --max-memory-gib 40 --cpu-offload-gib 300
    log "lane G53B (stage 2) done" ;;
esac
