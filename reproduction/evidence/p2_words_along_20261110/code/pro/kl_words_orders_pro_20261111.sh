#!/bin/bash
# P2 ordinary words in next-token prediction, the rest on PRO (11-11, user: KL words in the natural order of the appendix too; "一次
# 就把没跑的模型的两个一次跑完"; "不重叠"). kl_words_orders_20261111.py: one load per model, the orders it still lacks:
#   natural only   the appendix natural-order models whose main order is done: Qwen3.8-27B non-thinking, gpt-oss-20b, ERNIE
#                  (weights kept by hard links), then GLM-4.6V, OLMo-2-7B, MiMo-7B, Moonlight, phi-4, Apertus (downloaded again)
#   main + natural Granite-4.1, Gemma-4, EXAONE-4.0.1, Seed-OSS, R1-Distill-70B (appendix natural-order list, not run yet)
#   main only      Qwen3.6-27B, Qwen3.5-27B, Granite-4.2, GLM-4.7, gpt-oss-120b, Qwen3.6-35B, Flash-Next non-thinking
#   depth (p3)     gpt-oss-120b and Qwen3.6-27B on the 200 phase-3 bodies (kl_words_p3_20261111.py), on the same download
# Each model runs with the arguments and environment of its paper KL runs (template kwargs, language, date, memory, offload,
# network, PYTHONPATH, wrapper; Flash-Next as the 11-07 KL run, with the venv's fla) and is checked against them: main against
# modeswitch/ctrl/<tag>, natural against modeswitch/nat/<tag> (content split: Qwen3.8, gpt-oss-20b) or modeswitch/<tag> (full KL).
# Natural3 (Ministral, DeepSeek-V4.1) is not in this queue.
#   DL   one download at a time, in table order, each once free disk >= size + 10 GiB (hf-mirror, no xet, pinned commits)
#   RUN  after "lane p3 DEPTH done" (one task on the GPU at a time); deletes each model's weights by literal path once all its
#        outputs are DONE
# Stop: touch words_along/STOP_KLW; no signals.
# usage (tmux session klw, windows ORDDL and ORDRUN, in the crossmodel directory): bash kl_words_orders_pro_20261111.sh DL|RUN
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next; PPT=/root/p2_scale_20261015/pydeps_tiktoken_20261108
B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; B3=$C/phase3_bodies_zh.jsonl; E3=$C/phase3_bodies_en.json
WJ=$C/words_v2.json; MS=$C/modeswitch; MC=$MS/ctrl
O=$C/words_along/kl; STOP=$C/words_along/STOP_KLW
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
LANE=${1:?lane DL or RUN}
log() { echo "[klw orders $LANE] $(date '+%F %T') $*" | tee -a $O/queue.log; }
free_gib() { df --output=avail -BG /root | tail -1 | tr -dc 0-9; }
NT='{"enable_thinking": false}'
P=/root/hf_models_p2nat; W=/root/hf_models_words
# tag|repo|commit|weight dir|GiB|lang|template kwargs|max GPU GiB|CPU offload GiB|wrapper|PYTHONPATH|net|fixed date|orders|p3
MODELS="glm46v_flash|zai-org/GLM-4.6V-Flash|411bb4d77144a3f03accbf4b780f5acb8b7cde4e|$P|20|zh|$NT|44|0||||-|natural|0
olmo2_7b_en|allenai/OLMo-2-1124-7B-Instruct|470b1fba1ae01581f270116362ee4aa1b97f4c84|$W|14|en|{}|44|0||||-|natural|0
mimo7b_rl_0530|XiaomiMiMo/MiMo-7B-RL-0530|ee8dd14ecd59a1655ebe2ede9f7a06b40a8c8c5b|$P|15|zh|$NT|44|0||||-|natural|0
moonlight16b_a3b_instruct|moonshotai/Moonlight-16B-A3B-Instruct|4e735b07a89f73647dfab71ab91b840f362ede5b|$P|30|zh|{}|44|0|moonlight_compat_20261103.py|$PPT||-|natural|0
phi4_en|microsoft/phi-4|2db69c1c3e91a05d2c64a3185acfbaf36f744e25|$P|28|en|{}|44|0||||-|natural|0
apertus8b_en|swiss-ai/Apertus-8B-Instruct-2509|b946d40447b2b597999b9c86d44bee0b452c919f|$W|15|en|{}|44|0||||2026-11-03|natural|0
qwen36_27b|Qwen/Qwen3.6-27B|6a9e13bd6fc8f0983b9b99948120bc37f49c13e9|$P|52|zh|$NT|90|0||||-|main|1
qwen35_27b|Qwen/Qwen3.5-27B|fc05daec18b0a78c049392ed2e771dde82bdf654|$P|52|zh|$NT|90|0||||-|main|0
granite41_30b_en|ibm-granite/granite-4.1-30b|4fae6278f7132abf5e971f9de49ebbad09c54cce|$P|54|en|{}|90|0||||-|main,natural|0
granite42_30b_en|ibm-granite/granite-4.2-30b|9e668ce1c538387ef24d3644e9b0606647762636|$P|55|en|$NT|90|0||||-|main|0
gemma4_31b_en|google/gemma-4-31B-it|842da3794eaa0b77d5f08bae87a17459d91ff475|$P|59|en|{}|90|0||||-|main,natural|0
glm47_flash|zai-org/GLM-4.7-Flash|7dd20894a642a0aa287e9827cb1a1f7f91386b67|$P|59|zh|$NT|90|0||||-|main|0
exaone4_32b_en|LGAI-EXAONE/EXAONE-4.0.1-32B|a6b0bd4b0fb47cd3c644d60d12dba8ecdb9dc810|$P|60|en|$NT|90|0||||-|main,natural|0
gpt_oss_120b_en|openai/gpt-oss-120b|b5c939de8f754692c1647ca79fbf85e8c1e70f8a|$W|61|en|{}|90|0||$TF19|1|2026-10-19|main|1
qwen36_35b_a3b|Qwen/Qwen3.6-35B-A3B|995ad96eacd98c81ed38be0c5b274b04031597b0|$P|67|zh|$NT|90|0||||-|main|0
seed_oss_36b|ByteDance-Seed/Seed-OSS-36B-Instruct|497f1dca95ebdec98e41d517b9f060ee753c902f|$P|68|zh|{\"thinking_budget\": 0}|90|0||||-|main,natural|0
r1_distill_llama70b|deepseek-ai/DeepSeek-R1-Distill-Llama-70B|b1c0b44b4369b597ad119a196caf79a9c40e141e|$P|132|zh|{}|84|60||||-|main,natural|0
flashnext_nt|Qwen/Qwen3.8-Flash-Next-FP8|236dfdf285828023ca3bcd3f37366c58a3469b13|$W|173|zh|$NT|84|52||$TF19||-|main|0"

natref() {  # natural-order reference and its kind
  case $1 in qwen38_nt|gpt_oss_20b_en) echo "$MS/nat/$1 nat" ;; *) echo "$MS/$1 full" ;; esac
}
all_done() {  # tag orders p3
  local o
  for o in ${2//,/ }; do
    if [ $o = main ]; then [ -e $O/$1/DONE ] || return 1; else [ -e $O/natural/$1/DONE ] || return 1; fi
  done
  [ $3 = 1 ] && { [ -e $O/p3_$1/DONE ] || return 1; }
  return 0
}
del_weights() {  # literal paths of this queue's downloads only
  case $1 in
    glm46v_flash) rm -rf /root/hf_models_p2nat/zai-org/GLM-4.6V-Flash ;;
    olmo2_7b_en) rm -rf /root/hf_models_words/allenai/OLMo-2-1124-7B-Instruct ;;
    mimo7b_rl_0530) rm -rf /root/hf_models_p2nat/XiaomiMiMo/MiMo-7B-RL-0530 ;;
    moonlight16b_a3b_instruct) rm -rf /root/hf_models_p2nat/moonshotai/Moonlight-16B-A3B-Instruct ;;
    phi4_en) rm -rf /root/hf_models_p2nat/microsoft/phi-4 ;;
    apertus8b_en) rm -rf /root/hf_models_words/swiss-ai/Apertus-8B-Instruct-2509 ;;
    qwen36_27b) rm -rf /root/hf_models_p2nat/Qwen/Qwen3.6-27B ;;
    qwen35_27b) rm -rf /root/hf_models_p2nat/Qwen/Qwen3.5-27B ;;
    granite41_30b_en) rm -rf /root/hf_models_p2nat/ibm-granite/granite-4.1-30b ;;
    granite42_30b_en) rm -rf /root/hf_models_p2nat/ibm-granite/granite-4.2-30b ;;
    gemma4_31b_en) rm -rf /root/hf_models_p2nat/google/gemma-4-31B-it ;;
    glm47_flash) rm -rf /root/hf_models_p2nat/zai-org/GLM-4.7-Flash ;;
    exaone4_32b_en) rm -rf /root/hf_models_p2nat/LGAI-EXAONE/EXAONE-4.0.1-32B ;;
    gpt_oss_120b_en) rm -rf /root/hf_models_words/openai/gpt-oss-120b ;;
    qwen36_35b_a3b) rm -rf /root/hf_models_p2nat/Qwen/Qwen3.6-35B-A3B ;;
    seed_oss_36b) rm -rf /root/hf_models_p2nat/ByteDance-Seed/Seed-OSS-36B-Instruct ;;
    r1_distill_llama70b) rm -rf /root/hf_models_p2nat/deepseek-ai/DeepSeek-R1-Distill-Llama-70B ;;
    flashnext_nt) rm -rf /root/hf_models_words/Qwen/Qwen3.8-Flash-Next-FP8 ;;
    ernie_keep) rm -rf /root/hf_models_keep_20261111/baidu/ERNIE-4.5-21B-A3B-Thinking ;;
    *) log "del_weights: unknown tag $1, nothing deleted" ;;
  esac
}

run_orders() {  # tag model lang kw mem off wrap pp net date orders
  local tag=$1 m=$2 lang=$3 kw=$4 mem=$5 off=$6 wrap=$7 pp=$8 net=$9 day=${10} ords=${11} todo="" o nr nk a lg=$O/orders_$1.log
  for o in ${ords//,/ }; do
    if [ $o = main ]; then [ -e $O/$tag/DONE ] || todo="$todo,main"; else [ -e $O/natural/$tag/DONE ] || todo="$todo,natural"; fi
  done
  todo=${todo#,}
  [ -z "$todo" ] && { log "already DONE: $tag ($ords)"; return 0; }
  [ -e $STOP ] && { log "stop flag: not starting $tag"; exit 2; }
  nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: lane stops before $tag"; exit 3; }
  a=(--model $m --tmpl-kwargs "$kw" --bodies $B1 --words $WJ --orders $todo --lang $lang --max-memory-gib $mem --cpu-offload-gib $off
     --stop-file $STOP)
  [ $lang = en ] && a+=(--bodies-en bodies_en_150.json)
  [ "$day" != - ] && a+=(--fixed-date $day)
  case $todo in *main*) a+=(--out-main $O/$tag --check-ctrl $MC/$tag) ;; esac
  case $todo in *natural*) read nr nk <<< "$(natref $tag)"; a+=(--out-natural $O/natural/$tag --check-natural $nr --check-natural-kind $nk) ;; esac
  mkdir -p $O/natural
  log "start $tag ($todo)"
  local s=(kl_words_orders_20261111.py); [ -n "$wrap" ] && s=($wrap $m kl_words_orders_20261111.py)
  if [ "$net" = 1 ]; then
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 \
      PYTHONPATH=$pp $PY "${s[@]}" "${a[@]}" >> $lg 2>&1 < /dev/null
  else
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY "${s[@]}" "${a[@]}" >> $lg 2>&1 < /dev/null
  fi
  log "ended $tag: $(grep -a -E 'summary \(content|CHECK FAILED|Error|stop' $lg | tail -3 | tr '\n' ' ')"
}

run_p3() {  # tag model lang kw mem off pp net date
  local tag=$1 m=$2 lang=$3 kw=$4 mem=$5 off=$6 pp=$7 net=$8 day=$9 a lg=$O/p3_$1.log
  [ -e $O/p3_$tag/DONE ] && { log "already DONE: p3_$tag"; return 0; }
  [ -e $STOP ] && { log "stop flag: not starting p3_$tag"; exit 2; }
  a=(--model $m --tmpl-kwargs "$kw" --bodies $B1 --p3-bodies $B3 --words $WJ --out $O/p3_$tag --check-ctrl $MC/$tag --lang $lang
     --max-memory-gib $mem --cpu-offload-gib $off --stop-file $STOP)
  [ $lang = en ] && a+=(--bodies-en bodies_en_150.json --p3-bodies-en $E3)
  [ "$day" != - ] && a+=(--fixed-date $day)
  log "start p3_$tag"
  if [ "$net" = 1 ]; then
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 \
      PYTHONPATH=$pp $PY kl_words_p3_20261111.py "${a[@]}" >> $lg 2>&1 < /dev/null
  else
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY kl_words_p3_20261111.py "${a[@]}" >> $lg 2>&1 < /dev/null
  fi
  log "ended p3_$tag: $(grep -a -E 'summary \(content|CHECK FAILED|Error|stop' $lg | tail -2 | tr '\n' ' ')"
}

case $LANE in
  DL)
    while IFS='|' read -r tag repo rev base gb lang kw mem off wrap pp net day ords p3; do
      [ -z "$tag" ] && continue
      [ -e $O/dlo_$tag.ok ] && continue
      all_done $tag $ords $p3 && continue
      n=0
      until [ $(free_gib) -ge $(( gb + 10 )) ]; do [ -e $STOP ] && exit 2
        [ $((n % 10)) = 0 ] && log "waiting for $(( gb + 10 )) GiB free ($(free_gib)) for $repo"; n=$((n + 1)); sleep 60; done
      log "download $repo@${rev:0:12} ($gb GiB, $(free_gib) GiB free)"
      ok=0
      for i in 1 2 3 4 5 6; do
        [ -e $STOP ] && exit 2
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY -c \
          "from huggingface_hub import snapshot_download; print('snapshot', snapshot_download('$repo', revision='$rev', local_dir='$base/$repo', max_workers=8))" \
          >> $O/dlo_$tag.log 2>&1 < /dev/null && { ok=1; break; }
        log "$repo attempt $i failed"; sleep 120
      done
      if [ $ok = 1 ]; then touch $O/dlo_$tag.ok; log "downloaded $repo $(du -sh $base/$repo | cut -f1)"; else touch $O/dlo_$tag.failed; log "download FAILED $repo"; fi
    done <<< "$MODELS"
    log "lane orders DL done" ;;
  RUN)
    until grep -q 'lane p3 DEPTH done' $O/queue.log 2>/dev/null; do [ -e $STOP ] && exit 2; sleep 60; done
    log "the depth part is finished: natural order and the rest start"
    run_orders qwen38_nt /root/hf_models/Qwen/Qwen3.8-27B zh "$NT" 90 0 '' '' 0 - natural
    run_orders gpt_oss_20b_en /root/hf_models/openai/gpt-oss-20b en '{}' 44 0 '' $TF19 1 2026-10-19 natural
    EK=/root/hf_models_keep_20261111/baidu/ERNIE-4.5-21B-A3B-Thinking
    if [ -d $EK ]; then
      run_orders ernie45_21b_a3b_thinking $EK zh '{}' 60 0 '' '' 0 - natural
      [ -e $O/natural/ernie45_21b_a3b_thinking/DONE ] && { del_weights ernie_keep; log "kept ERNIE weights deleted ($(free_gib) GiB free)"; }
    else log "no kept ERNIE weights at $EK: ERNIE natural not run"; fi
    while IFS='|' read -r tag repo rev base gb lang kw mem off wrap pp net day ords p3; do
      [ -z "$tag" ] && continue
      if ! all_done $tag $ords $p3; then
        until [ -e $O/dlo_$tag.ok ] || [ -e $O/dlo_$tag.failed ]; do [ -e $STOP ] && exit 2; sleep 60; done
        [ -e $O/dlo_$tag.failed ] && { log "skip $tag: download failed"; continue; }
        run_orders $tag $base/$repo $lang "$kw" $mem $off "$wrap" "$pp" "$net" $day $ords
        [ $p3 = 1 ] && run_p3 $tag $base/$repo $lang "$kw" $mem $off "$pp" "$net" $day
      fi
      if all_done $tag $ords $p3; then
        [ -e $O/dlo_$tag.ok ] && { del_weights $tag; log "weights of $tag deleted ($(free_gib) GiB free)"; }
      else log "NOT DONE $tag (weights kept), see $O/orders_$tag.log"; fi
    done <<< "$MODELS"
    log "lane orders RUN done" ;;
esac
