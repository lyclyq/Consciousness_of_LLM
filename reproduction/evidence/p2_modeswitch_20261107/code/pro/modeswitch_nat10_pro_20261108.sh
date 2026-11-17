#!/bin/bash
# P2 mode switch, the 10 natural-order models of the appendix that are not on PRO (user 11-08: download them to PRO in turn,
# run, delete; "你可以删 就你来"). Settings as the server3 persistence queue (persist_s3_20261106.sh) and the paper's
# Table tab:self_config: same template arguments, item language, Moonlight compat wrapper, revision pinned to the paper's commit.
# No layer gate (the G references are on server3); before each run a render-only check (--dry-run) and a decoded dump of body 0.
# Per model: mode_switch_20261107.py (main + natural, full KL) and mode_switch_ctrl_20261107.py (main, control-token split);
# when both are DONE the weights are deleted (literal paths below) and the download lane uses the space.
# Lanes (tmux modesw, one window each):  DL  downloads in size order while free disk >= size + 30 GiB;  RUN  one GPU job at a
# time under the exclusive flock on persistence/gpu.lock (PRO policy: at most two runs; this queue uses one).
# Stop: touch modeswitch/STOP_MS (checked before every download, task and body); no signals.
# usage: bash modeswitch_nat10_pro_20261108.sh DL|RUN
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python
D=/root/hf_models_p2nat; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
P=$C/modeswitch; M=$P/nat10; STOP=$P/STOP_MS; LOCK=$C/persistence/gpu.lock
F=mode_switch_20261107.py; X=mode_switch_ctrl_20261107.py
mkdir -p $D $M $P/ctrl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LANE=${1:?lane DL or RUN}
log() { echo "[nat10 $LANE] $(date '+%F %T') $*" | tee -a $M/queue_nat10.log >> $P/queue.log; }
free_gib() { df --output=avail -BG /root | tail -1 | tr -dc 0-9; }

# tag|repo|commit|lang|template kwargs|GiB|wrapper|max GPU GiB|CPU offload GiB
MODELS='mimo7b_rl_0530|XiaomiMiMo/MiMo-7B-RL-0530|ee8dd14ecd59a1655ebe2ede9f7a06b40a8c8c5b|zh|{"enable_thinking": false}|15||44|0
glm46v_flash|zai-org/GLM-4.6V-Flash|411bb4d77144a3f03accbf4b780f5acb8b7cde4e|zh|{"enable_thinking": false}|20||44|0
phi4_en|microsoft/phi-4|2db69c1c3e91a05d2c64a3185acfbaf36f744e25|en|{}|28||44|0
moonlight16b_a3b_instruct|moonshotai/Moonlight-16B-A3B-Instruct|4e735b07a89f73647dfab71ab91b840f362ede5b|zh|{}|30|moonlight_compat_20261103.py|44|0
ernie45_21b_a3b_thinking|baidu/ERNIE-4.5-21B-A3B-Thinking|4341bb42644d5422859509fa25d41544c57181f8|zh|{}|41||60|0
granite41_30b_en|ibm-granite/granite-4.1-30b|4fae6278f7132abf5e971f9de49ebbad09c54cce|en|{}|54||90|0
gemma4_31b_en|google/gemma-4-31B-it|842da3794eaa0b77d5f08bae87a17459d91ff475|en|{}|59||90|0
exaone4_32b_en|LGAI-EXAONE/EXAONE-4.0.1-32B|a6b0bd4b0fb47cd3c644d60d12dba8ecdb9dc810|en|{"enable_thinking": false}|60||90|0
seed_oss_36b|ByteDance-Seed/Seed-OSS-36B-Instruct|497f1dca95ebdec98e41d517b9f060ee753c902f|zh|{"thinking_budget": 0}|68||90|0
r1_distill_llama70b|deepseek-ai/DeepSeek-R1-Distill-Llama-70B|b1c0b44b4369b597ad119a196caf79a9c40e141e|zh|{}|132||84|60'

del_weights() {  # literal paths only
  case $1 in
    mimo7b_rl_0530) rm -rf /root/hf_models_p2nat/XiaomiMiMo/MiMo-7B-RL-0530 ;;
    glm46v_flash) rm -rf /root/hf_models_p2nat/zai-org/GLM-4.6V-Flash ;;
    phi4_en) rm -rf /root/hf_models_p2nat/microsoft/phi-4 ;;
    moonlight16b_a3b_instruct) rm -rf /root/hf_models_p2nat/moonshotai/Moonlight-16B-A3B-Instruct ;;
    ernie45_21b_a3b_thinking) rm -rf /root/hf_models_p2nat/baidu/ERNIE-4.5-21B-A3B-Thinking ;;
    granite41_30b_en) rm -rf /root/hf_models_p2nat/ibm-granite/granite-4.1-30b ;;
    gemma4_31b_en) rm -rf /root/hf_models_p2nat/google/gemma-4-31B-it ;;
    exaone4_32b_en) rm -rf /root/hf_models_p2nat/LGAI-EXAONE/EXAONE-4.0.1-32B ;;
    seed_oss_36b) rm -rf /root/hf_models_p2nat/ByteDance-Seed/Seed-OSS-36B-Instruct ;;
    r1_distill_llama70b) rm -rf /root/hf_models_p2nat/deepseek-ai/DeepSeek-R1-Distill-Llama-70B ;;
    *) log "del_weights: unknown tag $1, nothing deleted" ;;
  esac
}

dl() {  # tag repo commit gib
  local tag=$1 repo=$2 rev=$3 gb=$4 dir=$D/$2
  [ -e $M/dl_$tag.ok ] && return 0
  until [ $(free_gib) -ge $(( gb + 30 )) ]; do [ -e $STOP ] && exit 2; sleep 120; done
  log "download $repo@${rev:0:12} ($gb GiB, $(free_gib) GiB free)"
  for i in 1 2 3 4 5 6; do
    [ -e $STOP ] && exit 2
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY -c "from huggingface_hub import snapshot_download; p = snapshot_download('$repo', revision='$rev', local_dir='$dir', max_workers=8); print('snapshot', p)" >> $M/dl_$tag.log 2>&1 && {
      log "downloaded $repo $(du -sh $dir | cut -f1)"; touch $M/dl_$tag.ok; return 0; }
    log "$repo attempt $i failed"; sleep 120
  done
  touch $M/dl_$tag.failed; log "download FAILED $repo"
}

task() {  # script outdir tag model lang kw wrap mem off [extra]
  local s=$1 od=$2 tag=$3 m=$4 lang=$5 kw=$6 wrap=$7 mem=$8 off=$9; shift 9
  local out=$od/$tag lg=$od/$tag.log a
  [ -e $STOP ] && { log "stop flag: not starting $out"; exit 2; }
  [ -e $out/DONE ] && { log "$out already DONE"; return 0; }
  nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: lane stops before $out"; exit 3; }
  a=(--model $m --tmpl-kwargs "$kw" --bodies $B1 --out $out --stop-file $STOP --max-memory-gib $mem --cpu-offload-gib $off)
  [ $lang = en ] && a+=(--lang en --bodies-en bodies_en_150.json)
  a+=("$@")
  (
    flock -x 9
    log "start $out"
    if [ -n "$wrap" ]; then HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 $PY $wrap $m $s "${a[@]}" >> $lg 2>&1
    else HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 $PY $s "${a[@]}" >> $lg 2>&1; fi
  ) 9>$LOCK
  if [ -e $out/DONE ]; then log "DONE $out: $(grep '^summary:' $lg | tail -1 | cut -c1-500)"
  elif [ -e $STOP ]; then log "stopped in $out (stop flag)"; exit 2
  else log "FAILED $out, see $lg"; fi
}

run_one() {  # tag repo lang kw wrap mem off
  local tag=$1 m=$D/$2 lang=$3 kw=$4 wrap=$5 mem=$6 off=$7 e=()
  [ -e $M/run_$tag.done ] && return 0
  while [ ! -e $M/dl_$tag.ok ] && [ ! -e $M/dl_$tag.failed ]; do [ -e $STOP ] && exit 2; sleep 60; done
  [ -e $M/dl_$tag.failed ] && { log "skip $tag: download failed"; return 0; }
  [ $lang = en ] && e=(--lang en --bodies-en bodies_en_150.json)
  local w=(); [ -n "$wrap" ] && w=($wrap $m)
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY "${w[@]}" render_dump_20261108.py --model $m --tmpl-kwargs "$kw" --bodies $B1 "${e[@]}" > $M/$tag.render.txt 2>&1
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY "${w[@]}" $F --model $m --tmpl-kwargs "$kw" --bodies $B1 "${e[@]}" --out $M/dry_$tag --dry-run > $M/$tag.dry.log 2>&1
  grep -q 'DRY RUN OK' $M/$tag.dry.log || { log "render check FAILED for $tag (weights kept), see $M/$tag.dry.log"; return 0; }
  log "render check OK $tag: $(grep -o '"natural_system_ok": [a-z]*' $M/$tag.dry.log | head -1)"
  task $F $P $tag $m $lang "$kw" "$wrap" $mem $off --min-free-gib 4
  task $X $P/ctrl $tag $m $lang "$kw" "$wrap" $mem $off
  if [ -e $P/$tag/DONE ] && [ -e $P/ctrl/$tag/DONE ]; then
    del_weights $tag; touch $M/run_$tag.done; log "finished $tag; weights deleted ($(free_gib) GiB free)"
  else log "$tag not complete: weights kept"; fi
}

case $LANE in
  DL)
    while IFS="|" read -r -u 3 tag repo rev lang kw gb wrap mem off; do dl $tag $repo $rev $gb; done 3<<< "$MODELS"
    log "all downloads finished" ;;
  RUN)
    while IFS="|" read -r -u 3 tag repo rev lang kw gb wrap mem off; do run_one $tag $repo $lang "$kw" "$wrap" $mem $off; done 3<<< "$MODELS"
    log "nat10 queue done" ;;
esac
