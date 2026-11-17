#!/bin/bash
# P2 mode switch on the six models of the paper's main comparison that are not yet covered and fit PRO.
# User 11-08: fill in the main-index KL for every model that can be run; non-thinking rendering where the template has one,
# as the main index table. The models: Qwen3.6-27B, Qwen3.5-27B, Qwen3.6-35B-A3B, GLM-4.7-Flash, Granite-4.2-30B, OLMo-2-13B.
# GLM-5.3-Flash and DeepSeek-V4.1-Flash wait for server3.
# Same pipeline as modeswitch_nat10_pro_20261108.sh:
#   - commit pinned to the paper's Table tab:self_config, downloaded into /root/hf_models_p2nat;
#   - render dump and dry run;
#   - mode_switch_20261107.py (main + natural, full KL) and mode_switch_ctrl_20261107.py (main, control-token split);
#   - weights deleted (literal paths) when both are DONE.
# The natural order of these six is not in the paper's natural-order appendix (the smallest model per developer). The full script
# computes it anyway; it is recorded but not part of the protocol table.
# transformers 5.8.0 (the venv), as the server3 runs of these models in the paper.
# Disk: two other lanes download on PRO (nat10 DL, then nat3's Ministral). Before each download this lane reserves the full size
# of the model those lanes download next, so the lanes together never overfill the disk.
# Lanes (tmux modesw): DL and RUN.  Stop: touch modeswitch/STOP_MS; no signals.
# usage: bash modeswitch_main6_pro_20261108.sh DL|RUN
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python
D=/root/hf_models_p2nat; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
P=$C/modeswitch; M=$P/main6; N10=$P/nat10; STOP=$P/STOP_MS; LOCK=$C/persistence/gpu.lock
F=mode_switch_20261107.py; X=mode_switch_ctrl_20261107.py
mkdir -p $D $M $P/ctrl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LANE=${1:?lane DL or RUN}
log() { echo "[main6 $LANE] $(date '+%F %T') $*" | tee -a $M/queue_main6.log >> $P/queue.log; }
free_gib() { df --output=avail -BG /root | tail -1 | tr -dc 0-9; }

# tag|repo|commit|lang|template kwargs|GiB|max GPU GiB
MODELS='olmo2_13b_en|allenai/OLMo-2-1124-13B-Instruct|3a5c85baefbb1896a54d56fe2e76c0395627ddf4|en|{}|26|44
qwen36_27b|Qwen/Qwen3.6-27B|6a9e13bd6fc8f0983b9b99948120bc37f49c13e9|zh|{"enable_thinking": false}|52|90
qwen35_27b|Qwen/Qwen3.5-27B|fc05daec18b0a78c049392ed2e771dde82bdf654|zh|{"enable_thinking": false}|52|90
granite42_30b_en|ibm-granite/granite-4.2-30b|9e668ce1c538387ef24d3644e9b0606647762636|en|{"enable_thinking": false}|55|90
glm47_flash|zai-org/GLM-4.7-Flash|7dd20894a642a0aa287e9827cb1a1f7f91386b67|zh|{"enable_thinking": false}|59|90
qwen36_35b_a3b|Qwen/Qwen3.6-35B-A3B|995ad96eacd98c81ed38be0c5b274b04031597b0|zh|{"enable_thinking": false}|67|90'

del_weights() {  # literal paths only
  case $1 in
    olmo2_13b_en) rm -rf /root/hf_models_p2nat/allenai/OLMo-2-1124-13B-Instruct ;;
    qwen36_27b) rm -rf /root/hf_models_p2nat/Qwen/Qwen3.6-27B ;;
    qwen35_27b) rm -rf /root/hf_models_p2nat/Qwen/Qwen3.5-27B ;;
    granite42_30b_en) rm -rf /root/hf_models_p2nat/ibm-granite/granite-4.2-30b ;;
    glm47_flash) rm -rf /root/hf_models_p2nat/zai-org/GLM-4.7-Flash ;;
    qwen36_35b_a3b) rm -rf /root/hf_models_p2nat/Qwen/Qwen3.6-35B-A3B ;;
    *) log "del_weights: unknown tag $1, nothing deleted" ;;
  esac
}

reserve_others() {  # full size of the model the other lanes download next (in progress counts in full)
  local t
  for t in exaone4_32b_en:60 seed_oss_36b:68 r1_distill_llama70b:132; do
    [ -e $N10/dl_${t%%:*}.ok ] || [ -e $N10/dl_${t%%:*}.failed ] || { echo ${t##*:}; return; }
  done
  for t in ministral3_14b_instruct ministral3_14b_reasoning; do
    [ -e $N10/dl_$t.ok ] || { echo 26; return; }
  done
  echo 0
}

dl() {  # tag repo commit gib
  local tag=$1 repo=$2 rev=$3 gb=$4 dir=$D/$2
  [ -e $M/dl_$tag.ok ] && return 0
  until [ $(free_gib) -ge $(( gb + $(reserve_others) + 30 )) ]; do [ -e $STOP ] && exit 2; sleep 120; done
  log "download $repo@${rev:0:12} ($gb GiB, $(free_gib) GiB free, $(reserve_others) GiB reserved for the other lanes)"
  for i in 1 2 3 4 5 6; do
    [ -e $STOP ] && exit 2
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY -c "from huggingface_hub import snapshot_download; p = snapshot_download('$repo', revision='$rev', local_dir='$dir', max_workers=8); print('snapshot', p)" >> $M/dl_$tag.log 2>&1 && {
      log "downloaded $repo $(du -sh $dir | cut -f1)"; touch $M/dl_$tag.ok; return 0; }
    log "$repo attempt $i failed"; sleep 120
  done
  touch $M/dl_$tag.failed; log "download FAILED $repo"
}

task() {  # script outdir tag model lang kw mem [extra]
  local s=$1 od=$2 tag=$3 m=$4 lang=$5 kw=$6 mem=$7; shift 7
  local out=$od/$tag lg=$od/$tag.log a
  [ -e $STOP ] && { log "stop flag: not starting $out"; exit 2; }
  [ -e $out/DONE ] && { log "$out already DONE"; return 0; }
  nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: lane stops before $out"; exit 3; }
  a=(--model $m --tmpl-kwargs "$kw" --bodies $B1 --out $out --stop-file $STOP --max-memory-gib $mem --cpu-offload-gib 0)
  [ $lang = en ] && a+=(--lang en --bodies-en bodies_en_150.json)
  a+=("$@")
  (
    flock -x 9
    log "start $out"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 $PY $s "${a[@]}" >> $lg 2>&1 < /dev/null
  ) 9>$LOCK
  if [ -e $out/DONE ]; then log "DONE $out: $(grep '^summary:' $lg | tail -1 | cut -c1-500)"
  elif [ -e $STOP ]; then log "stopped in $out (stop flag)"; exit 2
  else log "FAILED $out, see $lg"; fi
}

run_one() {  # tag repo lang kw mem
  local tag=$1 m=$D/$2 lang=$3 kw=$4 mem=$5 e=()
  [ -e $M/run_$tag.done ] && return 0
  while [ ! -e $M/dl_$tag.ok ] && [ ! -e $M/dl_$tag.failed ]; do [ -e $STOP ] && exit 2; sleep 60; done
  [ -e $M/dl_$tag.failed ] && { log "skip $tag: download failed"; return 0; }
  [ $lang = en ] && e=(--lang en --bodies-en bodies_en_150.json)
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY render_dump_20261108.py --model $m --tmpl-kwargs "$kw" --bodies $B1 "${e[@]}" > $M/$tag.render.txt 2>&1 < /dev/null
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY $F --model $m --tmpl-kwargs "$kw" --bodies $B1 "${e[@]}" --out $M/dry_$tag --dry-run > $M/$tag.dry.log 2>&1 < /dev/null
  grep -q 'DRY RUN OK' $M/$tag.dry.log || { log "render check FAILED for $tag (weights kept), see $M/$tag.dry.log"; return 0; }
  log "render check OK $tag: $(grep -o '"natural_system_ok": [a-z]*' $M/$tag.dry.log | head -1)"
  task $F $P $tag $m $lang "$kw" $mem --min-free-gib 4
  task $X $P/ctrl $tag $m $lang "$kw" $mem
  if [ -e $P/$tag/DONE ] && [ -e $P/ctrl/$tag/DONE ]; then
    del_weights $tag; touch $M/run_$tag.done; log "finished $tag; weights deleted ($(free_gib) GiB free)"
  else log "$tag not complete: weights kept"; fi
}

case $LANE in
  DL)
    while IFS="|" read -r -u 3 tag repo rev lang kw gb mem; do dl $tag $repo $rev $gb; done 3<<< "$MODELS"
    log "all downloads finished" ;;
  RUN)
    while IFS="|" read -r -u 3 tag repo rev lang kw gb mem; do run_one $tag $repo $lang "$kw" $mem; done 3<<< "$MODELS"
    log "main6 queue done" ;;
esac
