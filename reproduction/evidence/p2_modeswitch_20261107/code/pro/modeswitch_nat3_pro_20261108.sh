#!/bin/bash
# P2 mode switch, the rest of the natural-order protocol on PRO (11-08), with mode_switch_nat_20261108.py:
#   1. natural-order control-token split for the models whose weights are on PRO: Qwen3.8-27B (both modes) and gpt-oss-20b
#      (the ctrl script did the main order only; Qwen3.8 is the model whose next-token mass is mostly control tokens)
#   2. Ministral-3-14B-Instruct-2512-BF16 and Ministral-3-14B-Reasoning-2512 (paper Table tab:self_natural3, natural order only):
#      natural3 renderings of crossmodel_natural3.py, transformers 5.19 (tf_next) as in the paper's run, commits pinned to the
#      paper's (3cea74c1ebaf, 51f9210f3cd2), HF shards only (consolidated.safetensors, the Mistral-format copy, not
#      downloaded); started after the nat10 download lane has finished (so the two lanes never compete for disk); weights
#      deleted (literal paths) when the run is DONE.
# Every GPU job takes the exclusive flock on persistence/gpu.lock. Stop: touch modeswitch/STOP_MS; no signals.
# usage: bash modeswitch_nat3_pro_20261108.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; S=mode_switch_nat_20261108.py
P=$C/modeswitch; M=$P/nat10; STOP=$P/STOP_MS; LOCK=$C/persistence/gpu.lock; D=/root/hf_models_p2nat
mkdir -p $P/nat $P/nat3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[nat3] $(date '+%F %T') $*" | tee -a $M/queue_nat10.log >> $P/queue.log; }
free_gib() { df --output=avail -BG /root | tail -1 | tr -dc 0-9; }

task() {  # order outdir tag model lang kw pythonpath net mem [extra]
  local order=$1 od=$2 tag=$3 m=$4 lang=$5 kw=$6 pp=$7 net=$8 mem=$9; shift 9
  local out=$od/$tag lg=$od/$tag.log a
  [ -e $STOP ] && { log "stop flag: not starting $out"; exit 2; }
  [ -e $out/DONE ] && { log "$out already DONE"; return 0; }
  a=(--order $order --model $m --tmpl-kwargs "$kw" --bodies $B1 --out $out --stop-file $STOP --max-memory-gib $mem)
  [ $lang = en ] && a+=(--lang en --bodies-en bodies_en_150.json)
  a+=("$@")
  if [ $net = 1 ]; then E=(env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1)
  else E=(env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1); fi
  "${E[@]}" PYTHONPATH=$pp $PY $S "${a[@]}" --dry-run > $od/$tag.dry.log 2>&1
  grep -q 'DRY RUN OK' $od/$tag.dry.log || { log "render check FAILED $out, see $od/$tag.dry.log"; return 1; }
  log "render check OK $out: $(tr -d '\n' < $od/$tag.dry.log | grep -o '"roles": \[[^]]*\]' | head -1)"
  (
    flock -x 9
    log "start $out"
    "${E[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $S "${a[@]}" >> $lg 2>&1
  ) 9>$LOCK
  if [ -e $out/DONE ]; then log "DONE $out: $(grep '^summary:' $lg | tail -1 | cut -c1-500)"; return 0
  elif [ -e $STOP ]; then log "stopped in $out (stop flag)"; exit 2
  else log "FAILED $out, see $lg"; return 1; fi
}

dl() {  # tag repo commit gib
  local tag=$1 repo=$2 rev=$3 gb=$4 dir=$D/$2
  [ -e $M/dl_$tag.ok ] && return 0
  until [ $(free_gib) -ge $(( gb + 30 )) ]; do [ -e $STOP ] && exit 2; sleep 120; done
  log "download $repo@${rev:0:12} ($gb GiB, $(free_gib) GiB free)"
  for i in 1 2 3 4 5 6; do
    [ -e $STOP ] && exit 2
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY -c "from huggingface_hub import snapshot_download; p = snapshot_download('$repo', revision='$rev', local_dir='$dir', max_workers=8, ignore_patterns=['consolidated*']); print('snapshot', p)" >> $M/dl_$tag.log 2>&1 && {
      log "downloaded $repo $(du -sh $dir | cut -f1)"; touch $M/dl_$tag.ok; return 0; }
    log "$repo attempt $i failed"; sleep 120
  done
  log "download FAILED $repo"; return 1
}

NT='{"enable_thinking": false}'
task natural $P/nat qwen38_nt /root/hf_models/Qwen/Qwen3.8-27B zh "$NT" '' 0 90
task natural $P/nat qwen38_think /root/hf_models/Qwen/Qwen3.8-27B zh '{}' '' 0 90
task natural $P/nat gpt_oss_20b_en /root/hf_models/openai/gpt-oss-20b en '{}' $TF19 1 44 --fixed-date 2026-10-19

log "waiting for the nat10 download lane to finish before the Ministral downloads"
until grep -q 'all downloads finished' $M/queue_nat10.log; do [ -e $STOP ] && exit 2; sleep 300; done
dl ministral3_14b_instruct mistralai/Ministral-3-14B-Instruct-2512-BF16 3cea74c1ebaf5ce5f5a2553de470e2ceab825142 26 && \
  task natural3 $P/nat3 ministral3_14b_instruct_en $D/mistralai/Ministral-3-14B-Instruct-2512-BF16 en '{}' $TF19 0 60 && \
  { rm -rf /root/hf_models_p2nat/mistralai/Ministral-3-14B-Instruct-2512-BF16; log "Ministral Instruct weights deleted"; }
dl ministral3_14b_reasoning mistralai/Ministral-3-14B-Reasoning-2512 51f9210f3cd20f3452a80d5819d15dc61cc50630 26 && \
  task natural3 $P/nat3 ministral3_14b_reasoning_en $D/mistralai/Ministral-3-14B-Reasoning-2512 en '{}' $TF19 0 60 && \
  { rm -rf /root/hf_models_p2nat/mistralai/Ministral-3-14B-Reasoning-2512; log "Ministral Reasoning weights deleted"; }
log "nat3 queue done"
