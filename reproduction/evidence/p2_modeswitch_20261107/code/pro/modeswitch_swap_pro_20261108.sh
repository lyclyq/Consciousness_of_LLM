#!/bin/bash
# P2 mode switch under role-coordinate replacement (user 11-08: "gpt oss 20b 120b 本地是不是应该先做一下 kl 替换消融"), the four
# readouts of the mechanism group on PRO, with mode_switch_swap_20261108.py. The fits are r05r06_fits/<g20|g120|qsp|qth>/fit_m.pt;
# the template arguments are those of r05r06_pro.sh and of the mode-switch runs.
#   g20 and qsp/qth first (weights on PRO). gpt-oss-120b is downloaded again meanwhile: pinned commit b5c939de8f75 (the paper's),
#   original/ and metal/ skipped as before, into /root/hf_models_p2nat. Its run follows, and its weights are then deleted
#   (literal path).
# gpt-oss runs online (hf-mirror, for the MXFP4 kernel) with the mode-switch fixed date 2026-10-19. The fits were made on
# 10-30; fit_diag.json shows overlap 0.99+ with earlier fits of other dates, so the subspace does not depend on the date line.
# Every GPU job takes the exclusive flock on persistence/gpu.lock. Stop: touch modeswitch/STOP_MS; no signals.
# usage: bash modeswitch_swap_pro_20261108.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; S=mode_switch_swap_20261108.py
P=$C/modeswitch; SW=$P/swap; M=$P/nat10; STOP=$P/STOP_MS; LOCK=$C/persistence/gpu.lock
mkdir -p $SW
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[swap] $(date '+%F %T') $*" | tee -a $SW/queue_swap.log >> $P/queue.log; }
free_gib() { df --output=avail -BG /root | tail -1 | tr -dc 0-9; }

task() {  # tag model lang kw fit pythonpath net mem [extra]
  local tag=$1 m=$2 lang=$3 kw=$4 fit=$5 pp=$6 net=$7 mem=$8; shift 8
  local out=$SW/$tag lg=$SW/$tag.log a
  [ -e $STOP ] && { log "stop flag: not starting $tag"; exit 2; }
  [ -e $out/DONE ] && { log "$tag already DONE"; return 0; }
  a=(--model $m --tmpl-kwargs "$kw" --bodies $B1 --fit $fit --out $out --stop-file $STOP --max-memory-gib $mem)
  [ $lang = en ] && a+=(--lang en --bodies-en bodies_en_150.json)
  a+=("$@")
  if [ $net = 1 ]; then E=(env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1)
  else E=(env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1); fi
  "${E[@]}" PYTHONPATH=$pp $PY $S "${a[@]}" --dry-run > $SW/$tag.dry.log 2>&1
  grep -q 'DRY RUN OK' $SW/$tag.dry.log || { log "align check FAILED $tag, see $SW/$tag.dry.log"; return 1; }
  (
    flock -x 9
    log "start $tag"
    "${E[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $S "${a[@]}" >> $lg 2>&1
  ) 9>$LOCK
  if [ -e $out/DONE ]; then log "DONE $tag: $(grep '^summary:' $lg | tail -1 | cut -c1-600)"; return 0
  elif [ -e $STOP ]; then log "stopped in $tag (stop flag)"; exit 2
  else log "FAILED $tag, see $lg"; return 1; fi
}

dl120() {
  local dir=/root/hf_models_p2nat/openai/gpt-oss-120b
  [ -e $SW/dl_g120.ok ] && return 0
  until [ $(free_gib) -ge 91 ]; do [ -e $STOP ] && return 2; sleep 120; done
  log "download openai/gpt-oss-120b@b5c939de8f75 ($(free_gib) GiB free)"
  for i in 1 2 3 4 5 6; do
    [ -e $STOP ] && return 2
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY -c "from huggingface_hub import snapshot_download; p = snapshot_download('openai/gpt-oss-120b', revision='b5c939de8f754692c1647ca79fbf85e8c1e70f8a', local_dir='$dir', max_workers=8, ignore_patterns=['original/*', 'metal/*']); print('snapshot', p)" >> $SW/dl_g120.log 2>&1 && {
      log "downloaded gpt-oss-120b $(du -sh $dir | cut -f1)"; touch $SW/dl_g120.ok; return 0; }
    log "gpt-oss-120b attempt $i failed"; sleep 120
  done
  log "download FAILED gpt-oss-120b"; return 1
}

dl120 &
DLPID=$!
NT='{"enable_thinking": false}'
task g20 /root/hf_models/openai/gpt-oss-20b en '{}' r05r06_fits/g20/fit_m.pt $TF19 1 44 --fixed-date 2026-10-19
task qsp /root/hf_models/Qwen/Qwen3.8-27B zh "$NT" r05r06_fits/qsp/fit_m.pt '' 0 90
task qth /root/hf_models/Qwen/Qwen3.8-27B zh '{}' r05r06_fits/qth/fit_m.pt '' 0 90
wait $DLPID
if [ -e $SW/dl_g120.ok ]; then
  task g120 /root/hf_models_p2nat/openai/gpt-oss-120b en '{}' r05r06_fits/g120/fit_m.pt $TF19 1 90 --fixed-date 2026-10-19 && \
    { rm -rf /root/hf_models_p2nat/openai/gpt-oss-120b; log "gpt-oss-120b weights deleted"; }
fi
log "swap queue done"
