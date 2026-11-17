#!/bin/bash
# Moonlight-16B-A3B-Instruct for the nat10 queue (modeswitch_nat10_pro_20261108.sh), which skipped it: its render check failed
# because the PRO venv has no tiktoken (the repository tokenizer imports it). tiktoken 0.14.0 and requests were installed with
# --no-deps into a separate directory (pydeps_tiktoken_20261108), added by PYTHONPATH for this model only; the venv is unchanged.
# Same steps as run_one in the nat10 queue: render dump, dry run, full + ctrl, delete the weights (literal path) when both DONE.
# usage: bash modeswitch_moonlight_pro_20261108.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
P=$C/modeswitch; M=$P/nat10; STOP=$P/STOP_MS; LOCK=$C/persistence/gpu.lock
tag=moonlight16b_a3b_instruct; m=/root/hf_models_p2nat/moonshotai/Moonlight-16B-A3B-Instruct; W=moonlight_compat_20261103.py
export PYTHONPATH=/root/p2_scale_20261015/pydeps_tiktoken_20261108 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[nat10 moonlight] $(date '+%F %T') $*" | tee -a $M/queue_nat10.log >> $P/queue.log; }
[ -e $M/run_$tag.done ] && { log "$tag already done"; exit 0; }
$PY $W $m render_dump_20261108.py --model $m --tmpl-kwargs '{}' --bodies $B1 > $M/$tag.render.txt 2>&1
$PY $W $m mode_switch_20261107.py --model $m --tmpl-kwargs '{}' --bodies $B1 --out $M/dry_$tag --dry-run > $M/$tag.dry.log 2>&1
grep -q 'DRY RUN OK' $M/$tag.dry.log || { log "render check FAILED again for $tag, see $M/$tag.dry.log"; exit 1; }
log "render check OK $tag: $(grep -o '"natural_system_ok": [a-z]*' $M/$tag.dry.log | head -1)"
for job in full ctrl; do
  [ -e $STOP ] && { log "stop flag"; exit 2; }
  if [ $job = full ]; then s=mode_switch_20261107.py; out=$P/$tag; extra=(--min-free-gib 4); else s=mode_switch_ctrl_20261107.py; out=$P/ctrl/$tag; extra=(); fi
  [ -e $out/DONE ] && continue
  (
    flock -x 9
    log "start $out"
    CUDA_VISIBLE_DEVICES=0 $PY $W $m $s --model $m --tmpl-kwargs '{}' --bodies $B1 --out $out --stop-file $STOP --max-memory-gib 44 --cpu-offload-gib 0 "${extra[@]}" >> $out.log 2>&1
  ) 9>$LOCK
  if [ -e $out/DONE ]; then log "DONE $out: $(grep '^summary:' $out.log | tail -1 | cut -c1-500)"; else log "FAILED $out, see $out.log"; fi
done
if [ -e $P/$tag/DONE ] && [ -e $P/ctrl/$tag/DONE ]; then
  rm -rf /root/hf_models_p2nat/moonshotai/Moonlight-16B-A3B-Instruct
  touch $M/run_$tag.done; log "finished $tag; weights deleted"
else log "$tag not complete: weights kept"; fi
