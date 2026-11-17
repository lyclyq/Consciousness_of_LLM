#!/bin/bash
# Ministral-3-14B-Reasoning-2512, natural3 KL of the ordinary words (11-12). Step 5 of g120_kl_words_20261111.sh (v3 ea0a6e64)
# skipped this cell: all six download attempts failed on hf-mirror's 503 for the repository's .gitattributes. The six weight
# shards, tokenizer.json, tekken.json and SYSTEM_PROMPT.txt were already on disk (27 GiB), but the snapshot never completed, so
# the small config files are missing. This script completes the download at the same commit (51f9210f) with the paper's
# ignore_patterns plus '.gitattributes' and '*.md' (README.md gave the same 503), which no loader reads. It then runs the cell with the command line of step 5 and deletes
# the weights once natural/<tag>/DONE exists.
# It starts only if no compute process is on the GPU. The queue's next GPU job (R1) waits for its 132 GiB download.
# Stop: touch words_along/STOP_KLW.  usage (tmux klw, window MINR, in the crossmodel directory): bash ministral_r_fix_20261112.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; WJ=$C/words_v2.json; MS=$C/modeswitch
O=$C/words_along/kl; STOP=$C/words_along/STOP_KLW
tag=ministral3_14b_reasoning_en; repo=mistralai/Ministral-3-14B-Reasoning-2512; rev=51f9210f3cd20f3452a80d5819d15dc61cc50630
MM=/root/hf_models_p2nat/$repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
log() { echo "[klw minr] $(date '+%F %T') $*" | tee -a $O/queue.log; }
free_gib() { df --output=avail -BG /root | tail -1 | tr -dc 0-9; }
[ -e $O/natural/$tag/DONE ] && { log "already DONE: natural/$tag"; exit 0; }
[ -e $STOP ] && { log "stop flag"; exit 2; }
if [ ! -e $O/dlo_$tag.ok ]; then
  # v3 (11-12 04:40 UTC): snapshot_download failed six times again (503 on .gitattributes, README.md and chat_template.jinja), so the
  # completion goes file by file (hf_complete_20261112.py, 20 tries per file), same mirror, same commit, same ignore patterns
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY hf_complete_20261112.py \
    $repo $rev $MM --ignore 'consolidated*' '.gitattributes' '*.md' --tries 20 --sleep 20 >> $O/dlo_$tag.log 2>&1 < /dev/null \
    && touch $O/dlo_$tag.ok
  log "file-by-file completion: $(grep -a '^complete:' $O/dlo_$tag.log | tail -1 | cut -c1-200)"
  [ -e $O/dlo_$tag.ok ] || { log "download completion FAILED $repo"; exit 1; }
  log "download completed (ignore_patterns consolidated*, .gitattributes, *.md): $(du -sh $MM | cut -f1), $(ls $MM | tr '\n' ' ')"
fi
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { log "GPU busy: not starting"; exit 1; }
log "start natural/$tag (natural3)"
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TF19 $PY kl_words_orders_20261112.py --model $MM --tmpl-kwargs '{}' \
  --bodies $B1 --words $WJ --orders natural3 --lang en --bodies-en bodies_en_150.json --max-memory-gib 60 --cpu-offload-gib 0 --stop-file $STOP \
  --out-natural $O/natural/$tag --check-natural $MS/nat3/$tag --check-natural-kind nat >> $O/orders_$tag.log 2>&1 < /dev/null
log "ended natural/$tag: $(grep -a -E 'summary \(content|CHECK FAILED|Error|stop' $O/orders_$tag.log | tail -2 | tr '\n' ' ')"
if [ -e $O/natural/$tag/DONE ]; then
  if [ -d $MM ] && [ ! -L $MM ]; then rm -rf /root/hf_models_p2nat/mistralai/Ministral-3-14B-Reasoning-2512; log "weights of $tag deleted ($(free_gib) GiB free)"; fi
else log "natural/$tag not DONE: weights kept at $MM"; fi
