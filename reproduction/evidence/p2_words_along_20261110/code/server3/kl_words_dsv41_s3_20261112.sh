#!/bin/bash
# P2 ordinary words in next-token prediction, DeepSeek-V4.1-Flash-1017, natural order only (Table tab:self_natural3), server3, 11-12.
# dsv41/dsv41_kl_words_nat_20261112.py in the environment of the paper-run KL of this model (run_dsv41_kl_s3_20261108.sh: venv_t210,
# the official files below, one process per GPU on every 4th body):
#   1. the official files must be those of the paper's runs (inference/model.py e5decd76, inference/engram.py 11f35ecb,
#      encoding/encoding.py 13231c59, tokenizer.json c90dfa01)
#   2. dry run (CPU, tokenizer and encoder): word renderings identical to the words-along DeepSeek run, control ids as the KL run
#   3. starts once no compute process has been on any GPU for 5 minutes in a row
#   4. smoke on GPU 0: the first 2 held-out bodies, role KLs against modeswitch/nat3/dsv41_flash (relative difference <= 0.02)
#   5. only on a passed check: 4 shards (GPUs 0-3, every 4th held-out body), then --finalize (CPU): summary, DONE
# About 67 forwards per body (3 roles + 32 pairs x 2), ~630 s per body as the words-along DeepSeek run, ~3.5 h with 4 shards.
# Stop: touch words_along/kl/STOP_DSV41KL (checked before each step and each group of bodies). No signals: never kill anything here.
# usage (tmux, its own window, in the crossmodel directory): bash kl_words_dsv41_s3_20261112.sh
set -u
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C || exit 1
PY=/root/data/venv_t210/bin/python; S=dsv41/dsv41_kl_words_nat_20261112.py
MD=/root/data/hf_models/deepseek-ai/DeepSeek-V4.1-Flash-1017; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
WJ=$C/words_v2.json; REF=$C/modeswitch/nat3/dsv41_flash; K=$C/words_along/kl; OUT=$K/natural/dsv41_flash; STOP=$K/STOP_DSV41KL
mkdir -p $K/natural
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
log() { echo "[klw dsv41] $(date '+%F %T') $*" | tee -a $K/queue_dsv41.log >> $C/words_along/queue_s3.log; }
stop() { [ -e $STOP ] && { log "stop flag"; exit 2; }; }
[ -e $OUT/DONE ] && { log "$OUT already DONE"; exit 0; }
stop
log "start (script $(sha256sum $S | cut -c1-8))"
for pair in inference/model.py:e5decd76 inference/engram.py:11f35ecb encoding/encoding.py:13231c59 tokenizer.json:c90dfa01; do
  f=${pair%%:*}; h=${pair##*:}
  [ "$(sha256sum $MD/$f | cut -c1-8)" = $h ] || { log "$f is not the paper's file ($h): stop"; exit 1; }
done
log "official files match the paper's runs"
CUDA_VISIBLE_DEVICES= $PY $S --model-dir $MD --bodies $B --words $WJ --check-ref $REF --dry-run > $K/dsv41_flash.dry.log 2>&1 < /dev/null
grep -q 'DRY RUN OK' $K/dsv41_flash.dry.log || { log "dry run FAILED, see $K/dsv41_flash.dry.log"; exit 1; }
log "dry run OK: $(grep -o '"forwards": [0-9]*, "max_tokens": [0-9]*' $K/dsv41_flash.dry.log | head -1)"
idle=0
while [ $idle -lt 5 ]; do
  stop
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; then idle=$((idle + 1)); else idle=0; fi
  sleep 60
done
log "GPUs idle for 5 minutes"
if ! { [ -e $OUT/check.json ] && grep -q '"ok": true' $OUT/check.json; }; then
  stop
  log "smoke start (GPU 0, first 2 held-out bodies)"
  CUDA_VISIBLE_DEVICES=0 $PY $S --model-dir $MD --bodies $B --words $WJ --check-ref $REF --out $OUT --smoke --stop-file $STOP \
    > $K/dsv41_flash_smoke.log 2>&1 < /dev/null
  if [ -e $OUT/check.json ] && grep -q '"ok": true' $OUT/check.json; then
    log "smoke check PASSED: $(grep -a '^check (natural3)' $K/dsv41_flash_smoke.log | tail -1 | cut -c1-300)"
  else log "smoke check FAILED or the smoke failed: the full run does not start; see $K/dsv41_flash_smoke.log"; exit 1; fi
fi
stop
log "full start: 4 shards"
pids=()
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k $PY $S --model-dir $MD --bodies $B --words $WJ --check-ref $REF --out $OUT --shard $k --n-shards 4 \
    --stop-file $STOP > $K/dsv41_flash_shard$k.log 2>&1 < /dev/null &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { log "a shard failed or stopped (see $K/dsv41_flash_shard*.log)"; exit 1; }
stop
CUDA_VISIBLE_DEVICES= $PY $S --model-dir $MD --bodies $B --words $WJ --check-ref $REF --out $OUT --finalize > $K/dsv41_flash_finalize.log 2>&1 < /dev/null
if [ -e $OUT/DONE ]; then log "DONE $OUT: $(grep -a 'summary (content' $K/dsv41_flash_finalize.log | tail -1 | cut -c1-500)"
else log "finalize FAILED, see $K/dsv41_flash_finalize.log"; exit 1; fi
