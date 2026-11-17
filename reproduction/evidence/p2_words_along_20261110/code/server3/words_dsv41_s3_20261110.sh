#!/bin/bash
# DeepSeek-V4.1-Flash words along the message on server3 (11-10, ../../PROTOCOL_zh.md; user 11-10: own reader, smoke with the gate
# against the paper's natural3 features first, 4 shards only if the gate passes). Called by lane D of words_s3_20261110.sh after
# lane R, and with "smoke" by lane C (one GPU, while lanes U/A hold GPUs 0,1).
#   smoke: the official files must be the paper's (sha256 as in run_dsv41_kl_s3_20261108.sh); dry run; GPU $SMOKE_GPU, the gate on
#          the first 2 held-out bodies (cos >= 0.999, rel <= 0.03 against suite_nat3/dsv41_flash_1017/N) and those 2 bodies.
#   full:  only if the smoke's gate.json holds ok: 4 shards (one process per GPU on every 4th held-out body, --gate-from the smoke),
#          then --finalize and extract_gram_20261110.py. If the gate failed: nothing runs, the log says so.
# Environment: venv_t210 (the paper's DeepSeek environment). Stop: words_along/STOP_WORDS; no signals.
# usage: bash words_dsv41_s3_20261110.sh [smoke|full]   (default full)
set -u
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C || exit 1
PY=/root/data/venv_t210/bin/python; S=dsv41/dsv41_words_along_20261110.py
MD=/root/data/hf_models/deepseek-ai/DeepSeek-V4.1-Flash-1017; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
O=$C/words_along; SM=$O/dsv41_smoke; OUT=$O/b150_natural/dsv41_flash; STOP=$O/STOP_WORDS; PN=$C/suite_nat3/dsv41_flash_1017/N
MODE=${1:-full}   # lane D calls it without an argument
log() { echo "[words dsv41 $MODE] $(date '+%F %T') $*" | tee -a $O/queue_s3.log; }
stop() { [ -e $STOP ] && { log "stop flag"; exit 2; }; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for pair in inference/model.py:e5decd76 inference/engram.py:11f35ecb encoding/encoding.py:13231c59 tokenizer.json:c90dfa01; do
  f=${pair%%:*}; h=${pair##*:}
  [ "$(sha256sum $MD/$f | cut -c1-8)" = $h ] || { log "$f is not the paper's file ($h): stop"; exit 1; }
done
if [ $MODE = smoke ]; then
  stop
  [ -e $SM/gate.json ] && grep -q '"ok": true' $SM/gate.json && { log "smoke already passed"; exit 0; }
  CUDA_VISIBLE_DEVICES= $PY $S --model-dir $MD --bodies $B --words words_v2.json --dry-run > $O/dsv41.dry.log 2>&1 < /dev/null
  grep -q 'DRY RUN OK' $O/dsv41.dry.log || { log "dry run FAILED, see $O/dsv41.dry.log"; exit 1; }
  log "smoke start on GPU ${SMOKE_GPU:-2}"
  CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-2} $PY $S --model-dir $MD --bodies $B --words words_v2.json --out $SM --gate --paper-n $PN \
    --limit 2 --stop-file $STOP > $O/dsv41_smoke.log 2>&1 < /dev/null
  if [ -e $SM/gate.json ] && grep -q '"ok": true' $SM/gate.json; then
    log "smoke gate PASSED: $(tr -d '\n' < $SM/gate.json | cut -c1-600); bodies on disk: $(ls $SM/*.json 2>/dev/null | grep -vc gate.json)"
  else log "smoke gate FAILED or the smoke failed: the full run will not start; see $O/dsv41_smoke.log"; exit 1; fi
  exit 0
fi
# full
[ -e $SM/gate.json ] && grep -q '"ok": true' $SM/gate.json || { log "no passed smoke gate: DeepSeek-V4.1-Flash is not run"; exit 1; }
[ -e $OUT/DONE ] && { log "$OUT already DONE"; exit 0; }
stop
log "full start: 4 shards"
pids=()
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k $PY $S --model-dir $MD --bodies $B --words words_v2.json --out $OUT --gate-from $SM/gate.json \
    --shard $k --n-shards 4 --stop-file $STOP > $O/dsv41_shard$k.log 2>&1 < /dev/null &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { log "a shard failed (see $O/dsv41_shard*.log)"; exit 1; }
stop
CUDA_VISIBLE_DEVICES= $PY $S --model-dir $MD --bodies $B --words words_v2.json --out $OUT --finalize > $O/dsv41_finalize.log 2>&1 < /dev/null
if [ -e $OUT/DONE ]; then
  log "DONE $OUT"
  mkdir -p $O/extracts/b150/natural/dsv41_flash
  /root/recursive_state_moe_server3_20260711/venv/bin/python extract_gram_20261110.py $OUT $O/extracts/b150/natural/dsv41_flash/words >> $O/extract.log 2>&1 \
    && log "extracted b150/natural/dsv41_flash/words" || log "extraction FAILED $OUT"
else log "finalize FAILED, see $O/dsv41_finalize.log"; exit 1; fi
