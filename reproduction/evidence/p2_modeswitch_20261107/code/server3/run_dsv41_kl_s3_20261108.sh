#!/bin/bash
# P2 mode switch (KL), DeepSeek-V4.1-Flash-1017 natural order (Table tab:self_natural3) on server3, 11-08.
# Waits for: the re-download (download_dsv41_flash_1017_20261108.log DONE) and the end of the GLM-5.3-Flash queue (msglm), then
# no compute process on any GPU for 5 minutes in a row.
# Then:
#   1. the official files must be those of the paper's runs (sha256 recorded in suite_nat3/dsv41_flash_1017/identity.json):
#      inference/model.py e5decd76, inference/engram.py 11f35ecb, encoding/encoding.py 13231c59, tokenizer.json c90dfa01;
#   2. dry run (tokenizer and encoder);
#   3. smoke: GPU 0, 3 bodies, --check; passes if 3 bodies are on disk, every KL is finite and the last-position log-probabilities
#      equal those of dsv41_natural3.forward_seqs (same argmax, max abs difference <= 1e-3);
#   4. all 150 bodies, one process per GPU on every 4th body (as the paper's natural3 run), then --finalize.
# Script dsv41/dsv41_modeswitch_nat3_20261108.py, venv_t210 (the paper's DeepSeek environment). Stop: touch modeswitch/STOP_MS
# (checked before each step and before each group of bodies). No signals. Runs in tmux.
set -u
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C || exit 1
PY=/root/data/venv_t210/bin/python; S=dsv41/dsv41_modeswitch_nat3_20261108.py
MD=/root/data/hf_models/deepseek-ai/DeepSeek-V4.1-Flash-1017; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
P=$C/modeswitch; N=$P/nat3; STOP=$P/STOP_MS; DL=$C/download_dsv41_flash_1017_20261108.log
OUT=$N/dsv41_flash; SM=$N/dsv41_flash_smoke
mkdir -p $N
log() { echo "[dsv41 kl] $(date '+%F %T') $*" | tee -a $P/queue.log >> $N/dsv41_queue.log; }
stop() { [ -e $STOP ] && { log "stop flag"; exit 2; }; }
log "waiting for the download and the GLM-5.3 queue"
until grep -q ' DONE ' $DL 2>/dev/null; do stop; grep -q 'FAILED\|listing differs' $DL 2>/dev/null && { log "download failed, see $DL"; exit 1; }; sleep 300; done
until grep -q 'glm53 queue done\|\[glm53 ms\].*FAILED' $P/queue.log || ! tmux has-session -t msglm 2>/dev/null; do stop; sleep 300; done
idle=0
while [ $idle -lt 5 ]; do
  stop
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; then idle=$((idle + 1)); else idle=0; fi
  sleep 60
done
log "download done, GPUs idle"
for pair in inference/model.py:e5decd76 inference/engram.py:11f35ecb encoding/encoding.py:13231c59 tokenizer.json:c90dfa01; do
  f=${pair%%:*}; h=${pair##*:}
  [ "$(sha256sum $MD/$f | cut -c1-8)" = $h ] || { log "$f is not the paper's file ($h): stop"; exit 1; }
done
log "official files match the paper's runs"
CUDA_VISIBLE_DEVICES= $PY $S --model-dir $MD --bodies $B --dry-run > $N/dsv41_flash.dry2.log 2>&1 < /dev/null
grep -q 'DRY RUN OK' $N/dsv41_flash.dry2.log || { log "dry run FAILED, see $N/dsv41_flash.dry2.log"; exit 1; }
stop
log "smoke start"
CUDA_VISIBLE_DEVICES=0 $PY $S --model-dir $MD --bodies $B --out $SM --limit 3 --check --stop-file $STOP > $N/dsv41_flash_smoke.log 2>&1 < /dev/null
verdict=$($PY - $SM <<'EOF'
import json, math, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((d / 'bodies').glob('*.json'))]
c = json.loads((d / 'check_shard0.json').read_text(encoding='utf-8')) if (d / 'check_shard0.json').exists() else None
fin = all(math.isfinite(v) for r in rows for pr in r['pairs'].values() for v in pr.values())
ok = len(rows) == 3 and fin and c is not None and c['same_argmax'] and c['max_abs_last_logprob_diff'] <= 1e-3
print('PASS' if ok else 'FAIL', 'bodies', len(rows), 'finite', fin, 'check', c,
      '| first', json.dumps(rows[0]['pairs']) if rows else None, '| p_ctrl', json.dumps({k: v['p_ctrl'] for k, v in rows[0]['roles'].items()}) if rows else None)
EOF
)
log "smoke: $verdict"
[ "${verdict%% *}" = PASS ] || { log "smoke failed: not starting the full run"; exit 1; }
stop
log "full start: 4 shards"
pids=()
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k $PY $S --model-dir $MD --bodies $B --out $OUT --shard $k --n-shards 4 --check --stop-file $STOP \
    > $N/dsv41_flash_shard$k.log 2>&1 < /dev/null &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { log "a shard failed (see $N/dsv41_flash_shard*.log)"; exit 1; }
stop
CUDA_VISIBLE_DEVICES= $PY $S --model-dir $MD --bodies $B --out $OUT --finalize > $N/dsv41_flash_finalize.log 2>&1 < /dev/null
if [ -e $OUT/DONE ]; then log "DONE $OUT: $(grep '^summary:' $N/dsv41_flash_finalize.log | tail -1 | cut -c1-500)"
else log "finalize FAILED, see $N/dsv41_flash_finalize.log"; exit 1; fi
log "dsv41 queue done"
