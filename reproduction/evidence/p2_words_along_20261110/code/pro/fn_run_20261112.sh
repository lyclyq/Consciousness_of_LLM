#!/bin/bash
# Flash-Next non-thinking, main probe, KL of the ordinary words (11-12). pro_tail_20261112.sh downloads it file by file
# (hf_complete_20261112.py). The mirror kept failing on .gitattributes and LICENSE, which no loader reads, so the helper reports
# the download as incomplete and the tail skips the run. This script waits for the helper's verdict. It accepts the download
# when every file of the repository listing except .gitattributes, *.md and LICENSE* is on disk, and every safetensors shard
# named in model.safetensors.index.json is present. Then it runs the cell with the tail's own command line (the lane's row
# flashnext_nt: PYTHONPATH tf_next, 84 GiB, CPU offload 52 GiB, offline, zh, non-thinking, check against modeswitch/ctrl/flashnext_nt)
# and deletes the weights once DONE. pro_r1_20261112.sh waits for 140 GiB free disk, so it cannot start a GPU job meanwhile.
# Stop: touch words_along/STOP_KLW.  usage (tmux klw, window FNRUN, in the crossmodel directory): bash fn_run_20261112.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; WJ=$C/words_v2.json; MC=$C/modeswitch/ctrl
O=$C/words_along/kl; STOP=$C/words_along/STOP_KLW; NT='{"enable_thinking": false}'
tag=flashnext_nt; FN=/root/hf_models_words/Qwen/Qwen3.8-Flash-Next-FP8; REV=236dfdf285828023ca3bcd3f37366c58a3469b13
log() { echo "[klw fnrun] $(date '+%F %T') $*" | tee -a $O/queue.log; }
free_gib() { df --output=avail -BG /root | tail -1 | tr -dc 0-9; }
[ -e $O/$tag/DONE ] && { log "already DONE"; exit 0; }
until grep -a -q '^\[klw tail\] .* Flash-Next download:' $O/queue.log; do [ -e $STOP ] && { log "stop flag"; exit 2; }; sleep 60; done
if [ ! -e $O/dlo_$tag.ok ]; then
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY - "$FN" "$REV" <<'EOF' >> $O/dlo_$tag.log 2>&1
import fnmatch, json, sys, time
from pathlib import Path
from huggingface_hub import HfApi
d, rev = Path(sys.argv[1]), sys.argv[2]
files = None
for t in range(20):
    try:
        files = [s.rfilename for s in HfApi().model_info('Qwen/Qwen3.8-Flash-Next-FP8', revision=rev).siblings]
        break
    except Exception as e:
        print('listing try', t + 1, type(e).__name__); time.sleep(30)
if files is None:
    print('ACCEPT no: listing failed'); sys.exit(1)
skip = ('.gitattributes', '*.md', 'LICENSE*')
need = [f for f in files if not any(fnmatch.fnmatch(f, p) for p in skip)]
missing = [f for f in need if not (d / f).exists()]
idx = json.loads((d / 'model.safetensors.index.json').read_text()) if (d / 'model.safetensors.index.json').exists() else {'weight_map': {}}
shards = sorted(set(idx['weight_map'].values()))
miss_sh = [s for s in shards if not (d / s).exists()]
print(f'ACCEPT {"yes" if not missing and not miss_sh and shards else "no"}: {len(need)} needed, missing {missing[:10]}, '
      f'{len(shards)} shards in the index, missing shards {miss_sh[:10]}')
sys.exit(0 if not missing and not miss_sh and shards else 1)
EOF
  rc=$?
  log "download check: $(grep -a '^ACCEPT' $O/dlo_$tag.log | tail -1 | cut -c1-300)"
  [ $rc = 0 ] || { log "Flash-Next download not complete: not run"; exit 1; }
  touch $O/dlo_$tag.ok
fi
[ -e $STOP ] && { log "stop flag"; exit 2; }
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { log "GPU busy: not starting"; exit 1; }
log "start $tag (main)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TF19 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  $PY kl_words_orders_20261111.py --model $FN --tmpl-kwargs "$NT" --bodies $B1 --words $WJ --orders main --lang zh \
  --max-memory-gib 84 --cpu-offload-gib 52 --stop-file $STOP --out-main $O/$tag --check-ctrl $MC/$tag >> $O/orders_$tag.log 2>&1 < /dev/null
log "ended $tag: $(grep -a -E 'summary \(content|CHECK FAILED|Error|stop' $O/orders_$tag.log | tail -3 | tr '\n' ' ' | cut -c1-500)"
if [ -e $O/$tag/DONE ] && [ -d $FN ] && [ ! -L $FN ]; then
  rm -rf /root/hf_models_words/Qwen/Qwen3.8-Flash-Next-FP8; log "weights of $tag deleted ($(free_gib) GiB free)"
fi
