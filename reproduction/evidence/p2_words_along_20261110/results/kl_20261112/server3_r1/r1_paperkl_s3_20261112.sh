#!/bin/bash
# DeepSeek-R1-Distill-Llama-70B: the paper's two KL runs redone on server3 (11-12, user: "所有没在 server3 上的都补完，更干净").
# All other R1 measurements of the paper (geometry, thinking, natural order, retention) were made on server3 with this weight
# copy (hf_models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B, commit b1c0b44b). The two KL runs were made on PRO
# (modeswitch_nat10_pro_20261108.sh, 11-08: mode_switch_20261107.py 171e7c53 = main + natural, full KL; then
# mode_switch_ctrl_20261107.py d1e71212 = main, control-token split). They are redone here with the same scripts and arguments
# (zh, '{}', role_bodies.jsonl, --min-free-gib 4 for the full run), the same render-only dry run first, and placement on the four
# GPUs without offload (PRO: 84 GiB + 60 GiB CPU offload). The KL-words run of 11-12 (out/, same weights, same directory) then
# has a reference from the same environment. Its role KLs are compared with these runs at the end (compare_r1_20261112.log).
# The PRO R1 job (pro_r1_20261112.sh) was cancelled before it started.
# Starts once no compute process has been on any GPU for 5 minutes. Stop: touch STOP_R1KL here; never kill anything.
# usage (tmux words, window R1PK, in crossmodel/r1kl_20261112): bash r1_paperkl_s3_20261112.sh
set -u
D=/root/data/p2_small_checks_20261014/crossmodel/r1kl_20261112; cd $D || exit 1
PY=/root/recursive_state_moe_server3_20260711/venv/bin/python
M=/root/data/hf_models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
STOP=$D/STOP_R1KL; P=$D/s3kl; tag=r1_distill_llama70b
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
log() { echo "[r1pk s3] $(date '+%F %T') $*" | tee -a $D/queue_r1kl.log; }
mkdir -p $P/ctrl
log "start (full $(sha256sum mode_switch_20261107.py | cut -c1-8), ctrl $(sha256sum mode_switch_ctrl_20261107.py | cut -c1-8), suite $(sha256sum crossmodel_suite.py | cut -c1-8))"
$PY mode_switch_20261107.py --model $M --tmpl-kwargs '{}' --bodies $B --out $P/dry_$tag --dry-run > $P/$tag.dry.log 2>&1
grep -q 'DRY RUN OK' $P/$tag.dry.log || { log "render check FAILED, see $P/$tag.dry.log"; exit 1; }
log "render check OK: $(grep -o '"natural_system_ok": [a-z]*' $P/$tag.dry.log | head -1)"
idle=0
while [ $idle -lt 5 ]; do [ -e $STOP ] && { log "stop flag"; exit 2; }
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; then idle=$((idle + 1)); else idle=0; fi; sleep 60; done
log "GPUs idle for 5 minutes"
for job in "mode_switch_20261107.py $P/$tag --min-free-gib 4" "mode_switch_ctrl_20261107.py $P/ctrl/$tag"; do
  read -r s out extra <<< "$job"
  [ -e $out/DONE ] && { log "$out already DONE"; continue; }
  [ -e $STOP ] && { log "stop flag"; exit 2; }
  log "start $s -> $out"
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY $s --model $M --tmpl-kwargs '{}' --bodies $B --out $out --stop-file $STOP --max-memory-gib 44 \
    --cpu-offload-gib 0 $extra >> $out.log 2>&1 < /dev/null
  if [ -e $out/DONE ]; then log "DONE $out: $(grep -a '^summary:' $out.log | tail -1 | cut -c1-400)"; else log "FAILED $out, see $out.log"; exit 1; fi
done
# compare the KL-words run's role KLs (out/) with these same-environment references
$PY - > $D/compare_r1_20261112.log 2>&1 <<'EOF'
import json, glob, statistics as st
from pathlib import Path
D = Path('/root/data/p2_small_checks_20261014/crossmodel/r1kl_20261112')
for order, refdir, getref in (('main', D / 's3kl/ctrl/r1_distill_llama70b', lambda r, pn: r['pairs'][pn]['kl_content']),
                              ('natural', D / 's3kl/r1_distill_llama70b', lambda r, pn: r['natural']['pairs'][pn]['kl_body'])):
    od = D / ('out/r1_distill_llama70b' if order == 'main' else 'out/natural/r1_distill_llama70b')
    key = 'kl_content' if order == 'main' else 'kl_body'
    rels = []
    for f in sorted((od / 'bodies').glob('*.json')):
        rec = json.loads(f.read_text())
        ref = json.loads((refdir / 'bodies' / f.name).read_text())
        for pn, v in rec['roles'].items():
            r = getref(ref, pn)
            rels.append(abs(v[key] - r) / max(abs(r), 1e-12))
    print(order, 'bodies', len(list((od / 'bodies').glob('*.json'))), 'values', len(rels), 'max rel %.2e median rel %.2e' % (max(rels), st.median(rels)))
EOF
log "words-run roles vs server3 references: $(tr '\n' ' ' < $D/compare_r1_20261112.log | cut -c1-300)"
log "done"
