#!/bin/bash
# Smoke runs of words_along_20261110.py on PRO (11-10), MiMo-V2.6-Distill-Qwen-9B, under the shared lock of persistence/gpu.lock,
# outputs in words_along/_smoke (not analysed):
#   1. b150 main, 28 new pairs, gate against the pilot run (states, role decomposition, the four pilot word pairs), 2 bodies
#   2. p3 main, the first pair of each category, gate against the pilot's lean p3main run (states, role decomposition), 1 body
#   3. b150 main, roles + the first pair of each category, gate against the paper's G features (every layer), 1 body
# then extract_gram_20261110.py on each output. usage: bash smoke_words_pro_20261110.sh  (in tmux; no signals are sent)
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python
M=/root/hf_models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
O=$C/words_along/_smoke; mkdir -p $O
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
A=(--model $M --tmpl-kwargs '{"enable_thinking": false}' --words words_v2.json --max-memory-gib 44 --min-free-gib 8)
(
  flock -s 9
  echo "smoke start $(date '+%F %T')"
  $PY words_along_20261110.py "${A[@]}" --bodies $B1 --out $O/b150_main_mimo26 --order main --pairs new --gate-pilot persistence/main/mimo_v26_distill_qwen9b --limit 2 > $O/1_main_pilotgate.log 2>&1
  echo "1 rc=$? $(tail -1 $O/1_main_pilotgate.log)"
  $PY words_along_20261110.py "${A[@]}" --bodies phase3_bodies_zh.jsonl --out $O/p3_main_mimo26 --order main --pairs first --gate-pilot persistence/p3main/mimo_v26_distill_qwen9b --limit 1 > $O/2_p3_pilotgate.log 2>&1
  echo "2 rc=$? $(tail -1 $O/2_p3_pilotgate.log)"
  $PY words_along_20261110.py "${A[@]}" --bodies $B1 --out $O/b150_main_mimo26_roles --order main --roles --pairs first --gate-g suite/mimo_v26_distill_qwen9b --limit 1 > $O/3_main_ggate.log 2>&1
  echo "3 rc=$? $(tail -1 $O/3_main_ggate.log)"
) 9>$C/persistence/gpu.lock
for d in b150_main_mimo26 p3_main_mimo26 b150_main_mimo26_roles; do $PY extract_gram_20261110.py $O/$d $O/extract_$d >> $O/extract.log 2>&1; echo "extract $d rc=$?"; done
echo "smoke end $(date '+%F %T')"
