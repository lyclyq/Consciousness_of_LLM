#!/bin/bash
# Smoke runs of words_along_20261110.py on server3 (11-10), phi-4 on GPU 0 only, outputs in words_along/_smoke (not analysed):
#   1. main, 28 new pairs, gate against the pilot run (A/U/S states, role decomposition, the four pilot word pairs), 2 bodies
#   2. natural, 32 pairs, gate against the pilot's natural run, 1 body
#   3. main, roles + the first pair of each category, gate against the paper's G features (every layer), 1 body
# then extract_gram_20261110.py on each output. usage: bash smoke_words_s3_20261110.sh  (in tmux; no signals are sent)
set -u
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C || exit 1
PY=/root/recursive_state_moe_server3_20260711/venv/bin/python
M=/root/data/hf_models/microsoft/phi-4; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
O=$C/words_along/_smoke; mkdir -p $O
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
A=(--model $M --tmpl-kwargs '{}' --bodies $B1 --lang en --bodies-en bodies_en_150.json --words words_v2.json --max-memory-gib 44 --min-free-gib 4)
echo "smoke start $(date '+%F %T')"
$PY words_along_20261110.py "${A[@]}" --out $O/b150_main_phi4_en --order main --pairs new --gate-pilot persistence/main/phi4_en --limit 2 > $O/1_main_pilotgate.log 2>&1
echo "1 rc=$? $(grep -c '^gate' $O/1_main_pilotgate.log) gate lines; $(tail -1 $O/1_main_pilotgate.log)"
$PY words_along_20261110.py "${A[@]}" --out $O/b150_natural_phi4_en --order natural --request 'Please write a short description.' --pairs all --gate-pilot persistence/natural/phi4_en --limit 1 > $O/2_natural_pilotgate.log 2>&1
echo "2 rc=$? $(tail -1 $O/2_natural_pilotgate.log)"
$PY words_along_20261110.py "${A[@]}" --out $O/b150_main_phi4_en_roles --order main --roles --pairs first --gate-g suite/phi4_en --limit 1 > $O/3_main_ggate.log 2>&1
echo "3 rc=$? $(tail -1 $O/3_main_ggate.log)"
for d in b150_main_phi4_en b150_natural_phi4_en b150_main_phi4_en_roles; do $PY extract_gram_20261110.py $O/$d $O/extract_$d >> $O/extract.log 2>&1; echo "extract $d rc=$?"; done
echo "smoke end $(date '+%F %T')"
