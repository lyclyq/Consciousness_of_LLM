#!/bin/bash
# PRO, user 10-29 (OLMo-2-7B and EXAONE-1.2B deleted to make room, "就够了"): allenai/OLMo-2-1124-13B-Instruct, a second case of
# format difference without a stable index, from a stronger model of the same family. Its chat template renders exactly as the 7B one
# (checked 10-29 from the tokenizer alone: <|assistant|>\n + body + <|endoftext|> for a history Assistant message, <|user|>\n + body
# for a User message). Same measurement as olmo2_7b_pro.sh: suite phase G, non-thinking (the model has no thinking mode), geometry
# (body mean, response onset) and native letter probabilities P(you), English bodies (crossmodel_suite.py 516f117e, transformers 5.8).
#   0. waits until at least 30 GB are free (the user deletes the two small models); download (hf-mirror, 27.4 GB)
#   1. suite G non-thinking
# Runs in tmux session olmo13.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/allenai/OLMo-2-1124-13B-Instruct
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/olmo13.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> olmo2_13b_status.log; }
avail() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
st "0 waiting for 30G free (now $(avail)G)"
until [ "$(avail)" -ge 30 ]; do sleep 120; done
st "0 download start; free $(avail)G"
for i in 1 2 3 4; do
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('allenai/OLMo-2-1124-13B-Instruct', local_dir='$M', max_workers=8))" \
    >> olmo2_13b_download.log 2>&1 && { st "0 download DONE $(du -sh $M | cut -f1); free $(avail)G"; break; }
  st "0 download attempt $i failed"; sleep 120
done
[ -e $M/config.json ] || { st "0 download FAILED"; exit 1; }
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1 \
  $PY crossmodel_suite.py --model $M --tmpl-kwargs '{}' --bodies $B --problems problems.json --out suite/olmo2_13b_en --phases G --max-memory-gib 60 \
  --lang en --bodies-en bodies_en_150.json > olmo2_13b_G.log 2>&1
st "1 suite G exit $?"
st "done"
