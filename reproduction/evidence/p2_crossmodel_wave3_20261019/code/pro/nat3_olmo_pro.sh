#!/bin/bash
# Natural-order check for OLMo-2 on PRO (user 10-29: OLMo enters the main comparison, so its developer needs a natural-order row in
# the appendix, as every other developer; the smallest model is the appendix's representative, OLMo-2-7B; the larger one as well).
# The template keeps all three roles in place after a user request, User as its own message (checked 10-29 with the tokenizer:
# <|user|>\nREQ\n<|role|>\nBODY ... <|user|>\nQ\n<|assistant|>\n). crossmodel_natural3.py 83bff549 as for Ministral (geometry and
# single-pass attribution), English bodies, the template's default call, transformers 5.8 as in the OLMo suite runs.
#   1. OLMo-2-13B, whose weights are on disk now
#   2. waits until at least 16 GB are free (the user deletes the 13B weights), downloads OLMo-2-7B again (hf-mirror) and runs it
# Runs in tmux session olmonat. Skip with STOP_NAT3_OLMO.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; A=/root/hf_models/allenai
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/nat3_olmo.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> nat3_olmo_status.log; }
avail() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
stop() { [ -e STOP_NAT3_OLMO ] && { st "stopped by STOP_NAT3_OLMO"; exit 0; }; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
run() {
  st "$1 start (natural3 $(sha256sum crossmodel_natural3.py | cut -c1-8))"
  $PY crossmodel_natural3.py --model $A/$2 --tmpl-kwargs '{}' --bodies $B --lang en --bodies-en bodies_en_150.json --out suite_nat3/$1 \
    --max-memory-gib 60 > nat3_$1.log 2>&1
  st "$1 exit $?"
}
stop
[ -e $A/OLMo-2-1124-13B-Instruct/config.json ] && run olmo2_13b_en OLMo-2-1124-13B-Instruct
st "waiting for 16G free for OLMo-2-7B (now $(avail)G)"
until [ "$(avail)" -ge 16 ]; do stop; sleep 120; done
st "7B download start; free $(avail)G"
for i in 1 2 3 4; do
  HF_HUB_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('allenai/OLMo-2-1124-7B-Instruct', local_dir='$A/OLMo-2-1124-7B-Instruct', max_workers=8))" \
    >> nat3_olmo_download.log 2>&1 && { st "7B download DONE $(du -sh $A/OLMo-2-1124-7B-Instruct | cut -f1); free $(avail)G"; break; }
  st "7B download attempt $i failed"; sleep 120
done
[ -e $A/OLMo-2-1124-7B-Instruct/config.json ] || { st "7B download FAILED"; exit 1; }
stop
run olmo2_7b_en OLMo-2-1124-7B-Instruct
st "done"
