#!/bin/bash
# PRO, user 11-02 ("行，补 64 维思考"): with the English items and thinking, installing the eight own coordinates into a User message
# hardly creates a claim (swA8_U +0.04 [+0.01, +0.07], Chinese +0.60), the reasoning reading "there is a user message" in 125 of 150
# items. The paper attributes the weaker installation with eight dimensions mostly to the dimension of the edit (Chinese thinking with
# 64: +0.97). This run tests that with the English items: installation with 64 dimensions and its random control, plus the unedited
# User message as the paired base, with the script, fit and settings of the Chinese thinking k64 run (thinking_swap/qwen38_27b_all_k64:
# crossmodel_role_swap.py f2362716, thinking_ablate4 fit, band 0-63, batch 16, 3072 tokens); removal with 64 is not rerun (removal
# with eight is already complete in English). Waits until the other three queues (q38en, q38xl, q38er) have ended. tmux q38ek;
# STOP_Q38EK stops before the run.
D=/root/p2_scale_20261015/crossmodel_qwen38_en_20261102; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/Qwen/Qwen3.8-27B
SPL=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
cd $D || exit 1
mkdir q38ek.lock 2>/dev/null || { echo "q38ek.lock present"; exit 0; }
st() { echo "$(date '+%F %T') $1" >> q38ek_status.log; }
st "waiting for q38en, q38xl and q38er"
while [ -d q38en.lock ] || [ -d q38xl.lock ] || [ -d q38er.lock ]; do [ -e STOP_Q38EK ] && { st "stopped by STOP_Q38EK"; rmdir q38ek.lock; exit 0; }; sleep 60; done
[ -e STOP_Q38EK ] && { st "stopped by STOP_Q38EK"; rmdir q38ek.lock; exit 0; }
sha256sum -c --quiet SHA256SUMS_code >> q38ek_status.log 2>&1 || { st "code or item hash mismatch"; rmdir q38ek.lock; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
       PYTHONUNBUFFERED=1
O=thinking_swap/qwen38_27b_en_all_k64
mkdir -p thinking_swap && touch $O.GO
st "th_k64 start"
$PY crossmodel_role_swap.py --mode thinking --model $M --on-kwargs '{}' --lang en --bodies bodies_en_150_list.json --splits $SPL \
  --fit-from thinking_ablate4/qwen38_27b_en/fit.pt --band 0-63 \
  --conds "base_U:user:-:-:0,swA64_U:user:assistant:V:64,rswA64_U:user:assistant:R:64" --batch 16 --max-new 3072 --max-memory-gib 88 \
  --save-gen --out $O > q38en_th_k64.log 2>&1
st "th_k64 exit $?"
st "done"
rmdir q38ek.lock
