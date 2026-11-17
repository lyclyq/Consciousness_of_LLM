#!/bin/bash
# PRO, user 11-10 ("排上"): the removal half of the English thinking k64 run. qwen38_en_k64_pro.sh (11-02) ran only installation
# with 64 dimensions (base_U, swA64_U, rswA64_U); the Chinese thinking k64 run (server3 thinking_swap/qwen38_27b_all_k64) and the
# gpt-oss k64 runs have both directions. This adds the unedited own message, removal with 64 and its random control, with the
# script, suite, fit and settings of the 11-02 English k64 run (crossmodel_role_swap.py f2362716, crossmodel_suite.py ad284bb6,
# thinking_ablate4/qwen38_27b_en/fit.pt, band 0-63, all positions, kwargs {}, batch 16, 3072 tokens, 88 GiB), checked against
# SHA256SUMS_code before the run. Separate folder thinking_swap/qwen38_27b_en_all_k64_rm. Waits until at most one other compute
# process is on the GPU and 92 GiB are free. tmux q38ekr; STOP_Q38EKR stops before the run. Status q38ekr_status.log.
D=/root/p2_scale_20261015/crossmodel_qwen38_en_20261102; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/Qwen/Qwen3.8-27B
SPL=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
cd $D || exit 1
mkdir q38ekr.lock 2>/dev/null || { echo "q38ekr.lock present"; exit 0; }
st() { echo "$(date '+%F %T') $1" >> q38ekr_status.log; }
fin() { rmdir q38ekr.lock; exit $1; }
sha256sum -c --quiet SHA256SUMS_code >> q38ekr_status.log 2>&1 || { st "code or item hash mismatch"; fin 1; }
[ "$(sha256sum crossmodel_role_swap.py | cut -c1-8)" = f2362716 ] || { st "crossmodel_role_swap.py is not f2362716"; fin 1; }
[ "$(sha256sum crossmodel_suite.py | cut -c1-8)" = ad284bb6 ] || { st "crossmodel_suite.py is not ad284bb6"; fin 1; }
w=0
while n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); \
      [ "$n" -gt 1 ] || [ $((f / 1024)) -lt 92 ]; do
  [ -e STOP_Q38EKR ] && { st "stopped by STOP_Q38EKR"; fin 0; }
  [ $w = 0 ] && st "waiting for a GPU slot ($n other jobs, $((f / 1024)) GiB free)"; w=1; sleep 60
done
[ -e STOP_Q38EKR ] && { st "stopped by STOP_Q38EKR"; fin 0; }
st "GPU slot ($n other job(s), $((f / 1024)) GiB free)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
       PYTHONUNBUFFERED=1
O=thinking_swap/qwen38_27b_en_all_k64_rm
mkdir -p thinking_swap && touch $O.GO
st "th_k64_rm start"
$PY crossmodel_role_swap.py --mode thinking --model $M --on-kwargs '{}' --lang en --bodies bodies_en_150_list.json --splits $SPL \
  --fit-from thinking_ablate4/qwen38_27b_en/fit.pt --band 0-63 \
  --conds "base_A:assistant:-:-:0,swU64_A:assistant:user:V:64,rswU64_A:assistant:user:R:64" --batch 16 --max-new 3072 --max-memory-gib 88 \
  --save-gen --out $O < /dev/null > q38en_th_k64_rm.log 2>&1
st "th_k64_rm exit $? | $(tail -1 q38en_th_k64_rm.log | cut -c1-400)"
st "done"
fin 0
