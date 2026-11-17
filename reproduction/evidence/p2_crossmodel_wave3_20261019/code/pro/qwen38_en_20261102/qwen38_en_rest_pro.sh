#!/bin/bash
# PRO, user 11-02 ("thinking 的机制英文要做齐点，但是这个中英文互换做个单次也足够了"): the rest of the English thinking-mode causal tests
# of Qwen3.8-27B, after the core queue stopped at th_attr (STOP_Q38EN) and after the cross-language queue (q38xl). Steps 5 (content,
# arithmetic) and 6 (games) of qwen38_en_core_pro.sh, with the same commands, fit and settings; th_attr is not rerun.
# Waits until q38en.lock, q38xl.lock and q38xg.lock are gone (11-02 11:30 UTC: the first launch, which waited for q38en and q38xl
# only, was stopped with STOP_Q38ER before it had run anything and relaunched with this file, so that the cross-language games run
# first). Runs in tmux session q38er; STOP_Q38ER stops between steps.
D=/root/p2_scale_20261015/crossmodel_qwen38_en_20261102; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/Qwen/Qwen3.8-27B
SPL=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
cd $D || exit 1
mkdir q38er.lock 2>/dev/null || { echo "q38er.lock present"; exit 0; }
st() { echo "$(date '+%F %T') $1" >> q38er_status.log; }
stop() { [ -e STOP_Q38ER ] && { st "stopped by STOP_Q38ER"; rmdir q38er.lock; exit 0; }; }
st "waiting for the core and cross-language queues"
while [ -d q38en.lock ] || [ -d q38xl.lock ] || [ -d q38xg.lock ]; do stop; sleep 60; done
grep -q 'th_attr exit 0' q38en_status.log || { st "th_attr did not finish"; rmdir q38er.lock; exit 1; }
sha256sum -c --quiet SHA256SUMS_code >> q38er_status.log 2>&1 || { st "code or item hash mismatch"; rmdir q38er.lock; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
       PYTHONUNBUFFERED=1
KT='{}'
CON="base_A:assistant:-:-:0,swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,base_U:user:-:-:0,swA8_U:user:assistant:V:8"
C6="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
FT=thinking_ablate4/qwen38_27b_en/fit.pt
BOD="--bodies bodies_en_150_list.json --splits $SPL"
MEM="--max-memory-gib 88"
run() { local tag=$1 out=$2; shift 2; stop; [ -n "$out" ] && { mkdir -p "$(dirname "$out")"; touch "$out.GO"; }; st "$tag start"; "$@" > q38en_$tag.log 2>&1; local rc=$?; st "$tag exit $rc"; return $rc; }

# 5 (without th_attr)
run th_content thinking_swap/qwen38_27b_en_all_content $PY crossmodel_role_swap_v3_2b032831.py --mode thinking --model $M --on-kwargs "$KT" --lang en $BOD \
  --fit-from $FT --band 0-63 --conds "$CON" --question content --batch 16 --max-new 3072 $MEM --save-gen --out thinking_swap/qwen38_27b_en_all_content
run th_arith thinking_swap/qwen38_27b_en_all_arith $PY crossmodel_role_swap.py --mode thinking --model $M --on-kwargs "$KT" --lang en $BOD \
  --fit-from $FT --band 0-63 --conds "$CON" --question arith --batch 16 --max-new 3072 $MEM --save-gen --out thinking_swap/qwen38_27b_en_all_arith

# 6
for q in main control; do
  O=games_swap/qwen38_27b_en_s6_$q
  run th_game_$q $O $PY crossmodel_games.py --mode thinking --model $M --on-kwargs "$KT" --items games/games_en_secret6.json --games secret --lang en \
    --fit-from $FT --band 0-63 --conds "$C6" --qkind $q --batch 16 --max-new 3072 $MEM --save-gen --out $O
done
st "done"
rmdir q38er.lock
