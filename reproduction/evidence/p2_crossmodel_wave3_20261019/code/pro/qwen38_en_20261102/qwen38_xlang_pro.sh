#!/bin/bash
# PRO, user 11-02 ("中文版的那个空间是否能影响英文版的那个空间，只做归属"; "只做归属题单次向前的中英文互相影响够了，没必要做 thinking"):
# cross-language transfer of the self-index subspace of Qwen3.8-27B, single pass, attribution question only. Same scripts, conditions,
# band, rank and batch as the single-pass replacement runs (qwen38_en_core_pro.sh step 3, the Chinese server3 runs).
#   1. Chinese single-pass fit on PRO (crossmodel_singlepass_ablate.py e848ae5c, --lang zh)     -> singlepass_ablate/qwen38_27b_zh
#   2. Chinese fit on English items  (crossmodel_role_swap_v3_2b032831.py, ALL conditions)      -> singlepass_swap/qwen38_27b_xl_zhfit_en
#   3. English fit on Chinese items                                                            -> singlepass_swap/qwen38_27b_xl_enfit_zh
#   4. Chinese fit on Chinese items, same machine (separates machine and refit from language)  -> singlepass_swap/qwen38_27b_zh_all_pro
# The English fit on English items is singlepass_swap/qwen38_27b_en_all (core run). The random bases R_l are QR of a Gaussian with
# seed 20261022 + l in both fits, so the random controls use the same bases. Reading rule, fixed before any result: transfer ratio =
# paired change with the other language's fit / paired change with the same language's fit (both on this machine), for removal
# (own <- User) and installation (User <- own) separately; the random controls and the identity control are reported as they are.
# Waits until the core queue has ended (q38en.lock gone). Runs in tmux session q38xl; STOP_Q38XL stops between steps.
D=/root/p2_scale_20261015/crossmodel_qwen38_en_20261102; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/Qwen/Qwen3.8-27B
SPL=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
cd $D || exit 1
mkdir q38xl.lock 2>/dev/null || { echo "q38xl.lock present"; exit 0; }
st() { echo "$(date '+%F %T') $1" >> q38xl_status.log; }
stop() { [ -e STOP_Q38XL ] && { st "stopped by STOP_Q38XL"; rmdir q38xl.lock; exit 0; }; }
st "waiting for the core queue (q38en.lock)"
while [ -d q38en.lock ]; do stop; sleep 60; done
sha256sum -c --quiet SHA256SUMS_code SHA256SUMS_xlang >> q38xl_status.log 2>&1 || { st "code or item hash mismatch"; rmdir q38xl.lock; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
       PYTHONUNBUFFERED=1
KS='{"enable_thinking": false}'
ALL="base_A:assistant:-:-:0,base_U:user:-:-:0,base_S:system:-:-:0,idA8_A:assistant:assistant:V:8,swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8,rswA8_U:user:assistant:R:8,swA8_S:system:assistant:V:8"
FZ=singlepass_ablate/qwen38_27b_zh/fit.pt; FE=singlepass_ablate/qwen38_27b_en/fit.pt
MEM="--max-memory-gib 88"
run() { local tag=$1 out=$2; shift 2; stop; [ -n "$out" ] && { mkdir -p "$(dirname "$out")"; touch "$out.GO"; }; st "$tag start"; "$@" > q38xl_$tag.log 2>&1; local rc=$?; st "$tag exit $rc"; return $rc; }
swap() { local tag=$1 lang=$2 fit=$3 out=$4
  run $tag $out $PY crossmodel_role_swap_v3_2b032831.py --mode single --model $M --on-kwargs "$KS" --lang $lang --bodies bodies_${lang}_150_list.json \
    --splits $SPL --fit-from $fit --band 0-63 --conds "$ALL" --batch 16 --max-new 3072 $MEM --out $out; }

# 1
[ -e $FZ ] || run sp_fit_zh "" $PY crossmodel_singlepass_ablate.py --model $M --lang zh --band 0-63 --bodies bodies_zh_150_list.json --splits $SPL $MEM \
  --out singlepass_ablate/qwen38_27b_zh
[ -e $FZ ] && [ -e $FE ] || { st "missing fit"; rmdir q38xl.lock; exit 1; }
# 2-4
swap xl_zhfit_en en $FZ singlepass_swap/qwen38_27b_xl_zhfit_en
swap xl_enfit_zh zh $FE singlepass_swap/qwen38_27b_xl_enfit_zh
swap zh_all_pro zh $FZ singlepass_swap/qwen38_27b_zh_all_pro
st "done"
rmdir q38xl.lock
