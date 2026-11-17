#!/bin/bash
# PRO, user 11-02 ("中英文双向，不如对游戏题也做一下"): cross-language transfer of the single-pass self-index subspace of Qwen3.8-27B on
# the secret game v6, main and control questions, with the script, conditions and settings of the single-pass game runs
# (crossmodel_games.py 5b29640d; the Chinese server3 runs games_swap/qwen38_27b_sp_s6_*):
#   1. English fit on the Chinese game          -> games_swap/qwen38_27b_xl_enfit_zh_sp_s6_{main,control}
#   2. Chinese fit on the English game          -> games_swap/qwen38_27b_xl_zhfit_en_sp_s6_{main,control}
#   3. Chinese fit on the Chinese game, same machine (the reference of 1) -> games_swap/qwen38_27b_zh_pro_sp_s6_{main,control}
# The English fit on the English game is games_swap/qwen38_27b_en_sp_s6_* (core run); its unedited User baseline is not clean
# (X read as own in 35 of 100 games), so direction 2 is reported, not compared. Reading rule as in qwen38_xlang_pro.sh.
# Waits until q38en and q38xl have ended (the Chinese fit of q38xl is needed). tmux q38xg; STOP_Q38XG stops between steps.
D=/root/p2_scale_20261015/crossmodel_qwen38_en_20261102; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/Qwen/Qwen3.8-27B
cd $D || exit 1
mkdir q38xg.lock 2>/dev/null || { echo "q38xg.lock present"; exit 0; }
st() { echo "$(date '+%F %T') $1" >> q38xg_status.log; }
stop() { [ -e STOP_Q38XG ] && { st "stopped by STOP_Q38XG"; rmdir q38xg.lock; exit 0; }; }
st "waiting for q38en and q38xl"
while [ -d q38en.lock ] || [ -d q38xl.lock ]; do stop; sleep 30; done
sha256sum -c --quiet SHA256SUMS_code SHA256SUMS_xlang_games >> q38xg_status.log 2>&1 || { st "code or item hash mismatch"; rmdir q38xg.lock; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
       PYTHONUNBUFFERED=1
KS='{"enable_thinking": false}'
C6="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
FZ=singlepass_ablate/qwen38_27b_zh/fit.pt; FE=singlepass_ablate/qwen38_27b_en/fit.pt
[ -e $FZ ] && [ -e $FE ] || { st "missing fit"; rmdir q38xg.lock; exit 1; }
run() { local tag=$1 out=$2; shift 2; stop; mkdir -p "$(dirname "$out")"; touch "$out.GO"; st "$tag start"; "$@" > q38xg_$tag.log 2>&1; local rc=$?; st "$tag exit $rc"; return $rc; }
game() { local tag=$1 lang=$2 fit=$3 pre=$4
  for q in main control; do
    run ${tag}_$q games_swap/${pre}_sp_s6_$q $PY crossmodel_games.py --mode single --model $M --on-kwargs "$KS" --items games/games_${lang}_secret6.json \
      --games secret --lang $lang --fit-from $fit --band 0-63 --conds "$C6" --qkind $q --batch 16 --max-new 3072 --max-memory-gib 88 --save-gen \
      --out games_swap/${pre}_sp_s6_$q
  done; }
game xg_enfit_zh zh $FE qwen38_27b_xl_enfit_zh
game xg_zhfit_en en $FZ qwen38_27b_xl_zhfit_en
game xg_zh_pro zh $FZ qwen38_27b_zh_pro
st "done"
rmdir q38xg.lock
