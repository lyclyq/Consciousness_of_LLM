#!/bin/bash
# PRO, user 10-28 ("看来还是需要 100 局看看直出和 thinking 的格式更替后"; "模板变动影响到的都得重来，如果变动前后一样可以省算力复用"):
# full runs of Qwen3.8-27B with preserve_thinking=false after the smoke (qwen38_nothink_smoke_pro.sh). Release rule of the smoke:
# game, both modes, pass (own 1.00, User 0.00, finished 8 and >= 7 of 8, control 8/8); attribution with thinking passes (8/8 vs 0/8);
# attribution without thinking does not (P(you) 0.41 < 0.5 for the model's own X), so only its unedited conditions are run here
# (the edited ones wait for the user). Smoke items are copied into the full runs' folders (same machine, prompts, fit, settings)
# and skipped. Order:
#   1. secret game v6, 100 games, six conditions: thinking main, single main, thinking control, single control
#   2. attribution with thinking, the ten conditions of the other thinking readouts (150 held-out bodies x 2 orders)
#   3. attribution without thinking, unedited only: base_A base_U base_S
# Settings as qwen38_nothink_smoke_pro.sh (batch 16 here). Runs in tmux session q38nf; skip with STOP_Q38NF (between steps).
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/Qwen/Qwen3.8-27B
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/q38nf.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> qwen38_nothink_full_status.log; }
stop() { [ -e STOP_Q38NF ] && { st "stopped by STOP_Q38NF"; exit 0; }; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
       PYTHONUNBUFFERED=1
KT='{"preserve_thinking": false}'
KS='{"enable_thinking": false, "preserve_thinking": false}'
C6="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
ALL="base_A:assistant:-:-:0,base_U:user:-:-:0,base_S:system:-:-:0,idA8_A:assistant:assistant:V:8,swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8,rswA8_U:user:assistant:R:8,swA8_S:system:assistant:V:8"
B3="base_A:assistant:-:-:0,base_U:user:-:-:0,base_S:system:-:-:0"
FT=thinking_ablate4/qwen38_27b_nothink_fit/fit.pt; FS=thinking_ablate4/qwen38_27b_sp_nothink_fit/fit.pt
MEM="--max-memory-gib 88"
grep -q '^.* done$' qwen38_nothink_status.log || { st "smoke pipeline not finished"; exit 1; }
seed() { mkdir -p $2/items && cp -n $1/items/*.json $2/items/ && touch $2.GO; }

# 1
for q in main control; do
  for m in thinking single; do
    stop
    K=$KT; F=$FT; P=qwen38_27b_nothink; [ $m = single ] && { K=$KS; F=$FS; P=qwen38_27b_sp_nothink; }
    O=games_swap/${P}_s6_$q; seed games_swap/${P}_s6_smoke_$q $O
    st "1 game $m $q start ($(ls $O/items | wc -l) items on disk)"
    $PY crossmodel_games.py --mode $m --model $M --on-kwargs "$K" --items games/games_zh_secret6.json --games secret --qkind $q --lang zh \
      --fit-from $F --band 0-63 --conds "$C6" --batch 16 --max-new 3072 $MEM --save-gen --out $O > qwen38_nothink_full_g6_${m}_$q.log 2>&1
    st "1 game $m $q exit $?"
  done
done

# 2
stop
O=thinking_swap/qwen38_27b_nothink_all; seed thinking_swap/qwen38_27b_nothink_smoke $O
st "2 attribution thinking start ($(ls $O/items | wc -l) items on disk)"
$PY crossmodel_role_swap.py --mode thinking --model $M --on-kwargs "$KT" --lang zh --bodies bodies_zh_150_list.json --splits $B --fit-from $FT \
  --band 0-63 --conds "$ALL" --batch 16 --max-new 3072 $MEM --save-gen --out $O > qwen38_nothink_full_attr_thinking.log 2>&1
st "2 attribution thinking exit $?"

# 3
stop
O=singlepass_swap/qwen38_27b_nothink_all; mkdir -p $O/items && cp -n singlepass_swap/qwen38_27b_nothink_smoke/items/base_[AU]__*.json $O/items/ && touch $O.GO
st "3 attribution single unedited start ($(ls $O/items | wc -l) items on disk)"
$PY crossmodel_role_swap.py --mode single --model $M --on-kwargs "$KS" --lang zh --bodies bodies_zh_150_list.json --splits $B --fit-from $FS \
  --band 0-63 --conds "$B3" --batch 16 $MEM --out $O > qwen38_nothink_full_attr_single.log 2>&1
st "3 attribution single unedited exit $?"
st "done"
