#!/bin/bash
# PRO, user 11-02 ("其实就差 qwen 27b 最好用英文吧机制组都做一遍"; "pro 6000 并排跑 qwen 行了"): the core causal tests of Qwen3.8-27B
# with the English items. Scripts are the as-run versions of the Chinese runs on server3 (hashes checked against SHA256SUMS_code
# before anything runs; crossmodel_suite.py is ad284bb6, the suite of those runs); conditions, layers, ranks, batch and token limits
# are those of the Chinese runs (from their identity.json); only --lang en, the English bodies (bodies_en_150_list.json, as the
# gpt-oss runs) and the English game items (games_en_secret6.json, as the gpt-oss runs) differ. One Qwen job at a time, next to the
# RoBERTa queue on the same GPU (user's approval); --max-memory-gib 88 as the earlier Qwen3.8-27B runs on PRO.
# The thinking fit uses --conds base_A,base_U: only the fit.pt of the Chinese thinking_ablate4 run enters the paper (source map),
# and the fit does not depend on the conditions (K_MAX = 64 is a constant), so its removal conditions are not rerun.
# Order:
#   0. tokenizer-only alignment check of the role swap, both renderings
#   1. single-pass fit and removal (crossmodel_singlepass_ablate.py e848ae5c)    -> singlepass_ablate/qwen38_27b_en
#   2. thinking fit and baselines (crossmodel_thinking_ablate2.py ba2208a6)       -> thinking_ablate4/qwen38_27b_en
#   -- waits for GO_Q38EN_MAIN, written by hand after 0-2 are read (rendering, P(you), parsed answers vs raw text) --
#   3. single-pass replacement: attribution (2b032831, ten conditions), content (2b032831), arithmetic (f2362716)
#   4. single-pass games, secret v6, main and control (crossmodel_games.py 5b29640d)
#   5. thinking replacement: attribution, content, arithmetic
#   6. thinking games, main and control
# Runs in tmux session q38en; STOP_Q38EN stops between steps. Every step resumes from the items on disk.
D=/root/p2_scale_20261015/crossmodel_qwen38_en_20261102; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/Qwen/Qwen3.8-27B
SPL=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
cd $D || exit 1
mkdir q38en.lock 2>/dev/null || { echo "q38en.lock present"; exit 0; }
st() { echo "$(date '+%F %T') $1" >> q38en_status.log; }
stop() { [ -e STOP_Q38EN ] && { st "stopped by STOP_Q38EN"; rmdir q38en.lock; exit 0; }; }
sha256sum -c --quiet SHA256SUMS_code >> q38en_status.log 2>&1 || { st "code or item hash mismatch"; rmdir q38en.lock; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
       PYTHONUNBUFFERED=1
KT='{}'
KS='{"enable_thinking": false}'
ALL="base_A:assistant:-:-:0,base_U:user:-:-:0,base_S:system:-:-:0,idA8_A:assistant:assistant:V:8,swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8,rswA8_U:user:assistant:R:8,swA8_S:system:assistant:V:8"
CON="base_A:assistant:-:-:0,swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,base_U:user:-:-:0,swA8_U:user:assistant:V:8"
C6="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
FS=singlepass_ablate/qwen38_27b_en/fit.pt; FT=thinking_ablate4/qwen38_27b_en/fit.pt
BOD="--bodies bodies_en_150_list.json --splits $SPL"
MEM="--max-memory-gib 88"
# 11-02 07:58 UTC: the copy on PRO ran without the mkdir below; the touch failed for thinking_ablate4/qwen38_27b_en.GO (the
# parent folder did not exist yet), th_fit waited 33 min, and the folders and that .GO were made by hand (no other change).
run() { local tag=$1 out=$2; shift 2; stop; [ -n "$out" ] && { mkdir -p "$(dirname "$out")"; touch "$out.GO"; }; st "$tag start"; "$@" > q38en_$tag.log 2>&1; local rc=$?; st "$tag exit $rc"; return $rc; }

# 0
for m in single thinking; do
  K=$KT; [ $m = single ] && K=$KS
  run align_$m "" $PY crossmodel_role_swap.py --check-align --mode $m --model $M --on-kwargs "$K" --lang en $BOD --fit-from none \
    --band 0-63 --conds "$ALL" --out align_$m || { rmdir q38en.lock; exit 1; }
done

# 1
[ -e $FS ] || run sp_fit "" $PY crossmodel_singlepass_ablate.py --model $M --lang en --band 0-63 $BOD $MEM --out singlepass_ablate/qwen38_27b_en
[ -e $FS ] || { st "no single-pass fit"; rmdir q38en.lock; exit 1; }

# 2
O=thinking_ablate4/qwen38_27b_en
[ -e $FT ] || run th_fit $O $PY crossmodel_thinking_ablate2.py --suite full --no-gen-edit --edit-end before_gen --model $M --lang en --main-layer 39 \
  --band 0-63 --layers band $BOD --batch 16 --max-new 3072 $MEM --save-gen --conds "base_A:assistant:-:0,base_U:user:-:0" --out $O
[ -e $FT ] || { st "no thinking fit"; rmdir q38en.lock; exit 1; }

st "waiting for GO_Q38EN_MAIN"
while [ ! -e GO_Q38EN_MAIN ]; do stop; sleep 60; done

# 3
run sp_attr singlepass_swap/qwen38_27b_en_all $PY crossmodel_role_swap_v3_2b032831.py --mode single --model $M --on-kwargs "$KS" --lang en $BOD \
  --fit-from $FS --band 0-63 --conds "$ALL" --batch 16 --max-new 3072 $MEM --out singlepass_swap/qwen38_27b_en_all
run sp_content singlepass_swap/qwen38_27b_en_all_content $PY crossmodel_role_swap_v3_2b032831.py --mode single --model $M --on-kwargs "$KS" --lang en $BOD \
  --fit-from $FS --band 0-63 --conds "$CON" --question content --batch 16 --max-new 3072 $MEM --out singlepass_swap/qwen38_27b_en_all_content
run sp_arith singlepass_swap/qwen38_27b_en_all_arith $PY crossmodel_role_swap.py --mode single --model $M --on-kwargs "$KS" --lang en $BOD \
  --fit-from $FS --band 0-63 --conds "$CON" --question arith --batch 16 --max-new 3072 $MEM --out singlepass_swap/qwen38_27b_en_all_arith

# 4
for q in main control; do
  O=games_swap/qwen38_27b_en_sp_s6_$q
  run sp_game_$q $O $PY crossmodel_games.py --mode single --model $M --on-kwargs "$KS" --items games/games_en_secret6.json --games secret --lang en \
    --fit-from $FS --band 0-63 --conds "$C6" --qkind $q --batch 16 --max-new 3072 $MEM --save-gen --out $O
done

# 5
run th_attr thinking_swap/qwen38_27b_en_all $PY crossmodel_role_swap_v3_2b032831.py --mode thinking --model $M --on-kwargs "$KT" --lang en $BOD \
  --fit-from $FT --band 0-63 --conds "$ALL" --batch 16 --max-new 3072 $MEM --save-gen --out thinking_swap/qwen38_27b_en_all
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
rmdir q38en.lock
