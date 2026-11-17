#!/bin/bash
# PRO, user 10-24: is the asymmetry (removal complete, installation partial) a limit of the 8-dim edit? Diagnostic with the leading
# 64 dimensions of the same fit (k_max 64), every layer, gpt-oss-20b, thinking: (1) the attribution question on the 75 held-out
# bodies x 2 orders (as thinking_swap/gpt_oss_20b_all); (2) the games secret and score, main question (as games_swap/
# gpt_oss_20b_main). The unedited conditions are copied from those runs (identical prompts; the scripts skip items present) so
# only the 64-dim conditions run. Not a paper row (the paper reports rank 8); an appendix diagnostic. Waits until
# gptoss_games_full_pro.sh has exited. Watchdog as gptoss_ctrl_pro.sh. Skip with STOP_GPTOSS_K64.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-20b
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
WD_MIN=15
K64="swU64_A:assistant:user:V:64,swA64_U:user:assistant:V:64,rswU64_A:assistant:user:R:64,rswA64_U:user:assistant:R:64"
mkdir "$C/gptoss_k64.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss_k8_status.log; }
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
PROG='^[A-Za-z0-9_]+ [0-9]+/[0-9]+ [0-9]+s|waiting for '
wd() {  # wd NAME LOG -- command...: run under the watchdog, at most two attempts
  local name=$1 log=$2; shift 2
  local t pid start ok rc
  for t in 1 2; do
    "$@" > "$log.attempt$t" 2>&1 &
    pid=$!; start=$(date +%s); ok=0
    while kill -0 $pid 2>/dev/null; do
      grep -qE "$PROG" "$log.attempt$t" && ok=1
      if [ $ok = 0 ] && [ $(( $(date +%s) - start )) -gt $(( WD_MIN * 60 )) ]; then
        st "$name watchdog: no progress after $WD_MIN min, stopping pid $pid (attempt $t)"
        kill -TERM $pid; sleep 30; kill -9 $pid 2>/dev/null
        break
      fi
      sleep 30
    done
    wait $pid; rc=$?
    cp "$log.attempt$t" "$log"
    if [ $ok = 1 ] || [ $rc = 0 ]; then return $rc; fi
    st "$name attempt $t ended without progress (exit $rc)"
  done
  return 1
}
until ! pgrep -f '[g]ptoss_games_full_pro.sh' > /dev/null; do [ -e STOP_GPTOSS_K64 ] && exit 0; sleep 60; done
[ -e STOP_GPTOSS_K64 ] && exit 0
# (1) attribution question, every layer, 64 dims
O1=thinking_swap/gpt_oss_20b_all_k64
mkdir -p $O1/items && cp -n thinking_swap/gpt_oss_20b_all/items/base_A__* thinking_swap/gpt_oss_20b_all/items/base_U__* $O1/items/ && touch $O1.GO
st "D k64 attribution start ($(ls $O1/items | wc -l) unedited items copied)"
wd D_attr gptoss_k64_attr.log $PY crossmodel_role_swap.py --mode thinking --model $M --lang en --bodies bodies_en_150_list.json --splits $B \
   --batch 4 --max-new 3072 --max-memory-gib 30 --save-gen --fit-from thinking_ablate4/gpt_oss_20b_smoke/fit.pt --band 0-23 \
   --conds "base_A:assistant:-:-:0,base_U:user:-:-:0,$K64" --out $O1
st "D k64 attribution exit $?"
[ -e STOP_GPTOSS_K64 ] && exit 0
# (2) games, main question, every layer, 64 dims
O2=games_swap/gpt_oss_20b_main_k64
mkdir -p $O2/items && cp -n games_swap/gpt_oss_20b_main/items/base_A__* games_swap/gpt_oss_20b_main/items/base_U__* $O2/items/ && touch $O2.GO
st "D k64 games start ($(ls $O2/items | wc -l) unedited items copied)"
wd D_games gptoss_k64_games.log $PY crossmodel_games.py --mode thinking --model $M --on-kwargs '{}' --items games/games_en.json \
   --games secret,score --lang en --fit-from thinking_ablate4/gpt_oss_20b_smoke/fit.pt --band 0-23 \
   --conds "base_A:assistant:-:-:0,base_U:user:-:-:0,$K64" --batch 4 --max-new 3072 --max-memory-gib 30 --save-gen --qkind main --out $O2
st "D k64 games exit $?"
