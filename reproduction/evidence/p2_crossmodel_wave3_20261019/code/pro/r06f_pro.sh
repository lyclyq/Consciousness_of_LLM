#!/bin/bash
# R06 follow-up on PRO (user 10-31: "可以先跑加幅度的"): the random-subspace controls of R06 with their edit scaled up, layer by
# layer and item by item, to at least the realized relative size of the self-index (V) edit on the same item
# (crossmodel_role_swap_r06f.py --dose-floor, the V doses of <dir>_r05q_all). Per readout (settings and fits as r05r06_pro.sh):
#   1. smoke: 2 held-out bodies x 2 orders, all positions, fit_q: frswU8_A, frswA8_U; rule in r06f_smoke_check.py
#   2. full runs, 75 held-out bodies x 2 orders, all positions: <dir>_r06f_s0 (fit_q.pt, base seed 20261022) and <dir>_r06f_s1..4
#      (fit_seed1..4.pt, base seeds 20261023..20261026): frswU8_A (floor: swU8_A), frswA8_U (floor: swA8_U); the unedited items
#      of r05q_all are copied in first, so each summary has its baseline
# Lanes as r05r06_pro.sh (PRO: at most two jobs at once): a = Qwen single pass, then thinking; b = gpt-oss-20b, then gpt-oss-120b
# once lane a has ended (Qwen and the 120b model do not fit together). Status: r06f_status.log. Stop between steps: touch STOP_R06F.
# usage: bash r06f_pro.sh a|b   (in tmux sessions r06fa / r06fb)
LANE=$1
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=crossmodel_role_swap_r06f.py; LOG=r06f_status.log
cd $C
st() { echo "$(date '+%F %T') [$LANE] $1" >> $LOG; }
stop() { [ -e STOP_R06F ] && { st "stopped by STOP_R06F"; st "lane $LANE ended"; exit 0; }; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
QENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
GENV="HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1"
CONDS="frswU8_A:assistant:user:R:8:swU8_A,frswA8_U:user:assistant:R:8:swA8_U"

readout() {
  case $1 in
    qsp)  MODE=single;   MODEL=/root/hf_models/Qwen/Qwen3.8-27B;   KW='{"enable_thinking": false}'; LG=zh; BD=bodies_zh_150_list.json; BAND=0-63; BT=16; G=60; PP=; DIR=singlepass_swap/qwen38_27b; ENVX=$QENV ;;
    qth)  MODE=thinking; MODEL=/root/hf_models/Qwen/Qwen3.8-27B;   KW='{}'; LG=zh; BD=bodies_zh_150_list.json; BAND=0-63; BT=16; G=60; PP=; DIR=thinking_swap/qwen38_27b; ENVX=$QENV ;;
    g20)  MODE=thinking; MODEL=/root/hf_models/openai/gpt-oss-20b;  KW='{}'; LG=en; BD=bodies_en_150_list.json; BAND=0-23; BT=4; G=30; PP=/root/p2_scale_20261015/tf_next; DIR=thinking_swap/gpt_oss_20b; ENVX=$GENV ;;
    g120) MODE=thinking; MODEL=/root/hf_models/openai/gpt-oss-120b; KW='{}'; LG=en; BD=bodies_en_150_list.json; BAND=0-35; BT=8; G=90; PP=/root/p2_scale_20261015/tf_next; DIR=thinking_swap/gpt_oss_120b; ENVX=$GENV ;;
  esac
  FD=r05r06_fits/$1; RD=$1
}

run_swap() {  # OUT FIT [LIMIT]
  env $ENVX PYTHONPATH=$PP $PY $S --mode $MODE --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B --fit-from $2 --band $BAND \
    --conds "$CONDS" --region all --dose-floor ${DIR}_r05q_all --batch $BT --max-new 3072 --max-memory-gib $G --save-gen --out $1 \
    ${3:+--limit-bodies $3} > r06f_${RD}_$(basename $1).log 2>&1
}

do_readout() {
  local R=$1
  readout $R; stop
  local Q=${DIR}_r05q_all
  [ -s $FD/fit_seed4.pt ] && [ $(ls $Q/items | grep -c -E '^(swU8_A|swA8_U)__') -eq 300 ] || { st "$R: fits or V items of r05q_all missing"; return 1; }
  local SM=${DIR}_r06f_smoke
  if [ ! -e $SM.PASS ]; then
    st "$R smoke start"
    run_swap $SM $FD/fit_q.pt 2
    st "$R smoke exit $?"
    $PY r06f_smoke_check.py $SM $Q $MODE >> $LOG 2>&1 && touch $SM.PASS || { st "$R smoke FAILED"; return 1; }
  fi
  for j in 0 1 2 3 4; do
    local O=${DIR}_r06f_s$j fit=fit_seed$j
    [ $j = 0 ] && fit=fit_q
    stop
    mkdir -p $O/items && cp -n $Q/items/base_* $O/items/
    st "$R r06f_s$j start"; touch $O.GO
    run_swap $O $FD/$fit.pt
    st "$R r06f_s$j exit $?"
  done
  st "$R done"
}

if [ "$LANE" = a ]; then
  rm -f r06f_lane_a.ended
  do_readout qsp && do_readout qth
  st "lane a ended"; touch r06f_lane_a.ended
else
  do_readout g20
  # the marker file, not the log line (as r05r06_pro.sh)
  until [ -e r06f_lane_a.ended ] && ! tmux has-session -t r06fa 2>/dev/null; do stop; sleep 120; done
  do_readout g120
  st "lane b ended"
fi
