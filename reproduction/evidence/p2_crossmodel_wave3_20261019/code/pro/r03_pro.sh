#!/bin/bash
# R03 of the supervisor's review on PRO (user 10-31: "r03 还是可以机制组的四个模型排上"): does the readout of the source go through the
# self-index subspace or through the residual outside it? Four readouts (Qwen3.8-27B single pass and thinking, gpt-oss-20b,
# gpt-oss-120b), fit_q.pt of r05r06_fits, the settings of r05r06_pro.sh.
#   A  crossmodel_role_swap_r03.py, all positions: <dir>_r03c_all with cswU8_A / cswA8_U (own <- User, User <- own: everything but
#      the eight coordinates replaced, basis C) and fswU8_A / fswA8_U (the full state replaced, basis F); the unedited items of
#      <dir>_r05q_all are copied in first (one baseline with R05/R06/R08). Smoke (2 bodies) with r08_smoke_check.py.
#   B  crossmodel_readout_fd_r03.py: <dir>_r03fd, derivatives of the "you" - "user" letter logit along the whole own-User difference,
#      its part in the subspace, the rest, and its part in a random subspace, at every layer (Qwen: every second layer), at both
#      endpoints. Smoke (2 bodies) with r03fd_smoke_check.py.
# Starts only after both lanes of r08_pro.sh have ended (PRO: at most two jobs at once; the 120b model needs the whole card).
# Lanes as r05r06_pro.sh: a = Qwen single pass, then thinking; b = gpt-oss-20b, then gpt-oss-120b once lane a has ended.
# Status: r03_status.log. Stop between steps: touch STOP_R03.
# usage: bash r03_pro.sh a|b   (in tmux sessions r03a / r03b)
LANE=$1
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
LOG=r03_status.log
cd $C
st() { echo "$(date '+%F %T') [$LANE] $1" >> $LOG; }
stop() { [ -e STOP_R03 ] && { st "stopped by STOP_R03"; st "lane $LANE ended"; exit 0; }; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
QENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
GENV="HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1"
CONDS="cswU8_A:assistant:user:C:8,cswA8_U:user:assistant:C:8,fswU8_A:assistant:user:F:8,fswA8_U:user:assistant:F:8"
CHECK="cswU8_A,cswA8_U,fswU8_A,fswA8_U"

readout() {
  case $1 in
    qsp)  MODE=single;   MODEL=/root/hf_models/Qwen/Qwen3.8-27B;   KW='{"enable_thinking": false}'; LG=zh; BD=bodies_zh_150_list.json; BAND=0-63; BT=16; G=60; PP=; DIR=singlepass_swap/qwen38_27b; ENVX=$QENV; STEP=2 ;;
    qth)  MODE=thinking; MODEL=/root/hf_models/Qwen/Qwen3.8-27B;   KW='{}'; LG=zh; BD=bodies_zh_150_list.json; BAND=0-63; BT=16; G=60; PP=; DIR=thinking_swap/qwen38_27b; ENVX=$QENV; STEP=2 ;;
    g20)  MODE=thinking; MODEL=/root/hf_models/openai/gpt-oss-20b;  KW='{}'; LG=en; BD=bodies_en_150_list.json; BAND=0-23; BT=4; G=30; PP=/root/p2_scale_20261015/tf_next; DIR=thinking_swap/gpt_oss_20b; ENVX=$GENV; STEP=1 ;;
    g120) MODE=thinking; MODEL=/root/hf_models/openai/gpt-oss-120b; KW='{}'; LG=en; BD=bodies_en_150_list.json; BAND=0-35; BT=8; G=90; PP=/root/p2_scale_20261015/tf_next; DIR=thinking_swap/gpt_oss_120b; ENVX=$GENV; STEP=1 ;;
  esac
  FD=r05r06_fits/$1; RD=$1
}

run_a() {  # OUT [LIMIT]
  env $ENVX PYTHONPATH=$PP $PY crossmodel_role_swap_r03.py --mode $MODE --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B \
    --fit-from $FD/fit_q.pt --band $BAND --conds "$CONDS" --region all --batch $BT --max-new 3072 --max-memory-gib $G --save-gen --out $1 \
    ${2:+--limit-bodies $2} > r03_${RD}_$(basename $1).log 2>&1
}

run_b() {  # OUT [LIMIT]
  env $ENVX PYTHONPATH=$PP $PY crossmodel_readout_fd_r03.py --mode $MODE --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B \
    --fit-from $FD/fit_q.pt --band $BAND --layer-step $STEP --batch $BT --max-new 3072 --max-memory-gib $G --out $1 ${2:+--limit-bodies $2} \
    > r03_${RD}_$(basename $1).log 2>&1
}

do_readout() {
  local R=$1
  readout $R; stop
  local Q=${DIR}_r05q_all
  [ -s $FD/fit_q.pt ] && [ $(ls $Q/items | grep -c '^base_') -eq 300 ] || { st "$R: fit or unedited items of r05q_all missing"; return 1; }
  local SA=${DIR}_r03c_smoke
  if [ ! -e $SA.PASS ]; then
    st "$R A smoke start"; run_a $SA 2; st "$R A smoke exit $?"
    $PY r08_smoke_check.py $SA $MODE $CHECK >> $LOG 2>&1 && touch $SA.PASS || { st "$R A smoke FAILED"; return 1; }
  fi
  stop
  local OA=${DIR}_r03c_all
  mkdir -p $OA/items && cp -n $Q/items/base_* $OA/items/
  st "$R A start"; touch $OA.GO; run_a $OA; st "$R A exit $?"
  stop
  local SB=${DIR}_r03fd_smoke
  if [ ! -e $SB.PASS ]; then
    st "$R B smoke start"; run_b $SB 2; st "$R B smoke exit $?"
    $PY r03fd_smoke_check.py $SB $MODE >> $LOG 2>&1 && touch $SB.PASS || { st "$R B smoke FAILED"; return 1; }
  fi
  stop
  local OB=${DIR}_r03fd
  st "$R B start"; touch $OB.GO; run_b $OB; st "$R B exit $?"
  st "$R done"
}

until ! tmux has-session -t r08a 2>/dev/null && ! tmux has-session -t r08b 2>/dev/null \
      && ! tmux has-session -t r06fa 2>/dev/null && ! tmux has-session -t r06fb 2>/dev/null; do stop; sleep 120; done
st "lane $LANE: r06f and r08 have ended, starting"
if [ "$LANE" = a ]; then
  rm -f r03_lane_a.ended
  do_readout qsp && do_readout qth
  st "lane a ended"; touch r03_lane_a.ended
else
  do_readout g20
  until [ -e r03_lane_a.ended ] && ! tmux has-session -t r03a 2>/dev/null; do stop; sleep 120; done
  do_readout g120
  st "lane b ended"
fi
