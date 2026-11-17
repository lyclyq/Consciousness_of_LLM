#!/bin/bash
# R08 of the supervisor's review on PRO (user 10-31: "r08 也可以用这三模型 4 形式做"): do the self-index coordinates tell one external
# source from the other? A User and a System message exchange their eight coordinates (User <- System, System <- User), with the
# random control of base seed 20261022; crossmodel_role_swap_r05.py (01ab63dd) with fit_q.pt of r05r06_fits at all positions, the
# settings of r05r06_pro.sh. Per readout (Qwen3.8-27B single pass and thinking, gpt-oss-20b, gpt-oss-120b):
#   1. smoke: 2 held-out bodies x 2 orders, all five conditions; rule in r08_smoke_check.py (execution only)
#   2. full run, 75 held-out bodies x 2 orders: <dir>_r08_us with base_S (System, unedited), swS8_U (User <- System),
#      swU8_S (System <- User), rswS8_U and rswU8_S (the same in the random subspace); the unedited own and User items of
#      <dir>_r05q_all are copied in first, so R05, R06 and R08 share one baseline
# Starts only after both lanes of r06f_pro.sh have ended (PRO: at most two jobs at once; the 120b model needs the whole card).
# Lanes as r05r06_pro.sh: a = Qwen single pass, then thinking; b = gpt-oss-20b, then gpt-oss-120b once lane a has ended.
# Status: r08_status.log. Stop between steps: touch STOP_R08.
# usage: bash r08_pro.sh a|b   (in tmux sessions r08a / r08b)
LANE=$1
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=crossmodel_role_swap_r05.py; LOG=r08_status.log
cd $C
st() { echo "$(date '+%F %T') [$LANE] $1" >> $LOG; }
stop() { [ -e STOP_R08 ] && { st "stopped by STOP_R08"; st "lane $LANE ended"; exit 0; }; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
QENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
GENV="HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1"
CONDS="base_S:system:-:-:0,swS8_U:user:system:V:8,swU8_S:system:user:V:8,rswS8_U:user:system:R:8,rswU8_S:system:user:R:8"
CHECK="base_S,swS8_U,swU8_S,rswS8_U,rswU8_S"

readout() {
  case $1 in
    qsp)  MODE=single;   MODEL=/root/hf_models/Qwen/Qwen3.8-27B;   KW='{"enable_thinking": false}'; LG=zh; BD=bodies_zh_150_list.json; BAND=0-63; BT=16; G=60; PP=; DIR=singlepass_swap/qwen38_27b; ENVX=$QENV ;;
    qth)  MODE=thinking; MODEL=/root/hf_models/Qwen/Qwen3.8-27B;   KW='{}'; LG=zh; BD=bodies_zh_150_list.json; BAND=0-63; BT=16; G=60; PP=; DIR=thinking_swap/qwen38_27b; ENVX=$QENV ;;
    g20)  MODE=thinking; MODEL=/root/hf_models/openai/gpt-oss-20b;  KW='{}'; LG=en; BD=bodies_en_150_list.json; BAND=0-23; BT=4; G=30; PP=/root/p2_scale_20261015/tf_next; DIR=thinking_swap/gpt_oss_20b; ENVX=$GENV ;;
    g120) MODE=thinking; MODEL=/root/hf_models/openai/gpt-oss-120b; KW='{}'; LG=en; BD=bodies_en_150_list.json; BAND=0-35; BT=8; G=90; PP=/root/p2_scale_20261015/tf_next; DIR=thinking_swap/gpt_oss_120b; ENVX=$GENV ;;
  esac
  FD=r05r06_fits/$1; RD=$1
}

run_swap() {  # OUT [LIMIT]
  env $ENVX PYTHONPATH=$PP $PY $S --mode $MODE --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B --fit-from $FD/fit_q.pt \
    --band $BAND --conds "$CONDS" --region all --batch $BT --max-new 3072 --max-memory-gib $G --save-gen --out $1 ${2:+--limit-bodies $2} \
    > r08_${RD}_$(basename $1).log 2>&1
}

do_readout() {
  local R=$1
  readout $R; stop
  local Q=${DIR}_r05q_all
  [ -s $FD/fit_q.pt ] && [ $(ls $Q/items | grep -c '^base_') -eq 300 ] || { st "$R: fit or unedited items of r05q_all missing"; return 1; }
  local SM=${DIR}_r08_smoke
  if [ ! -e $SM.PASS ]; then
    st "$R smoke start"
    run_swap $SM 2
    st "$R smoke exit $?"
    $PY r08_smoke_check.py $SM $MODE $CHECK >> $LOG 2>&1 && touch $SM.PASS || { st "$R smoke FAILED"; return 1; }
  fi
  stop
  local O=${DIR}_r08_us
  mkdir -p $O/items && cp -n $Q/items/base_* $O/items/
  st "$R r08_us start"; touch $O.GO
  run_swap $O
  st "$R r08_us exit $?"
  st "$R done"
}

until ! tmux has-session -t r06fa 2>/dev/null && ! tmux has-session -t r06fb 2>/dev/null; do stop; sleep 120; done
st "lane $LANE: r06f has ended, starting"
if [ "$LANE" = a ]; then
  rm -f r08_lane_a.ended
  do_readout qsp && do_readout qth
  st "lane a ended"; touch r08_lane_a.ended
else
  do_readout g20
  until [ -e r08_lane_a.ended ] && ! tmux has-session -t r08a 2>/dev/null; do stop; sleep 120; done
  do_readout g120
  st "lane b ended"
fi
