#!/bin/bash
# R05 / R06 of the supervisor's review on PRO (user 10-30: "继续完成那两个任务的部署和排上跑"; protocol: the mechanism group, three
# models and four readouts; random seeds: the paper's base seed 20261022 and the next four, consecutive). Script
# crossmodel_role_swap_r05.py (crossmodel_role_swap.py f2362716 + region "message", --make-fits, per-item dose).
# Per readout (Qwen3.8-27B single pass and thinking, gpt-oss-20b, gpt-oss-120b; the settings of the paper's runs):
#   1. fits on the 75 discovery bodies: fit_q (the paper's positions), fit_m (body and end of message only), fit_seed1..4
#      (V of fit_q, random R of base seed 20261023..20261026); gpt-oss fit_q is compared with the paper's fit.pt
#   2. smoke: 2 held-out bodies x 2 orders, message region, fit_m: base_A, base_U, swU8_A, swA8_U, rswU8_A; rule in
#      r05r06_smoke_check.py (execution only); the full runs start only after it passes
#   3. full runs, 75 held-out bodies x 2 orders (the edited conditions of the paper: removal swU8_A, installation swA8_U):
#      <dir>_r05q_all  fit_q, all positions: base_A, base_U, swU8_A, swA8_U, rswU8_A, rswA8_U (paper's procedure on PRO, seed 1 of 5)
#      <dir>_r05q_msg  fit_q, message region: swU8_A, swA8_U
#      <dir>_r05m_msg  fit_m, message region: swU8_A, swA8_U        (R05: neither fit nor edit sees the question)
#      <dir>_r05m_all  fit_m, all positions: swU8_A, swA8_U
#      <dir>_r06_s1..4 fit_seed1..4, all positions: rswU8_A, rswA8_U (R06: seeds 2..5 of 5)
#      the unedited items of r05q_all are copied into the other folders before they run, so each summary has its baseline
# Lanes (PRO: at most two jobs at once): a = Qwen single pass, then thinking; b = gpt-oss-20b, then gpt-oss-120b once lane a has
# ended (Qwen and the 120b model do not fit together). Status: r05r06_status.log. Stop between steps: touch STOP_R05.
# usage: bash r05r06_pro.sh a|b   (in tmux sessions r05a / r05b)
LANE=$1
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=crossmodel_role_swap_r05.py; LOG=r05r06_status.log
cd $C
st() { echo "$(date '+%F %T') [$LANE] $1" >> $LOG; }
stop() { [ -e STOP_R05 ] && { st "stopped by STOP_R05"; st "lane $LANE ended"; exit 0; }; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
# Qwen loads offline; gpt-oss resolves its triton kernels through the mirror, as in the earlier PRO launchers (10-30 22:46 the
# first lane b stopped at once with HF_HUB_OFFLINE=1: "gpt-oss-triton-kernels ... not available in the local cache")
QENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
GENV="HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1"

readout() {
  case $1 in
    qsp)  MODE=single;   MODEL=/root/hf_models/Qwen/Qwen3.8-27B;   KW='{"enable_thinking": false}'; LG=zh; BD=bodies_zh_150_list.json; BAND=0-63; BT=16; G=60; PP=; CMP=; DIR=singlepass_swap/qwen38_27b; ENVX=$QENV ;;
    qth)  MODE=thinking; MODEL=/root/hf_models/Qwen/Qwen3.8-27B;   KW='{}'; LG=zh; BD=bodies_zh_150_list.json; BAND=0-63; BT=16; G=60; PP=; CMP=; DIR=thinking_swap/qwen38_27b; ENVX=$QENV ;;
    g20)  MODE=thinking; MODEL=/root/hf_models/openai/gpt-oss-20b;  KW='{}'; LG=en; BD=bodies_en_150_list.json; BAND=0-23; BT=4; G=30; PP=/root/p2_scale_20261015/tf_next; CMP=thinking_ablate4/gpt_oss_20b_smoke/fit.pt; DIR=thinking_swap/gpt_oss_20b; ENVX=$GENV ;;
    g120) MODE=thinking; MODEL=/root/hf_models/openai/gpt-oss-120b; KW='{}'; LG=en; BD=bodies_en_150_list.json; BAND=0-35; BT=8; G=90; PP=/root/p2_scale_20261015/tf_next; CMP=thinking_ablate4/gpt_oss_120b_fit/fit.pt; DIR=thinking_swap/gpt_oss_120b; ENVX=$GENV ;;
  esac
  FD=r05r06_fits/$1; RD=$1
}

run_swap() {  # OUT FIT REGION CONDS [LIMIT]
  env $ENVX PYTHONPATH=$PP $PY $S --mode $MODE --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B --fit-from $2 --band $BAND \
    --conds "$4" --region $3 --batch $BT --max-new 3072 --max-memory-gib $G --save-gen --out $1 ${5:+--limit-bodies $5} > r05_${RD}_$(basename $1).log 2>&1
}

do_readout() {
  local R=$1
  readout $R; stop
  if [ ! -s $FD/fit_seed4.pt ]; then
    st "$R fits start"
    env $ENVX PYTHONPATH=$PP $PY $S --make-fits $FD ${CMP:+--compare-fit $CMP} --mode $MODE --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD \
      --splits $B --band $BAND --max-memory-gib $G > r05_fits_$R.log 2>&1
    st "$R fits exit $? | $(tail -1 r05_fits_$R.log | cut -c1-220)"
  fi
  [ -s $FD/fit_seed4.pt ] || { st "$R fits FAILED"; return 1; }
  stop
  local SM=${DIR}_r05_smoke
  if [ ! -e $SM.PASS ]; then
    st "$R smoke start"
    run_swap $SM $FD/fit_m.pt message "base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8" 2
    st "$R smoke exit $?"
    $PY r05r06_smoke_check.py $SM $MODE >> $LOG 2>&1 && touch $SM.PASS || { st "$R smoke FAILED"; return 1; }
  fi
  stop
  local Q=${DIR}_r05q_all
  st "$R r05q_all start"; touch $Q.GO
  run_swap $Q $FD/fit_q.pt all "base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
  st "$R r05q_all exit $?"
  [ $(ls $Q/items | grep -c '^base_') -eq 300 ] || { st "$R r05q_all: unedited items incomplete"; return 1; }
  for spec in "r05q_msg fit_q message swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8" \
              "r05m_msg fit_m message swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8" \
              "r05m_all fit_m all swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8" \
              "r06_s1 fit_seed1 all rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8" \
              "r06_s2 fit_seed2 all rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8" \
              "r06_s3 fit_seed3 all rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8" \
              "r06_s4 fit_seed4 all rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"; do
    read -r name fit reg conds <<< "$spec"
    local O=${DIR}_$name
    stop
    mkdir -p $O/items && cp -n $Q/items/base_* $O/items/
    st "$R $name start"; touch $O.GO
    run_swap $O $FD/$fit.pt $reg "$conds"
    st "$R $name exit $?"
  done
  st "$R done"
}

if [ "$LANE" = a ]; then
  rm -f r05_lane_a.ended
  do_readout qsp && do_readout qth
  st "lane a ended"; touch r05_lane_a.ended
else
  do_readout g20
  # the marker file, not the log line, since a restarted lane a leaves earlier "lane a ended" lines in the log (10-31)
  until [ -e r05_lane_a.ended ] && ! tmux has-session -t r05a 2>/dev/null; do stop; sleep 120; done
  do_readout g120
  st "lane b ended"
fi
