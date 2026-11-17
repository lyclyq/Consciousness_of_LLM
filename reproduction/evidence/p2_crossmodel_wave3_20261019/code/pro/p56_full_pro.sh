#!/bin/bash
# Full runs of the R05/R06 signal pilots on PRO (user 10-31: "自动接"; rule of 10-31: a full run only for a test with a signal,
# "如果有信号我们才做，不然不做也不写"). Reads the verdicts of pilot_check_p56.py (rules fixed before any pilot ran) and runs, in all four
# readouts and with the settings of p56_pilot_pro.sh, every test whose verdict is "signal" (T5: "valid"); each full run is reported as
# it comes out.
#   T1 <dir>_p56full_alt       attr_alt, fit_m, message only, BASE5 (75 held-out bodies x 2 orders)
#   T2 <dir>_p56full_num       attr_num, fit_m, message only, BASE5 (75 bodies, the third option order)
#   T4 <dir>_p56full_yn_self, <dir>_p56full_yn_user   yes/no questions, fit_m, message only, BASE5 (75 bodies x 2 orders)
#   T5 <dir>_p56full_cdir      attribution question, basis P/M at the realized size of the V edit of <dir>_r05q_all, all positions;
#      the unedited items of <dir>_r05q_all are copied in first (one baseline with R05/R06/R08)
#      <dir>_p56full_cdir_content  content question: the V removal and P/M (75 bodies, order 0)
#   T3 games_swap/<model>_p56full_game_{main,control}  the secret game v6 (all games), fit_m, X only, BASE5
# Lanes: a = Qwen single pass, then thinking, after the pilots and the R03 retry (tmux r03q1) have ended; b = gpt-oss-20b after the
# pilots (alongside the retry or lane a: two jobs at most), then gpt-oss-120b once lane a and the retry have ended (whole card).
# Status: p56full_status.log; verdicts: p56_verdict_<lane>.txt. Stop between steps: touch STOP_P56FULL.
# usage: bash p56_full_pro.sh a|b   (in tmux sessions p56fa / p56fb)
LANE=$1
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
LOG=p56full_status.log
cd $C
st() { echo "$(date '+%F %T') [$LANE] $1" >> $LOG; }
stop() { [ -e STOP_P56FULL ] && { st "stopped by STOP_P56FULL"; [ "$LANE" = a ] && touch p56full_lane_a.ended; st "lane $LANE ended"; exit 0; }; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
QENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
GENV="HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1"
BASE5="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8"
CDF="pcsU8_A:assistant:user:P:8:swU8_A,mcsU8_A:assistant:user:M:8:swU8_A,pcsA8_U:user:assistant:P:8:swA8_U,mcsA8_U:user:assistant:M:8:swA8_U"
CC="base_A:assistant:-:-:0,swU8_A:assistant:user:V:8,pcsU8_A:assistant:user:P:8:swU8_A,mcsU8_A:assistant:user:M:8:swU8_A"
V=p56_verdict_$LANE.txt

readout() {
  case $1 in
    qsp)  MODE=single;   MODEL=/root/hf_models/Qwen/Qwen3.8-27B;   KW='{"enable_thinking": false}'; LG=zh; BD=bodies_zh_150_list.json; BAND=0-63; BT=16; G=60; PP=; DIR=singlepass_swap/qwen38_27b; GN=qwen38_27b_sp; ENVX=$QENV ;;
    qth)  MODE=thinking; MODEL=/root/hf_models/Qwen/Qwen3.8-27B;   KW='{}'; LG=zh; BD=bodies_zh_150_list.json; BAND=0-63; BT=16; G=60; PP=; DIR=thinking_swap/qwen38_27b; GN=qwen38_27b_th; ENVX=$QENV ;;
    g20)  MODE=thinking; MODEL=/root/hf_models/openai/gpt-oss-20b;  KW='{}'; LG=en; BD=bodies_en_150_list.json; BAND=0-23; BT=4; G=30; PP=/root/p2_scale_20261015/tf_next; DIR=thinking_swap/gpt_oss_20b; GN=gpt_oss_20b; ENVX=$GENV ;;
    g120) MODE=thinking; MODEL=/root/hf_models/openai/gpt-oss-120b; KW='{}'; LG=en; BD=bodies_en_150_list.json; BAND=0-35; BT=8; G=90; PP=/root/p2_scale_20261015/tf_next; DIR=thinking_swap/gpt_oss_120b; GN=gpt_oss_120b; ENVX=$GENV ;;
  esac
  FD=r05r06_fits/$1; RD=$1; CF=p56_fits/$1/fit_content.pt
}

swap() {  # OUT QUESTION FIT REGION CONDS [more arguments]; a full run (no --limit-bodies) starts on OUT.GO
  local o=$1 q=$2 f=$3 r=$4 c=$5; shift 5
  touch $o.GO
  env $ENVX PYTHONPATH=$PP $PY crossmodel_role_swap_p56.py --mode $MODE --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B \
    --fit-from $f --band $BAND --conds "$c" --region $r --question $q --batch $BT --max-new 3072 --max-memory-gib $G --save-gen \
    "$@" --out $o > p56full_${RD}_$(basename $o).log 2>&1
  st "$RD $(basename $o) exit $?"
}

game() {  # QKIND
  local o=games_swap/${GN}_p56full_game_$1
  touch $o.GO
  env $ENVX PYTHONPATH=$PP $PY crossmodel_games_p56.py --mode $MODE --model $MODEL --on-kwargs "$KW" --lang $LG --max-new 3072 --save-gen \
    --items games/games_${LG}_secret6.json --games secret --conds "$BASE5" --qkind $1 --fit-from $FD/fit_m.pt --band $BAND --region x \
    --batch $BT --max-memory-gib $G --out $o > p56full_${RD}_game_$1.log 2>&1
  st "$RD game_$1 exit $?"
}

has() { grep -q "$1" $V; }

do_readout() {
  local R=$1
  readout $R; stop
  [ -s $FD/fit_m.pt ] && [ -s $FD/fit_q.pt ] || { st "$R: fits missing"; return 1; }
  st "$R start"
  if has 'T1 attr_alt: signal'; then stop; swap ${DIR}_p56full_alt attr_alt $FD/fit_m.pt message "$BASE5"; fi
  if has 'T2 attr_num: signal'; then stop; swap ${DIR}_p56full_num attr_num $FD/fit_m.pt message "$BASE5"; fi
  if has 'T4 polarity: signal'; then
    stop; swap ${DIR}_p56full_yn_self yn_self $FD/fit_m.pt message "$BASE5"
    stop; swap ${DIR}_p56full_yn_user yn_user $FD/fit_m.pt message "$BASE5"
  fi
  if has 'T5 content direction: valid'; then
    if [ -s $CF ] && [ $(ls ${DIR}_r05q_all/items | grep -c '^base_') -eq 300 ]; then
      stop; mkdir -p ${DIR}_p56full_cdir/items && cp -n ${DIR}_r05q_all/items/base_* ${DIR}_p56full_cdir/items/
      swap ${DIR}_p56full_cdir attr $FD/fit_q.pt all "$CDF" --dose-floor ${DIR}_r05q_all --content-fit $CF
      stop; swap ${DIR}_p56full_cdir_content content $FD/fit_q.pt all "$CC" --dose-floor ${DIR}_r05q_all --content-fit $CF --floor-order 0 --floor-ratio-only
    else
      st "$R: content fit or unedited items of r05q_all missing, T5 not run"
    fi
  fi
  if has 'T3 game: signal'; then stop; game main; stop; game control; fi
  st "$R done"
}

[ "$LANE" = a ] && rm -f p56full_lane_a.ended
until ! tmux has-session -t p56a 2>/dev/null && ! tmux has-session -t p56b 2>/dev/null; do stop; sleep 120; done
$PY pilot_check_p56.py . > $V 2>&1
st "lane $LANE: the pilots have ended | $(grep '^VERDICT' $V || echo 'no VERDICT line')"
if ! grep -q -E 'T[1-4] [a-z_ ]+: signal|T5 content direction: valid' $V; then
  st "lane $LANE: no test with a signal, nothing to run"
  [ "$LANE" = a ] && touch p56full_lane_a.ended
  st "lane $LANE ended"; exit 0
fi
if [ "$LANE" = a ]; then
  until ! tmux has-session -t r03q1 2>/dev/null; do stop; sleep 120; done
  do_readout qsp; do_readout qth
  touch p56full_lane_a.ended; st "lane a ended"
else
  do_readout g20
  until [ -e p56full_lane_a.ended ] && ! tmux has-session -t p56fa 2>/dev/null && ! tmux has-session -t r03q1 2>/dev/null; do stop; sleep 120; done
  do_readout g120
  st "lane b ended"
fi
