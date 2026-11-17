#!/bin/bash
# Retry 1 of R03 for Qwen3.8-27B with thinking (user 10-31: option 1, run with the same settings and report unfinished items as
# such). The A smoke of r03_pro.sh failed its rule on the finished count only (cswU8_A: 2 of 4 items finished within 3072 new
# tokens; the two others were still reasoning about who wrote the message; no execution error), which ended lane a before the A and
# B parts of this readout. Here A runs with the settings of r03_pro.sh unchanged; the failed smoke stays as it is and is marked
# <dir>_r03c_smoke.OVERRIDE (not .PASS). Then the B smoke with its rule (r03fd_smoke_check.py) and B.
# Starts only after the signal pilots (tmux sessions p56a / p56b) and lane b of r03_pro.sh have ended (PRO: at most two jobs at once).
# Status: r03_status.log (lane r1). Stop between steps: touch STOP_R03.
# usage: bash r03_qth_retry1_pro.sh   (in tmux session r03q1)
LANE=r1
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
LOG=r03_status.log
cd $C
st() { echo "$(date '+%F %T') [$LANE] $1" >> $LOG; }
stop() { [ -e STOP_R03 ] && { st "stopped by STOP_R03"; st "lane $LANE ended"; exit 0; }; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
ENVX="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
CONDS="cswU8_A:assistant:user:C:8,cswA8_U:user:assistant:C:8,fswU8_A:assistant:user:F:8,fswA8_U:user:assistant:F:8"
MODE=thinking; MODEL=/root/hf_models/Qwen/Qwen3.8-27B; KW='{}'; LG=zh; BD=bodies_zh_150_list.json; BAND=0-63; BT=16; G=60; PP=
DIR=thinking_swap/qwen38_27b; STEP=2; FD=r05r06_fits/qth; RD=qth

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

until ! tmux has-session -t p56a 2>/dev/null && ! tmux has-session -t p56b 2>/dev/null && ! tmux has-session -t r03b 2>/dev/null; do
  stop; sleep 120
done
st "lane r1: the p56 pilots and r03 lane b have ended, starting qth (retry 1)"
Q=${DIR}_r05q_all; SA=${DIR}_r03c_smoke
[ -s $FD/fit_q.pt ] && [ $(ls $Q/items | grep -c '^base_') -eq 300 ] || { st "qth: fit or unedited items of r05q_all missing"; st "lane r1 ended"; exit 1; }
[ -e $SA.OVERRIDE ] || echo "A smoke failed the finished-count rule only (cswU8_A 2 of 4 finished within 3072 new tokens, the others still reasoning; no execution error). User 10-31: run A with the same settings and report unfinished items as such (retry 1, r03_qth_retry1_pro.sh)." > $SA.OVERRIDE
stop
OA=${DIR}_r03c_all
mkdir -p $OA/items && cp -n $Q/items/base_* $OA/items/
st "qth A start (retry 1)"; touch $OA.GO; run_a $OA; st "qth A exit $?"
stop
SB=${DIR}_r03fd_smoke
if [ ! -e $SB.PASS ]; then
  st "qth B smoke start"; run_b $SB 2; st "qth B smoke exit $?"
  $PY r03fd_smoke_check.py $SB $MODE >> $LOG 2>&1 && touch $SB.PASS || { st "qth B smoke FAILED"; st "lane r1 ended"; exit 1; }
fi
stop
OB=${DIR}_r03fd
st "qth B start"; touch $OB.GO; run_b $OB; st "qth B exit $?"
st "qth done"; st "lane r1 ended"
