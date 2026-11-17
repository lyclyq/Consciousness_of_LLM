#!/bin/bash
# Wording variant of R05 for Qwen3.8-27B in English (user 11-09: Qwen is the only model of the mechanism group run in both
# languages, "补"). There is no English R05 cell for Qwen: the 11-02 English rerun (crossmodel_qwen38_en_20261102) used the
# paper's design (paper fit, all positions, the question included), so this queue makes the English R05 cell first, the same
# way r05r06_pro.sh made the Chinese one, and then the two wordings:
#   1. fits on the 75 discovery bodies with crossmodel_role_swap_r05.py --make-fits (fit_m is used; fit_q is compared with the
#      11-02 English paper fit, a diagnostic only), in r05r06_fits/qsp_en and r05r06_fits/qth_en;
#   2. per wording: paper = crossmodel_role_swap_r05.py unchanged (Q_ATTR_EN, "You (the AI assistant) / The user / The system"),
#      pron and role = wording_swap_v2_20261109.py --wording pron|role (the options only); alignment check (tokenizer only),
#      smoke (2 held-out bodies x 2 orders, r05r06_smoke_check.py), full run (75 held-out bodies x 2 orders), all three with the
#      same six conditions, fit_m, region message, 8 dims at every layer.
# Settings of the Chinese R05 readouts in r05r06_pro.sh (qsp single pass with enable_thinking false, qth thinking with kwargs {}):
# band 0-63, batch 16, 60 GiB, 3072 new tokens, offline; only --lang en and bodies_en_150_list.json change (the same file as the
# 11-02 English rerun). Order qsp_en, then qth_en. Starts after the queue wsw3 (wording_swap_pro_v3_20261109.sh) has ended, so
# this adds no second job of mine; each GPU step also waits until at most one other compute process is on the GPU and enough
# memory is free. Stop between steps: touch STOP_WQEN. Status: wording_swap_v2_status.log, lines tagged [qen]. tmux wqen.
set -u
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
E=/root/p2_scale_20261015/crossmodel_qwen38_en_20261102
S=crossmodel_role_swap_r05.py; W=wording_swap_v2_20261109.py; LOG=wording_swap_v2_status.log
MODEL=/root/hf_models/Qwen/Qwen3.8-27B; LG=en; BD=bodies_en_150_list.json; BAND=0-63; BT=16; G=60; NEED=64
cd $C || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
st() { echo "$(date '+%F %T') [qen] $1" >> $LOG; }
stop() { [ -e STOP_WQEN ] && { st "stopped by STOP_WQEN"; exit 0; }; }
slot() {  # NEED_GIB: at most one other compute process and NEED_GIB free on the GPU
  local n f w=0
  while n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); \
        [ "$n" -gt 1 ] || [ $((f / 1024)) -lt $1 ]; do
    stop; [ $w = 0 ] && st "waiting for a GPU slot ($n other jobs, $((f / 1024)) GiB free, need $1)"; w=1; sleep 60
  done
  st "GPU slot ($n other job(s), $((f / 1024)) GiB free)"
}
for pair in $S:01ab63dd crossmodel_suite.py:516f117e r05r06_smoke_check.py:10c2dc30 $W:bcff63a9; do
  f=${pair%%:*}; h=${pair##*:}; [ "$(sha256sum $f | cut -c1-8)" = $h ] || { echo "$f is not $h" >> $LOG; exit 1; }
done
cmp -s $BD $E/$BD || { st "$BD differs from the 11-02 English rerun: stop"; exit 1; }
st "start (queue $(sha256sum $0 | cut -c1-8)); waiting for wsw3 to end"
while tmux has-session -t wsw3 2>/dev/null; do stop; sleep 120; done
st "wsw3 ended"
CONDS="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
readout() {
  case $1 in
    qsp_en) MODE=single;   KW='{"enable_thinking": false}'; CMP=$E/singlepass_ablate/qwen38_27b_en/fit.pt; OUTD=singlepass_swap/wording_20261109 ;;
    qth_en) MODE=thinking; KW='{}';                          CMP=$E/thinking_ablate4/qwen38_27b_en/fit.pt;  OUTD=thinking_swap/wording_20261109 ;;
  esac
  FD=r05r06_fits/$1
}
run() {  # wording extra-args...
  local w=$1; shift
  if [ $w = paper ]; then set -- $S "$@"; else set -- $W --wording $w "$@"; fi
  $PY "$@" --mode $MODE --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B --band $BAND < /dev/null
}
mkdir -p singlepass_swap/wording_20261109 thinking_swap/wording_20261109
for R in qsp_en qth_en; do
  stop; readout $R
  if [ ! -s $FD/fit_seed4.pt ]; then
    slot $NEED
    st "$R fits start"
    $PY $S --make-fits $FD --compare-fit $CMP --mode $MODE --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B \
      --band $BAND --max-memory-gib $G < /dev/null > r05_fits_$R.log 2>&1
    st "$R fits exit $? | $(tail -1 r05_fits_$R.log | cut -c1-260)"
  fi
  [ -s $FD/fit_m.pt ] || { st "$R fits FAILED"; exit 1; }
  for w in paper pron role; do
    [ $w != paper ] && { CUDA_VISIBLE_DEVICES= run $w --show-render > $OUTD/${R}_${w}.render.txt 2>&1; st "$R $w render exit $? | $(grep -m1 '^wording' $OUTD/${R}_${w}.render.txt | cut -c1-200)"; }
    CUDA_VISIBLE_DEVICES= run $w --check-align --conds "$CONDS" > $OUTD/${R}_${w}.align.txt 2>&1
    st "$R $w align exit $? | $(grep '<-' $OUTD/${R}_${w}.align.txt | tr '\n' ' ' | cut -c1-300)"
  done
  for w in paper pron role; do
    O=$OUTD/${R}_${w}
    if [ -s $O/summary.json ]; then st "$R $w full already done: skip"; continue; fi
    stop; slot $NEED
    SM=$OUTD/${R}_${w}_smoke
    st "$R $w smoke start"
    run $w --fit-from $FD/fit_m.pt --conds "$CONDS" --region message --batch $BT --max-new 3072 --max-memory-gib $G --save-gen --out $SM --limit-bodies 2 > $SM.log 2>&1
    st "$R $w smoke exit $?"
    $PY r05r06_smoke_check.py $SM $MODE >> $LOG 2>&1 || { st "$R $w smoke FAILED: stop"; exit 1; }
    stop; slot $NEED
    mkdir -p $O; touch $O.GO
    st "$R $w full start"
    run $w --fit-from $FD/fit_m.pt --conds "$CONDS" --region message --batch $BT --max-new 3072 --max-memory-gib $G --save-gen --out $O > $O.log 2>&1
    st "$R $w full exit $? | $(tail -1 $O.log | cut -c1-400)"
    [ -s $O/summary.json ] || { st "$R $w full: no summary, stop"; exit 1; }
  done
done
st "qwen en wording queue done"
