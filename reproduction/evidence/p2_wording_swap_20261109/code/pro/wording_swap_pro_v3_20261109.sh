#!/bin/bash
# Resume of wording_swap_pro_v2_20261109.sh (a065ef09) after the 11-09 PRO pause (user: "pro 6000 暂时 先别占用", later
# "好像pro 6000 也空了"). v2 finished m26 (pron, role) and was stopped by STOP_WSW2 at 02:28 UTC while the qth pron smoke was
# loading weights (that process was ended by me, exit 143); the smoke left no items. This queue runs the remaining readouts
# qth, g20, g120 with exactly the v2 settings (same wrapper bcff63a9, same R05 scripts, same per-readout model, band, batch,
# memory, environment, fixed gpt-oss date 2026-10-30, same conditions, smoke check and full run). Differences from v2 only:
#   - m26 is not in the loop; a wording whose full run already has summary.json is skipped;
#   - the empty qth_pron_smoke directory of the stopped smoke is moved to $OUTD/_stopped_20261109/ (not deleted) first;
#   - stop flag STOP_WSW3 (STOP_WSW2 stays in place for the old queue); same status log wording_swap_v2_status.log.
# The English readouts (g20, g120) still wait for the user's OK on the English options (touch GO_WSW2_EN).
# Every GPU step waits until at most one other compute process is on the GPU and enough memory is free (PRO: 2 jobs at once).
# tmux wsw3. No signals.
set -u
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
W=wording_swap_v2_20261109.py; LOG=wording_swap_v2_status.log; OUTD=thinking_swap/wording_20261109
cd $C || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
QENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
GENV="HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1"
st() { echo "$(date '+%F %T') [v3] $1" >> $LOG; }
stop() { [ -e STOP_WSW3 ] && { st "stopped by STOP_WSW3"; exit 0; }; }
slot() {  # NEED_GIB: at most one other compute process and NEED_GIB free on the GPU
  local n f w=0
  while n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); \
        [ "$n" -gt 1 ] || [ $((f / 1024)) -lt $1 ]; do
    stop; [ $w = 0 ] && st "waiting for a GPU slot ($n other jobs, $((f / 1024)) GiB free, need $1)"; w=1; sleep 60
  done
  st "GPU slot ($n other job(s), $((f / 1024)) GiB free)"
}
for pair in crossmodel_role_swap_r05.py:01ab63dd crossmodel_suite.py:516f117e r05r06_smoke_check.py:10c2dc30 $W:bcff63a9; do
  f=${pair%%:*}; h=${pair##*:}; [ "$(sha256sum $f | cut -c1-8)" = $h ] || { st "$f is not the expected version ($h): stop"; exit 1; }
done
st "start (queue $(sha256sum $0 | cut -c1-8), wrapper $(sha256sum $W | cut -c1-8))"
S0=$OUTD/qth_pron_smoke
if [ -d $S0 ] && [ -z "$(ls -A $S0/items 2>/dev/null)" ] && [ ! -e $S0/summary.json ]; then
  mkdir -p $OUTD/_stopped_20261109 && mv $S0 $S0.log $OUTD/_stopped_20261109/ && st "moved the stopped qth pron smoke (no items) to $OUTD/_stopped_20261109/"
fi
CONDS="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
readout() {
  case $1 in
    qth)  MODEL=/root/hf_models/Qwen/Qwen3.8-27B; LG=zh; BD=bodies_zh_150_list.json; BAND=0-63; BT=16; G=60; PP=; ENVX=$QENV; DAY= ;;
    g20)  MODEL=/root/hf_models/openai/gpt-oss-20b; LG=en; BD=bodies_en_150_list.json; BAND=0-23; BT=4; G=30; PP=/root/p2_scale_20261015/tf_next; ENVX=$GENV; DAY=2026-10-30 ;;
    g120) MODEL=/root/hf_models/openai/gpt-oss-120b; LG=en; BD=bodies_en_150_list.json; BAND=0-35; BT=8; G=90; PP=/root/p2_scale_20261015/tf_next; ENVX=$GENV; DAY=2026-10-30 ;;
  esac
  FIT=r05r06_fits/$1/fit_m.pt; NEED=$((G + 4))
}
wrap() {  # wording extra-args...
  local w=$1; shift
  env $ENVX PYTHONPATH=$PP $PY $W --wording $w ${DAY:+--fixed-date $DAY} --mode thinking --model $MODEL --on-kwargs '{}' --lang $LG \
    --bodies $BD --splits $B --band $BAND --region message "$@" < /dev/null
}
mkdir -p $OUTD
for R in qth g20 g120; do
  stop; readout $R
  if [ $LG = en ] && [ ! -e GO_WSW2_EN ]; then  # the English options wait for the user's review (11-09: "我得先看题目设计")
    st "waiting for the user's OK on the English options (touch GO_WSW2_EN)"
    until [ -e GO_WSW2_EN ]; do stop; sleep 120; done
    st "GO_WSW2_EN present"
  fi
  if [ $R = g120 ]; then
    until [ -e dl_g120_20261109.ok ]; do stop; grep -q FAILED dl_g120_20261109.log 2>/dev/null && { st "g120 download FAILED: stop"; exit 1; }; sleep 120; done
  fi
  for w in pron role; do
    CUDA_VISIBLE_DEVICES= wrap $w --show-render > $OUTD/${R}_${w}.render.txt 2>&1
    st "$R $w render exit $? | $(grep -m1 '^wording' $OUTD/${R}_${w}.render.txt | cut -c1-200)"
    if [ -n "$DAY" ] && ! grep -q "$DAY" $OUTD/${R}_${w}.render.txt; then st "$R: date $DAY not in the rendering: stop"; exit 1; fi
    CUDA_VISIBLE_DEVICES= wrap $w --check-align --conds "$CONDS" > $OUTD/${R}_${w}.align.txt 2>&1
    st "$R $w align exit $? | $(grep '<-' $OUTD/${R}_${w}.align.txt | tr '\n' ' ' | cut -c1-300)"
  done
  for w in pron role; do
    O=$OUTD/${R}_${w}
    if [ -s $O/summary.json ]; then st "$R $w full already done: skip"; continue; fi
    stop; slot $NEED
    SM=$OUTD/${R}_${w}_smoke
    st "$R $w smoke start"
    wrap $w --fit-from $FIT --conds "$CONDS" --batch $BT --max-new 3072 --max-memory-gib $G --save-gen --out $SM --limit-bodies 2 > $SM.log 2>&1
    st "$R $w smoke exit $?"
    $PY r05r06_smoke_check.py $SM thinking >> $LOG 2>&1 || { st "$R $w smoke FAILED: stop"; exit 1; }
    stop; slot $NEED
    mkdir -p $O; touch $O.GO
    st "$R $w full start"
    wrap $w --fit-from $FIT --conds "$CONDS" --batch $BT --max-new 3072 --max-memory-gib $G --save-gen --out $O > $O.log 2>&1
    st "$R $w full exit $? | $(tail -1 $O.log | cut -c1-400)"
    [ -s $O/summary.json ] || { st "$R $w full: no summary, stop"; exit 1; }
  done
done
st "wording v3 queue done"
