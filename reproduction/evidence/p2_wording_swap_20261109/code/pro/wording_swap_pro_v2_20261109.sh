#!/bin/bash
# Wording variant of R05 for the other four readouts of the mechanism group on PRO (user 11-09: "主要是对归属题做这个消融就足够了 ...
# 4 模型 5 模式 ... 补齐"; qsp done with wording_swap_pro_20261109.sh). Per readout, the settings of its anchor <dir>_r05m_msg:
# r05r06_pro.sh readout() for qth, g20, g120 (10-30) and mech_mimo26_pro_20261108.sh for m26 (11-08): fit_m, region message,
# layers, batch, memory, environment; gpt-oss with tf_next (transformers 5.19.0) and the template date fixed to 2026-10-30, the day
# the gpt-oss anchors ran (UTC) without a fixed date. Script wording_swap_v2_20261109.py (only the answer options change).
# Per readout: render dump and alignment check (tokenizer only); then per wording (pron, role) a smoke (2 held-out bodies x 2 orders,
# r05r06_smoke_check.py, execution only) and the full run (75 held-out bodies x 2 orders, the six conditions of the qsp wording runs).
# Order m26, qth, g20, g120 (g120 after its re-download, dl_g120_20261109.ok). Every GPU step waits until at most one other compute
# process is on the GPU and enough memory is free (PRO: 2 jobs at once). Stop between steps: touch STOP_WSW2. Status:
# wording_swap_v2_status.log. tmux wsw2. No signals.
set -u
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
W=wording_swap_v2_20261109.py; LOG=wording_swap_v2_status.log; OUTD=thinking_swap/wording_20261109
cd $C || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
QENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
GENV="HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1"
st() { echo "$(date '+%F %T') $1" >> $LOG; }
stop() { [ -e STOP_WSW2 ] && { st "stopped by STOP_WSW2"; exit 0; }; }
slot() {  # NEED_GIB: at most one other compute process and NEED_GIB free on the GPU
  local n f w=0
  while n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); \
        [ "$n" -gt 1 ] || [ $((f / 1024)) -lt $1 ]; do
    stop; [ $w = 0 ] && st "waiting for a GPU slot ($n other jobs, $((f / 1024)) GiB free, need $1)"; w=1; sleep 60
  done
  st "GPU slot ($n other job(s), $((f / 1024)) GiB free)"
}
for pair in crossmodel_role_swap_r05.py:01ab63dd crossmodel_suite.py:516f117e r05r06_smoke_check.py:10c2dc30; do
  f=${pair%%:*}; h=${pair##*:}; [ "$(sha256sum $f | cut -c1-8)" = $h ] || { st "$f is not the R05 version ($h): stop"; exit 1; }
done
st "start (wrapper $(sha256sum $W | cut -c1-8))"
CONDS="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
readout() {
  case $1 in
    m26)  MODEL=/root/hf_models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B; LG=zh; BD=bodies_zh_150_list.json; BAND=0-31; BT=16; G=44; PP=; ENVX=$QENV; DAY= ;;
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
for R in m26 qth g20 g120; do
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
    stop; slot $NEED
    SM=$OUTD/${R}_${w}_smoke
    st "$R $w smoke start"
    wrap $w --fit-from $FIT --conds "$CONDS" --batch $BT --max-new 3072 --max-memory-gib $G --save-gen --out $SM --limit-bodies 2 > $SM.log 2>&1
    st "$R $w smoke exit $?"
    $PY r05r06_smoke_check.py $SM thinking >> $LOG 2>&1 || { st "$R $w smoke FAILED: stop"; exit 1; }
    stop; slot $NEED
    O=$OUTD/${R}_${w}; mkdir -p $O; touch $O.GO
    st "$R $w full start"
    wrap $w --fit-from $FIT --conds "$CONDS" --batch $BT --max-new 3072 --max-memory-gib $G --save-gen --out $O > $O.log 2>&1
    st "$R $w full exit $? | $(tail -1 $O.log | cut -c1-400)"
    [ -s $O/summary.json ] || { st "$R $w full: no summary, stop"; exit 1; }
  done
done
st "wording v2 queue done"
