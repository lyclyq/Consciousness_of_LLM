#!/bin/bash
# Wording variant of R05 for Qwen3.8-27B without thinking on PRO (user 11-09). Settings of the qsp r05m_msg run of r05r06_pro.sh
# (the anchor, 10-30): PRO venv, /root/hf_models/Qwen/Qwen3.8-27B, enable_thinking false, Chinese bodies, the same splits,
# r05r06_fits/qsp/fit_m.pt, layers 0-63, region message, batch 16, 60 GiB. Only the answer options change (wording_swap_20261109.py).
# Per wording (pron, then role):
#   1. smoke: 2 held-out bodies x 2 orders, rule r05r06_smoke_check.py (execution only: items present, letter mass >= 0.9, edits
#      recorded); the full run starts only after it passes
#   2. full: 75 held-out bodies x 2 orders, base_A, base_U, removal swU8_A, installation swA8_U, random rswU8_A, rswA8_U
# One job at a time. Stop between steps: touch STOP_WSW. Status: wording_swap_status.log. tmux wsw. No signals.
set -u
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
W=wording_swap_20261109.py; LOG=wording_swap_status.log; OUTD=singlepass_swap/wording_20261109
cd $C || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
st() { echo "$(date '+%F %T') $1" >> $LOG; }
stop() { [ -e STOP_WSW ] && { st "stopped by STOP_WSW"; exit 0; }; }
for pair in crossmodel_role_swap_r05.py:01ab63dd crossmodel_suite.py:516f117e r05r06_smoke_check.py:10c2dc30; do
  f=${pair%%:*}; h=${pair##*:}; [ "$(sha256sum $f | cut -c1-8)" = $h ] || { st "$f is not the R05 version ($h): stop"; exit 1; }
done
st "start (wrapper $(sha256sum $W | cut -c1-8), r05 01ab63dd, suite 516f117e, smoke check 10c2dc30)"
CONDS="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
run() {  # wording out [limit]
  $PY $W --wording $1 --mode single --model /root/hf_models/Qwen/Qwen3.8-27B --on-kwargs '{"enable_thinking": false}' --lang zh \
    --bodies bodies_zh_150_list.json --splits $B --fit-from r05r06_fits/qsp/fit_m.pt --band 0-63 --conds "$CONDS" --region message \
    --batch 16 --max-new 3072 --max-memory-gib 60 --save-gen --out $2 ${3:+--limit-bodies $3} > $2.log 2>&1 < /dev/null
}
mkdir -p $OUTD
for w in pron role; do
  stop
  SM=$OUTD/${w}_smoke
  st "$w smoke start"; run $w $SM 2; st "$w smoke exit $?"
  $PY r05r06_smoke_check.py $SM single >> $LOG 2>&1 || { st "$w smoke FAILED"; exit 1; }
  stop
  O=$OUTD/$w; mkdir -p $O; touch $O.GO
  st "$w full start"; run $w $O; st "$w full exit $? | $(tail -1 $O.log | cut -c1-400)"
  [ -s $O/summary.json ] || { st "$w full: no summary"; exit 1; }
done
st "wording queue done"
