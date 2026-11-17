#!/bin/bash
# MiMo-V2.6-Distill-Qwen-9B as a further mechanism readout (user 11-08). It is the model that claims its own messages only in the
# default call of its template (0.35, against 0.05 and 0.12; the call answers without written reasoning). Exploratory: if the
# replacement has no effect, the user takes it as a too marginal self-reference.
# The steps follow r05r06_pro.sh for each readout of the mechanism group, with crossmodel_role_swap_r05.py (01ab63dd):
#   1. fits on the 75 discovery bodies, both option orders (fit_q, fit_m, fit_seed1..4) -> r05r06_fits/m26
#   2. smoke: 2 held-out bodies, message region, fit_m (base_A, base_U, swU8_A, swA8_U, rswU8_A); r05r06_smoke_check.py
#   3. the paper's procedure, fit_q, all positions, 75 held-out bodies x 2 orders:
#      base_A, base_U, swU8_A (removal), swA8_U (installation), rswU8_A, rswA8_U (random 8-dim)
#   4. the paper's non-self controls, same subspace and edits: content question and two-digit arithmetic
#      (base_A, base_U, swU8_A, swA8_U; one option order)
#   5. KL replacement: mode_switch_swap_20261108.py with fit_m, as for the other four readouts
# Mode: --mode thinking with the default call '{}' (greedy generation, answer after </think>), as the paper's MiMo-V2.6 entry.
# The two modes of this template render identical tokens up to the end of the body, so the body-token KL swap is the same in both.
# Every GPU step takes the exclusive flock on persistence/gpu.lock. Stop: touch modeswitch/STOP_MS; no signals.
# usage: bash mech_mimo26_pro_20261108.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=crossmodel_role_swap_r05.py; LOCK=$C/persistence/gpu.lock; STOP=$C/modeswitch/STOP_MS
MODEL=/root/hf_models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B; KW='{}'; LG=zh; BD=bodies_zh_150_list.json; BAND=0-31; BT=16; G=44
FD=r05r06_fits/m26; DIR=thinking_swap/mimo_v26; W=$C/modeswitch/mech_m26
mkdir -p $W thinking_swap
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
st() { echo "[mech m26] $(date '+%F %T') $*" | tee -a $W/status.log >> $C/modeswitch/queue.log; }
stop() { [ -e $STOP ] && { st "stop flag"; exit 2; }; }
gpu() { ( flock -x 9; CUDA_VISIBLE_DEVICES=0 "$@" ) 9>$LOCK; }
run_swap() {  # OUT FIT REGION CONDS QUESTION [LIMIT]
  gpu $PY $S --mode thinking --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B --fit-from $2 --band $BAND \
    --conds "$4" --region $3 --question $5 --batch $BT --max-new 3072 --max-memory-gib $G --save-gen --out $1 ${6:+--limit-bodies $6} \
    > $W/$(basename $1).log 2>&1 < /dev/null
}

stop
if [ ! -s $FD/fit_seed4.pt ]; then
  st "fits start"
  gpu $PY $S --make-fits $FD --mode thinking --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B --band $BAND \
    --max-memory-gib $G > $W/fits.log 2>&1 < /dev/null
  st "fits exit $? | $(tail -1 $W/fits.log | cut -c1-200)"
fi
[ -s $FD/fit_seed4.pt ] || { st "fits FAILED"; exit 1; }
stop
SM=${DIR}_r05_smoke
if [ ! -e $SM.PASS ]; then
  st "smoke start"
  run_swap $SM $FD/fit_m.pt message "base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8" attr 2
  $PY r05r06_smoke_check.py $SM thinking >> $W/status.log 2>&1 && touch $SM.PASS || { st "smoke FAILED"; exit 1; }
  st "smoke PASS"
fi
stop
Q=${DIR}_r05q_all
st "r05q_all start"; touch $Q.GO
run_swap $Q $FD/fit_q.pt all "base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8" attr
st "r05q_all exit $? | $(ls $Q/items 2>/dev/null | wc -l) items"
for qn in content arith; do
  stop
  O=${DIR}_${qn}_all
  st "$qn control start"; touch $O.GO
  run_swap $O $FD/fit_q.pt all "base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8" $qn
  st "$qn control exit $? | $(ls $O/items 2>/dev/null | wc -l) items"
done
stop
st "KL swap start"
gpu $PY mode_switch_swap_20261108.py --model $MODEL --tmpl-kwargs "$KW" --bodies $B --fit $FD/fit_m.pt --out $C/modeswitch/swap/m26 \
  --max-memory-gib $G --stop-file $STOP > $C/modeswitch/swap/m26.log 2>&1 < /dev/null
if [ -e $C/modeswitch/swap/m26/DONE ]; then st "KL swap DONE: $(grep '^summary:' $C/modeswitch/swap/m26.log | tail -1 | cut -c1-500)"; else st "KL swap FAILED, see modeswitch/swap/m26.log"; fi
st "mech m26 done"
