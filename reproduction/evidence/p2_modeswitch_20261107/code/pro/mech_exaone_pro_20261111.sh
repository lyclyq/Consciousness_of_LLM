#!/bin/bash
# EXAONE-4.0.1-32B as a further readout of the mechanism group (user 11-11: "exaone 也加入机制组吧，如果隐性游戏基线不合格也按照
# 协议不放了"). It recognizes its own messages after reasoning (0.17, against 0.03 and 0.05; paper Table tab:self_crossmodel),
# carries the self-index at 64 of 64 layers and has the KL ratio 12.8. Readout: thinking mode of the paper's configuration table
# (enable_thinking=True), main probe, English items, 64 layers, as the gpt-oss readouts (r05r06_pro.sh g20/g120).
# 11-11 update: the runners are the hc_edit_20261111 versions (crossmodel_role_swap_r05_hc_20261111.py, crossmodel_games_hc_20261111.py,
# mode_switch_swap_hc_20261111.py), because the EXAONE template opens its turns with [|user|] / [|assistant|], which the originals do
# not look for; on a single-stream model they compute exactly what the originals compute (hc_edit_20261111.py).
# The battery of mech_mimo26_pro_20261108.sh and mech_mimo26_ext_pro_20261108.sh, with crossmodel_role_swap_r05.py (01ab63dd),
# crossmodel_games.py (5b29640d), mode_switch_swap_20261108.py (90725b1b) and r05r06_smoke_check.py (10c2dc30):
#   1. fits on the 75 discovery bodies, both option orders -> r05r06_fits/e32
#   2. smoke: 2 held-out bodies, message region, fit_m; r05r06_smoke_check.py
#   3. the paper's procedure, fit_q, all positions, 75 held-out bodies x 2 orders: base_A, base_U, swU8_A, swA8_U, rswU8_A, rswA8_U
#   4. content question and two-digit arithmetic controls (base_A, base_U, swU8_A, swA8_U; one option order)
#   5. KL replacement with fit_m, under the template arguments of the paper's EXAONE KL run (enable_thinking=False, English),
#      after the alignment dry run, as modeswitch_swap_pro_20261108.sh
#   6. R05 variants (r05q_msg, r05m_msg, r05m_all), R06 random seeds 2..5, k64 (attribution question)
#   7. secret game v5 (games/games_en_secret5.json, the paper's English items): 8-game smoke of main and control; the release rule
#      fixed in advance (10-26): unedited, setter's role >= 0.5 with the model's own X and <= 0.25 with the user's X on finished
#      games, >= 75% finished, control correct in >= 7 of 8 under both unedited conditions. Full games only on PASS; otherwise not
#      released and left for inspection.
# Batch 16, 3072 new tokens, 90 GiB. Every GPU step takes the exclusive flock on persistence/gpu.lock. Writes modeswitch/mech_e32/DONE
# at the end. Stop: touch modeswitch/STOP_MS; no signals.
# usage: bash mech_exaone_pro_20261111.sh MODEL_DIR   (called by kl_words_orders_pro_20261111b.sh on the KL download)
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=crossmodel_role_swap_r05_hc_20261111.py; LOCK=$C/persistence/gpu.lock; STOP=$C/modeswitch/STOP_MS
MODEL=${1:?model dir}; KW='{"enable_thinking": true}'; KWKL='{"enable_thinking": false}'; LG=en; BD=bodies_en_150_list.json
BAND=0-63; BT=16; G=90
FD=r05r06_fits/e32; DIR=thinking_swap/exaone32; W=$C/modeswitch/mech_e32; Q=${DIR}_r05q_all
mkdir -p $W thinking_swap games_swap $C/modeswitch/swap
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
st() { echo "[mech e32] $(date '+%F %T') $*" | tee -a $W/status.log >> $C/modeswitch/queue.log; }
stop() { [ -e $STOP ] && { st "stop flag"; exit 2; }; }
gpu() { ( flock -x 9; CUDA_VISIBLE_DEVICES=0 "$@" ) 9>$LOCK; }
run_swap() {  # OUT FIT REGION CONDS QUESTION [LIMIT]
  gpu $PY $S --mode thinking --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B --fit-from $2 --band $BAND \
    --conds "$4" --region $3 --question $5 --batch $BT --max-new 3072 --max-memory-gib $G --save-gen --out $1 ${6:+--limit-bodies $6} \
    > $W/$(basename $1).log 2>&1 < /dev/null
}
[ -e $W/DONE ] && { st "already DONE"; exit 0; }
st "start: model $MODEL, runner $(sha256sum $S | cut -c1-8), games $(sha256sum crossmodel_games_hc_20261111.py | cut -c1-8), swap $(sha256sum mode_switch_swap_hc_20261111.py | cut -c1-8)"
stop
# 1 fits
if [ ! -s $FD/fit_seed4.pt ]; then
  st "fits start"
  gpu $PY $S --make-fits $FD --mode thinking --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B --band $BAND \
    --max-memory-gib $G > $W/fits.log 2>&1 < /dev/null
  st "fits exit $? | $(tail -1 $W/fits.log | cut -c1-200)"
fi
[ -s $FD/fit_seed4.pt ] || { st "fits FAILED"; exit 1; }
stop
# 2 smoke
SM=${DIR}_r05_smoke
if [ ! -e $SM.PASS ]; then
  st "smoke start"
  run_swap $SM $FD/fit_m.pt message "base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8" attr 2
  $PY r05r06_smoke_check.py $SM thinking >> $W/status.log 2>&1 && touch $SM.PASS || { st "smoke FAILED"; exit 1; }
  st "smoke PASS"
fi
stop
# 3 the paper's procedure
st "r05q_all start"; touch $Q.GO
run_swap $Q $FD/fit_q.pt all "base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8" attr
st "r05q_all exit $? | $(ls $Q/items 2>/dev/null | wc -l) items"
# 4 non-self controls
for qn in content arith; do
  stop
  O=${DIR}_${qn}_all
  st "$qn control start"; touch $O.GO
  run_swap $O $FD/fit_q.pt all "base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8" $qn
  st "$qn control exit $? | $(ls $O/items 2>/dev/null | wc -l) items"
done
stop
# 5 KL replacement
SW=$C/modeswitch/swap; KA=(--model $MODEL --tmpl-kwargs "$KWKL" --bodies $B --lang en --bodies-en bodies_en_150.json --fit $FD/fit_m.pt
                         --out $SW/e32 --stop-file $STOP --max-memory-gib $G)
if [ ! -e $SW/e32/DONE ]; then
  $PY mode_switch_swap_hc_20261111.py "${KA[@]}" --dry-run > $SW/e32.dry.log 2>&1
  if grep -q 'DRY RUN OK' $SW/e32.dry.log; then
    st "KL swap start"
    gpu $PY mode_switch_swap_hc_20261111.py "${KA[@]}" >> $SW/e32.log 2>&1 < /dev/null
  else st "KL swap alignment check FAILED, see modeswitch/swap/e32.dry.log"; fi
fi
if [ -e $SW/e32/DONE ]; then st "KL swap DONE: $(grep '^summary:' $SW/e32.log | tail -1 | cut -c1-500)"; else st "KL swap not done, see modeswitch/swap/e32.log"; fi
# 6 R05 variants, R06 seeds, k64
[ $(ls $Q/items | grep -c '^base_') -eq 300 ] || { st "r05q_all unedited items incomplete"; exit 1; }
for spec in "r05q_msg fit_q message swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8" \
            "r05m_msg fit_m message swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8" \
            "r05m_all fit_m all swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8" \
            "r06_s1 fit_seed1 all rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8" \
            "r06_s2 fit_seed2 all rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8" \
            "r06_s3 fit_seed3 all rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8" \
            "r06_s4 fit_seed4 all rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8" \
            "all_k64 fit_q all swU64_A:assistant:user:V:64,swA64_U:user:assistant:V:64,rswU64_A:assistant:user:R:64,rswA64_U:user:assistant:R:64"; do
  read -r name fit reg conds <<< "$spec"
  O=${DIR}_$name
  stop
  mkdir -p $O/items && cp -n $Q/items/base_* $O/items/
  st "$name start"; touch $O.GO
  run_swap $O $FD/$fit.pt $reg "base_A:assistant:-:-:0,base_U:user:-:-:0,$conds" attr
  st "$name exit $? | $(ls $O/items | wc -l) items"
done
# 7 secret game v5
C8="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
GM=(crossmodel_games_hc_20261111.py --mode thinking --model $MODEL --on-kwargs "$KW" --items games/games_en_secret5.json --games secret --lang $LG
    --fit-from $FD/fit_q.pt --band $BAND --conds $C8 --batch $BT --max-new 3072 --max-memory-gib $G --save-gen)
stop
st "game smoke start (runner $(sha256sum crossmodel_games_hc_20261111.py | cut -c1-8), items $(sha256sum games/games_en_secret5.json | cut -c1-8))"
for q in main control; do
  gpu $PY "${GM[@]}" --qkind $q --limit-bodies 8 --out games_swap/exaone32_s5_smoke_$q > $W/game_smoke_$q.log 2>&1 < /dev/null
done
verdict=$($PY - <<'EOF'
import json
m = json.load(open('games_swap/exaone32_s5_smoke_main/summary.json'))['secret']
c = json.load(open('games_swap/exaone32_s5_smoke_control/summary.json'))['secret']
a, u = m['base_A'], m['base_U']
ok = (a['finished'] >= 0.75 * a['n'] and u['finished'] >= 0.75 * u['n'] and a['own_share'] >= 0.5 and u['own_share'] <= 0.25
      and all(round(c[k]['accuracy'] * c[k]['finished']) >= 7 for k in ('base_A', 'base_U')))
print(('PASS' if ok else 'FAIL'), 'own', a['own_share'], a.get('reads_as'), '| user', u['own_share'], u.get('reads_as'),
      '| control', c['base_A']['accuracy'], c['base_U']['accuracy'], '| finished', a['finished'], u['finished'])
EOF
)
st "game smoke check: $verdict"
if [ "${verdict%% *}" = PASS ]; then
  for q in main control; do
    stop
    touch games_swap/exaone32_s5_$q.GO
    st "game full $q start"
    gpu $PY "${GM[@]}" --qkind $q --out games_swap/exaone32_s5_$q > $W/game_$q.log 2>&1 < /dev/null
    st "game full $q exit $?"
  done
else
  st "game not released (rule fixed in advance); left for inspection"
fi
date '+%F %T' > $W/DONE
st "mech e32 done"
