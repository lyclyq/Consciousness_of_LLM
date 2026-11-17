#!/bin/bash
# MiMo-V2.6-Distill-Qwen-9B mechanism readout, the rest of the mechanism-group battery (user 11-08: "补上呗 既然效应这么好").
# Starts after mech_mimo26_pro_20261108.sh has ended ("mech m26 done"). Same model, mode (thinking, default call '{}'), language
# (zh), fits (r05r06_fits/m26), band (L0-L31), batch 16, 3072 new tokens and 44 GiB as there.
#   1. R05 variants, as r05r06_pro.sh: r05q_msg (fit_q, message region), r05m_msg (fit_m, message), r05m_all (fit_m, all positions):
#      swU8_A, swA8_U
#   2. R06 random seeds 2..5: fit_seed1..4, all positions: rswU8_A, rswA8_U
#   3. k64 diagnostic, as gptoss_k64_diag_pro.sh (attribution question only): fit_q, all positions, the leading 64 dimensions:
#      swU64_A, swA64_U, rswU64_A, rswA64_U
#   4. secret game v5, as gptoss_games5_pro.sh / add_games5_s3.py: crossmodel_games.py, games/games_zh_secret5.json (a712030b, the
#      file of the paper's Chinese runs), eight dimensions, fit_q.
#      - smoke: 8 games, main question and control.
#      - The release rule is the one fixed in advance (10-26). Unedited, the main question must separate the sources: the
#        setter's role >= 0.5 with the model's own X and <= 0.25 with the user's X, on finished games. At least 75% must be
#        finished, and the control must be answered correctly in at least 7 of 8 games under both unedited conditions.
#      - Only if it passes: full main (100 games, 6 conditions), then control.
# The unedited items of thinking_swap/mimo_v26_r05q_all are copied into steps 1-3 (identical prompts; the runner skips items present).
# Every GPU step takes the exclusive flock on persistence/gpu.lock. Stop: touch modeswitch/STOP_MS; no signals.
# usage: bash mech_mimo26_ext_pro_20261108.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=crossmodel_role_swap_r05.py; LOCK=$C/persistence/gpu.lock; STOP=$C/modeswitch/STOP_MS
MODEL=/root/hf_models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B; KW='{}'; LG=zh; BD=bodies_zh_150_list.json; BAND=0-31; BT=16; G=44
FD=r05r06_fits/m26; DIR=thinking_swap/mimo_v26; W=$C/modeswitch/mech_m26; Q=${DIR}_r05q_all
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
st() { echo "[mech m26 ext] $(date '+%F %T') $*" | tee -a $W/status.log >> $C/modeswitch/queue.log; }
stop() { [ -e $STOP ] && { st "stop flag"; exit 2; }; }
gpu() { ( flock -x 9; CUDA_VISIBLE_DEVICES=0 "$@" ) 9>$LOCK; }
run_swap() {  # OUT FIT REGION CONDS
  gpu $PY $S --mode thinking --model $MODEL --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B --fit-from $2 --band $BAND \
    --conds "$4" --region $3 --question attr --batch $BT --max-new 3072 --max-memory-gib $G --save-gen --out $1 \
    > $W/$(basename $1).log 2>&1 < /dev/null
}
until grep -q 'mech m26 done' $W/status.log; do stop; sleep 60; done
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
  run_swap $O $FD/$fit.pt $reg "base_A:assistant:-:-:0,base_U:user:-:-:0,$conds"
  st "$name exit $? | $(ls $O/items | wc -l) items"
done
C8="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
GM=(crossmodel_games.py --mode thinking --model $MODEL --on-kwargs "$KW" --items games/games_zh_secret5.json --games secret --lang $LG
    --fit-from $FD/fit_q.pt --band $BAND --conds $C8 --batch $BT --max-new 3072 --max-memory-gib $G --save-gen)
mkdir -p games_swap
stop
st "game smoke start (runner $(sha256sum crossmodel_games.py | cut -c1-8), items $(sha256sum games/games_zh_secret5.json | cut -c1-8))"
for q in main control; do
  gpu $PY "${GM[@]}" --qkind $q --limit-bodies 8 --out games_swap/mimo_v26_s5_smoke_$q > $W/game_smoke_$q.log 2>&1 < /dev/null
done
verdict=$($PY - <<'EOF'
import json
m = json.load(open('games_swap/mimo_v26_s5_smoke_main/summary.json'))['secret']
c = json.load(open('games_swap/mimo_v26_s5_smoke_control/summary.json'))['secret']
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
    touch games_swap/mimo_v26_s5_$q.GO
    st "game full $q start"
    gpu $PY "${GM[@]}" --qkind $q --out games_swap/mimo_v26_s5_$q > $W/game_$q.log 2>&1 < /dev/null
    st "game full $q exit $?"
  done
else
  st "game not released (rule fixed in advance); left for inspection"
fi
st "mech m26 ext done"
