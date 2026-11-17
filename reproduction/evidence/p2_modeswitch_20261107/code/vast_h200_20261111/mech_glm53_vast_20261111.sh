#!/bin/bash
# GLM-5.3-Flash as a readout of the mechanism group, core battery (11-11, user: "咬牙测 ... glm5.3 flash 机制协议", "如果隐性游戏不合格
# 则不跑对应部分"; edits per stream; baseline rule fixed in advance). Part 2 on the rented machine, after part 1 (run.log "part 1 done").
# Thinking mode (the template has only this mode; default call '{}'), main probe, Chinese items, 45 layers x 4 streams. Scripts:
# crossmodel_role_swap_r05_hc_20261111.py, crossmodel_games_hc_20261111.py, mode_switch_swap_hc_20261111.py (the mechanism-group
# runners with hc_edit_20261111: turn-start tokens of the GLM template, one subspace per stream per layer), r05r06_smoke_check.py.
#   1. fits on the 75 discovery bodies, both option orders, per stream -> out/mech/fits
#   2. identity check, 1 held-out body x 2 orders, batch 1, 512 new tokens: own message with its own coordinates (idA8_A) must give
#      the same text as base_A; checks the per-stream write path on the model itself
#   3. smoke: 2 held-out bodies, message region, fit_m (base_A, base_U, swU8_A, swA8_U, rswU8_A); r05r06_smoke_check.py
#   4. baseline gate (rule fixed 11-11 before any GLM-5.3 run): unedited base_A and base_U on 75 held-out bodies x 2 orders; continue
#      only if each has >= 75% finished and the share of own messages answered "you" exceeds the User share by >= 0.20; otherwise the
#      edits are not tested ("unedited baseline does not separate the sources")
#   5. the paper's procedure, fit_q, all positions: swU8_A, swA8_U, rswU8_A, rswA8_U (the unedited items of step 4 are reused)
#   6. content question and two-digit arithmetic controls (base_A, base_U, swU8_A, swA8_U; one option order)
#   7. KL replacement with fit_m, template arguments of the paper's GLM-5.3 KL run ('{}', Chinese), after the alignment dry run
#   8. secret game v5 (games_zh_secret5.json): 8-game smoke of main and control, the release rule of 10-26; full only on PASS
# 11-11 order change (user: "怎么节省时间尽快开始"): the mechanism runs first, as soon as the weights and the environment are ready;
# part 1 (KL, retention) waits for out/mech/DONE. Batch 16 as the Qwen and MiMo readouts (KV estimate ~40 GB over three H200).
# 11-11 v4: GLM-5.3 is re-run as a whole in this environment (user: "连同几何一块重跑"; the H200 and 4090 states differ beyond the
# protocol's tolerance, MoE routing); the battery waits for this environment's KL ctrl run; no secret game (user, cost).
# Batch BT (default 8), 3072 new tokens, MEM GiB per GPU and OFF GiB CPU as part 1. Writes out/mech/DONE. Stop: touch /root/glm53/STOP.
# usage (tmux glm53, window mech): bash mech_glm53_vast_20261111.sh
set -u
G=/root/glm53; cd $G/up_code || exit 1
PY=/root/venv_t26/bin/python; M=/dev/shm/hf_models/zai-org/GLM-5.3-Flash; R=$G/ref; O=$G/out/mech; STOP=$G/STOP
S=crossmodel_role_swap_r05_hc_20261111.py; GS=crossmodel_games_hc_20261111.py; SW=mode_switch_swap_hc_20261111.py
KW='{}'; LG=zh; BD=$R/bodies_zh_150_list.json; B=$R/role_bodies.jsonl; BAND=0-44
BT=${BT:-8}; MEM=${MEM:-72}; OFF=${OFF:-40}
FD=$O/fits; DIR=$O/glm53; Q=${DIR}_r05q_all
mkdir -p $O/logs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 P2_FP8_FORCE_FALLBACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3
st() { echo "[mech] $(date '+%F %T') $*" | tee -a $G/run.log; }
stop() { [ -e $STOP ] && { st "stop flag"; exit 2; }; }
run_swap() {  # OUT FIT REGION CONDS QUESTION [LIMIT] [BATCH] [MAXNEW]
  $PY $S --mode thinking --model $M --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B --fit-from $2 --band $BAND \
    --conds "$4" --region $3 --question $5 --batch ${7:-$BT} --max-new ${8:-3072} --max-memory-gib $MEM --cpu-offload-gib $OFF \
    --save-gen --out $1 ${6:+--limit-bodies $6} >> $O/logs/$(basename $1).log 2>&1 < /dev/null
}
[ -e $O/DONE ] && { st "already DONE"; exit 0; }
until [ -e $G/out/paper_rerun/modeswitch/ctrl/glm53_flash/DONE ]; do stop; sleep 30; done
st "start (runner $(sha256sum $S | cut -c1-8), games $(sha256sum $GS | cut -c1-8), swap $(sha256sum $SW | cut -c1-8), hc_edit $(sha256sum hc_edit_20261111.py | cut -c1-8), BT $BT)"
# 1 fits
if [ ! -s $FD/fit_seed4.pt ]; then
  st "fits start"
  $PY $S --make-fits $FD --mode thinking --model $M --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B --band $BAND \
    --max-memory-gib $MEM --cpu-offload-gib $OFF > $O/logs/fits.log 2>&1 < /dev/null
  st "fits exit $? | $(tail -1 $O/logs/fits.log | cut -c1-300)"
fi
[ -s $FD/fit_seed4.pt ] || { st "fits FAILED"; exit 1; }
stop
# 2 identity check
ID=${DIR}_idcheck
if [ ! -e $ID.PASS ]; then
  run_swap $ID $FD/fit_m.pt message "base_A:assistant:-:-:0,idA8_A:assistant:assistant:V:8" attr 1 1 512
  v=$($PY - "$ID" <<'EOF'
import json, sys
from pathlib import Path
it = {p.name: json.loads(p.read_text(encoding='utf-8')) for p in (Path(sys.argv[1]) / 'items').glob('*.json')}
pairs = [(it[k], it.get(k.replace('base_A__', 'idA8_A__'))) for k in it if k.startswith('base_A__')]
ok = len(pairs) == 2 and all(b is not None and a['gen'] == b['gen'] for a, b in pairs) and all(b['dose']['n_pos'] > 0 for _, b in pairs)
print(('PASS' if ok else 'FAIL'), [(a['n_new'], b['n_new'] if b else None, (a['gen'] == b['gen']) if b else None) for a, b in pairs])
EOF
)
  st "identity check: $v"
  [ "${v%% *}" = PASS ] && touch $ID.PASS || { st "identity check FAILED: the per-stream write path changes an unedited state"; exit 1; }
fi
stop
# 3 smoke
SM=${DIR}_r05_smoke
if [ ! -e $SM.PASS ]; then
  st "smoke start"
  run_swap $SM $FD/fit_m.pt message "base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8" attr 2
  $PY r05r06_smoke_check.py $SM thinking >> $G/run.log 2>&1 && touch $SM.PASS || { st "smoke FAILED"; exit 1; }
  st "smoke PASS"
fi
stop
# 4 baseline gate
if [ ! -e $Q.BASELINE_PASS ]; then
  st "baseline start"; touch $Q.GO
  run_swap $Q $FD/fit_q.pt all "base_A:assistant:-:-:0,base_U:user:-:-:0" attr
  nb=$(ls $Q/items 2>/dev/null | grep -c '^base_')
  [ "$nb" = 300 ] || { st "baseline incomplete ($nb of 300 items): a run error, not the gate; see logs/$(basename $Q).log"; exit 1; }
  v=$($PY - "$Q" <<'EOF'
import json, sys
from pathlib import Path
s = json.loads((Path(sys.argv[1]) / 'summary.json').read_text(encoding='utf-8'))
a, u = s['base_A'], s['base_U']
ok = a['finished'] >= 0.75 * a['n'] and u['finished'] >= 0.75 * u['n'] and a['you_share'] - u['you_share'] >= 0.20
print(('PASS' if ok else 'FAIL'), 'own', a['you_share'], f"{a['finished']}/{a['n']}", '| user', u['you_share'], f"{u['finished']}/{u['n']}",
      '| difference', round(a['you_share'] - u['you_share'], 4))
EOF
)
  st "baseline gate: $v"
  if [ "${v%% *}" = PASS ]; then touch $Q.BASELINE_PASS
  else touch $Q.NOT_TESTED; st "edits not tested (rule fixed in advance); mech stops here"; date '+%F %T' > $O/DONE; exit 0; fi
fi
stop
# 5 the paper's procedure
st "r05q_all edits start"
run_swap $Q $FD/fit_q.pt all "base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8" attr
st "r05q_all exit $? | $(ls $Q/items | wc -l) items | $(tail -c 600 $O/logs/$(basename $Q).log | tr '\n' ' ' | cut -c1-400)"
# 6 controls
for qn in content arith; do
  stop
  C6=${DIR}_${qn}_all
  st "$qn control start"; touch $C6.GO
  run_swap $C6 $FD/fit_q.pt all "base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8" $qn
  st "$qn control exit $? | $(ls $C6/items 2>/dev/null | wc -l) items"
done
stop
# 7 KL replacement
KA=(--model $M --tmpl-kwargs "$KW" --bodies $B --lang zh --fit $FD/fit_m.pt --out $O/swap_glm53 --stop-file $STOP --max-memory-gib $MEM
    --cpu-offload-gib $OFF)
if [ ! -e $O/swap_glm53/DONE ]; then
  $PY $SW "${KA[@]}" --dry-run > $O/logs/swap.dry.log 2>&1
  if grep -q 'DRY RUN OK' $O/logs/swap.dry.log; then st "KL swap start"; $PY $SW "${KA[@]}" >> $O/logs/swap.log 2>&1 < /dev/null
  else st "KL swap alignment check FAILED, see logs/swap.dry.log"; fi
fi
if [ -e $O/swap_glm53/DONE ]; then st "KL swap DONE: $(grep '^summary:' $O/logs/swap.log | tail -1 | cut -c1-500)"; else st "KL swap not done"; fi
# 8 secret game v5: not run (user 11-11: "不跑游戏了 跑不起了")
st "secret game not run (user decision 11-11, cost)"
date '+%F %T' > $O/DONE
st "mech done"
