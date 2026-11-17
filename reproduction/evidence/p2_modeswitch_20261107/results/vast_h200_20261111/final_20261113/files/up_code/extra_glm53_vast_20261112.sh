#!/bin/bash
# GLM-5.3-Flash on the rented 3xH200, after part 1 (11-12, user: "行 跑 我出钱", then "但是先不跑正式游戏"):
#   1. secret game v5 smoke (games_zh_secret5.json, the first 8 items of the game), main and control, the six conditions of the
#      mechanism batteries (MiMo-V2.6, EXAONE), and the release rule fixed on 10-26, computed and logged:
#      unedited setter's role >= 0.5 with the model's own X and <= 0.25 with the user's X on finished games, >= 75% finished,
#      control correct in >= 7 of 8 under both unedited conditions. The full game is NOT run here, whatever the verdict (user).
#   2. k64 as the batteries (all positions, fit_q, attribution question, 75 held-out bodies x 2 orders): swU64_A, swA64_U, rswU64_A,
#      rswA64_U; the unedited items of glm53_r05q_all are reused (copied, as mech_exaone_pro_20261111.sh does).
# Same environment, placement, fits, runners and batch as mech_glm53_vast_20261111.sh (00cf86b1). Waits for "part 1 done" in
# run.log so that it never shares the GPUs with the other two queues. Writes out/extra/DONE. Stop: touch /root/glm53/STOP.
# usage (tmux glm53, window extra): BT=16 MEM=125 OFF=20 bash extra_glm53_vast_20261112.sh
set -u
G=/root/glm53; cd $G/up_code || exit 1
PY=/root/venv_t26/bin/python; M=/dev/shm/hf_models/zai-org/GLM-5.3-Flash; R=$G/ref; O=$G/out/mech; X=$G/out/extra; STOP=$G/STOP
S=crossmodel_role_swap_r05_hc_20261111.py; GS=crossmodel_games_hc_20261111.py
KW='{}'; LG=zh; BD=$R/bodies_zh_150_list.json; B=$R/role_bodies.jsonl; BAND=0-44; GI=$R/games/games_zh_secret5.json
BT=${BT:-16}; MEM=${MEM:-125}; OFF=${OFF:-20}
FD=$O/fits; DIR=$O/glm53; Q=${DIR}_r05q_all
mkdir -p $X $O/logs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 P2_FP8_FORCE_FALLBACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3
st() { echo "[extra] $(date '+%F %T') $*" | tee -a $G/run.log; }
stop() { [ -e $STOP ] && { st "stop flag"; exit 2; }; }
[ -e $X/DONE ] && { st "already DONE"; exit 0; }
until grep -a -q '^\[part1\] .* part 1 done' $G/run.log; do stop; sleep 60; done
stop
st "start after part 1 (runner $(sha256sum $S | cut -c1-8), games $(sha256sum $GS | cut -c1-8), items $(sha256sum $GI | cut -c1-8), BT $BT)"

# 1 secret game smoke, main and control (no full game)
C6="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
for q in main control; do
  stop
  GO=$X/glm53_s5_smoke_$q
  if [ ! -e $GO/summary.json ]; then
    st "game smoke $q start"
    $PY $GS --mode thinking --model $M --on-kwargs "$KW" --items $GI --games secret --lang $LG --fit-from $FD/fit_q.pt --band $BAND \
      --conds $C6 --batch $BT --max-new 3072 --max-memory-gib $MEM --cpu-offload-gib $OFF --save-gen --qkind $q --limit-bodies 8 \
      --out $GO >> $O/logs/game_smoke_$q.log 2>&1 < /dev/null
    st "game smoke $q exit $?"
  fi
done
if [ -e $X/glm53_s5_smoke_main/summary.json ] && [ -e $X/glm53_s5_smoke_control/summary.json ]; then
  v=$($PY - "$X" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
m = json.loads((d / 'glm53_s5_smoke_main' / 'summary.json').read_text(encoding='utf-8'))['secret']
c = json.loads((d / 'glm53_s5_smoke_control' / 'summary.json').read_text(encoding='utf-8'))['secret']
a, u = m['base_A'], m['base_U']
ok = (a['finished'] >= 0.75 * a['n'] and u['finished'] >= 0.75 * u['n'] and a['own_share'] >= 0.5 and u['own_share'] <= 0.25
      and all(round(c[k]['accuracy'] * c[k]['finished']) >= 7 for k in ('base_A', 'base_U')))
print(('PASS' if ok else 'FAIL'), 'own', a['own_share'], a.get('reads_as'), '| user', u['own_share'], u.get('reads_as'),
      '| control', c['base_A']['accuracy'], c['base_U']['accuracy'], '| finished', a['finished'], u['finished'])
EOF
)
  st "game smoke check (release rule of 10-26): $v; full game not run (user 11-12)"
  echo "$v" > $X/game_smoke_verdict.txt
else st "game smoke incomplete: no verdict"; fi

# 2 k64
stop
K64=${DIR}_all_k64
mkdir -p $K64/items && cp -n $Q/items/base_* $K64/items/
st "all_k64 start ($(ls $K64/items | grep -c '^base_') unedited items reused)"
$PY $S --mode thinking --model $M --on-kwargs "$KW" --lang $LG --bodies $BD --splits $B --fit-from $FD/fit_q.pt --band $BAND \
  --conds "base_A:assistant:-:-:0,base_U:user:-:-:0,swU64_A:assistant:user:V:64,swA64_U:user:assistant:V:64,rswU64_A:assistant:user:R:64,rswA64_U:user:assistant:R:64" \
  --region all --question attr --batch $BT --max-new 3072 --max-memory-gib $MEM --cpu-offload-gib $OFF --save-gen --out $K64 \
  >> $O/logs/$(basename $K64).log 2>&1 < /dev/null
st "all_k64 exit $? | $(ls $K64/items | wc -l) items | $(tail -c 500 $O/logs/$(basename $K64).log | tr '\n' ' ' | cut -c1-300)"
date '+%F %T' > $X/DONE
st "extra done"
