#!/bin/bash
# GLM-5.3-Flash secret game v5, main game, reduced full run: the first 50 of the 100 items, the six conditions of the batteries
# (300 games). User 11-13: "要不 glm 不跑慢的 900 局了，做 300 局行了，然后作为一个标注" (cost limit), then "可以 就这么做吧".
# The smoke (8 items, main and control) passed the 10-26 release rule, so the control game is not repeated here.
# Same runner, items, fits, environment, placement and batch as extra_glm53_vast_20261112.sh (b3fd9aa6). --limit-bodies 50
# selects the first 50 items, and the runner then does not wait for a GO file. Starts after "extra done" in run.log. Writes
# out/extra/glm53_s5_main50/DONE. Stop: touch /root/glm53/STOP.
# usage (tmux glm53, window game300): BT=16 MEM=125 OFF=20 bash game300_glm53_vast_20261113.sh
set -u
G=/root/glm53; cd $G/up_code || exit 1
PY=/root/venv_t26/bin/python; M=/dev/shm/hf_models/zai-org/GLM-5.3-Flash; R=$G/ref; O=$G/out/mech; X=$G/out/extra; STOP=$G/STOP
GS=crossmodel_games_hc_20261111.py; GI=$R/games/games_zh_secret5.json; FD=$O/fits; BAND=0-44
BT=${BT:-16}; MEM=${MEM:-125}; OFF=${OFF:-20}; OUT=$X/glm53_s5_main50
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 P2_FP8_FORCE_FALLBACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3
st() { echo "[game300] $(date '+%F %T') $*" | tee -a $G/run.log; }
[ -e $OUT/DONE ] && { st "already DONE"; exit 0; }
until grep -a -q '^\[extra\] .* extra done' $G/run.log; do [ -e $STOP ] && { st "stop flag"; exit 2; }; sleep 60; done
[ -e $STOP ] && { st "stop flag"; exit 2; }
C6="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
st "start: main game, first 50 items, 6 conditions (runner $(sha256sum $GS | cut -c1-8), items $(sha256sum $GI | cut -c1-8), BT $BT)"
$PY $GS --mode thinking --model $M --on-kwargs '{}' --items $GI --games secret --lang zh --fit-from $FD/fit_q.pt --band $BAND \
  --conds $C6 --batch $BT --max-new 3072 --max-memory-gib $MEM --cpu-offload-gib $OFF --save-gen --qkind main --limit-bodies 50 \
  --out $OUT >> $O/logs/game_main50.log 2>&1 < /dev/null
rc=$?
if [ -e $OUT/summary.json ]; then date '+%F %T' > $OUT/DONE; st "main50 exit $rc | $(tail -c 700 $O/logs/game_main50.log | tr '\n' ' ' | cut -c1-500)"
else st "main50 exit $rc, no summary; see logs/game_main50.log"; fi
