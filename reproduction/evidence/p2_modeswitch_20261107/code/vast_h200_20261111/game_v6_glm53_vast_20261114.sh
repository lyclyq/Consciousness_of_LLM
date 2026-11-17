#!/bin/bash
# GLM-5.3-Flash secret game v6 (11-14). The earlier H200 game runs (extra_glm53_vast_20261112.sh, game300_glm53_vast_20261113.sh)
# used the v5 items (games_zh_secret5.json a712030b), but the paper's game is v6 for every readout (games_zh_secret6.json 4d27d8a5:
# the System message adds "如果用户问的不是游戏里的一步，就直接回答所问的问题"). User 11-14: rerun v6 on a rented 4 x H200 NVL machine,
# "只做 v6 游戏别多做". Everything else as the v5 runs: runner crossmodel_games_hc_20261111.py (13463899), fit_q.pt of the 11-11 fits
# (07b80490), band 0-44, the six conditions, batch 16, max-new 3072, placement MEM 125 / OFF 20 on three GPUs (as on the 3-GPU machine
# of 11-11..13), environment venv_t26 from the same freeze, weights commit 46a050be, P2_FP8_FORCE_FALLBACK=1.
#   0. waits for the weights and the environment; hc_edit unit tests (CPU, seconds)
#   1. refuses to run unless the items, the runner and the fit have the expected sha256 (11-14 rule: check protocol inputs)
#   2. smoke: 8 items, main and control; release rule of 10-26
#   3. on PASS only: main game, first NMAIN items (default 50, as the v5 run) -> out/v6/glm53_s6_main, DONE
# Stop: touch /root/glm53/STOP. usage (tmux glm53, window v6): bash game_v6_glm53_vast_20261114.sh
set -u
G=/root/glm53; cd $G/up_code || exit 1
PY=/root/venv_t26/bin/python; M=/dev/shm/hf_models/zai-org/GLM-5.3-Flash; R=$G/ref; X=$G/out/v6; STOP=$G/STOP
GS=crossmodel_games_hc_20261111.py; GI=$R/games/games_zh_secret6.json; FIT=$G/out/mech/fits/fit_q.pt; BAND=0-44
BT=16; MEM=125; OFF=20; NMAIN=${NMAIN:-50}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 P2_FP8_FORCE_FALLBACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2
mkdir -p $X/logs
st() { echo "[v6] $(date '+%F %T') $*" | tee -a $G/run.log; }
stop() { [ -e $STOP ] && { st "stop flag"; exit 2; }; }
until [ -e $G/weights.ok ] && grep -q "venv OK" $G/setup.log; do stop; sleep 30; done
st "weights and environment ready: $(grep 'venv OK' $G/setup.log | tail -1)"
$PY test_hc_edit_20261111.py > $G/test_hc_edit.log 2>&1; st "hc_edit tests: $(tail -1 $G/test_hc_edit.log)"
chk() { [ "$(sha256sum $1 | cut -c1-64)" = "$2" ] || { st "REFUSED: $1 is not $2 (protocol input check)"; exit 1; }; }
chk $GI 4d27d8a56e0225042b7fb69df118e74b23d709f656306f9c0b06b52dd3d87e05
chk $GS 134638992c2d854063a5c8e6f471c84bd3167bbdd07f9219b213581e36ee44cd
chk $FIT 07b804907f2299f5d77d2f9eb4f18ff350c8f04aad396196bf49954f04c1ef52
st "protocol inputs checked: items v6 4d27d8a5, runner 13463899, fit_q 07b80490"
C6="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
GM=($GS --mode thinking --model $M --on-kwargs '{}' --items $GI --games secret --lang zh --fit-from $FIT --band $BAND --conds $C6
    --batch $BT --max-new 3072 --max-memory-gib $MEM --cpu-offload-gib $OFF --save-gen)
for q in main control; do
  stop
  if [ ! -e $X/glm53_s6_smoke_$q/summary.json ]; then
    st "smoke $q start"
    $PY "${GM[@]}" --qkind $q --limit-bodies 8 --out $X/glm53_s6_smoke_$q >> $X/logs/smoke_$q.log 2>&1 < /dev/null
    st "smoke $q exit $?"
  fi
done
v=$($PY - "$X" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
m = json.loads((d / 'glm53_s6_smoke_main' / 'summary.json').read_text(encoding='utf-8'))['secret']
c = json.loads((d / 'glm53_s6_smoke_control' / 'summary.json').read_text(encoding='utf-8'))['secret']
a, u = m['base_A'], m['base_U']
ok = (a['finished'] >= 0.75 * a['n'] and u['finished'] >= 0.75 * u['n'] and a['own_share'] >= 0.5 and u['own_share'] <= 0.25
      and all(round(c[k]['accuracy'] * c[k]['finished']) >= 7 for k in ('base_A', 'base_U')))
print(('PASS' if ok else 'FAIL'), 'own', a['own_share'], a.get('reads_as'), '| user', u['own_share'], u.get('reads_as'),
      '| control', c['base_A']['accuracy'], c['base_U']['accuracy'], '| finished', a['finished'], u['finished'])
EOF
)
echo "$v" > $X/smoke_verdict.txt
st "smoke check (release rule of 10-26): $v"
[ "${v%% *}" = PASS ] || { st "not released: no main game"; exit 0; }
stop
st "main game start: first $NMAIN items, 6 conditions"
$PY "${GM[@]}" --qkind main --limit-bodies $NMAIN --out $X/glm53_s6_main >> $X/logs/main.log 2>&1 < /dev/null
rc=$?
if [ -e $X/glm53_s6_main/summary.json ]; then date '+%F %T' > $X/glm53_s6_main/DONE; st "main exit $rc | $(tail -c 600 $X/logs/main.log | tr '\n' ' ' | cut -c1-500)"
else st "main exit $rc, no summary; see $X/logs/main.log"; fi
st "v6 done"
