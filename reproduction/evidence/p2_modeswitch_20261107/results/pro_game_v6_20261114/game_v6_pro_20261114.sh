#!/bin/bash
# Secret game v6 for MiMo-V2.6-Distill-Qwen-9B and EXAONE-4.0.1-32B on PRO (11-14, user: "你先把 mimo v 2.6 和 exaone ver 6 补上吧",
# "pro 在本地 跑行了"; the side job on the rented H200 was cancelled, that machine runs GLM-5.3 only). The v5 game steps of
# mech_mimo26_ext_pro_20261108.sh and mech_exaone_pro_20261111.sh used the v5 items (zh a712030b, en d24c0a2f); the paper's game is v6
# (zh 4d27d8a5, en 26553b13: the System message adds the sentence about questions that are not a game move). Only the game is run
# ("只做 v6 游戏别多做"); every argument as the v5 game step of that model, except the items:
#   MiMo   crossmodel_games.py (5b29640d), thinking, call '{}', zh, fit r05r06_fits/m26/fit_q.pt (2b77d870, restored from the local
#          mirror; the PRO copy had been retired), band 0-31, 44 GiB
#   EXAONE crossmodel_games_hc_20261111.py (13463899), thinking, call {"enable_thinking": true}, en, fit r05r06_fits/e32/fit_q.pt
#          (8e863193), band 0-63, 90 GiB
#   both   the six conditions, batch 16, 3072 new tokens, crossmodel_suite.py 516f117e, the same venv (unchanged since 10-25)
# Weights were deleted after the v5 runs; downloaded again at the same commits (MiMo 1f09ddcc, EXAONE a6b0bd4b) to the same paths
# (hf-mirror, no xet). EXAONE downloads while MiMo plays.
# Steps per model: 8-game smoke, main and control; release rule fixed in advance (10-26): unedited, setter's role >= 0.5 with the
# model's own X and <= 0.25 with the user's X on finished games, >= 75% finished, control correct in >= 7 of 8 under both unedited
# conditions. Full main (100 games) and control only on PASS; otherwise not released and left for inspection. Finished steps (a
# summary.json) are not rerun.
# Refuses to run unless items, runners, suite and fits have the expected sha256 (11-14 rule). Every GPU step takes the exclusive
# flock on persistence/gpu.lock. Writes modeswitch/game_v6_20261114/DONE. Stop: touch modeswitch/STOP_MS; no signals.
# usage (tmux modesw, window g6): bash game_v6_pro_20261114.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python; LOCK=$C/persistence/gpu.lock; STOP=$C/modeswitch/STOP_MS
W=$C/modeswitch/game_v6_20261114; mkdir -p $W games_swap
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
st() { echo "[game v6] $(date '+%F %T') $*" | tee -a $W/status.log >> $C/modeswitch/queue.log; }
stop() { [ -e $STOP ] && { st "stop flag"; exit 2; }; }
gpu() { ( flock -x 9; CUDA_VISIBLE_DEVICES=0 "$@" ) 9>$LOCK; }
chk() { [ "$(sha256sum $1 | cut -c1-64)" = "$2" ] || { st "REFUSED: $1 is not $2 (protocol input check)"; exit 1; }; }
[ -e $W/DONE ] && { st "already DONE"; exit 0; }
# protocol inputs and environment
chk games/games_zh_secret6.json 4d27d8a56e0225042b7fb69df118e74b23d709f656306f9c0b06b52dd3d87e05
chk games/games_en_secret6.json 26553b1333e457646b0621aae5266bd349dee5db749b4073e7c469d931ecdb1e
chk crossmodel_games.py 5b29640df6a80a83526c7a4a1de55688c1ae37c3b2b077fa04d27f859ef820a1
chk crossmodel_games_hc_20261111.py 134638992c2d854063a5c8e6f471c84bd3167bbdd07f9219b213581e36ee44cd
chk crossmodel_suite.py 516f117e330e81c95d705c75f07531cbfef23484732d74105415811865573f31
chk r05r06_fits/m26/fit_q.pt 2b77d870291c60c383a88909de1d1847f2ca8ec8d969fdf852619a98e3757066
chk r05r06_fits/e32/fit_q.pt 8e863193913393ec6e57eac4b378d11dc90599d5c21269107ef95ccb1d4e63cb
ENV=$($PY -c 'import sys, torch, transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__)')
[ "$ENV" = "3.12.3 2.11.0+cu130 5.8.0" ] || { st "REFUSED: environment is '$ENV'"; exit 1; }
st "protocol inputs checked: items zh 4d27d8a5 / en 26553b13, runners 5b29640d / 13463899, suite 516f117e, fits m26 2b77d870 / e32 8e863193; env $ENV"
# weights
dl() {  # repo rev dir okfile
  [ -e $W/$4 ] && return 0
  for i in 1 2 3 4; do
    stop
    st "download $1@${2:0:12} attempt $i ($(df -BG --output=avail /root | tail -1 | tr -d ' ') free)"
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY -c \
      "from huggingface_hub import snapshot_download; print('snapshot', snapshot_download('$1', revision='$2', local_dir='$3', max_workers=8))" \
      >> $W/download.log 2>&1 && { st "downloaded $1 $(du -sh $3 | cut -f1)"; touch $W/$4; return 0; }
    sleep 60
  done
  st "download FAILED $1"; return 1
}
MM=/root/hf_models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B; ME=/root/hf_models_p2nat/LGAI-EXAONE/EXAONE-4.0.1-32B
dl XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B 1f09ddcc1a0ea45751ae0dfe2af20d533168c46f $MM w_mimo.ok
( dl LGAI-EXAONE/EXAONE-4.0.1-32B a6b0bd4b0fb47cd3c644d60d12dba8ecdb9dc810 $ME w_exaone.ok ) &
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
C6="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
game() {  # name runner model kwargs items lang fit band gib
  local n=$1
  local GM=($2 --mode thinking --model $3 --on-kwargs "$4" --items $5 --games secret --lang $6 --fit-from $7 --band $8 --conds $C6
            --batch 16 --max-new 3072 --max-memory-gib $9 --save-gen)
  for q in main control; do
    stop
    [ -e games_swap/${n}_s6_smoke_$q/summary.json ] && continue
    st "$n smoke $q start"
    gpu $PY "${GM[@]}" --qkind $q --limit-bodies 8 --out games_swap/${n}_s6_smoke_$q > $W/${n}_smoke_$q.log 2>&1 < /dev/null
    st "$n smoke $q exit $?"
  done
  local v
  v=$($PY - "$n" <<'EOF'
import json, sys
n = sys.argv[1]
m = json.load(open(f'games_swap/{n}_s6_smoke_main/summary.json'))['secret']
c = json.load(open(f'games_swap/{n}_s6_smoke_control/summary.json'))['secret']
a, u = m['base_A'], m['base_U']
ok = (a['finished'] >= 0.75 * a['n'] and u['finished'] >= 0.75 * u['n'] and a['own_share'] >= 0.5 and u['own_share'] <= 0.25
      and all(round(c[k]['accuracy'] * c[k]['finished']) >= 7 for k in ('base_A', 'base_U')))
print(('PASS' if ok else 'FAIL'), 'own', a['own_share'], a.get('reads_as'), '| user', u['own_share'], u.get('reads_as'),
      '| control', c['base_A']['accuracy'], c['base_U']['accuracy'], '| finished', a['finished'], u['finished'])
EOF
)
  echo "$v" > $W/${n}_smoke_verdict.txt
  st "$n smoke check (release rule of 10-26): $v"
  if [ "${v%% *}" = PASS ]; then
    for q in main control; do
      stop
      [ -e games_swap/${n}_s6_$q/summary.json ] && continue
      touch games_swap/${n}_s6_$q.GO
      st "$n full $q start"
      gpu $PY "${GM[@]}" --qkind $q --out games_swap/${n}_s6_$q > $W/${n}_$q.log 2>&1 < /dev/null
      st "$n full $q exit $?"
    done
  else
    st "$n not released (rule fixed in advance); left for inspection"
  fi
}
game mimo_v26 crossmodel_games.py $MM '{}' games/games_zh_secret6.json zh r05r06_fits/m26/fit_q.pt 0-31 44
wait
[ -e $W/w_exaone.ok ] || { st "EXAONE weights missing, stop"; exit 1; }
game exaone32 crossmodel_games_hc_20261111.py $ME '{"enable_thinking": true}' games/games_en_secret6.json en r05r06_fits/e32/fit_q.pt 0-63 90
date '+%F %T' > $W/DONE
st "game v6 done"
