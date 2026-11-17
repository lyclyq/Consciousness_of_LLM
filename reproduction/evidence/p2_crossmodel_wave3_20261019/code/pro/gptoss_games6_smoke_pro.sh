#!/bin/bash
# PRO, user 10-27: secret game v6 (make_games_secret6.py: v5 plus one System sentence that allows a question which is not a move to be
# answered) on the two gpt-oss models, smoke only: the first 8 games, main question and control, the six conditions of the v5 runs,
# with the v5 settings of each model (crossmodel_games.py 5b29640d; thinking rendering, English, native MXFP4, every layer, rank 8;
# 120b: its 2x2 fit, batch 8, 90 GiB; 20b: its fit, batch 4, 30 GiB). Verdict by the release rule of v5 (main: own >= 0.5, user
# <= 0.25 on finished games, >= 75% finished; control: correct in >= 7 of 8 games under both unedited conditions). Nothing full is
# started. Runs in tmux session g6s. Skip with STOP_GPTOSS_G6.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python
mkdir "$C/gptoss_g6s.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss_g6_status.log; }
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
C6="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
st "start (games $(sha256sum crossmodel_games.py | cut -c1-8), items $(sha256sum games/games_en_secret6.json | cut -c1-8))"
for m in 120b 20b; do
  if [ $m = 120b ]; then
    G="--model /root/hf_models/openai/gpt-oss-120b --fit-from thinking_ablate4/gpt_oss_120b_fit/fit.pt --band 0-35 --batch 8 --max-memory-gib 90"
  else
    G="--model /root/hf_models/openai/gpt-oss-20b --fit-from thinking_ablate4/gpt_oss_20b_smoke/fit.pt --band 0-23 --batch 4 --max-memory-gib 30"
  fi
  for q in main control; do
    [ -e STOP_GPTOSS_G6 ] && { st "stopped by STOP_GPTOSS_G6"; exit 0; }
    $PY crossmodel_games.py --mode thinking $G --on-kwargs {} --lang en --max-new 3072 --save-gen --items games/games_en_secret6.json \
      --games secret --conds $C6 --qkind $q --limit-bodies 8 --out games_swap/gpt_oss_${m}_s6_smoke_$q > gptoss${m}_g6_smoke_$q.log 2>&1
    st "$m smoke $q exit $?"
  done
  v=$($PY - $m <<'EOF'
import json, sys
m = sys.argv[1]
M = json.load(open(f'games_swap/gpt_oss_{m}_s6_smoke_main/summary.json'))['secret']
C = json.load(open(f'games_swap/gpt_oss_{m}_s6_smoke_control/summary.json'))['secret']
a, u = M['base_A'], M['base_U']
ok = (a['finished'] >= 0.75 * a['n'] and u['finished'] >= 0.75 * u['n'] and a['own_share'] >= 0.5 and u['own_share'] <= 0.25
      and all(round(C[k]['accuracy'] * C[k]['finished']) >= 7 for k in ('base_A', 'base_U')))
d = lambda c: (M.get(c, {}).get('own_minus_base') or {}).get('mean')
print('PASS' if ok else 'FAIL', f"own {a['own_share']} user {u['own_share']} | control {C['base_A']['accuracy']} {C['base_U']['accuracy']} | "
      f"removal {d('swU8_A')} installation {d('swA8_U')} random {d('rswU8_A')} {d('rswA8_U')}")
EOF
)
  st "$m smoke verdict: $v"
done
st "done"
