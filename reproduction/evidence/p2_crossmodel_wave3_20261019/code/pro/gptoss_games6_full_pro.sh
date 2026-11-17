#!/bin/bash
# PRO, user 10-27: secret game v6 full runs for the two gpt-oss models (100 games, main question then control, six conditions), each
# only if its v6 smoke passed the v5 release rule (gptoss_games6_smoke_pro.sh writes "<m> smoke verdict: PASS ..." to
# gptoss_g6_status.log). Settings as the v5 full runs of each model (crossmodel_games.py 5b29640d; thinking rendering, English,
# native MXFP4, every layer, rank 8; 120b: batch 8, 90 GiB; 20b: batch 4, 30 GiB). Waits for the smoke driver. Runs in tmux g6f.
# Skip with STOP_GPTOSS_G6.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python
mkdir "$C/gptoss_g6f.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss_g6_status.log; }
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
C6="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
until ! pgrep -f '[g]ptoss_games6_smoke_pro.sh' > /dev/null; do sleep 30; done
for m in 120b 20b; do
  grep -q "$m smoke verdict: PASS" gptoss_g6_status.log || { st "$m full not started: smoke did not pass"; continue; }
  if [ $m = 120b ]; then
    G="--model /root/hf_models/openai/gpt-oss-120b --fit-from thinking_ablate4/gpt_oss_120b_fit/fit.pt --band 0-35 --batch 8 --max-memory-gib 90"
  else
    G="--model /root/hf_models/openai/gpt-oss-20b --fit-from thinking_ablate4/gpt_oss_20b_smoke/fit.pt --band 0-23 --batch 4 --max-memory-gib 30"
  fi
  for q in main control; do
    [ -e STOP_GPTOSS_G6 ] && { st "stopped by STOP_GPTOSS_G6"; exit 0; }
    touch games_swap/gpt_oss_${m}_s6_$q.GO
    st "$m full $q start"
    $PY crossmodel_games.py --mode thinking $G --on-kwargs {} --lang en --max-new 3072 --save-gen --items games/games_en_secret6.json \
      --games secret --conds $C6 --qkind $q --out games_swap/gpt_oss_${m}_s6_$q > gptoss${m}_g6_$q.log 2>&1
    st "$m full $q exit $?"
  done
done
st "full done"
