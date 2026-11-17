#!/bin/bash
# PRO, user 10-26: the layer-window map of the replacement for gpt-oss-20b (thinking), 8-layer windows moving by 4 as for the other
# readouts: L0-L7, L4-L11, L8-L15, L12-L19, L16-L23. L0-L7, L8-L15 and L16-L23 are the third runs (thinking_swap/gpt_oss_20b_third_*,
# same conditions, settings and subspace), so only L4-L11 and L12-L19 run here: conditions and settings of the third runs (thinking,
# English, fit thinking_ablate4/gpt_oss_20b_smoke/fit.pt, batch 4, 30 GiB, 3072 new tokens), unedited own and User items copied from
# thinking_swap/gpt_oss_20b_all (identical prompts). Waits for the gpt-oss-120b 2x2 driver. Runs in tmux session g20w.
# Skip with STOP_GPTOSS20_WIN.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-20b
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/gptoss20_win.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss20_windows_status.log; }
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
REG="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8"
SW="--mode thinking --model $M --lang en --bodies bodies_en_150_list.json --splits $B --batch 4 --max-new 3072 --max-memory-gib 30 --save-gen \
    --fit-from thinking_ablate4/gpt_oss_20b_smoke/fit.pt --conds $REG"
until ! pgrep -f '[g]ptoss120b_2x2_pro.sh' > /dev/null; do sleep 60; done
for w in 4-11 12-19; do
  [ -e STOP_GPTOSS20_WIN ] && { st "stopped by STOP_GPTOSS20_WIN"; exit 0; }
  a=${w%-*}; b=${w#*-}; O=thinking_swap/gpt_oss_20b_window_L$(printf %02d $a)_$(printf %02d $b)
  mkdir -p $O/items && cp -n thinking_swap/gpt_oss_20b_all/items/base_[AU]__*.json $O/items/ && touch $O.GO
  st "window L$a-L$b start ($(ls $O/items | wc -l) unedited items copied; role_swap $(sha256sum crossmodel_role_swap.py | cut -c1-8))"
  $PY crossmodel_role_swap.py $SW --band $w --out $O > gptoss20_window_L${a}_${b}.log 2>&1
  st "window L$a-L$b exit $?"
done
st "done"
