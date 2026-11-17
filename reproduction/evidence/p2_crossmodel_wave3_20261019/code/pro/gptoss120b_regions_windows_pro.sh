#!/bin/bash
# PRO, user 10-28 ("位置分组（正文 / 消息结尾 / 问题）这个得有"; "三分段没太多价值，被脑图（层窗口图）替代了"): the two localisation
# analyses that gpt-oss-120b still lacks, with the conditions of the gpt-oss-20b runs (gptoss_region_full_pro.sh,
# gptoss20b_windows_pro.sh): base_A, base_U, swU8_A, rswU8_A, swA8_U; unedited items copied from thinking_swap/gpt_oss_120b_all
# (identical prompts). Settings of the 120b 2x2: thinking, English, fit thinking_ablate4/gpt_oss_120b_fit/fit.pt, batch 8, 90 GiB,
# 3072 new tokens; crossmodel_role_swap.py f2362716.
#   1. position groups, every layer L0-L35: --region body, end, question
#   2. layer windows of 8 layers moving by 4: L0-7, L4-11, ..., L28-35 (8 windows)
# (gpt-oss-20b has all five of its windows: L0-7, L8-15, L16-23 are its third runs, L4-11 and L12-19 were run separately.)
# Waits until the gpt-oss-20b masked 2x2 retry (session g20r) has finished. Runs in tmux session g120l; skip with STOP_G120L.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-120b
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/g120l.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss120b_loc_status.log; }
stop() { [ -e STOP_G120L ] && { st "stopped by STOP_G120L"; exit 0; }; }
until grep -q 'retry done' ablation_fill_status.log && ! tmux has-session -t g20r 2>/dev/null; do stop; sleep 120; done
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
REG="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8"
SW="--mode thinking --model $M --on-kwargs {} --lang en --bodies bodies_en_150_list.json --splits $B --batch 8 --max-new 3072 --max-memory-gib 90 \
    --save-gen --fit-from thinking_ablate4/gpt_oss_120b_fit/fit.pt --conds $REG"
seed() { mkdir -p $1/items && cp -n thinking_swap/gpt_oss_120b_all/items/base_[AU]__*.json $1/items/ && touch $1.GO; }
st "start (role_swap $(sha256sum crossmodel_role_swap.py | cut -c1-8))"
# 1
for reg in body end question; do
  stop
  O=thinking_swap/gpt_oss_120b_region_$reg; seed $O
  $PY crossmodel_role_swap.py $SW --band 0-35 --region $reg --out $O > gptoss120b_region_$reg.log 2>&1
  st "1 region $reg exit $?"
done
# 2
for a in 0 4 8 12 16 20 24 28; do
  stop
  b=$((a + 7)); O=thinking_swap/gpt_oss_120b_window_L$(printf %02d $a)_$(printf %02d $b); seed $O
  $PY crossmodel_role_swap.py $SW --band $a-$b --out $O > gptoss120b_window_L${a}_${b}.log 2>&1
  st "2 window L$a-L$b exit $?"
done
st "done"
