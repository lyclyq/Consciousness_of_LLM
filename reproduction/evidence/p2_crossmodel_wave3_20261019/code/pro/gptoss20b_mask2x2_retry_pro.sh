#!/bin/bash
# PRO, 10-28: retry of the gpt-oss-20b masked 2x2 of ablation_fill_pro.sh, which stopped at model load (08:14) because the publisher
# trust check of the MXFP4 kernel repository kernels-community/gpt-oss-triton-kernels through hf-mirror failed; the same kernel loaded
# for the 20b fit just before and for the 120b fit after, so the failure was transient. Same command; the unedited items on disk are
# skipped. Up to three attempts, 10 minutes apart. Waits until the queue (qposf) and the position reruns (qpost) have finished.
# Runs in tmux session g20r.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/g20r.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> ablation_fill_status.log; }
until grep -q 'rerun done' qwen38_pos_status.log && ! tmux has-session -t qposf 2>/dev/null && ! tmux has-session -t qpost 2>/dev/null; do sleep 120; done
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
E4="swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
O=thinking_swap/gpt_oss_20b_mask
for i in 1 2 3; do
  $PY crossmodel_role_swap_mask.py --mode thinking --model /root/hf_models/openai/gpt-oss-20b --fit-from thinking_ablate4/gpt_oss_20b_mask_fit/fit.pt \
    --band 0-23 --batch 4 --max-memory-gib 30 --on-kwargs '{}' --lang en --bodies bodies_en_150_list.json --splits $B --max-new 3072 --save-gen \
    --mask-extra --conds "$E4" --out $O > ablation_fill_gptoss20b_2x2_retry$i.log 2>&1
  r=$?; st "2 gpt-oss-20b mask 2x2 retry $i exit $r ($(ls $O/items | wc -l) items)"
  [ $r = 0 ] && break
  sleep 600
done
st "retry done"
