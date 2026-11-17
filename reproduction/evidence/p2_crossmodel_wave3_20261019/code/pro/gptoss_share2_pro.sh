#!/bin/bash
# PRO, 10-23: the gpt-oss-20b epsilon check again with crossmodel_subspace_share.py using the two-segment alignment of
# crossmodel_role_swap.align (the first run, 05:10 UTC, stopped at the three-role common-suffix assertion: gpt-oss renders a
# System message as a developer message with two extra newlines after the body). Waits until gptoss_swap2_pro.sh has exited.
# Hub timeouts set (the MXFP4 loader fetches its kernel from the hub at every load). Skip with STOP_GPTOSS_SHARE2.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-20b
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/gptoss_share2.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss_k8_status.log; }
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
until ! pgrep -f '[g]ptoss_swap2_pro.sh' > /dev/null; do [ -e STOP_GPTOSS_SHARE2 ] && exit 0; sleep 120; done
mkdir -p subspace_share
st "E2 subspace share (two-segment alignment) start"
$PY crossmodel_subspace_share.py --model $M --on-kwargs '{}' --fit-from thinking_ablate4/gpt_oss_20b_smoke/fit.pt --lang en \
  --bodies bodies_en_150_list.json --splits $B --k 8 --max-memory-gib 30 --out subspace_share/gpt_oss_20b_think.json > gptoss_share2.log 2>&1
st "E2 exit $?"
