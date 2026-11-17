#!/bin/bash
# PRO, 10-27: the subspace-share check (crossmodel_subspace_share.py, as gptoss_share2_pro.sh for gpt-oss-20b) for the gpt-oss-120b
# fit used in its 2x2 (thinking_ablate4/gpt_oss_120b_fit/fit.pt, thinking rendering, English, k 8, every layer L0-L35): the share
# of a held-out body's own-User difference inside the eight fitted dimensions, against a random subspace, and the own-System share.
# Waits until the gpt-oss-120b secret game (gptoss120b_game5_release_pro.sh) has exited, so the two never share the GPU. Runs in
# tmux session g120s. Skip with STOP_GPTOSS120_SHARE.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-120b
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/gptoss120_share.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss120_g5_status.log; }
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
until ! pgrep -f '[g]ptoss120b_game5_release_pro.sh' > /dev/null; do [ -e STOP_GPTOSS120_SHARE ] && exit 0; sleep 60; done
[ -e STOP_GPTOSS120_SHARE ] && { st "share stopped by STOP_GPTOSS120_SHARE"; exit 0; }
mkdir -p subspace_share
st "share start (script $(sha256sum crossmodel_subspace_share.py | cut -c1-8), fit $(sha256sum thinking_ablate4/gpt_oss_120b_fit/fit.pt | cut -c1-8))"
$PY crossmodel_subspace_share.py --model $M --on-kwargs '{}' --fit-from thinking_ablate4/gpt_oss_120b_fit/fit.pt --lang en \
  --bodies bodies_en_150_list.json --splits $B --k 8 --band 0-35 --max-memory-gib 90 --out subspace_share/gpt_oss_120b_think.json > gptoss120b_share.log 2>&1
st "share exit $?"
