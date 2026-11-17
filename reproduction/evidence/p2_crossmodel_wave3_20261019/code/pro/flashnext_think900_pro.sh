#!/bin/bash
# user 10-20: complete the Qwen3.8-Flash-Next thinking attribution to the full sample (150 bodies x 3 roles x 2 orders = 900 items)
# on the same path as its first 180 items (10-17, pro_chain6.sh): PRO, the official FP8 release with the hub FP8 kernel, the same
# environment, the same arguments (batch 40, 70 GiB weight budget, 66 GiB CPU offload, up to 3072 new tokens), and the same
# generation code (crossmodel_thinking_b_b2e4.py = b2e46749; it differs from fafd3562, used on 10-17, only by the --skip-off flag).
# The runner skips the 180 items on disk, so only the other 720 are generated.
#   1. download Qwen/Qwen3.8-Flash-Next-FP8 at the revision of the 10-16 download (refs/main then: 236dfdf2...), via hf-mirror
#   2. thinking, bodies_zh_150_list.json, into thinking/flashnext
# Skip with STOP_FN900.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/Qwen/Qwen3.8-Flash-Next-FP8
REV=236dfdf285828023ca3bcd3f37366c58a3469b13
mkdir "$C/flashnext_think900.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> flashnext_think900_status.log; }
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/root/p2_scale_20261015/hf_home_dl
st "download start rev $REV; free $(df -h / | tail -1 | awk '{print $4}')"
ok=0
for i in 1 2 3 4 5 6 7 8; do
  [ -e STOP_FN900 ] && exit 0
  $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.8-Flash-Next-FP8', revision='$REV', local_dir='$M', max_workers=8))" >> flashnext_think900_download.log 2>&1 && { ok=1; break; }
  st "download attempt $i failed; free $(df -h / | tail -1 | awk '{print $4}')"; sleep 120
done
[ $ok = 1 ] || { st "download FAILED"; exit 1; }
st "download DONE $(du -sh $M | cut -f1); free $(df -h / | tail -1 | awk '{print $4}')"
[ -e STOP_FN900 ] && exit 0
st "thinking 900 start ($(ls thinking/flashnext/items | wc -l) items on disk)"
PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY crossmodel_thinking_b_b2e4.py --model $M --lang zh --off-kwargs '{"enable_thinking": false}' --on-kwargs '{}' \
  --bodies-zh bodies_zh_150_list.json --out thinking/flashnext --max-new 3072 --batch 40 --max-memory-gib 70 --cpu-offload-gib 66 \
  > thinking_flashnext_900.log 2>&1
st "thinking 900 exit $? ($(ls thinking/flashnext/items | wc -l) items on disk)"
