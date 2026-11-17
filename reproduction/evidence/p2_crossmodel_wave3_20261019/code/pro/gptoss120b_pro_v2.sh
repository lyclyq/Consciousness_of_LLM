#!/bin/bash
# gpt-oss-120b on PRO, v2 (10-19): replaces gptoss120b_pro.sh, which exits through STOP_GPTOSS120. Every gpt-oss run uses one
# path, native MXFP4 (user 10-19); 120b (~65 GB) fits the single 96 GB GPU only in this format.
# 1. Download (original/* and metal/* skipped) once >= 80 GB is free (the user deletes Qwen3.8-Flash-Next-FP8 first).
# 2. Run alone on the GPU: wait until the split-patching run (attrsplit150) has finished and the gpt-oss-20b 900-item run
#    (gptoss20b_think900_pro.sh) has ended.
# 3. G (geometry only, English, 150 bodies) and the 180-item thinking sample, same settings as gpt-oss-20b.
# Skip with STOP_GPTOSS120_V2. Status in gptoss120b_status.log.
C=/root/p2_scale_20261015/crossmodel; W=/root/p2_scale_20261015; PY=/root/consciousness_a6000pro/.venv/bin/python
M=/root/hf_models/openai/gpt-oss-120b
mkdir "$C/gptoss120b_pro_v2.lock" 2>/dev/null || exit 0
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss120b_status.log; }
until [ "$(df -BG / | tail -1 | awk '{print $4}' | tr -d G)" -ge 80 ]; do [ -e STOP_GPTOSS120_V2 ] && exit 0; sleep 600; done
if ! grep -q DONE download_gptoss120b_pro.log 2>/dev/null; then
  st "download start"
  for i in 1 2 3 4 5 6; do
    $PY -c "from huggingface_hub import snapshot_download; snapshot_download('openai/gpt-oss-120b', local_dir='$M', max_workers=8, ignore_patterns=['original/*', 'metal/*'])" >> download_gptoss120b_pro.log 2>&1 && { echo "$(date '+%F %T') DONE $(du -sh $M | cut -f1)" >> download_gptoss120b_pro.log; break; }
    echo "$(date '+%F %T') attempt $i failed" >> download_gptoss120b_pro.log; sleep 120
  done
fi
grep -q DONE download_gptoss120b_pro.log || { st "download failed"; exit 1; }
until grep -q '^EXIT' $W/attrsplit150.log 2>/dev/null && tail -1 gptoss_mxfp4_status.log | grep -q 'think900 exit'; do
  [ -e STOP_GPTOSS120_V2 ] && exit 0; sleep 300
done
[ -e STOP_GPTOSS120_V2 ] && exit 0
st "G start"
$PY crossmodel_suite.py --model $M --tmpl-kwargs '{}' --bodies /root/selfindex_three_actual_20261011_v2/role_bodies.jsonl --problems problems.json --out suite_think/gpt_oss_120b_en --phases G --geometry-only --max-memory-gib 90 --lang en --bodies-en bodies_en_150.json > gptoss120b_G.log 2>&1
st "G exit $?"
[ -e STOP_GPTOSS120_V2 ] && exit 0
st "think180 start"
$PY crossmodel_thinking_b.py --model $M --off-kwargs '{}' --on-kwargs '{}' --skip-off --bodies-zh pilot_bodies_en_list.json --lang en --out thinking/gpt_oss_120b_en --batch 16 --max-new 3072 --max-memory-gib 90 > gptoss120b_think180.log 2>&1
st "think180 exit $?"
