#!/bin/bash
# Ministral-3-14B (Instruct BF16 and Reasoning) for the natural-order calibration on PRO (user 10-20: Mistral results go to an
# appendix). HF-format shards only (consolidated.safetensors, Mistral's own format, skipped), pinned to the commits below,
# then every LFS file checked against the hub's sha256 (verify_sha.sh). Status in download_ministral_pro.log.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python
mkdir "$C/download_ministral_pro.lock" 2>/dev/null || exit 0
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
cd $C
for spec in "mistralai/Ministral-3-14B-Instruct-2512-BF16 3cea74c1ebaf5ce5f5a2553de470e2ceab825142" "mistralai/Ministral-3-14B-Reasoning-2512 51f9210f3cd20f3452a80d5819d15dc61cc50630"; do
  set -- $spec; R=$1; REV=$2; N=$(basename $R); D=/root/hf_models/$R
  for i in 1 2 3 4 5 6; do
    $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$R', revision='$REV', local_dir='$D', max_workers=8, ignore_patterns=['consolidated.safetensors'])" >> download_ministral_pro.log 2>&1 && break
    echo "$(date '+%F %T') $N attempt $i failed" >> download_ministral_pro.log; sleep 60
  done
  bash verify_sha.sh $D $C/expected_$N.txt $C/verify_$N.txt
  echo "$(date '+%F %T') $N verify $(grep -c '^ok' verify_$N.txt) ok $(grep -c '^BAD' verify_$N.txt) bad" >> download_ministral_pro.log
done
echo "$(date '+%F %T') DONE" >> download_ministral_pro.log
