#!/bin/bash
# GLM-5.3-Flash on the rented vast.ai machine (11-11): environment of server3's venv_t26 (pip freeze, 76 packages, torch 2.6.0
# cu124, transformers 5.19.0) and the weights at server3's commit 46a050bec13002ad29ee795c16e61cf4fa3c4a9c in /dev/shm (RAM disk; the
# root disk has 16 GB). Logs in /root/glm53/setup.log.
set -u
mkdir -p /root/glm53 /dev/shm/hf_models /dev/shm/uvcache
L=/root/glm53/setup.log
echo "[setup] $(date '+%F %T') start" >> $L
( export UV_CACHE_DIR=/dev/shm/uvcache
  uv venv /root/venv_t26 --python 3.12 >> $L 2>&1 && \
  uv pip install --python /root/venv_t26/bin/python -r /root/glm53/venv_t26_freeze.txt >> $L 2>&1 && \
  echo "[setup] $(date '+%F %T') venv OK: $(/root/venv_t26/bin/python -c 'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.device_count())')" >> $L \
  || echo "[setup] $(date '+%F %T') venv FAILED" >> $L ) &
( for i in 1 2 3 4; do
    /root/venv_dl/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('zai-org/GLM-5.3-Flash', revision='46a050bec13002ad29ee795c16e61cf4fa3c4a9c', local_dir='/dev/shm/hf_models/zai-org/GLM-5.3-Flash', max_workers=16))" >> /root/glm53/download.log 2>&1 \
      && { echo "[setup] $(date '+%F %T') weights OK $(du -sh /dev/shm/hf_models/zai-org/GLM-5.3-Flash | cut -f1)" >> $L; touch /root/glm53/weights.ok; break; }
    echo "[setup] $(date '+%F %T') weights attempt $i failed" >> $L; sleep 30
  done ) &
wait
echo "[setup] $(date '+%F %T') end" >> $L
