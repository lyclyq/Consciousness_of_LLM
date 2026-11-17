#!/bin/bash
# Re-download gpt-oss-120b on PRO for the wording test (user 11-09: the mechanism group's five readouts, "在 deepseek 没开始前补齐").
# Pinned to the paper's commit b5c939de8f75 (Table tab:self_config), to the path of the anchor thinking_swap/gpt_oss_120b_r05m_msg
# (/root/hf_models/openai/gpt-oss-120b), as on 11-08 (modeswitch_swap_pro_20261108.sh): hf-mirror, no xet, without original/* and
# metal/*. Afterwards every LFS file is checked against the sha256 of the Hub metadata of that commit. Network only, no GPU.
# Marker dl_g120_20261109.ok; log dl_g120_20261109.log. Stop: touch STOP_DLG120. Runs in tmux dlg120.
set -u
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python
D=/root/hf_models/openai/gpt-oss-120b; REV=b5c939de8f754692c1647ca79fbf85e8c1e70f8a; L=$C/dl_g120_20261109.log
cd $C || exit 1
[ -e $C/dl_g120_20261109.ok ] && exit 0
free=$(df -BG /root | tail -1 | awk '{print $4}' | tr -d G)
[ "$free" -ge 80 ] || { echo "$(date '+%F %T') only $free GiB free, need 80" >> $L; exit 1; }
echo "$(date '+%F %T') start openai/gpt-oss-120b@$REV -> $D ($free GiB free)" >> $L
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
for i in 1 2 3 4 5; do
  [ -e STOP_DLG120 ] && { echo "$(date '+%F %T') stop flag" >> $L; exit 2; }
  $PY -c "from huggingface_hub import snapshot_download; print('snapshot', snapshot_download('openai/gpt-oss-120b', revision='$REV', local_dir='$D', max_workers=8, ignore_patterns=['original/*', 'metal/*']))" >> $L 2>&1 && break
  echo "$(date '+%F %T') attempt $i failed" >> $L; sleep 120
done
$PY - "$D" "$REV" >> $L 2>&1 <<'EOF'
import hashlib, sys
from pathlib import Path
from huggingface_hub import HfApi
d, rev = Path(sys.argv[1]), sys.argv[2]
info = HfApi().model_info('openai/gpt-oss-120b', revision=rev, files_metadata=True)
bad, n = [], 0
for s in info.siblings:
    if not s.lfs or s.rfilename.startswith(('original/', 'metal/')):
        continue
    f = d / s.rfilename
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b''):
            h.update(chunk)
    n += 1
    if h.hexdigest() != s.lfs.sha256:
        bad.append(s.rfilename)
print('sha256 check:', n, 'LFS files,', len(bad), 'mismatched', bad)
sys.exit(1 if bad or n == 0 else 0)
EOF
if [ $? = 0 ]; then touch $C/dl_g120_20261109.ok; echo "$(date '+%F %T') DONE $(du -sh $D | cut -f1)" >> $L
else echo "$(date '+%F %T') FAILED (download or sha256)" >> $L; exit 1; fi
