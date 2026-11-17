#!/bin/bash
# Weights for the PRO lane P3 of the persistence pilot (11-06): gpt-oss-120b (the paper's commit b5c939de8f75; original/* and
# metal/* skipped, as in download_gptoss_pro.sh) and OLMo-2-1124-7B-Instruct. Waits until the disk has >= 90 GiB free (the user
# frees space by hand; nothing is deleted here), then downloads both through hf-mirror. Writes persistence/download_*.done.
# Stop: touch persistence/STOP_PERSIST.
C=/root/p2_scale_20261015/crossmodel; P=$C/persistence; LOG=$P/download_persist_pro.log
PY=/root/consciousness_a6000pro/.venv/bin/python
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next
until [ $(df --output=avail -BG /root | tail -1 | tr -dc 0-9) -ge 90 ]; do
  [ -e $P/STOP_PERSIST ] && exit 2
  sleep 300
done
echo "$(date '+%F %T') free space ok, start" >> $LOG
get() {  # repo dir marker [ignore patterns as python list]
  local repo=$1 dir=$2 mark=$3 ign=${4:-[]}
  for i in 1 2 3 4 5 6; do
    [ -e $P/STOP_PERSIST ] && exit 2
    $PY -c "from huggingface_hub import snapshot_download; p = snapshot_download('$repo', local_dir='$dir', max_workers=8, ignore_patterns=$ign); print('snapshot', p)" >> $LOG 2>&1 && {
      echo "$(date '+%F %T') DONE $repo $(du -sh $dir | cut -f1) commit $(cat $dir/.cache/huggingface/download/config.json.metadata 2>/dev/null | head -1)" >> $LOG; touch $P/$mark; return 0; }
    echo "$(date '+%F %T') $repo attempt $i failed" >> $LOG; sleep 120
  done
  return 1
}
get openai/gpt-oss-120b /root/hf_models/openai/gpt-oss-120b download_gptoss120b.done "['original/*', 'metal/*']"
get allenai/OLMo-2-1124-7B-Instruct /root/hf_models/allenai/OLMo-2-1124-7B-Instruct download_olmo2_7b.done
echo "$(date '+%F %T') all downloads finished" >> $LOG
