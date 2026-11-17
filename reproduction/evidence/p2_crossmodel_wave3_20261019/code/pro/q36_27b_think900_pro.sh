#!/bin/bash
# user 10-21 ("统一"): Qwen3.6-27B thinking attribution brought to the full sample (150 bodies x 3 roles x 2 orders = 900 items), as
# every other model, on the same path as its first 180 items (10-17, pro_chain5.sh): PRO, BF16, the same environment and
# PYTHONPATH, the same arguments (batch 1, 90 GiB, up to 3072 new tokens). Generation code crossmodel_thinking_b_b2e4.py
# (b2e46749); against the PRO copy of 10-19 it adds only the gpt-oss channel marker, the subprocess and subset options and
# --skip-off, none of which is active here. The runner skips the 180 items on disk.
# The weights were deleted after 10-17 (the NFS link is stale); they are downloaded again at Qwen/Qwen3.6-27B 6a9e13bd, the
# hub head since 2026-04-24 (weights unchanged since 04-22, so the same files as on 10-17), into the same path.
#   1. wait until the Flash-Next thinking run has exited; 2. wait for >= 70 GB free (the user deletes Flash-Next);
#   3. download; 4. recheck: one pilot body (6 of the 180 items) again into thinking/qwen36_27b_recheck; the non-thinking
#      probabilities, answers, lengths and tails must equal the 10-17 items, else stop; 5. the remaining 720 items.
# Skip with STOP_Q36_900. Status in q36_900_status.log.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/Qwen/Qwen3.6-27B
REV=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
mkdir "$C/q36_900.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> q36_900_status.log; }
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/root/p2_scale_20261015/hf_home_dl PYTHONPATH=/root/p2_scale_20261015/tf_next
st "waiting for the Flash-Next thinking run to exit"
until grep -q 'thinking 900 exit' flashnext_think900_status.log && ! pgrep -f '[t]hinking/flashnext' > /dev/null; do
  [ -e STOP_Q36_900 ] && exit 0; sleep 300; done
st "Flash-Next done; waiting for >= 70 GB free"
until [ "$(df -BG / | tail -1 | awk '{print $4}' | tr -d G)" -ge 70 ]; do [ -e STOP_Q36_900 ] && exit 0; sleep 300; done
[ -L $M ] && mv $M $M.nfs_link_20260927
st "download start rev $REV; free $(df -h / | tail -1 | awk '{print $4}')"
ok=0
for i in 1 2 3 4 5 6 7 8; do
  [ -e STOP_Q36_900 ] && exit 0
  $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.6-27B', revision='$REV', local_dir='$M', max_workers=8))" >> q36_900_download.log 2>&1 && { ok=1; break; }
  st "download attempt $i failed"; sleep 120
done
[ $ok = 1 ] || { st "download FAILED"; exit 1; }
st "download DONE $(du -sh $M | cut -f1); free $(df -h / | tail -1 | awk '{print $4}')"
$PY -c "import json; b=json.load(open('pilot_bodies_zh.json', encoding='utf-8')); json.dump(b[:1], open('q36_recheck_body.json', 'w', encoding='utf-8'), ensure_ascii=False)"
st "recheck start"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY crossmodel_thinking_b_b2e4.py --model $M --lang zh --off-kwargs '{"enable_thinking": false}' \
  --on-kwargs '{}' --bodies-zh q36_recheck_body.json --out thinking/qwen36_27b_recheck --max-new 3072 --batch 1 --max-memory-gib 90 \
  > thinking_qwen36_27b_recheck.log 2>&1 || { st "recheck run FAILED"; exit 1; }
$PY - > q36_recheck_compare.txt 2>&1 <<'EOF'
import glob, json
key = lambda d: (d['id'], d['role'], str(d['order']))
old = {key(d): d for d in (json.load(open(f, encoding='utf-8')) for f in glob.glob('thinking/qwen36_27b/items/*.json'))}
new = [json.load(open(f, encoding='utf-8')) for f in glob.glob('thinking/qwen36_27b_recheck/items/*.json')]
bad = [key(d) for d in new if any(d[k] != old[key(d)][k] for k in ('off', 'on_answer', 'on_letter', 'n_new', 'tail', 'finished'))]
print(len(new), 'rechecked;', 'IDENTICAL' if new and not bad else f'DIFFER {bad}')
EOF
grep -q IDENTICAL q36_recheck_compare.txt || { st "recheck DIFFERS, stop: $(cat q36_recheck_compare.txt)"; exit 1; }
st "recheck identical: $(cat q36_recheck_compare.txt)"
[ -e STOP_Q36_900 ] && exit 0
st "thinking 900 start ($(ls thinking/qwen36_27b/items | wc -l) items on disk)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY crossmodel_thinking_b_b2e4.py --model $M --lang zh --off-kwargs '{"enable_thinking": false}' \
  --on-kwargs '{}' --bodies-zh bodies_zh_150_list.json --out thinking/qwen36_27b --max-new 3072 --batch 1 --max-memory-gib 90 \
  > thinking_qwen36_27b_900.log 2>&1
st "thinking 900 exit $? ($(ls thinking/qwen36_27b/items | wc -l) items on disk)"
