#!/bin/bash
# user 10-21: own-step trust WITH thinking (crossmodel_thinking_m.py) for the models whose self-reference appears under thinking
# and whose path is PRO, then the Qwen3.6-27B thinking top-up; one GPU job after another. Replaces q36_27b_think900_pro.sh
# (stopped with STOP_Q36_900 while it was still waiting; its steps are repeated here unchanged).
#   0. wait for THINKM_PRO_GO (set after the server3 smoke of crossmodel_thinking_m.py) and for the Flash-Next thinking 900 to exit
#   1. Qwen3.8-Flash-Next thinking M: official FP8, the arguments of its thinking runs (batch 40, 70 GiB, 66 GiB CPU offload)
#   2. gpt-oss-20b thinking M: native MXFP4, English, the arguments of its 900-item run (batch 16, 30 GiB)
#   -- the user deletes Flash-Next after step 1 --
#   3. gpt-oss-120b: download b5c939de (hub head since 2025-08-26, so the files of 10-19), thinking M (batch 16, 90 GiB)
#   4. Qwen3.6-27B: download 6a9e13bd, recheck 6 of its 180 items, then the other 720 (as q36_27b_think900_pro.sh)
# Skip with STOP_PRO_THINKM. Status in pro_thinkm_status.log.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python
FN=/root/hf_models/Qwen/Qwen3.8-Flash-Next-FP8; G20=/root/hf_models/openai/gpt-oss-20b; G120=/root/hf_models/openai/gpt-oss-120b
Q36=/root/hf_models/Qwen/Qwen3.6-27B; R120=b5c939de8f754692c1647ca79fbf85e8c1e70f8a; R36=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
mkdir "$C/pro_thinkm.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> pro_thinkm_status.log; }
stop() { [ -e STOP_PRO_THINKM ] && { st "stopped by STOP_PRO_THINKM"; exit 0; }; }
free_gb() { df -BG / | tail -1 | awk '{print $4}' | tr -d G; }
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/root/p2_scale_20261015/hf_home_dl \
       PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
dl() {  # repo rev dir
  for i in 1 2 3 4 5 6 7 8; do stop
    $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$1', revision='$2', local_dir='$3', max_workers=8, ignore_patterns=['original/*', 'metal/*']))" >> pro_thinkm_download.log 2>&1 && return 0
    st "download $1 attempt $i failed"; sleep 120
  done; return 1; }
st "waiting for THINKM_PRO_GO and the Flash-Next thinking 900"
until [ -e THINKM_PRO_GO ] && grep -q 'thinking 900 exit' flashnext_think900_status.log && ! pgrep -f '[t]hinking/flashnext' > /dev/null; do stop; sleep 300; done
st "1 Flash-Next thinking M start"
$PY crossmodel_thinking_m.py --model $FN --on-kwargs '{}' --problems problems.json --lang zh --out thinking_m/flashnext \
  --batch 40 --max-new 3072 --max-memory-gib 70 --cpu-offload-gib 66 --save-gen > thinking_m_flashnext.log 2>&1
st "1 Flash-Next thinking M exit $? ($(ls thinking_m/flashnext/items 2>/dev/null | wc -l) items)"; stop
st "2 gpt-oss-20b thinking M start"
$PY crossmodel_thinking_m.py --model $G20 --on-kwargs '{}' --problems problems_en_120.json --lang en --out thinking_m/gpt_oss_20b_mxfp4_en \
  --batch 16 --max-new 3072 --max-memory-gib 30 --save-gen > thinking_m_gptoss20b.log 2>&1
st "2 gpt-oss-20b thinking M exit $? ($(ls thinking_m/gpt_oss_20b_mxfp4_en/items 2>/dev/null | wc -l) items)"; stop
st "3 waiting for >= 150 GB free (the user deletes $FN)"
until [ "$(free_gb)" -ge 150 ]; do stop; sleep 300; done
st "3 download gpt-oss-120b rev $R120; free $(free_gb) GB"
dl openai/gpt-oss-120b $R120 $G120 || { st "3 download FAILED"; exit 1; }
st "3 download DONE $(du -sh $G120 | cut -f1); gpt-oss-120b thinking M start"
$PY crossmodel_thinking_m.py --model $G120 --on-kwargs '{}' --problems problems_en_120.json --lang en --out thinking_m/gpt_oss_120b_en \
  --batch 16 --max-new 3072 --max-memory-gib 90 --save-gen > thinking_m_gptoss120b.log 2>&1
st "3 gpt-oss-120b thinking M exit $? ($(ls thinking_m/gpt_oss_120b_en/items 2>/dev/null | wc -l) items)"; stop
[ -L $Q36 ] && mv $Q36 $Q36.nfs_link_20260927
st "4 download Qwen3.6-27B rev $R36; free $(free_gb) GB"
dl Qwen/Qwen3.6-27B $R36 $Q36 || { st "4 download FAILED"; exit 1; }
st "4 download DONE $(du -sh $Q36 | cut -f1)"
$PY -c "import json; b=json.load(open('pilot_bodies_zh.json', encoding='utf-8')); json.dump(b[:1], open('q36_recheck_body.json', 'w', encoding='utf-8'), ensure_ascii=False)"
$PY crossmodel_thinking_b_b2e4.py --model $Q36 --lang zh --off-kwargs '{"enable_thinking": false}' --on-kwargs '{}' \
  --bodies-zh q36_recheck_body.json --out thinking/qwen36_27b_recheck --max-new 3072 --batch 1 --max-memory-gib 90 \
  > thinking_qwen36_27b_recheck.log 2>&1 || { st "4 recheck run FAILED"; exit 1; }
$PY - > q36_recheck_compare.txt 2>&1 <<'EOF'
import glob, json
key = lambda d: (d['id'], d['role'], str(d['order']))
old = {key(d): d for d in (json.load(open(f, encoding='utf-8')) for f in glob.glob('thinking/qwen36_27b/items/*.json'))}
new = [json.load(open(f, encoding='utf-8')) for f in glob.glob('thinking/qwen36_27b_recheck/items/*.json')]
bad = [key(d) for d in new if any(d[k] != old[key(d)][k] for k in ('off', 'on_answer', 'on_letter', 'n_new', 'tail', 'finished'))]
print(len(new), 'rechecked;', 'IDENTICAL' if new and not bad else f'DIFFER {bad}')
EOF
grep -q IDENTICAL q36_recheck_compare.txt || { st "4 recheck DIFFERS, stop: $(cat q36_recheck_compare.txt)"; exit 1; }
st "4 recheck identical: $(cat q36_recheck_compare.txt)"; stop
st "4 Qwen3.6-27B thinking 900 start ($(ls thinking/qwen36_27b/items | wc -l) items on disk)"
$PY crossmodel_thinking_b_b2e4.py --model $Q36 --lang zh --off-kwargs '{"enable_thinking": false}' --on-kwargs '{}' \
  --bodies-zh bodies_zh_150_list.json --out thinking/qwen36_27b --max-new 3072 --batch 1 --max-memory-gib 90 \
  > thinking_qwen36_27b_900.log 2>&1
st "4 Qwen3.6-27B thinking 900 exit $? ($(ls thinking/qwen36_27b/items | wc -l) items on disk)"
