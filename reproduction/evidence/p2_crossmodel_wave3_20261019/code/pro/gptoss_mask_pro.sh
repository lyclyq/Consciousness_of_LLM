#!/bin/bash
# PRO, user 10-28 ("gpt oss 只看了遮掩的几何，没有看遮掩的自我归属能力"; "false 模式作为遮掩组也算数，目的和作用都是消融"): the ablation of
# the own-message format tokens in gpt-oss-20b and gpt-oss-120b, the counterpart of Qwen3.8's template switch preserve_thinking=false.
# <|channel|>final of the Assistant rendering of X is masked out of attention; in generation the position ids follow the mask, so its
# position offset disappears too (crossmodel_role_swap_mask.py 7aaa2e11, from crossmodel_role_swap.py f2362716). Attribution question
# only (user 10-28), unedited conditions: base_A with the mask; base_U has nothing to mask and is copied from thinking_swap/
# gpt_oss_<m>_all (same prompts, machine and settings). Settings of the earlier gpt-oss attribution runs (gptoss_hdr_night_pro.sh).
#   1. smoke, gpt-oss-20b, first 4 held-out bodies x 2 orders; check: every item masks exactly the two tokens <|channel|> final and at
#      least 75% finish; then .GO for the full runs (otherwise .SKIP and stop)
#   2. geometry with the mask and mask-following positions (crossmodel_suite_mask2.py cdbd0b17 --mask-extra --mask-pos), 20b, 120b
#   3. full attribution: 20b, then 120b (the smoke items are copied in and skipped)
# Runs in tmux session gmask; skip with STOP_GMASK (between steps).
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/gmask.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss_mask_status.log; }
stop() { [ -e STOP_GMASK ] && { st "stopped by STOP_GMASK"; exit 0; }; }
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
opts() {
  if [ $1 = 120b ]; then
    echo "--model /root/hf_models/openai/gpt-oss-120b --fit-from thinking_ablate4/gpt_oss_120b_fit/fit.pt --band 0-35 --batch 8 --max-memory-gib 90"
  else
    echo "--model /root/hf_models/openai/gpt-oss-20b --fit-from thinking_ablate4/gpt_oss_20b_smoke/fit.pt --band 0-23 --batch 4 --max-memory-gib 30"
  fi
}
COMMON="--mode thinking --on-kwargs {} --lang en --bodies bodies_en_150_list.json --splits $B --max-new 3072 --save-gen --mask-extra"
# 1
st "start (role_swap_mask $(sha256sum crossmodel_role_swap_mask.py | cut -c1-8), suite_mask2 $(sha256sum crossmodel_suite_mask2.py | cut -c1-8))"
S=thinking_swap/gpt_oss_20b_mask_smoke
$PY crossmodel_role_swap_mask.py $COMMON $(opts 20b) --conds base_A:assistant:-:-:0 --limit-bodies 4 --out $S > gptoss20b_mask_smoke.log 2>&1
st "1 smoke exit $?"
$PY - > gptoss_mask_check.txt 2>&1 <<EOF
import glob, json
fs = sorted(glob.glob('$S/items/base_A__*.json'))
R = [json.load(open(f)) for f in fs]
toks = {tuple(r['masked_positions']) for r in R}
fin = sum(r.get('finished', False) for r in R)
ok = len(R) == 8 and all(len(r['masked_positions']) == 2 for r in R) and fin >= 6
print('items', len(R), 'masked positions', toks, 'finished', fin, 'answers', [r.get('on_answer') for r in R], 'PASS' if ok else 'FAIL')
EOF
st "1 check: $(tail -1 gptoss_mask_check.txt)"
grep -q PASS gptoss_mask_check.txt || { st "smoke check failed: stop"; exit 1; }
stop
# 2
for m in 20b 120b; do
  if [ $m = 20b ]; then O=suite_mask/gpt_oss_20b_mxfp4_en_maskpos; G=30; else O=suite_mask/gpt_oss_120b_en_maskpos; G=90; fi
  $PY crossmodel_suite_mask2.py --model /root/hf_models/openai/gpt-oss-$m --tmpl-kwargs '{}' --bodies $B --problems problems.json --out $O \
    --phases G --geometry-only --max-memory-gib $G --lang en --bodies-en bodies_en_150.json --mask-extra --mask-pos > gptoss${m}_maskpos_geo.log 2>&1
  st "2 geometry $m mask+pos exit $? ($(ls $O/G/*.json 2>/dev/null | wc -l) bodies)"
  stop
done
# 3
for m in 20b 120b; do
  O=thinking_swap/gpt_oss_${m}_mask
  mkdir -p $O/items && cp -n thinking_swap/gpt_oss_${m}_all/items/base_U__*.json $O/items/
  [ $m = 20b ] && cp -n $S/items/base_A__*.json $O/items/
  touch $O.GO
  st "3 attribution $m start ($(ls $O/items | wc -l) items on disk)"
  $PY crossmodel_role_swap_mask.py $COMMON $(opts $m) --conds base_A:assistant:-:-:0,base_U:user:-:-:0 --out $O > gptoss${m}_mask_attr.log 2>&1
  st "3 attribution $m exit $?"
  stop
done
st "done"
