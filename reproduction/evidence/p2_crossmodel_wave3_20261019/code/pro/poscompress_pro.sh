#!/bin/bash
# PRO, user 10-28 ("按照你说的，直出归属和所有的几何压缩行了"): the position-compression controls in the template's default rendering.
#   1. Qwen3.8-27B without thinking, attribution, P(you): default base_A and base_U on this machine, then base_A with the empty think
#      block at the position of the newline before it and every later token moved back by four (crossmodel_role_swap_pos.py 0b43df94
#      --pos-compress; the body has the User rendering's positions, the tokens stay visible). The fit file is loaded but not used
#      (unedited conditions only).
#   2. gpt-oss-20b (MXFP4) and gpt-oss-120b, geometry, default rendering with <|channel|>final compressed the same way
#      (crossmodel_suite_mask3.py 6e479c4b --pos-mode compress), against suite_think/gpt_oss_20b_mxfp4_en and suite_think/gpt_oss_120b_en
# (Qwen3.8's compressed geometry is suite_mask/qwen38_27b_compress and qwen38_27b_think_compress_v2.) No position increase and no
# thinking attribution under compression (user 10-28). Waits until gptoss120b_regions_windows_pro.sh (session g120l) has finished.
# Runs in tmux session pcomp.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/pcomp.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> poscompress_status.log; }
until [ "$(tail -1 gptoss120b_loc_status.log 2>/dev/null | awk '{print $3}')" = done ] && ! tmux has-session -t g120l 2>/dev/null; do sleep 120; done
st "start (role_swap_pos $(sha256sum crossmodel_role_swap_pos.py | cut -c1-8), suite_mask3 $(sha256sum crossmodel_suite_mask3.py | cut -c1-8))"
# 1
Q="--mode single --model /root/hf_models/Qwen/Qwen3.8-27B --on-kwargs {\"enable_thinking\":false} --lang zh --bodies bodies_zh_150_list.json --splits $B \
   --fit-from thinking_ablate4/qwen38_27b_sp_nothink_fit/fit.pt --band 0-63 --batch 16 --max-memory-gib 88"
for v in default compress; do
  O=singlepass_swap/qwen38_27b_pos_$v; touch $O.GO
  if [ $v = default ]; then X=""; CD="base_A:assistant:-:-:0,base_U:user:-:-:0"; else X="--pos-compress"; CD="base_A:assistant:-:-:0"; fi
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    $PY crossmodel_role_swap_pos.py $Q --conds $CD $X --out $O > poscompress_qwen38_$v.log 2>&1
  st "1 qwen38 single $v exit $?"
done
# 2
for m in 20b 120b; do
  if [ $m = 20b ]; then O=suite_mask/gpt_oss_20b_mxfp4_en_compress; G=30; else O=suite_mask/gpt_oss_120b_en_compress; G=90; fi
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1 $PY crossmodel_suite_mask3.py --model /root/hf_models/openai/gpt-oss-$m --tmpl-kwargs '{}' --bodies $B --problems problems.json \
    --out $O --phases G --geometry-only --max-memory-gib $G --lang en --bodies-en bodies_en_150.json --pos-mode compress > poscompress_gptoss$m.log 2>&1
  st "2 gpt-oss-$m compress exit $? ($(ls $O/G/*.json 2>/dev/null | wc -l) bodies): $(grep -h -E 'content:|onset:' poscompress_gptoss$m.log | cut -c1-110 | tr '\n' ' ')"
done
st "done"
