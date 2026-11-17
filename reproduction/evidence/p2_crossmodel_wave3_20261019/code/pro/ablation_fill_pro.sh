#!/bin/bash
# PRO, user 10-28 ("这两个类似遮掩模式应该跑完 gpt oss 20b、120b thinking，qwen 3.8 直出、thinking 的自我归属题的几何和干预"; "2 x 2 的模型
# 默认模式加这个模式就很干净"): the cells of the ablation column that are still missing, attribution question only.
#   1. Qwen3.8-27B without thinking, template switch preserve_thinking=false: the ten conditions of the other readouts
#      (crossmodel_role_swap.py f2362716, fit thinking_ablate4/qwen38_27b_sp_nothink_fit, every layer); its unedited own message is
#      claimed with P(you) 0.10 (below the release line 0.5), so removal has little room; run for the complete table (user 10-28)
#   2. gpt-oss-20b, then gpt-oss-120b, <|channel|>final masked: the fit under the mask (crossmodel_thinking_ablate2_mask.py c69600d7
#      --mask-extra; settings of the original fits: 20b main layer 15, band 0-23; 120b main layer 23, band 0-35), then removal,
#      installation and their random controls (crossmodel_role_swap_mask.py 7aaa2e11 --mask-extra) beside the unedited conditions of
#      gptoss_mask_pro.sh in thinking_swap/gpt_oss_<m>_mask
# Waits until gptoss_mask_pro.sh has finished. Runs in tmux session afill; skip with STOP_AFILL (between steps).
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/afill.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> ablation_fill_status.log; }
stop() { [ -e STOP_AFILL ] && { st "stopped by STOP_AFILL"; exit 0; }; }
until [ "$(tail -1 gptoss_mask_status.log 2>/dev/null | awk '{print $3}')" = done ] && ! tmux has-session -t gmask 2>/dev/null; do stop; sleep 120; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
ALL="base_A:assistant:-:-:0,base_U:user:-:-:0,base_S:system:-:-:0,idA8_A:assistant:assistant:V:8,swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8,rswA8_U:user:assistant:R:8,swA8_S:system:assistant:V:8"
E4="swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
# 1
st "1 qwen38 single no-think start"
O=singlepass_swap/qwen38_27b_nothink_all; touch $O.GO
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY crossmodel_role_swap.py --mode single --model /root/hf_models/Qwen/Qwen3.8-27B \
  --on-kwargs '{"enable_thinking": false, "preserve_thinking": false}' --lang zh --bodies bodies_zh_150_list.json --splits $B \
  --fit-from thinking_ablate4/qwen38_27b_sp_nothink_fit/fit.pt --band 0-63 --conds "$ALL" --batch 16 --max-memory-gib 88 --out $O \
  > ablation_fill_qwen38_single.log 2>&1
st "1 qwen38 single no-think exit $?"
stop
# 2
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next
for m in 20b 120b; do
  if [ $m = 120b ]; then ML=23; BAND=0-35; BT=8; G=90; else ML=15; BAND=0-23; BT=4; G=30; fi
  M=/root/hf_models/openai/gpt-oss-$m; F=thinking_ablate4/gpt_oss_${m}_mask_fit
  if [ ! -s $F/fit.pt ]; then
    $PY crossmodel_thinking_ablate2_mask.py --suite smoke --layers band --no-gen-edit --edit-end before_gen --main-layer $ML --band $BAND \
      --limit-bodies 1 --conds base_A:assistant:-:0 --model $M --on-kwargs '{}' --lang en --bodies bodies_en_150_list.json --splits $B \
      --batch $BT --max-new 3072 --max-memory-gib $G --mask-extra --out $F > ablation_fill_gptoss${m}_fit.log 2>&1
    st "2 gpt-oss-$m mask fit exit $? ($(ls -la $F/fit.pt 2>/dev/null | awk '{print $5}') bytes)"
  fi
  [ -s $F/fit.pt ] || { st "2 gpt-oss-$m mask fit FAILED"; continue; }
  stop
  O=thinking_swap/gpt_oss_${m}_mask; touch $O.GO
  st "2 gpt-oss-$m mask 2x2 start ($(ls $O/items | wc -l) items on disk)"
  $PY crossmodel_role_swap_mask.py --mode thinking --model $M --fit-from $F/fit.pt --band $BAND --batch $BT --max-memory-gib $G \
    --on-kwargs '{}' --lang en --bodies bodies_en_150_list.json --splits $B --max-new 3072 --save-gen --mask-extra --conds "$E4" --out $O \
    > ablation_fill_gptoss${m}_2x2.log 2>&1
  st "2 gpt-oss-$m mask 2x2 exit $?"
  stop
done
st "done"
