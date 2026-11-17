#!/bin/bash
# v2 (user 10-28 "thinking no think 几何出来了吗"): as inducer_geometry_pro.sh, which it replaces before that started any work
# (stopped with STOP_IND while waiting), with the thinking rendering of Qwen3.8-27B first: default ({}) and the template switch
# ({"preserve_thinking": false}), content and response onset. Runs in tmux session q38ind2; lock q38ind2.lock.
# PRO, user 10-28 ("诱因补一个 gpt oss 20、120"): content geometry with the Assistant-only header tokens masked out of attention
# (crossmodel_suite_mask.py c8a29196 = crossmodel_suite.py 516f117e + --mask-extra; phase G, geometry only, 150 bodies x 3 roles,
# per-layer mean over the body tokens). Tokenizer check (check_mask_extra.py): Qwen3.8-27B masks <think> \n\n </think> \n\n,
# gpt-oss masks <|channel|> final; with preserve_thinking=false Qwen has nothing to mask.
#   1. calibration on Qwen3.8-27B, non-thinking rendering, same script: default; template switch preserve_thinking=false;
#      default with --mask-extra (does masking reproduce the switch?)
#   2. gpt-oss-20b (MXFP4) and gpt-oss-120b with --mask-extra; arguments of their geometry runs (gptoss_mxfp4_pro.sh,
#      gptoss120b_pro_v2.sh: tmpl-kwargs {}, English bodies, transformers 5.19 from tf_next), compared with
#      suite_think/gpt_oss_20b_mxfp4_en and suite_think/gpt_oss_120b_en
# Waits until the Qwen3.8 no-think runs (q38nf, q38geo) have finished. Runs in tmux session q38ind; skip with STOP_IND.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/q38ind2.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> inducer_geometry_status.log; }
stop() { [ -e STOP_IND ] && { st "stopped by STOP_IND"; exit 0; }; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
       PYTHONUNBUFFERED=1
until [ "$(tail -1 qwen38_nothink_full_status.log | awk '{print $3}')" = done ] && ! tmux has-session -t q38nf 2>/dev/null \
      && ! tmux has-session -t q38geo 2>/dev/null; do stop; sleep 120; done
st "start (suite_mask $(sha256sum crossmodel_suite_mask.py | cut -c1-8))"
Q=/root/hf_models/Qwen/Qwen3.8-27B
for v in think_default think_switch; do
  stop
  K='{}'; [ $v = think_switch ] && K='{"preserve_thinking": false}'
  $PY crossmodel_suite_mask.py --model $Q --tmpl-kwargs "$K" --bodies $B --problems problems.json --out suite_mask/qwen38_27b_$v \
    --phases G --geometry-only --max-memory-gib 88 --lang zh > inducer_qwen38_$v.log 2>&1
  st "0 qwen38 $v exit $? ($(ls suite_mask/qwen38_27b_$v/G/*.json 2>/dev/null | wc -l) bodies)"
done
for v in default switch mask; do
  stop
  K='{"enable_thinking": false}'; X=""
  [ $v = switch ] && K='{"enable_thinking": false, "preserve_thinking": false}'
  [ $v = mask ] && X="--mask-extra"
  $PY crossmodel_suite_mask.py --model $Q --tmpl-kwargs "$K" --bodies $B --problems problems.json --out suite_mask/qwen38_27b_$v \
    --phases G --geometry-only --max-memory-gib 88 --lang zh $X > inducer_qwen38_$v.log 2>&1
  st "1 qwen38 $v exit $? ($(ls suite_mask/qwen38_27b_$v/G/*.json 2>/dev/null | wc -l) bodies)"
done
# gpt-oss as in its geometry runs (gptoss_mxfp4_pro.sh): online through hf-mirror, so that transformers can resolve the MXFP4 kernel
# kernels-community/gpt-oss-triton-kernels (offline mode failed on 10-28: version 1 not in the local cache)
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONPATH=/root/p2_scale_20261015/tf_next HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
for m in 20b 120b; do
  stop
  if [ $m = 20b ]; then O=suite_mask/gpt_oss_20b_mxfp4_en_mask; G=30; else O=suite_mask/gpt_oss_120b_en_mask; G=90; fi
  $PY crossmodel_suite_mask.py --model /root/hf_models/openai/gpt-oss-$m --tmpl-kwargs '{}' --bodies $B --problems problems.json --out $O \
    --phases G --geometry-only --max-memory-gib $G --lang en --bodies-en bodies_en_150.json --mask-extra > inducer_gptoss$m.log 2>&1
  st "2 gpt-oss-$m mask exit $? ($(ls $O/G/*.json 2>/dev/null | wc -l) bodies)"
done
st "done"
