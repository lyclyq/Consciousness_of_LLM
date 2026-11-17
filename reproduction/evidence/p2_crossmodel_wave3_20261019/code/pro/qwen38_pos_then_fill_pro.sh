#!/bin/bash
# PRO, user 10-28 ("qwen 27b 的位置重算还没做，我觉得也得排除位置的影响"): the position cells of Qwen3.8-27B's format ablation, then the
# ablation fill (ablation_fill_pro.sh, unchanged; its waiting copy in session afill was stopped with STOP_AFILL before it started any
# work). Geometry only, phase G, 150 bodies x 3 roles (crossmodel_suite_mask3.py cdd0691a, explicit position ids for the Assistant
# rendering; Qwen3.8 uses positions only in its 16 full-attention layers):
#   0. check: default non-thinking rendering with --pos-mode gap --pos-gap 0 (explicit positions equal to the rendered ones) must
#      reproduce suite_mask/qwen38_27b_default
#   1. non-thinking and thinking renderings: default + compress (empty think block visible, body at the User rendering's positions);
#      template switch + gap 4 (no think block, body four positions later, as in the default)
#   2. ablation_fill_pro.sh
# Waits until gptoss_mask_pro.sh has finished. Runs in tmux session qposf.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
Q=/root/hf_models/Qwen/Qwen3.8-27B
mkdir "$C/qposf.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> qwen38_pos_status.log; }
until [ "$(tail -1 gptoss_mask_status.log 2>/dev/null | awk '{print $3}')" = done ] && ! tmux has-session -t gmask 2>/dev/null; do sleep 120; done
st "start (suite_mask3 $(sha256sum crossmodel_suite_mask3.py | cut -c1-8))"
run() {  # name kwargs pos-mode gap
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1 \
    $PY crossmodel_suite_mask3.py --model $Q --tmpl-kwargs "$2" --bodies $B --problems problems.json --out suite_mask/qwen38_27b_$1 \
    --phases G --geometry-only --max-memory-gib 88 --lang zh --pos-mode $3 --pos-gap $4 > qwen38_pos_$1.log 2>&1
  st "$1 exit $? ($(ls suite_mask/qwen38_27b_$1/G/*.json 2>/dev/null | wc -l) bodies): $(grep -h -E 'content:|onset:' qwen38_pos_$1.log | cut -c1-110 | tr '\n' ' ')"
}
NT='{"enable_thinking": false}'; NTS='{"enable_thinking": false, "preserve_thinking": false}'; TS='{"preserve_thinking": false}'
run default_pos0 "$NT" gap 0
run compress "$NT" compress 0
run switch_gap4 "$NTS" gap 4
run think_compress '{}' compress 0
run think_switch_gap4 "$TS" gap 4
st "1 done; starting ablation_fill_pro.sh"
bash $C/ablation_fill_pro.sh
st "done"
