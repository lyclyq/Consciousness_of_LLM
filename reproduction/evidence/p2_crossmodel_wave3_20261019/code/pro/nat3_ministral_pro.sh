#!/bin/bash
# Natural-order calibration for Mistral on PRO (user 10-20; results go to an appendix, with the reason). Waits until the
# Ministral download has finished and every file passed its sha256 check, then runs crossmodel_natural3.py one model at a
# time on the single GPU (transformers 5.19 overlay, as for gpt-oss): Instruct with geometry and single-pass attribution,
# Reasoning with geometry only (its default system prompt makes it reason first, so a single-pass letter readout does not
# apply). English bodies, the template's default call. Skip with STOP_NAT3_MISTRAL. Status in nat3_ministral_status.log.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/mistralai
mkdir "$C/nat3_ministral_pro.lock" 2>/dev/null || exit 0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $C
st() { echo "$(date '+%F %T') $1" >> nat3_ministral_status.log; }
until grep -q DONE download_ministral_pro.log 2>/dev/null; do [ -e STOP_NAT3_MISTRAL ] && exit 0; sleep 120; done
for v in verify_Ministral-3-14B-Instruct-2512-BF16.txt verify_Ministral-3-14B-Reasoning-2512.txt; do
  if grep -q '^BAD' $v || ! grep -q '^END' $v; then st "verification failed: $v"; exit 1; fi
done
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
[ -e STOP_NAT3_MISTRAL ] && exit 0
st "instruct start"
$PY crossmodel_natural3.py --model $M/Ministral-3-14B-Instruct-2512-BF16 --tmpl-kwargs '{}' --bodies $B --lang en --bodies-en bodies_en_150.json --out suite_nat3/ministral3_14b_instruct_en --max-memory-gib 90 > nat3_ministral_instruct.log 2>&1
st "instruct exit $?"
[ -e STOP_NAT3_MISTRAL ] && exit 0
st "reasoning start"
$PY crossmodel_natural3.py --model $M/Ministral-3-14B-Reasoning-2512 --tmpl-kwargs '{}' --bodies $B --lang en --bodies-en bodies_en_150.json --out suite_nat3/ministral3_14b_reasoning_en --geometry-only --max-memory-gib 90 > nat3_ministral_reasoning.log 2>&1
st "reasoning exit $?"
