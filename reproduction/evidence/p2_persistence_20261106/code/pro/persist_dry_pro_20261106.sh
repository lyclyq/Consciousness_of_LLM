#!/bin/bash
# Dry runs (CPU, tokenizer + template only) of the PRO settings of the persistence pilot, both orders.
C=/root/p2_scale_20261015/crossmodel; cd $C
PY=/root/consciousness_a6000pro/.venv/bin/python
H=/root/hf_models; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; S=persistence_readout_20261106.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=
one() {  # tag model kw lang [pythonpath]
  local tag=$1 m=$H/$2 kw=$3 lang=$4 pp=${5:-}
  for order in main natural; do
    local a=(--model $m --tmpl-kwargs "$kw" --bodies $B --out /tmp/persist_dry --dry-run --order $order)
    [ $lang = en ] && a+=(--lang en --bodies-en bodies_en_150.json)
    [ $order = natural ] && { [ $lang = en ] && a+=(--request "Please write a short description.") || a+=(--request "请写一段简短的描述。"); }
    echo "== $tag $order"
    PYTHONPATH=$pp $PY $S "${a[@]}" 2>&1 | grep -v Warning | tail -2
  done
}
one gpt_oss_20b_en openai/gpt-oss-20b '{}' en /root/p2_scale_20261015/tf_next
one apertus8b_en swiss-ai/Apertus-8B-Instruct-2509 '{}' en
one mimo_v26_distill_qwen9b XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B '{"enable_thinking": false}' zh
one mimo_vl7b_rl_2508 XiaomiMiMo/MiMo-VL-7B-RL-2508 '{}' zh
echo ALL_DRY_DONE
