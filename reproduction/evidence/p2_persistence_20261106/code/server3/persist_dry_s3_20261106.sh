#!/bin/bash
# Dry runs (CPU, tokenizer + template only) of every server3 setting of the persistence pilot, both orders.
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C
PY=/root/recursive_state_moe_server3_20260711/venv/bin/python
H=/root/data/hf_models; B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; S=persistence_readout_20261106.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=
one() {  # tag model kw lang [wrap]
  local tag=$1 m=$H/$2 kw=$3 lang=$4 wrap=${5:-}
  for order in main natural; do
    local a=(--model $m --tmpl-kwargs "$kw" --bodies $B --out /tmp/persist_dry --dry-run --order $order)
    [ $lang = en ] && a+=(--lang en --bodies-en bodies_en_150.json)
    [ $order = natural ] && { [ $lang = en ] && a+=(--request "Please write a short description.") || a+=(--request "请写一段简短的描述。"); }
    echo "== $tag $order"
    if [ -n "$wrap" ]; then $PY $wrap $m $S "${a[@]}" 2>&1 | grep -v Warning | tail -2; else $PY $S "${a[@]}" 2>&1 | grep -v Warning | tail -2; fi
  done
}
one qwen38_27b Qwen/Qwen3.8-27B '{"enable_thinking": false}' zh
one qwen38_27b_think Qwen/Qwen3.8-27B '{"enable_thinking": true}' zh
one exaone4_32b_en LGAI-EXAONE/EXAONE-4.0.1-32B '{"enable_thinking": false}' en
one gemma4_31b_en google/gemma-4-31B-it '{}' en
one glm46v_flash zai-org/GLM-4.6V-Flash '{"enable_thinking": false}' zh
one granite41_30b_en ibm-granite/granite-4.1-30b '{}' en
one phi4_en microsoft/phi-4 '{}' en
one seed_oss_36b ByteDance-Seed/Seed-OSS-36B-Instruct '{"thinking_budget": 0}' zh
one ernie45_21b_a3b_thinking baidu/ERNIE-4.5-21B-A3B-Thinking '{}' zh
one mimo7b_rl_0530 XiaomiMiMo/MiMo-7B-RL-0530 '{"enable_thinking": false}' zh
one moonlight16b_a3b_instruct moonshotai/Moonlight-16B-A3B-Instruct '{}' zh moonlight_compat_20261103.py
one r1_distill_llama70b deepseek-ai/DeepSeek-R1-Distill-Llama-70B '{}' zh
echo ALL_DRY_DONE
