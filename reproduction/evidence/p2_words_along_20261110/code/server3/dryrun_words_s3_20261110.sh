#!/bin/bash
# Tokenizer-only dry runs of words_along_20261110.py on server3 (11-10, ../../PROTOCOL_zh.md section 3): every server3 cell, no GPU.
# Models not on disk: tokenizer files only (pinned commit, hf-mirror, no xet) into tok_words/, for the render check and the table
# of pairs whose two words have different token counts. Output: words_along/dry/<set>_<order>_<tag>.log and dry_summary.jsonl.
# usage: bash dryrun_words_s3_20261110.sh
set -u
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C || exit 1
PY=/root/recursive_state_moe_server3_20260711/venv/bin/python
H=/root/data/hf_models; T=/root/data/tok_words; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
S=words_along_20261110.py; WJ=$C/words_v2.json; O=$C/words_along; DR=$O/dry
mkdir -p $DR $T
REQ_ZH='请写一段简短的描述。'; REQ_EN='Please write a short description.'
SUM=$O/dry_summary.jsonl; : > $SUM

tokdl() {  # repo commit
  [ -e $T/$1/.tok_ok ] && return 0
  env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$1', revision='$2', local_dir='$T/$1', allow_patterns=['*.json', '*.txt', '*.model', '*.tiktoken', '*.jinja', 'tokeni*.py', '*.py'], ignore_patterns=['*.safetensors*', '*.bin', '*.pt'])" > $DR/tokdl_$(basename $1).log 2>&1 && touch $T/$1/.tok_ok
}

dry() {  # set order tag model kw lang mode [wrapper]
  local st=$1 order=$2 tag=$3 m=$4 kw=$5 lang=$6 mode=$7 wrap=${8:-} a lg
  lg=$DR/${st}_${order}_$tag.log
  a=(--model $m --tmpl-kwargs "$kw" --bodies $B1 --words $WJ --out $DR/x --order $order --dry-run)
  [ $lang = en ] && a+=(--lang en --bodies-en bodies_en_150.json)
  if [ $order = natural ]; then if [ $lang = en ]; then a+=(--request "$REQ_EN"); else a+=(--request "$REQ_ZH"); fi; fi
  case $mode in roles) a+=(--roles --pairs all) ;; new) a+=(--pairs new) ;; all) a+=(--pairs all) ;; esac
  if [ -n "$wrap" ]; then HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY $wrap $m $S "${a[@]}" > $lg 2>&1 < /dev/null
  else HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY $S "${a[@]}" > $lg 2>&1 < /dev/null; fi
  if grep -q 'DRY RUN OK' $lg; then
    $PY -c "import json,sys; d=json.loads([l for l in open(sys.argv[1]) if l.startswith('{')][-1]); d.update(set=sys.argv[2], tag=sys.argv[3]); print(json.dumps(d, ensure_ascii=False))" $lg $st $tag >> $SUM
    echo "OK   $st $order $tag"
  else echo "FAIL $st $order $tag (see $lg)"; fi
}

NT='{"enable_thinking": false}'
for o in main natural; do
  m=$([ $o = main ] && echo new || echo all)
  dry b150 $o qwen38_27b $H/Qwen/Qwen3.8-27B "$NT" zh $m
  dry b150 $o exaone4_32b_en $H/LGAI-EXAONE/EXAONE-4.0.1-32B "$NT" en $m
  dry b150 $o granite41_30b_en $H/ibm-granite/granite-4.1-30b '{}' en $m
  dry b150 $o seed_oss_36b $H/ByteDance-Seed/Seed-OSS-36B-Instruct '{"thinking_budget": 0}' zh $m
  dry b150 $o gemma4_31b_en $H/google/gemma-4-31B-it '{}' en $m
  dry b150 $o glm46v_flash $H/zai-org/GLM-4.6V-Flash "$NT" zh $m
  dry b150 $o phi4_en $H/microsoft/phi-4 '{}' en $m
  dry b150 $o ernie45_21b_a3b_thinking $H/baidu/ERNIE-4.5-21B-A3B-Thinking '{}' zh $m
  dry b150 $o mimo7b_rl_0530 $H/XiaomiMiMo/MiMo-7B-RL-0530 "$NT" zh $m
  dry b150 $o moonlight16b_a3b_instruct $H/moonshotai/Moonlight-16B-A3B-Instruct '{}' zh $m moonlight_compat_20261103.py
  dry b150 $o r1_distill_llama70b $H/deepseek-ai/DeepSeek-R1-Distill-Llama-70B '{}' zh $m
done
dry b150 main glm53_flash $H/zai-org/GLM-5.3-Flash '{}' zh roles
tokdl Qwen/Qwen3.6-35B-A3B 995ad96eacd98c81ed38be0c5b274b04031597b0 && dry b150 main qwen36_35b_a3b $T/Qwen/Qwen3.6-35B-A3B "$NT" zh roles
tokdl zai-org/GLM-4.7-Flash 7dd20894a642a0aa287e9827cb1a1f7f91386b67 && dry b150 main glm47_flash $T/zai-org/GLM-4.7-Flash "$NT" zh roles
tokdl ibm-granite/granite-4.2-30b 9e668ce1c538387ef24d3644e9b0606647762636 && dry b150 main granite42_30b_en $T/ibm-granite/granite-4.2-30b "$NT" en roles
tokdl Qwen/Qwen3.5-27B fc05daec18b0a78c049392ed2e771dde82bdf654 && dry b150 main qwen35_27b $T/Qwen/Qwen3.5-27B "$NT" zh roles
echo "dry runs finished: $(grep -c . $SUM) OK"
