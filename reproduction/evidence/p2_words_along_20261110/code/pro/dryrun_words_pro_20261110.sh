#!/bin/bash
# Tokenizer-only dry runs of words_along_20261110.py on PRO (11-10, ../../PROTOCOL_zh.md section 3): every PRO cell, no GPU.
# Models not on disk: tokenizer files only (pinned commit, hf-mirror, no xet) into /root/tok_words, for the render check and the
# table of pairs whose two words have different token counts. Output: words_along/dry/<set>_<order>_<tag>.log, dry_summary.jsonl.
# usage: bash dryrun_words_pro_20261110.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
H=/root/hf_models; T=/root/tok_words; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
B3=$C/phase3_bodies_zh.jsonl; E3=$C/phase3_bodies_en.json
S=words_along_20261110.py; WJ=$C/words_v2.json; O=$C/words_along; DR=$O/dry
mkdir -p $DR $T
REQ_ZH='请写一段简短的描述。'; REQ_EN='Please write a short description.'
SUM=$O/dry_summary.jsonl; : > $SUM

tokdl() {  # repo commit
  [ -e $T/$1/.tok_ok ] && return 0
  env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=$TF19 $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$1', revision='$2', local_dir='$T/$1', allow_patterns=['*.json', '*.txt', '*.model', '*.tiktoken', '*.jinja', '*.py'], ignore_patterns=['*.safetensors*', '*.bin', '*.pt', 'consolidated*'])" > $DR/tokdl_$(basename $1).log 2>&1 && touch $T/$1/.tok_ok
}

dry() {  # set order tag model kw lang mode pythonpath [extra]
  local st=$1 order=$2 tag=$3 m=$4 kw=$5 lang=$6 mode=$7 pp=$8 a lg; shift 8
  lg=$DR/${st}_${order}_$tag.log
  local bodies=$B1 ben=bodies_en_150.json
  [ $st = p3 ] && { bodies=$B3; ben=$E3; }
  a=(--model $m --tmpl-kwargs "$kw" --bodies $bodies --words $WJ --out $DR/x --order $order --dry-run)
  [ $lang = en ] && a+=(--lang en --bodies-en $ben)
  if [ $order = natural ]; then if [ $lang = en ]; then a+=(--request "$REQ_EN"); else a+=(--request "$REQ_ZH"); fi; fi
  case $mode in roles) a+=(--roles --pairs all) ;; new) a+=(--pairs new) ;; all) a+=(--pairs all) ;; esac
  a+=("$@")
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$pp $PY $S "${a[@]}" > $lg 2>&1 < /dev/null
  if grep -q 'DRY RUN OK' $lg; then
    $PY -c "import json,sys; d=json.loads([l for l in open(sys.argv[1]) if l.startswith('{')][-1]); d.update(set=sys.argv[2], tag=sys.argv[3]); print(json.dumps(d, ensure_ascii=False))" $lg $st $tag >> $SUM
    echo "OK   $st $order $tag"
  else echo "FAIL $st $order $tag (see $lg)"; fi
}

NT='{"enable_thinking": false}'; Q38T='{"enable_thinking": true}'
D19='--fixed-date 2026-10-19'; D03='--fixed-date 2026-11-03'
tokdl swiss-ai/Apertus-8B-Instruct-2509 b946d40447b2b597999b9c86d44bee0b452c919f
tokdl XiaomiMiMo/MiMo-VL-7B-RL-2508 ec4ebdc29b83b94aa7f62670c78376250dad8dbc
tokdl allenai/OLMo-2-1124-7B-Instruct 470b1fba1ae01581f270116362ee4aa1b97f4c84
tokdl allenai/OLMo-2-1124-13B-Instruct 3a5c85baefbb1896a54d56fe2e76c0395627ddf4
tokdl Qwen/Qwen3.6-27B 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
tokdl mistralai/Ministral-3-14B-Instruct-2512-BF16 3cea74c1ebaf5ce5f5a2553de470e2ceab825142
tokdl mistralai/Ministral-3-14B-Reasoning-2512 51f9210f3cd20f3452a80d5819d15dc61cc50630
tokdl Qwen/Qwen3.8-Flash-Next-FP8 236dfdf285828023ca3bcd3f37366c58a3469b13
dry b150 main qwen38_27b_think $H/Qwen/Qwen3.8-27B "$Q38T" zh new ''
dry b150 main gpt_oss_120b_en $H/openai/gpt-oss-120b '{}' en new $TF19 $D19
dry b150 main mimo_v26_distill_qwen9b $H/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B "$NT" zh new ''
dry b150 main mimo_vl7b_rl_2508 $T/XiaomiMiMo/MiMo-VL-7B-RL-2508 '{}' zh new ''
for o in main natural; do   # natural order: the paper's natural-order appendix only
  m=$([ $o = main ] && echo new || echo all)
  dry b150 $o gpt_oss_20b_en $H/openai/gpt-oss-20b '{}' en $m $TF19 $D19
  dry b150 $o apertus8b_en $T/swiss-ai/Apertus-8B-Instruct-2509 '{}' en $m '' $D03
  dry b150 $o olmo2_7b_en $T/allenai/OLMo-2-1124-7B-Instruct '{}' en $m ''
done
dry b150 main olmo2_13b_en $T/allenai/OLMo-2-1124-13B-Instruct '{}' en roles ''
dry b150 main qwen36_27b $T/Qwen/Qwen3.6-27B "$NT" zh roles ''
dry b150 main qwen38_flashnext $T/Qwen/Qwen3.8-Flash-Next-FP8 "$NT" zh roles $TF19
dry b150 natural ministral3_14b_instruct_en $T/mistralai/Ministral-3-14B-Instruct-2512-BF16 '{}' en roles $TF19
dry b150 natural ministral3_14b_reasoning_en $T/mistralai/Ministral-3-14B-Reasoning-2512 '{}' en roles $TF19
dry p3 main qwen38_27b $H/Qwen/Qwen3.8-27B "$NT" zh all ''
dry p3 main qwen38_27b_think $H/Qwen/Qwen3.8-27B "$Q38T" zh all ''
dry p3 main gpt_oss_20b_en $H/openai/gpt-oss-20b '{}' en all $TF19 $D19
dry p3 main gpt_oss_120b_en $H/openai/gpt-oss-120b '{}' en all $TF19 $D19
dry p3 main mimo_v26_distill_qwen9b $H/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B "$NT" zh all ''
dry p3 main qwen36_27b $T/Qwen/Qwen3.6-27B "$NT" zh roles ''
echo "dry runs finished: $(grep -c . $SUM) OK"
