#!/bin/bash
# retry of three tokenizer downloads (hf-mirror 521 on the first try) and their dry runs; sourced pieces of dryrun_words_s3_20261110.sh
set -u
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C || exit 1
PY=/root/recursive_state_moe_server3_20260711/venv/bin/python; T=/root/data/tok_words; B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
O=$C/words_along; DR=$O/dry; SUM=$O/dry_summary.jsonl; NT='{"enable_thinking": false}'
for spec in "Qwen/Qwen3.6-35B-A3B 995ad96eacd98c81ed38be0c5b274b04031597b0 qwen36_35b_a3b zh" "zai-org/GLM-4.7-Flash 7dd20894a642a0aa287e9827cb1a1f7f91386b67 glm47_flash zh" "ibm-granite/granite-4.2-30b 9e668ce1c538387ef24d3644e9b0606647762636 granite42_30b_en en"; do
  set -- $spec
  for i in 1 2 3 4 5; do
    [ -e $T/$1/.tok_ok ] && break
    env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$1', revision='$2', local_dir='$T/$1', allow_patterns=['*.json', '*.txt', '*.model', '*.tiktoken', '*.jinja', '*.py'], ignore_patterns=['*.safetensors*', '*.bin', '*.pt'])" > $DR/tokdl_retry_$3.log 2>&1 && touch $T/$1/.tok_ok || sleep 60
  done
  e=(); [ $4 = en ] && e=(--lang en --bodies-en bodies_en_150.json)
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY words_along_20261110.py --model $T/$1 --tmpl-kwargs "$NT" --bodies $B1 --words words_v2.json --out $DR/x --order main --roles --pairs all "${e[@]}" --dry-run > $DR/b150_main_$3.log 2>&1 < /dev/null
  if grep -q 'DRY RUN OK' $DR/b150_main_$3.log; then
    $PY -c "import json,sys; d=json.loads([l for l in open(sys.argv[1]) if l.startswith('{')][-1]); d.update(set='b150', tag=sys.argv[2]); print(json.dumps(d, ensure_ascii=False))" $DR/b150_main_$3.log $3 >> $SUM; echo "OK   b150 main $3"
  else echo "FAIL b150 main $3"; fi
done
echo "retry finished"
