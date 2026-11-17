#!/bin/bash
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C/dsv41 || exit 1
PY=/root/data/venv_t210/bin/python; MD=/root/data/hf_models/deepseek-ai/DeepSeek-V4.1-Flash-1017
O=$C/words_along/dsv41_prep; mkdir -p $O
echo "start $(date '+%F %T')"
CUDA_VISIBLE_DEVICES= $PY test_dsv41_words_along_20261110.py $MD/inference > $O/tiny_cpu_test.log 2>&1; echo "tiny rc=$? $(tail -1 $O/tiny_cpu_test.log)"
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY dsv41_words_along_20261110.py --model-dir $MD --bodies /root/selfindex_three_actual_20261011_v2/role_bodies.jsonl --words $C/words_v2.json --dry-run > $O/dry.log 2>&1; echo "dry rc=$? $(grep -c 'DRY RUN OK' $O/dry.log) $(grep '^{' $O/dry.log | cut -c1-600)"
echo "end $(date '+%F %T')"
