#!/bin/bash
# PRO, user 10-28 ("之后再测 OLMo-2-7B"): allenai/OLMo-2-1124-7B-Instruct, a candidate counterexample of the other kind: its template
# renders a history Assistant message <|assistant|>\n + body + <|endoftext|>, a User message <|user|>\n + body, i.e. the role word and an
# Assistant-only token after the body (as DeepSeek-R1-Distill-Llama-70B, whose response onset carries a strong index without
# self-reference). No thinking mode, so non-thinking only: suite phase G, geometry (body mean, response onset) and native letter
# probabilities P(you), English bodies (crossmodel_suite.py 516f117e, transformers 5.8).
#   0. waits until exaone12b_pro.sh (session exa) has finished and at least 17 GB are free; download (hf-mirror, about 15 GB)
#   1. suite G non-thinking
# Runs in tmux session olmo.
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/allenai/OLMo-2-1124-7B-Instruct
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/olmo.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> olmo2_status.log; }
until [ "$(tail -1 exaone12b_status.log 2>/dev/null | awk '{print $3}')" = done ] && ! tmux has-session -t exa 2>/dev/null; do sleep 120; done
free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "$free" -ge 17 ] || { st "0 only ${free}G free, need 17G: stop"; exit 1; }
st "0 download start; free ${free}G"
for i in 1 2 3 4; do
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('allenai/OLMo-2-1124-7B-Instruct', local_dir='$M', max_workers=8))" \
    >> olmo2_download.log 2>&1 && { st "0 download DONE $(du -sh $M | cut -f1); free $(df -h / | tail -1 | awk '{print $4}')"; break; }
  st "0 download attempt $i failed"; sleep 120
done
[ -e $M/config.json ] || { st "0 download FAILED"; exit 1; }
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1 \
  $PY crossmodel_suite.py --model $M --tmpl-kwargs '{}' --bodies $B --problems problems.json --out suite/olmo2_7b_en --phases G --max-memory-gib 40 \
  --lang en --bodies-en bodies_en_150.json > olmo2_G.log 2>&1
st "1 suite G exit $?"
st "done"
