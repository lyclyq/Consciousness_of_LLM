#!/bin/bash
# PRO, user 10-28 ("pro 6000 跑行了"; "先看冒烟，先下载，看 v6 效应是否正常化"): Qwen3.8-27B with the official template switch
# preserve_thinking=false. By default (preserve_thinking undefined) the template renders every history Assistant message as
# <|im_start|>assistant\n<think>\n\n</think>\n\n + content, so the model's own X carries four format tokens that a User X lacks; with
# the switch off, Assistant messages before the last user query are rendered <|im_start|>assistant\n + content (template line 116),
# so the two renderings of X differ only in the role word. enable_thinking only sets the generation prompt, so the switch applies to
# both modes. Smoke only (the full runs are decided after it):
#   0. wait for download_qwen38_27b_pro.sh; weights checked: sha256 of every safetensors file = its etag, and every etag = the etag of
#      the server3 download (qwen38_27b_etags_server3.txt)
#   1. rendering check: the text around X and the end of the prompt, default and switch off, both modes (qwen38_nothink_render.txt);
#      tokenizer alignment of the attribution renderings (role_swap --check-align), both modes
#   2. fit on the new renderings (crossmodel_thinking_ablate2_chunked.py, the fit of the thinking runs: top-8.. eigenvectors of
#      D^T D, D = h^A - h^U at aligned body and post-body positions of the 75 discovery bodies, both orders), thinking and single pass
#   3. secret game v6, first 8 games, main and control, thinking then single pass: base_A base_U swU8_A swA8_U rswU8_A rswA8_U
#   4. attribution, first 4 held-out bodies x 2 orders, thinking then single pass, the same six conditions
#   5. bridge: the default rendering (switch not passed), unedited game, first 8 games, thinking (PRO against server3)
# Settings of the server3 Qwen runs (crossmodel_role_swap.py f2362716, crossmodel_games.py 5b29640d, every layer 0-63, k 8, up to
# 3072 new tokens); transformers 5.8.0 as there (no tf_next on the path). crossmodel_suite.py here is 516f117e, whose difference to
# server3's ad284bb6 is GLM-5.3 loading and hyper-connection geometry only. One model on the GPU at a time.
# Runs in tmux session q38nt; skip with STOP_Q38NT (checked between steps).
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/Qwen/Qwen3.8-27B
REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/q38nt.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> qwen38_nothink_status.log; }
stop() { [ -e STOP_Q38NT ] && { st "stopped by STOP_Q38NT"; exit 0; }; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
       PYTHONUNBUFFERED=1
KT='{"preserve_thinking": false}'
KS='{"enable_thinking": false, "preserve_thinking": false}'
C6="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
FT=thinking_ablate4/qwen38_27b_nothink_fit/fit.pt; FS=thinking_ablate4/qwen38_27b_sp_nothink_fit/fit.pt
MEM="--max-memory-gib 88"

# 0
until [ -e "$M/.download_done_$REV" ]; do
  stop; grep -q FAILED download_qwen38_27b_pro.log && { st "0 download FAILED"; exit 1; }; sleep 60
done
st "0 download done ($(du -sh $M | cut -f1)); checking sha256"
bad=0
for f in $M/model-*.safetensors; do
  n=$(basename $f); e=$(sed -n 2p $M/.cache/huggingface/download/$n.metadata); s=$(grep " $n\$" qwen38_27b_etags_server3.txt | cut -d' ' -f1)
  h=$(sha256sum $f | cut -d' ' -f1)
  [ "$h" = "$e" ] && [ "$e" = "$s" ] || { bad=1; st "0 MISMATCH $n sha $h etag $e server3 $s"; }
done
[ $bad = 0 ] || { st "0 weights differ from server3: stop"; exit 1; }
st "0 weights identical to server3 ($(ls $M/model-*.safetensors | wc -l) files, sha256 = etag = server3 etag)"
stop

# 1
$PY - > qwen38_nothink_render.txt 2>&1 <<EOF
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$M')
g = json.load(open('games/games_zh_secret6.json', encoding='utf-8'))[0]
for kw in ({}, {'preserve_thinking': False}, {'enable_thinking': False}, {'enable_thinking': False, 'preserve_thinking': False}):
    for role in ('assistant', 'user'):
        msgs = [dict(m) for m in g['msgs']]
        msgs[g['edit']]['role'] = role
        msgs.append({'role': 'user', 'content': g['q']['main']})
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        i = t.index(g['msgs'][g['edit']]['content'])
        print(json.dumps(kw), role, 'before X:', repr(t[max(0, i - 50):i]), '| prompt end:', repr(t[-40:]))
EOF
st "1 rendering: $(grep -c . qwen38_nothink_render.txt) lines in qwen38_nothink_render.txt"
for m in thinking single; do
  K=$KT; [ $m = single ] && K=$KS
  $PY crossmodel_role_swap.py --check-align --mode $m --model $M --on-kwargs "$K" --lang zh --bodies bodies_zh_150_list.json --splits $B \
    --fit-from none --band 0-63 --conds "$C6" --batch 8 $MEM --out thinking_swap/qwen38_27b_nothink_align_$m > qwen38_nothink_align_$m.log 2>&1
  st "1 alignment $m exit $?: $(grep -c 'min 1.000' qwen38_nothink_align_$m.log) pairs with full coverage of $(grep -c ' <- ' qwen38_nothink_align_$m.log)"
done
stop

# 2
for m in thinking single; do
  K=$KT; F=$FT; [ $m = single ] && { K=$KS; F=$FS; }
  if [ ! -s $F ]; then
    st "2 fit $m start"
    $PY crossmodel_thinking_ablate2_chunked.py --suite smoke --layers band --no-gen-edit --edit-end before_gen --main-layer 39 --band 0-63 \
      --limit-bodies 1 --conds base_A:assistant:-:0 --model $M --on-kwargs "$K" --lang zh --bodies bodies_zh_150_list.json --splits $B \
      --batch 16 --max-new 3072 $MEM --out $(dirname $F) > qwen38_nothink_fit_$m.log 2>&1
    st "2 fit $m exit $? ($(ls -la $F 2>/dev/null | awk '{print $5}') bytes)"
  fi
  [ -s $F ] || { st "2 fit $m FAILED"; exit 1; }
done
stop

# 3
for m in thinking single; do
  K=$KT; F=$FT; P=qwen38_27b_nothink; [ $m = single ] && { K=$KS; F=$FS; P=qwen38_27b_sp_nothink; }
  for q in main control; do
    stop
    $PY crossmodel_games.py --mode $m --model $M --on-kwargs "$K" --items games/games_zh_secret6.json --games secret --qkind $q --lang zh \
      --fit-from $F --band 0-63 --conds "$C6" --limit-bodies 8 --batch 8 --max-new 3072 $MEM --save-gen \
      --out games_swap/${P}_s6_smoke_$q > qwen38_nothink_g6_${m}_$q.log 2>&1
    st "3 game $m $q exit $?"
  done
done

# 4
for m in thinking single; do
  stop
  K=$KT; F=$FT; O=thinking_swap/qwen38_27b_nothink_smoke; X="--max-new 3072 --save-gen"
  [ $m = single ] && { K=$KS; F=$FS; O=singlepass_swap/qwen38_27b_nothink_smoke; X=""; }
  $PY crossmodel_role_swap.py --mode $m --model $M --on-kwargs "$K" --lang zh --bodies bodies_zh_150_list.json --splits $B --fit-from $F \
    --band 0-63 --conds "$C6" --limit-bodies 4 --batch 8 $MEM $X --out $O > qwen38_nothink_attr_$m.log 2>&1
  st "4 attribution $m exit $?"
done

# 5
stop
$PY crossmodel_games.py --mode thinking --model $M --on-kwargs '{}' --items games/games_zh_secret6.json --games secret --qkind main --lang zh \
  --fit-from $FT --band 0-63 --conds "base_A:assistant:-:-:0,base_U:user:-:-:0" --limit-bodies 8 --batch 8 --max-new 3072 $MEM --save-gen \
  --out games_swap/qwen38_27b_default_pro_s6_smoke_main > qwen38_nothink_bridge.log 2>&1
st "5 bridge exit $?"
st "done"
