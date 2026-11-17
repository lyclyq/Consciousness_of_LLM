#!/bin/bash
# P2 ordinary words in next-token prediction, depth part on PRO (11-11, user: "机制组 KL 可以补 200 毕竟没有额外成本"): the four
# mechanism settings whose weights are on PRO, on the 200 held-out phase-3 bodies (kl_words_p3_20261111.py). Each setting uses the
# command of its 75-body run in kl_words_pro_20261111.sh (model, template kwargs, language, fixed date, PYTHONPATH, network, memory)
# and is checked against its paper ctrl run. gpt-oss-120b and Qwen3.6-27B (the depth no-index contrast) need downloads and wait for
# PRO disk with the other large models.
#   SMOKE  1 phase-3 body each for MiMo-V2.6 (zh) and gpt-oss-20b (en) into kl/_smoke_p3 (check on the 2 held-out bodies first)
#   DEPTH  waits until "lane ext RUN done" in queue.log (one task on the GPU at a time), then runs the four settings
# Stop: touch words_along/STOP_KLW (checked before every body and task); no signals.
# usage (tmux session klw, window DEPTH, in the crossmodel directory): bash kl_words_p3_pro_20261111.sh SMOKE|DEPTH
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; B3=$C/phase3_bodies_zh.jsonl; E3=$C/phase3_bodies_en.json
WJ=$C/words_v2.json; MC=$C/modeswitch/ctrl
O=$C/words_along/kl; STOP=$C/words_along/STOP_KLW
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
LANE=${1:?lane SMOKE or DEPTH}
log() { echo "[klw p3 $LANE] $(date '+%F %T') $*" | tee -a $O/queue.log; }

run() {  # tag(= paper ctrl run) out model kw lang net pythonpath mem [extra args]
  local tag=$1 out=$2 m=$3 kw=$4 lang=$5 net=$6 pp=$7 mem=$8; shift 8
  local lg=$out.log a
  mkdir -p $(dirname $out)
  [ -e $out/DONE ] && { log "already DONE: $out"; return 0; }
  [ -e $STOP ] && { log "stop flag: not starting $out"; exit 2; }
  nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: lane stops before $out"; exit 3; }
  a=(--model $m --tmpl-kwargs "$kw" --bodies $B1 --p3-bodies $B3 --words $WJ --out $out --check-ctrl $MC/$tag --lang $lang
     --max-memory-gib $mem --stop-file $STOP "$@")
  [ $lang = en ] && a+=(--bodies-en bodies_en_150.json --p3-bodies-en $E3)
  log "start $out"
  if [ $net = 1 ]; then
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 \
      PYTHONPATH=$pp $PY kl_words_p3_20261111.py "${a[@]}" >> $lg 2>&1 < /dev/null
  else
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY kl_words_p3_20261111.py "${a[@]}" >> $lg 2>&1 < /dev/null
  fi
  if [ -e $out/DONE ]; then log "DONE $out: $(grep 'summary (content)' $lg | tail -1)"
  else log "ended $out: $(grep -a -E 'check:|CHECK FAILED|Error|stop|smoke' $lg | tail -2 | tr '\n' ' ')"; fi
}

NT='{"enable_thinking": false}'
case $LANE in
  SMOKE)
    run mimo_v26_nt $O/_smoke_p3/mimo_v26_nt /root/hf_models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B "$NT" zh 0 '' 44 --max-bodies 1
    run gpt_oss_20b_en $O/_smoke_p3/gpt_oss_20b_en /root/hf_models/openai/gpt-oss-20b '{}' en 1 $TF19 44 --fixed-date 2026-10-19 --max-bodies 1
    log "lane SMOKE done" ;;
  DEPTH)
    until grep -q 'lane ext RUN done' $O/queue.log 2>/dev/null; do [ -e $STOP ] && exit 2; sleep 60; done
    log "the extension is finished: depth starts"
    run mimo_v26_nt $O/p3_mimo_v26_nt /root/hf_models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B "$NT" zh 0 '' 44
    run gpt_oss_20b_en $O/p3_gpt_oss_20b_en /root/hf_models/openai/gpt-oss-20b '{}' en 1 $TF19 44 --fixed-date 2026-10-19
    run qwen38_nt $O/p3_qwen38_nt /root/hf_models/Qwen/Qwen3.8-27B "$NT" zh 0 '' 90
    run qwen38_think $O/p3_qwen38_think /root/hf_models/Qwen/Qwen3.8-27B '{}' zh 0 '' 90
    log "lane p3 DEPTH done" ;;
esac
