#!/bin/bash
# P2 ordinary words in next-token prediction, PRO (11-11, user: "测测呗 看看值不值得"; scope: the four mechanism settings and
# MiMo-VL-7B as the no-index contrast). kl_words_20261111.py, one task at a time on the one GPU. Each setting uses the command
# of its paper KL ctrl run (model, template kwargs, language, fixed date, PYTHONPATH, network, memory) and is checked against
# that run. MiMo-VL-7B is downloaded again at the words run's pinned commit (lane DL, in parallel, no GPU) and deleted by its
# literal path once its run is DONE. Stop: touch words_along/STOP_KLW (checked before every body and task); no signals.
# usage (tmux session klw, one window per lane, in the crossmodel directory): bash kl_words_pro_20261111.sh RUN|DL
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; WJ=$C/words_v2.json; MC=$C/modeswitch/ctrl
O=$C/words_along/kl; STOP=$C/words_along/STOP_KLW
mkdir -p $O
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
LANE=${1:?lane RUN or DL}
log() { echo "[klw $LANE] $(date '+%F %T') $*" | tee -a $O/queue.log; }

run() {  # tag(= paper ctrl run) model kw lang net pythonpath mem [extra args]
  local tag=$1 m=$2 kw=$3 lang=$4 net=$5 pp=$6 mem=$7; shift 7
  local out=$O/$tag lg=$O/$tag.log a
  [ -e $out/DONE ] && { log "already DONE: $tag"; return 0; }
  [ -e $STOP ] && { log "stop flag: not starting $tag"; exit 2; }
  nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: lane stops before $tag"; exit 3; }
  a=(--model $m --tmpl-kwargs "$kw" --bodies $B1 --words $WJ --out $out --check-ctrl $MC/$tag --lang $lang --max-memory-gib $mem
     --stop-file $STOP "$@")
  log "start $tag"
  if [ $net = 1 ]; then
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 \
      PYTHONPATH=$pp $PY kl_words_20261111.py "${a[@]}" >> $lg 2>&1 < /dev/null
  else
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY kl_words_20261111.py "${a[@]}" >> $lg 2>&1 < /dev/null
  fi
  if [ -e $out/DONE ]; then log "DONE $tag: $(grep 'summary (content)' $lg | tail -1)"
  else log "NOT DONE $tag, see $lg: $(grep -a -E 'CHECK FAILED|Error|stop' $lg | tail -1)"; fi
}

case $LANE in
  DL)
    if [ ! -e $O/dl_mimo_vl.ok ]; then
      for i in 1 2 3 4 5 6; do
        [ -e $STOP ] && exit 2
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=$TF19 $PY -c \
          "from huggingface_hub import snapshot_download; print('snapshot', snapshot_download('XiaomiMiMo/MiMo-VL-7B-RL-2508', revision='ec4ebdc29b83b94aa7f62670c78376250dad8dbc', local_dir='/root/hf_models_words/XiaomiMiMo/MiMo-VL-7B-RL-2508', max_workers=8))" \
          >> $O/dl_mimo_vl.log 2>&1 && { touch $O/dl_mimo_vl.ok; log "downloaded MiMo-VL-7B $(du -sh /root/hf_models_words/XiaomiMiMo/MiMo-VL-7B-RL-2508 | cut -f1)"; break; }
        log "MiMo-VL-7B download attempt $i failed"; sleep 120
      done
    fi
    log "lane DL done" ;;
  RUN)
    NT='{"enable_thinking": false}'
    run mimo_v26_nt /root/hf_models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B "$NT" zh 0 '' 44
    run qwen38_nt /root/hf_models/Qwen/Qwen3.8-27B "$NT" zh 0 '' 90
    run qwen38_think /root/hf_models/Qwen/Qwen3.8-27B '{}' zh 0 '' 90
    run gpt_oss_20b_en /root/hf_models/openai/gpt-oss-20b '{}' en 1 $TF19 44 --bodies-en bodies_en_150.json --fixed-date 2026-10-19
    while [ ! -e $O/dl_mimo_vl.ok ]; do [ -e $STOP ] && exit 2; log "waiting for the MiMo-VL-7B download"; sleep 120; done
    run mimo_vl7b /root/hf_models_words/XiaomiMiMo/MiMo-VL-7B-RL-2508 '{}' zh 0 '' 44
    if [ -e $O/mimo_vl7b/DONE ]; then rm -rf /root/hf_models_words/XiaomiMiMo/MiMo-VL-7B-RL-2508; log "MiMo-VL-7B weights deleted"; fi
    log "lane RUN done" ;;
esac
