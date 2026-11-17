#!/bin/bash
# KL of the ordinary words for the cells that lane RUN of kl_words_orders_pro_20261111b.sh did not finish (11-11), run in sequence
# with the lane's own command lines (run_orders, run_p3), weights deleted only at the end:
#   gpt-oss-120b, main + 200-body depth: the lane's download had no ignore_patterns, so the repository's original/ (61 GiB, fetched)
#     and metal/ (one file above the mirror's plain-download limit, every attempt failed) were requested. They are not part of the
#     checkpoint the paper loads (the paper's downloads gptoss120b_pro_v2.sh, pro_chain*.sh used ignore_patterns ['original/*',
#     'metal/*']). original/ and metal/ removed 15:37 UTC. The download is completed with the paper's patterns at commit b5c939de.
#   Qwen3.6-35B-A3B (main) and Seed-OSS-36B (main, natural): 135 GiB were freed on PRO at 18:15 UTC outside this queue, the lane
#     downloaded both and started them while the EXAONE battery held the GPU. Both stopped at loading with CUDA out of memory
#     (18:39, 19:00), nothing written, the EXAONE battery unaffected (all its steps exit 0). Weights kept by the lane.
# Called by mech_e32_resume_20261111.sh after the EXAONE battery and before the EXAONE weights are deleted. The weights of these three
# are deleted only after all three runs, so that the queue's next download (R1, 142 GiB free needed) cannot start a GPU job meanwhile.
# v2 (23:45 UTC): Qwen3.6-35B and Seed added, deletions moved to the end (v1 0efd1ad1 had not been called yet).
# Stop: touch words_along/STOP_KLW.  usage (in the crossmodel directory): bash g120_kl_words_20261111.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python; TF19=/root/p2_scale_20261015/tf_next
B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; B3=$C/phase3_bodies_zh.jsonl; E3=$C/phase3_bodies_en.json
WJ=$C/words_v2.json; MS=$C/modeswitch; MC=$MS/ctrl; O=$C/words_along/kl; STOP=$C/words_along/STOP_KLW
NT='{"enable_thinking": false}'; SK='{"thinking_budget": 0}'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
log() { echo "[klw g120] $(date '+%F %T') $*" | tee -a $O/queue.log; }
free_gib() { df --output=avail -BG /root | tail -1 | tr -dc 0-9; }
NET=(env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1)
OFFL=(env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)
[ -e $STOP ] && { log "stop flag"; exit 2; }
nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails"; exit 3; }

# 1 gpt-oss-120b (en, '{}', 90 GiB, PYTHONPATH tf_next, network on, fixed date 2026-10-19), main + p3
tag=gpt_oss_120b_en; M=/root/hf_models_words/openai/gpt-oss-120b; REV=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
if [ ! -e $O/$tag/DONE ] || [ ! -e $O/p3_$tag/DONE ]; then
  if [ ! -e $O/dlo_$tag.fixed.ok ]; then
    [ -d $M/original ] && rm -rf /root/hf_models_words/openai/gpt-oss-120b/original
    [ -d $M/metal ] && rm -rf /root/hf_models_words/openai/gpt-oss-120b/metal
    rm -rf /root/hf_models_words/openai/gpt-oss-120b/.cache/huggingface/download/original /root/hf_models_words/openai/gpt-oss-120b/.cache/huggingface/download/metal
    log "completing the gpt-oss-120b download with the paper's ignore_patterns ($(free_gib) GiB free)"
    for i in 1 2 3 4 5 6; do
      [ -e $STOP ] && exit 2
      "${NET[@]}" $PY -c "from huggingface_hub import snapshot_download; print('snapshot', snapshot_download('openai/gpt-oss-120b', revision='$REV', local_dir='$M', max_workers=8, ignore_patterns=['original/*', 'metal/*']))" \
        >> $O/dlo_$tag.fixed.log 2>&1 < /dev/null && { touch $O/dlo_$tag.fixed.ok; break; }
      log "gpt-oss-120b download attempt $i failed"; sleep 120
    done
    if [ -e $O/dlo_$tag.fixed.ok ]; then log "downloaded $(du -sh $M | cut -f1)"; else log "gpt-oss-120b download FAILED"; fi
  fi
  if [ -e $O/dlo_$tag.fixed.ok ] && [ ! -e $O/$tag/DONE ]; then
    [ -e $STOP ] && exit 2
    log "start $tag (main)"
    "${NET[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TF19 $PY kl_words_orders_20261111.py --model $M --tmpl-kwargs '{}' --bodies $B1 --words $WJ \
      --orders main --lang en --max-memory-gib 90 --cpu-offload-gib 0 --stop-file $STOP --bodies-en bodies_en_150.json --fixed-date 2026-10-19 \
      --out-main $O/$tag --check-ctrl $MC/$tag >> $O/orders_$tag.log 2>&1 < /dev/null
    log "ended $tag: $(grep -a -E 'summary \(content|CHECK FAILED|Error|stop' $O/orders_$tag.log | tail -3 | tr '\n' ' ')"
  fi
  if [ -e $O/$tag/DONE ] && [ ! -e $O/p3_$tag/DONE ]; then
    [ -e $STOP ] && exit 2
    log "start p3_$tag"
    "${NET[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TF19 $PY kl_words_p3_20261111.py --model $M --tmpl-kwargs '{}' --bodies $B1 --p3-bodies $B3 \
      --words $WJ --out $O/p3_$tag --check-ctrl $MC/$tag --lang en --max-memory-gib 90 --cpu-offload-gib 0 --stop-file $STOP \
      --bodies-en bodies_en_150.json --p3-bodies-en $E3 --fixed-date 2026-10-19 >> $O/p3_$tag.log 2>&1 < /dev/null
    log "ended p3_$tag: $(grep -a -E 'summary \(content|CHECK FAILED|Error|stop' $O/p3_$tag.log | tail -2 | tr '\n' ' ')"
  fi
fi

# 2 Qwen3.6-35B-A3B (zh, non-thinking, 90 GiB, offline), main
tag=qwen36_35b_a3b; M2=/root/hf_models_p2nat/Qwen/Qwen3.6-35B-A3B
if [ ! -e $O/$tag/DONE ] && [ -d $M2 ]; then
  [ -e $STOP ] && exit 2
  log "start $tag (main)"
  "${OFFL[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH= $PY kl_words_orders_20261111.py --model $M2 --tmpl-kwargs "$NT" --bodies $B1 --words $WJ \
    --orders main --lang zh --max-memory-gib 90 --cpu-offload-gib 0 --stop-file $STOP --out-main $O/$tag --check-ctrl $MC/$tag \
    >> $O/orders_$tag.log 2>&1 < /dev/null
  log "ended $tag: $(grep -a -E 'summary \(content|CHECK FAILED|Error|stop' $O/orders_$tag.log | tail -3 | tr '\n' ' ')"
fi

# 3 Seed-OSS-36B (zh, thinking_budget 0, 90 GiB, offline), main + natural (natural reference modeswitch/seed_oss_36b, full KL)
tag=seed_oss_36b; M3=/root/hf_models_p2nat/ByteDance-Seed/Seed-OSS-36B-Instruct
if { [ ! -e $O/$tag/DONE ] || [ ! -e $O/natural/$tag/DONE ]; } && [ -d $M3 ]; then
  [ -e $STOP ] && exit 2
  todo=""
  [ -e $O/$tag/DONE ] || todo="$todo,main"
  [ -e $O/natural/$tag/DONE ] || todo="$todo,natural"
  todo=${todo#,}
  a=(--model $M3 --tmpl-kwargs "$SK" --bodies $B1 --words $WJ --orders $todo --lang zh --max-memory-gib 90 --cpu-offload-gib 0 --stop-file $STOP)
  case $todo in *main*) a+=(--out-main $O/$tag --check-ctrl $MC/$tag) ;; esac
  case $todo in *natural*) a+=(--out-natural $O/natural/$tag --check-natural $MS/$tag --check-natural-kind full) ;; esac
  mkdir -p $O/natural
  log "start $tag ($todo)"
  "${OFFL[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH= $PY kl_words_orders_20261111.py "${a[@]}" >> $O/orders_$tag.log 2>&1 < /dev/null
  log "ended $tag: $(grep -a -E 'summary \(content|CHECK FAILED|Error|stop' $O/orders_$tag.log | tail -3 | tr '\n' ' ')"
fi

# 4 deletions, each only once all its outputs are DONE (literal paths)
if [ -e $O/gpt_oss_120b_en/DONE ] && [ -e $O/p3_gpt_oss_120b_en/DONE ]; then
  if [ -d $M ] && [ ! -L $M ]; then rm -rf /root/hf_models_words/openai/gpt-oss-120b; log "weights of gpt_oss_120b_en deleted ($(free_gib) GiB free)"; fi
else log "gpt_oss_120b_en not DONE: weights kept"; fi
if [ -e $O/qwen36_35b_a3b/DONE ]; then
  if [ -d $M2 ] && [ ! -L $M2 ]; then rm -rf /root/hf_models_p2nat/Qwen/Qwen3.6-35B-A3B; log "weights of qwen36_35b_a3b deleted ($(free_gib) GiB free)"; fi
else log "qwen36_35b_a3b not DONE: weights kept"; fi
if [ -e $O/seed_oss_36b/DONE ] && [ -e $O/natural/seed_oss_36b/DONE ]; then
  if [ -d $M3 ] && [ ! -L $M3 ]; then rm -rf /root/hf_models_p2nat/ByteDance-Seed/Seed-OSS-36B-Instruct; log "weights of seed_oss_36b deleted ($(free_gib) GiB free)"; fi
else log "seed_oss_36b not DONE: weights kept"; fi
log "follow-up done"
