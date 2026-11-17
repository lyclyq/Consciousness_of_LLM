#!/bin/bash
# P2 ordinary words in next-token prediction, PRO extension (11-11, user: align with the words-along models; PRO disk holds only
# the small ones for now: "先跑放得下的 9 个" = MiMo-VL-7B in kl_words_pro_20261111.sh plus these 8). Each model uses the command
# of its paper KL ctrl run (repo, pinned commit, weight path, language, template kwargs, fixed date, wrapper, PYTHONPATH, memory)
# and is checked against that run by kl_words_20261111.py. Apertus-8B and OLMo-2-7B had no pinned commit in their KL queue; they
# use the commits of the words-along runs (resolved from the paper's configuration table).
#   DL   after the MiMo-VL-7B download of the other lane is ok, downloads one model at a time in this order, each once free disk
#        >= size + 10 GiB (hf-mirror, no xet), so that two downloads never run at once
#   RUN  waits until lane RUN of kl_words_pro_20261111.sh is done, then runs each model once its download is ok, one task on the
#        GPU at a time, and deletes its weights by literal path once its run is DONE
# Stop: touch words_along/STOP_KLW; no signals.
# usage (tmux session klw, windows EXTDL and EXTRUN, in the crossmodel directory): bash kl_words_ext_pro_20261111.sh DL|RUN
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
PY=/root/consciousness_a6000pro/.venv/bin/python
B1=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl; WJ=$C/words_v2.json; MC=$C/modeswitch/ctrl
O=$C/words_along/kl; STOP=$C/words_along/STOP_KLW
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
LANE=${1:?lane DL or RUN}
log() { echo "[klw ext $LANE] $(date '+%F %T') $*" | tee -a $O/queue.log; }
free_gib() { df --output=avail -BG /root | tail -1 | tr -dc 0-9; }
# tag|repo|commit|weight dir|GiB|lang|template kwargs|max GPU GiB|wrapper|PYTHONPATH|extra
MODELS='mimo7b_rl_0530|XiaomiMiMo/MiMo-7B-RL-0530|ee8dd14ecd59a1655ebe2ede9f7a06b40a8c8c5b|/root/hf_models_p2nat|15|zh|{"enable_thinking": false}|44|||
apertus8b_en|swiss-ai/Apertus-8B-Instruct-2509|b946d40447b2b597999b9c86d44bee0b452c919f|/root/hf_models_words|15|en|{}|44|||--fixed-date 2026-11-03
olmo2_7b_en|allenai/OLMo-2-1124-7B-Instruct|470b1fba1ae01581f270116362ee4aa1b97f4c84|/root/hf_models_words|14|en|{}|44|||
glm46v_flash|zai-org/GLM-4.6V-Flash|411bb4d77144a3f03accbf4b780f5acb8b7cde4e|/root/hf_models_p2nat|20|zh|{"enable_thinking": false}|44|||
olmo2_13b_en|allenai/OLMo-2-1124-13B-Instruct|3a5c85baefbb1896a54d56fe2e76c0395627ddf4|/root/hf_models_p2nat|26|en|{}|44|||
phi4_en|microsoft/phi-4|2db69c1c3e91a05d2c64a3185acfbaf36f744e25|/root/hf_models_p2nat|28|en|{}|44|||
moonlight16b_a3b_instruct|moonshotai/Moonlight-16B-A3B-Instruct|4e735b07a89f73647dfab71ab91b840f362ede5b|/root/hf_models_p2nat|30|zh|{}|44|moonlight_compat_20261103.py|/root/p2_scale_20261015/pydeps_tiktoken_20261108|
ernie45_21b_a3b_thinking|baidu/ERNIE-4.5-21B-A3B-Thinking|4341bb42644d5422859509fa25d41544c57181f8|/root/hf_models_p2nat|41|zh|{}|60|||'

del_weights() {  # literal paths of this extension's downloads only
  case $1 in
    mimo7b_rl_0530) rm -rf /root/hf_models_p2nat/XiaomiMiMo/MiMo-7B-RL-0530 ;;
    apertus8b_en) rm -rf /root/hf_models_words/swiss-ai/Apertus-8B-Instruct-2509 ;;
    olmo2_7b_en) rm -rf /root/hf_models_words/allenai/OLMo-2-1124-7B-Instruct ;;
    glm46v_flash) rm -rf /root/hf_models_p2nat/zai-org/GLM-4.6V-Flash ;;
    olmo2_13b_en) rm -rf /root/hf_models_p2nat/allenai/OLMo-2-1124-13B-Instruct ;;
    phi4_en) rm -rf /root/hf_models_p2nat/microsoft/phi-4 ;;
    moonlight16b_a3b_instruct) rm -rf /root/hf_models_p2nat/moonshotai/Moonlight-16B-A3B-Instruct ;;
    ernie45_21b_a3b_thinking) rm -rf /root/hf_models_p2nat/baidu/ERNIE-4.5-21B-A3B-Thinking ;;
    *) log "del_weights: unknown tag $1, nothing deleted" ;;
  esac
}

case $LANE in
  DL)
    while IFS='|' read -r tag repo rev base gb lang kw mem wrap pp extra; do
      [ -z "$tag" ] && continue
      [ -e $O/dl_$tag.ok ] && continue
      [ -e $O/$tag/DONE ] && continue
      until [ -e $O/dl_mimo_vl.ok ]; do [ -e $STOP ] && exit 2; sleep 60; done
      until [ $(free_gib) -ge $(( gb + 10 )) ]; do [ -e $STOP ] && exit 2; log "waiting for $(( gb + 10 )) GiB free ($(free_gib)) for $repo"; sleep 120; done
      log "download $repo@${rev:0:12} ($gb GiB, $(free_gib) GiB free)"
      ok=0
      for i in 1 2 3 4 5 6; do
        [ -e $STOP ] && exit 2
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 $PY -c \
          "from huggingface_hub import snapshot_download; print('snapshot', snapshot_download('$repo', revision='$rev', local_dir='$base/$repo', max_workers=8))" \
          >> $O/dl_$tag.log 2>&1 < /dev/null && { ok=1; break; }
        log "$repo attempt $i failed"; sleep 120
      done
      if [ $ok = 1 ]; then touch $O/dl_$tag.ok; log "downloaded $repo $(du -sh $base/$repo | cut -f1)"; else touch $O/dl_$tag.failed; log "download FAILED $repo"; fi
    done <<< "$MODELS"
    log "lane ext DL done" ;;
  RUN)
    until grep -q 'lane RUN done' $O/queue.log 2>/dev/null; do [ -e $STOP ] && exit 2; sleep 60; done
    log "the first five settings are finished: extension starts"
    while IFS='|' read -r tag repo rev base gb lang kw mem wrap pp extra; do
      [ -z "$tag" ] && continue
      out=$O/$tag; lg=$O/$tag.log; m=$base/$repo
      if [ ! -e $out/DONE ]; then
        until [ -e $O/dl_$tag.ok ] || [ -e $O/dl_$tag.failed ]; do [ -e $STOP ] && exit 2; sleep 60; done
        [ -e $O/dl_$tag.failed ] && { log "skip $tag: download failed"; continue; }
        [ -e $STOP ] && { log "stop flag: not starting $tag"; exit 2; }
        nvidia-smi > /dev/null 2>&1 || { log "nvidia-smi fails: lane stops before $tag"; exit 3; }
        a=(--model $m --tmpl-kwargs "$kw" --bodies $B1 --words $WJ --out $out --check-ctrl $MC/$tag --lang $lang --max-memory-gib $mem
           --stop-file $STOP)
        [ $lang = en ] && a+=(--bodies-en bodies_en_150.json)
        [ -n "$extra" ] && a+=($extra)
        log "start $tag"
        if [ -n "$wrap" ]; then
          HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY $wrap $m kl_words_20261111.py "${a[@]}" >> $lg 2>&1 < /dev/null
        else
          HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$pp $PY kl_words_20261111.py "${a[@]}" >> $lg 2>&1 < /dev/null
        fi
      fi
      if [ -e $out/DONE ]; then
        log "DONE $tag: $(grep 'summary (content)' $lg | tail -1)"
        del_weights $tag; log "weights of $tag deleted ($(free_gib) GiB free)"
      else log "NOT DONE $tag, see $lg (weights kept): $(grep -a -E 'CHECK FAILED|Error|stop' $lg | tail -1)"; fi
    done <<< "$MODELS"
    log "lane ext RUN done" ;;
esac
