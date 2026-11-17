#!/bin/bash
# Delete PRO's reference copies of Qwen3.8-27B, MiMo-V2.6-Distill-Qwen-9B and gpt-oss-20b once every KL-words output that uses
# them is DONE (11-11, user: "那就 1 呗": room for R1-Distill-70B and Qwen3.8-Flash-Next-FP8). Commits for a later download:
# Qwen3.8-27B 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, MiMo-V2.6-Distill-Qwen-9B 1f09ddcc1a0ea45751ae0dfe2af20d533168c46f,
# gpt-oss-20b 6cee5e81ee83917806bbde320786a8fb61efebee (config.json download metadata).
# Waits for: p3_mimo_v26_nt, p3_gpt_oss_20b_en, p3_qwen38_nt, p3_qwen38_think, natural/qwen38_nt, natural/gpt_oss_20b_en (DONE).
# Deletes only after no process maps or opens a file under the three directories, each a real directory; literal paths.
# Stop: touch words_along/STOP_KLW; no signals.
# usage (tmux session klw, window DELREFS): bash del_refs_pro_20261111.sh
set -u
O=/root/p2_scale_20261015/crossmodel/words_along/kl; STOP=/root/p2_scale_20261015/crossmodel/words_along/STOP_KLW
log() { echo "[klw refs] $(date '+%F %T') $*" | tee -a $O/queue.log; }
NEED="$O/p3_mimo_v26_nt/DONE $O/p3_gpt_oss_20b_en/DONE $O/p3_qwen38_nt/DONE $O/p3_qwen38_think/DONE $O/natural/qwen38_nt/DONE $O/natural/gpt_oss_20b_en/DONE"
log "waiting for the KL-words outputs on the three reference copies"
while :; do ok=1; for f in $NEED; do [ -e $f ] || ok=0; done; [ $ok = 1 ] && break; [ -e $STOP ] && exit 2; sleep 60; done
busy() { grep -l -s -E "/root/hf_models/(Qwen/Qwen3.8-27B|XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B|openai/gpt-oss-20b)/" /proc/[0-9]*/maps 2>/dev/null | head -1
         ls -l /proc/[0-9]*/fd 2>/dev/null | grep -E "/root/hf_models/(Qwen/Qwen3.8-27B|XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B|openai/gpt-oss-20b)/" | head -1; }
while [ -n "$(busy)" ]; do [ -e $STOP ] && exit 2; log "a process still uses the weights; waiting"; sleep 120; done
for d in /root/hf_models/Qwen/Qwen3.8-27B /root/hf_models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B /root/hf_models/openai/gpt-oss-20b; do
  if [ -d "$d" ] && [ ! -L "$d" ]; then s=$(du -sh "$d" | cut -f1); rm -rf "$d"; log "deleted $d ($s)"; else log "not a real directory, left alone: $d"; fi
done
log "reference copies deleted ($(df --output=avail -BG /root | tail -1 | tr -dc 0-9) GiB free)"
