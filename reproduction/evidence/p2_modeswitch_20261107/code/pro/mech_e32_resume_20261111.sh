#!/bin/bash
# EXAONE-4.0.1-32B mechanism battery, resumed after the fits re-run (11-11). The KL-words queue (kl_words_orders_pro_20261111b.sh,
# lane RUN) called the battery once, at 14:33 UTC; its fits were killed by the container memory limit, the lane went on and keeps
# the weights. The battery itself is unchanged (mech_exaone_pro_20261111.sh, b3221664) and finds the re-run fits in r05r06_fits/e32.
# While it runs no other GPU job of the queue can start: the next download (Qwen3.6-35B, 67 GiB) waits for 77 GiB free disk, which
# exists only once these weights (60 GiB) are deleted. After modeswitch/mech_e32/DONE the weights are deleted by literal path, as the
# lane's rule (delete only once DONE), after a check (/proc maps and fds) that no process has a file of the directory open or mapped.
# usage (tmux klw, window E32MECH, in the crossmodel directory): bash mech_e32_resume_20261111.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
M=/root/hf_models_p2nat/LGAI-EXAONE/EXAONE-4.0.1-32B; W=$C/modeswitch/mech_e32; Q=$C/words_along/kl/queue.log
st() { echo "[mech e32] $(date '+%F %T') $*" | tee -a $W/status.log >> $C/modeswitch/queue.log; }
in_use() {  # 0 if any process has a file under $1 open or mapped
  local p
  for p in /proc/[0-9]*; do
    grep -q -F "$1/" $p/maps 2>/dev/null && return 0
    ls -l $p/fd 2>/dev/null | grep -q -F "$1/" && return 0
  done
  return 1
}
[ -s r05r06_fits/e32/fit_seed4.pt ] || { st "resume: fits missing, not starting"; exit 1; }
st "resume (battery $(sha256sum mech_exaone_pro_20261111.sh | cut -c1-8))"
bash mech_exaone_pro_20261111.sh $M; rc=$?
st "resume: battery exit $rc"
# 11-11 15:40: gpt-oss-120b's KL words (skipped by lane RUN, download without the paper's ignore_patterns) run here, while these weights
# still hold the disk, so that no other GPU job of the queue starts meanwhile (p2_words_along_20261110 code/pro/g120_kl_words_20261111.sh)
[ -e g120_kl_words_20261111.sh ] && { st "gpt-oss-120b KL words (g120_kl_words_20261111.sh $(sha256sum g120_kl_words_20261111.sh | cut -c1-8))"; bash g120_kl_words_20261111.sh; st "g120 exit $?"; }
if [ -e $W/DONE ]; then
  if [ -d $M ] && [ ! -L $M ] && ! in_use $M; then
    rm -rf /root/hf_models_p2nat/LGAI-EXAONE/EXAONE-4.0.1-32B
    echo "[klw orders RUN] $(date '+%F %T') weights of exaone4_32b_en deleted after the mechanism battery ($(df --output=avail -BG /root | tail -1 | tr -dc 0-9) GiB free)" >> $Q
    st "weights deleted"
  else st "weights not deleted: missing, a link, or in use"; fi
else st "battery not DONE: weights kept"; fi
