#!/bin/bash
# P2 ordinary words along the message, PRO lane F (11-10, user decision "按论文环境重跑"): Qwen3.8-Flash-Next, main order,
# roles + 32 pairs, with the reference (torch) gated delta rule.
# Why: lane X's attempt (07:36 UTC) failed its gate against the paper's G features (suite_think/qwen38_flashnext, 10-17;
# cos 0.95-0.98, rel 0.23-0.32). flash-linear-attention 0.6.0 was installed in the venv on 10-25 09:39 UTC, after every paper
# Flash-Next run; those runs used the reference implementation (their logs say chunk_gated_delta_rule falls back to it).
# torch 2.11.0+cu130, transformers 5.19 (tf_next), the hub FP8 kernel and the 52 GiB CPU offload are unchanged.
# This lane hides fla for this run only: words_along/nofla_shim/sitecustomize.py sets sys.modules['fla'] = None, so
# transformers finds no fla and the hub-kernel decorator falls back to the torch function. The venv is not changed.
# The command is lane X's otherwise; the functions come from words_pro_20261110.sh (sourced with LANE=F, which matches no lane
# there; cf962886). The failed attempt is moved to words_along/_gatefail_fla_20261110/, not deleted.
# On success: weights deleted, run_qwen38_flashnext.done written, FN_RESERVE removed, so lanes S2 and X continue on their own.
# Lock: the exclusive persistence/gpu.lock, as lane X (never more than the PRO limit). Stop: touch words_along/STOP_WORDS.
# usage (tmux session words, window F, in the crossmodel directory): bash words_pro_fn_nofla_20261110.sh
set -u
C=/root/p2_scale_20261015/crossmodel; cd $C || exit 1
[ "$(sha256sum $C/words_pro_20261110.sh | cut -c1-8)" = cf962886 ] || { echo "words_pro_20261110.sh changed: lane F stops"; exit 1; }
set -- F; source $C/words_pro_20261110.sh
SH=$O/nofla_shim; mkdir -p $SH
printf '%s\n' 'import sys' "sys.modules['fla'] = None  # P2 words lane F (11-10): reference gated delta rule, as in the paper's Flash-Next runs" > $SH/sitecustomize.py
log "lane F start; shim $(sha256sum $SH/sitecustomize.py | cut -c1-8)"
[ -e $O/run_qwen38_flashnext.done ] && { log "Flash-Next already finished: nothing to do"; exit 0; }
[ -e $O/dl_qwen38_flashnext.ok ] && [ -d $D/Qwen/Qwen3.8-Flash-Next-FP8 ] || { log "Flash-Next weights missing: lane F stops"; exit 1; }
HF_HUB_OFFLINE=1 PYTHONPATH=$SH:$TF19 $PY -c "
import sys, transformers.utils.import_utils as u
v = u._is_package_available('fla')[0]
print('fla visible with the shim:', v)
sys.exit(1 if v else 0)" >> $O/laneF_check.log 2>&1 || { log "fla still visible with the shim: lane F stops"; exit 1; }
log "fla hidden with the shim (laneF_check.log)"
if [ ! -e $O/b150_main/qwen38_flashnext/DONE ] && grep -q "GATE FAILED" $O/b150_main_qwen38_flashnext.log 2>/dev/null; then
  R=$O/_gatefail_fla_20261110; mkdir -p $R
  mv $O/b150_main/qwen38_flashnext $R/b150_main_qwen38_flashnext && mv $O/b150_main_qwen38_flashnext.log $R/ && \
    log "moved the failed attempt (fla) to $R" || { log "could not move the failed attempt: lane F stops"; exit 1; }
fi
t x 84 b150 main qwen38_flashnext $D/Qwen/Qwen3.8-Flash-Next-FP8 "$NT" zh roles $SH:$TF19 1 --gate-g suite_think/qwen38_flashnext --gate-tmpl-kwargs '{}' -- --cpu-offload-gib 52
if grep -q "chunk_gated_delta_rule.*falling back" $O/b150_main_qwen38_flashnext.log; then log "log confirms the reference gated delta rule"
else log "WARNING: the log has no reference gated delta rule message"; fi
finish qwen38_flashnext b150_main/qwen38_flashnext
[ -e $O/run_qwen38_flashnext.done ] && { rm -f $O/FN_RESERVE; log "FN_RESERVE removed: lanes S2 and X continue"; }
log "lane F done"
