#!/bin/bash
# server3, 11-06 (user: "我们开始部署 qwen 3.8 的实验吧"): P2 self-space token states, Qwen3.8-27B BF16 on GPUs 0,1 (one run).
# Steps: code hashes -> wait until GPUs 0 and 1 have been idle for 3 minutes and >= 50 GiB are free -> CPU dry run of every
# rendering -> smoke (5 bodies, one sequence per forward; the first, batched smoke is NOT_USED) -> gate (reproduce the archived phase-G content features) -> all 150 bodies.
# tmux session selfspace; STOP_SELFSPACE before any step stops cleanly; Ctrl-C in the window stops the run (finished bodies stay,
# a rerun resumes). If the gate fails, the full run waits for SELFSPACE_GO (or stops on STOP_SELFSPACE).
C=/root/data/p2_small_checks_20261014/crossmodel
PY=/root/recursive_state_moe_server3_20260711/venv/bin/python
M=/root/data/hf_models/Qwen/Qwen3.8-27B
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
O=selfspace/qwen38_27b
cd $C || exit 1
mkdir selfspace.lock 2>/dev/null || { echo "selfspace.lock present"; exit 0; }
st() { echo "$(date '+%F %T') $1" | tee -a selfspace_status.log; }
stop() { [ -e STOP_SELFSPACE ] && { st "stopped by STOP_SELFSPACE"; rmdir selfspace.lock; exit 0; }; }
sha256sum -c --quiet SHA256SUMS_selfspace >> selfspace_status.log 2>&1 || { st "code hash mismatch"; rmdir selfspace.lock; exit 1; }
echo "c7a2ce298608594aa032d32848c3a78c51529b8d136397838c838bbe5c3b0029  $B" | sha256sum -c --quiet || { st "bodies hash mismatch"; rmdir selfspace.lock; exit 1; }
st "start (waiting for GPUs 0,1 idle and disk)"
free=0
while [ $free -lt 3 ]; do
  stop
  busy=$(nvidia-smi -i 0,1 --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
  [ "$busy" -eq 0 ] && free=$((free + 1)) || free=0
  sleep 60
done
until [ "$(df -BG --output=avail . | tail -1 | tr -dc 0-9)" -ge 50 ]; do stop; st "waiting for 50G free"; sleep 600; done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1
mkdir -p selfspace
st "dry run"
$PY selfspace_tokens_20261106.py --model $M --bodies $B --out ${O}_dry --dry-run >> selfspace_dry.log 2>&1 \
  && grep -q 'DRY RUN OK' selfspace_dry.log || { st "dry run failed (selfspace_dry.log)"; rmdir selfspace.lock; exit 1; }
stop
st "smoke start"
$PY selfspace_tokens_20261106.py --model $M --bodies $B --out ${O}_smoke1 --limit 5 --max-memory-gib 44 >> selfspace_smoke.log 2>&1
st "smoke exit $?"
$PY validate_selfspace_smoke_20261106.py ${O}_smoke1 suite/qwen38_27b_bf16_s3/G >> selfspace_gate.log 2>&1
if [ $? -eq 0 ]; then st "gate OK"; else
  st "gate FAIL (selfspace_gate.log): waiting for SELFSPACE_GO"
  until [ -e SELFSPACE_GO ]; do stop; sleep 120; done
  st "SELFSPACE_GO present"
fi
stop
st "full start"
$PY selfspace_tokens_20261106.py --model $M --bodies $B --out $O --max-memory-gib 44 >> selfspace_full.log 2>&1
st "full exit $?"
[ -e $O/DONE ] && st "done" || st "full run ended without DONE (rerun resumes)"
rmdir selfspace.lock
