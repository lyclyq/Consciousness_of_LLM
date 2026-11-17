#!/bin/bash
# Waits until the user has freed disk on server3 (>= 45 GiB free), then reopens the phase-3 lanes A3 and B3 of
# persist_s3_p3_20261107.sh in tmux session persistP3 (they stopped at the 32 GiB guard on 11-07 00:13 CST and resume from
# their per-body files). Nothing is deleted here. Stop: touch persistence/STOP_PERSIST.
C=/root/data/p2_small_checks_20261014/crossmodel; cd $C; P=$C/persistence
until [ $(df --output=avail -BG /root/data | tail -1 | tr -dc 0-9) -ge 45 ]; do [ -e $P/STOP_PERSIST ] && exit 2; sleep 300; done
echo "[queue resume] $(date "+%F %T") disk ok ($(df -h /root/data | tail -1 | awk "{print \$4}") free): reopening A3 and B3" | tee -a $P/queue_s3.log
tmux new-window -t persistP3 -n A3 "bash persist_s3_p3_20261107.sh A3"
tmux new-window -t persistP3 -n B3 "bash persist_s3_p3_20261107.sh B3"
