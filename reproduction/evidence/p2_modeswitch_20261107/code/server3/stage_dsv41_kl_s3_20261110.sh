#!/bin/bash
# Stage the DeepSeek-V4.1-Flash natural3 KL results on server3 for the local archive (11-10): everything under modeswitch/nat3 that
# belongs to dsv41 (the run, the passed smoke, the failed smoke kept in _failed_smoke_20261110, the queue, dry-run and shard logs),
# the queue log, the download log and listing, and the as-run scripts (KL port, kept old copy, device test, queue, finish script).
# Output crossmodel/sync_20261110/{dsv41_kl.tgz, SHA256SUMS.txt, dsv41_kl.tgz.sha256, READY}. Reads only, writes only there.
set -eu
cd /root/data/p2_small_checks_20261014/crossmodel
S=sync_20261110; mkdir -p $S; rm -f $S/READY
{ find modeswitch/nat3 -maxdepth 1 \( -name 'dsv41*' -o -name '_failed_smoke_20261110' \) -print
  ls modeswitch/queue.log download_dsv41_flash_1017_20261108.log dsv41_ms_files.tsv run_dsv41_kl_s3_20261108.sh \
     finish_dsv41_dl_s3_20261110.sh download_dsv41_ms_s3_v3_20261109.sh dsv41/dsv41_modeswitch_nat3_20261108.py \
     dsv41/dsv41_modeswitch_nat3_20261108.py.v6917870c dsv41/test_dsv41_body_record_device_20261110.py; } > $S/roots.txt
find $(cat $S/roots.txt) -type f | sort > $S/files.txt
xargs -d '\n' sha256sum < $S/files.txt > $S/SHA256SUMS.txt
tar czf $S/dsv41_kl.tgz -T $S/files.txt $S/SHA256SUMS.txt
sha256sum $S/dsv41_kl.tgz > $S/dsv41_kl.tgz.sha256
echo "$(wc -l < $S/files.txt) files, $(du -h $S/dsv41_kl.tgz | cut -f1)" > $S/READY
