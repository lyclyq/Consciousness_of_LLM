#!/bin/bash
# Stage the wording-test results made after the 11-09 mirror (w2.tgz) for the local archive: thinking_swap/wording_20261109 except
# m26_* (already mirrored), singlepass_swap/wording_20261109/qsp_en_*, the English Qwen fit diagnostics and logs, the status log and
# the two queues. fit .pt files stay on PRO (their sha256 are listed). Output sync_wording_v3_20261110/{w3.tgz, SHA256SUMS.txt,
# SHA256SUMS_fits_pt_on_pro.txt, w3.tgz.sha256, READY}. Reads only, writes only into that folder.
set -eu
cd /root/p2_scale_20261015/crossmodel
S=sync_wording_v3_20261110; mkdir -p $S; rm -f $S/READY
{ find thinking_swap/wording_20261109 -mindepth 1 -maxdepth 1 ! -name 'm26_*' -print
  ls -d singlepass_swap/wording_20261109/qsp_en_*
  ls r05r06_fits/qsp_en/fit_diag.json r05r06_fits/qth_en/fit_diag.json r05_fits_qsp_en.log r05_fits_qth_en.log \
     wording_swap_v2_status.log wording_swap_pro_v3_20261109.sh wording_swap_qwen_en_pro_20261109.sh; } > $S/roots.txt
find $(cat $S/roots.txt) -type f | sort > $S/files.txt
xargs -d '\n' sha256sum < $S/files.txt > $S/SHA256SUMS.txt
sha256sum r05r06_fits/qsp_en/*.pt r05r06_fits/qth_en/*.pt > $S/SHA256SUMS_fits_pt_on_pro.txt
tar czf $S/w3.tgz -T $S/files.txt $S/SHA256SUMS.txt $S/SHA256SUMS_fits_pt_on_pro.txt
sha256sum $S/w3.tgz > $S/w3.tgz.sha256
echo "$(wc -l < $S/files.txt) files, $(du -h $S/w3.tgz | cut -f1)" > $S/READY
