#!/bin/bash
# User 10-26: are the chat templates rendered on 10-26 (template_renderings_table2_20261026.txt, template_renderings_gptoss_20261026.txt)
# the ones the runs used? The run identity files record the model path, not the template hash, so this prints, for each model, the
# sha256 of chat_template.jinja / tokenizer_config.json, the Hugging Face commit and download time from .cache/huggingface/download,
# the file's last modification, and (server3) the first START of a task of that model in worker_done_s3.log. A template written once,
# before the first run, and not modified since is the template of every run. Read-only.
# usage: template_versions_20261026.sh HF_ROOT [WORKER_LOG] [MODEL ...]
root=$1; log=$2; shift 2
for m in "$@"; do
  d=$root/$m
  echo "== $m"
  for f in chat_template.jinja tokenizer_config.json; do
    [ -f "$d/$f" ] || continue
    md=$d/.cache/huggingface/download/$f.metadata
    echo "  $f sha256 $(sha256sum "$d/$f" | cut -c1-16) commit $(head -1 "$md" | cut -c1-12) downloaded $(date -d @"$(sed -n 3p "$md" | cut -d. -f1)" '+%F %T') modified $(stat -c %y "$d/$f" | cut -c1-19)"
  done
  echo "  commits over all downloaded files: $(cat "$d"/.cache/huggingface/download/*.metadata | awk 'NR%3==1' | sort -u | cut -c1-12 | tr '\n' ' ')"
  if [ -f "$log" ]; then
    echo "  first run: $(grep -F "\"$d\"" "$log" | grep -m1 START | cut -c1-19) $(grep -F "\"$d\"" "$log" | grep -m1 START | grep -o 'START [^ ]*')"
  fi
done
