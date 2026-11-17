"""Queue the two-role natural-order thinking run of DeepSeek-R1-Distill-Llama-70B on the server3 worker (user 10-20: one model
per developer in the natural order; its template moves a System message to the start, so Assistant and User only).
Same settings as r1_think_b16_v2 (batch 16, up to 3072 new tokens, 150 Chinese bodies x 2 orders), thinking only (--skip-off),
crossmodel_thinking_b.py --natural (7f62ca41). Atomic rewrite of the task file, snapshot kept.
usage (in the crossmodel folder on server3): python add_r1_natural_task.py"""
import json
import os
import shutil
import time

T = 'worker_tasks_s3.json'
task = {'name': 'r1_think_natural', 'kind': 'think',
        'args': {'model': '/root/data/hf_models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 'off_kwargs': '{}', 'on_kwargs': '{}',
                 'bodies_zh': 'bodies_zh_150_list.json', 'out': 'thinking_natural/r1_distill_llama70b', 'batch': 16, 'max_new': 3072,
                 'max_memory_gib': 44, 'skip_off': True, 'natural': True}}
tasks = json.load(open(T, encoding='utf-8'))
assert task['name'] not in {t['name'] for t in tasks}, 'already queued'
snap = f'worker_tasks_s3_before_r1_natural_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(tasks + [task], open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
print('queued', task['name'], 'snapshot', snap)
