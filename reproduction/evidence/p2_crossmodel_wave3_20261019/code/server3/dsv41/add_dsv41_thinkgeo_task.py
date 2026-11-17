"""Insert the DeepSeek-V4.1 thinking-rendering geometry (run_dsv41_think_geo.sh) into the server3 worker queue right after
q35fp8_G (the worker starts the first pending task in file order, so this runs next, before q35fp8_gthink). Requires q35fp8_G
done and the patched runner in place (sha check). Atomic rewrite, snapshot kept. usage (crossmodel folder): python add_dsv41_thinkgeo_task.py"""
import hashlib
import json
import os
import shutil
import time

T = 'worker_tasks_s3.json'
SHA = '9fb669a93cf2'
assert hashlib.sha256(open('dsv41/dsv41_natural3.py', 'rb').read()).hexdigest().startswith(SHA), 'runner not the patched version'
ENV = {'CUDA_VISIBLE_DEVICES': '0', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'HF_HUB_OFFLINE': '1',
       'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}  # as nat3_dsv41; the shell script sets the GPU per shard
new = {'name': 'nat3think_dsv41', 'kind': 'think', 'requires': [['worker_done_s3.log', 'DONE q35fp8_G#0']],
       'args': {'subprocess_cmd': ['/bin/bash', 'dsv41/run_dsv41_think_geo.sh'], 'subprocess_log': 'nat3think_dsv41.log',
                'subprocess_env': ENV}}
tasks = json.load(open(T, encoding='utf-8'))
names = [t['name'] for t in tasks]
assert new['name'] not in names, 'already queued'
i = names.index('q35fp8_G') + 1
snap = f'worker_tasks_s3_before_dsv41_thinkgeo_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(tasks[:i] + [new] + tasks[i:], open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
print('inserted', new['name'], 'after q35fp8_G, before', names[i], '| snapshot', snap)
