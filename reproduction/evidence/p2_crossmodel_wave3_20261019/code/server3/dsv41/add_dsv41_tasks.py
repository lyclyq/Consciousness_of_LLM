"""Append the DeepSeek-V4.1-Flash-1017 natural-order tasks to the server3 worker queue (atomic rewrite, snapshot kept).
  dsv41_tests        kernel checks and the small official-model forward checks on the GPU
  nat3_dsv41_smoke   4 bodies with the single-sequence check, after the weights, tokenizer and venv are in place
  nat3_dsv41         all 150 bodies, only after a manual DSV41_GO (set after reading the smoke results)
All run as children of the worker on GPU 0, after nat3_glm53f, one after another like every other task.
usage (in the crossmodel folder on server3): python add_dsv41_tasks.py"""
import json
import os
import shutil
import time

T = 'worker_tasks_s3.json'
PY = '/root/data/venv_t210/bin/python'
MD = '/root/data/hf_models/deepseek-ai/DeepSeek-V4.1-Flash-1017'
BODIES = '/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl'
ENV = {'CUDA_VISIBLE_DEVICES': '0', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'HF_HUB_OFFLINE': '1',
       'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
FILES_OK = [['download_dsv41_flash_1017.log', 'ok model-00047-of-00048.safetensors'],
            ['download_dsv41_flash_1017.log', 'ok model-00048-of-00048.safetensors'], ['DSV41_TOKENIZER_OK', '']]
new = [
    {'name': 'dsv41_tests', 'kind': 'think',
     'requires': [['worker_done_s3.log', 'DONE r1_think_natural#0'], ['VENV_T210_OK', '']],
     'args': {'subprocess_cmd': ['/bin/bash', '-c', f'cd dsv41 && {PY} test_dsv41_kernels.py cuda && {PY} test_dsv41_forward.py {MD}/inference cuda'],
              'subprocess_log': 'dsv41_tests.log', 'subprocess_env': ENV}},
    {'name': 'nat3_dsv41_smoke', 'kind': 'think',
     'requires': [['worker_done_s3.log', 'DONE dsv41_tests#0']] + FILES_OK,
     'args': {'subprocess_cmd': [PY, 'dsv41/dsv41_natural3.py', '--model-dir', MD, '--bodies', BODIES,
                                 '--out', 'suite_nat3/dsv41_flash_1017_smoke', '--limit', '4', '--check-single', '--wave-seqs', '64'],
              'subprocess_log': 'nat3_dsv41_smoke.log', 'subprocess_env': ENV}},
    {'name': 'nat3_dsv41', 'kind': 'think',
     'requires': [['worker_done_s3.log', 'DONE nat3_dsv41_smoke#0'], ['DSV41_GO', '']] + FILES_OK,
     'args': {'subprocess_cmd': [PY, 'dsv41/dsv41_natural3.py', '--model-dir', MD, '--bodies', BODIES,
                                 '--out', 'suite_nat3/dsv41_flash_1017', '--wave-seqs', '64'],
              'subprocess_log': 'nat3_dsv41.log', 'subprocess_env': ENV}},
]
tasks = json.load(open(T, encoding='utf-8'))
names = {t['name'] for t in tasks}
assert not names & {t['name'] for t in new}, 'tasks already queued'
snap = f'worker_tasks_s3_before_dsv41_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
tmp = T + '.tmp'
json.dump(tasks + new, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(tmp, T)
print('added', [t['name'] for t in new], 'snapshot', snap)
