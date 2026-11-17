"""server3 queue, user 10-22: full run of the fourth design for Qwen3.8-27B (message under test and question, every token, all 64
layers, mean replacement in the 64-dim subspace; generation prompt and generated tokens untouched) and its specificity control
(the same edits with a content question, --fit-from the full run's fit). Both right after qwen38_v1sweep_smoke; the GO files
are written here (the user approved after reading the smoke); q35fp8_think900 also waits for both. Atomic rewrite."""
import json
import os
import shutil
import time

T = 'worker_tasks_s3.json'
STABLE = '/root/recursive_state_moe_server3_20260711/venv/bin/python'
ENV = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONUNBUFFERED': '1',
       'OMP_NUM_THREADS': '32', 'MKL_NUM_THREADS': '32'}
COMMON = ['--no-gen-edit', '--edit-end', 'before_gen', '--model', '/root/data/hf_models/Qwen/Qwen3.8-27B', '--lang', 'zh',
          '--main-layer', '39', '--band', '0-63', '--layers', 'band', '--bodies', 'bodies_zh_150_list.json', '--splits',
          '/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl', '--batch', '16', '--max-new', '3072', '--max-memory-gib', '44', '--save-gen']


def task(name, suite, out, extra=(), req=()):
    return {'name': name, 'kind': 'think', 'requires': list(req),
            'args': {'subprocess_cmd': [STABLE, 'crossmodel_thinking_ablate2.py', '--suite', suite, *COMMON, '--out', out, *extra],
                     'subprocess_log': f'{name}.log', 'subprocess_env': ENV}}


full = task('qwen38_abl4_full', 'full', 'thinking_ablate4/qwen38_27b')
content = task('qwen38_abl4_content', 'content', 'thinking_ablate4/qwen38_27b_content',
               extra=('--fit-from', 'thinking_ablate4/qwen38_27b/fit.pt'), req=(['worker_done_s3.log', 'DONE qwen38_abl4_full#0'],))
tasks = json.load(open(T, encoding='utf-8'))
names = [x['name'] for x in tasks]
assert 'qwen38_abl4_full' not in names
os.makedirs('thinking_ablate4', exist_ok=True)
for go in ('thinking_ablate4/qwen38_27b.GO', 'thinking_ablate4/qwen38_27b_content.GO'):
    open(go, 'w').close()
q = tasks[names.index('q35fp8_think900')]
q['requires'] = q['requires'] + [['worker_done_s3.log', 'DONE qwen38_abl4_full#0'], ['worker_done_s3.log', 'DONE qwen38_abl4_content#0']]
i = names.index('qwen38_v1sweep_smoke') + 1
new = tasks[:i] + [full, content] + tasks[i:]
snap = f'worker_tasks_s3_before_qwen_abl4_full_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(new, open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
print([x['name'] for x in new][:7], '| snapshot', snap)
