"""server3 queue, user 10-22: the fourth thinking design without thinking for Qwen3.8-27B (crossmodel_singlepass_ablate.py, all 64
layers, the non-thinking rendering, letter probabilities at the onset) and its content-question control (--fit-from), after
qwen38_abl4_content and before qwen38_v1L39_full; q35fp8_think900 also waits for both. Atomic rewrite, snapshot kept."""
import json
import os
import shutil
import time

T = 'worker_tasks_s3.json'
STABLE = '/root/recursive_state_moe_server3_20260711/venv/bin/python'
ENV = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONUNBUFFERED': '1',
       'OMP_NUM_THREADS': '32', 'MKL_NUM_THREADS': '32'}
COMMON = ['--model', '/root/data/hf_models/Qwen/Qwen3.8-27B', '--lang', 'zh', '--band', '0-63', '--bodies', 'bodies_zh_150_list.json',
          '--splits', '/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl', '--max-memory-gib', '44']


def task(name, out, extra=(), req=()):
    return {'name': name, 'kind': 'think', 'requires': list(req),
            'args': {'subprocess_cmd': [STABLE, 'crossmodel_singlepass_ablate.py', *COMMON, '--out', out, *extra],
                     'subprocess_log': f'{name}.log', 'subprocess_env': ENV}}


a = task('qwen38_sp_ablate', 'singlepass_ablate/qwen38_27b')
c = task('qwen38_sp_ablate_content', 'singlepass_ablate/qwen38_27b_content', extra=('--question', 'content', '--fit-from', 'singlepass_ablate/qwen38_27b/fit.pt'),
         req=(['worker_done_s3.log', 'DONE qwen38_sp_ablate#0'],))
tasks = json.load(open(T, encoding='utf-8'))
names = [x['name'] for x in tasks]
assert 'qwen38_sp_ablate' not in names and os.path.exists('crossmodel_singlepass_ablate.py')
q = tasks[names.index('q35fp8_think900')]
q['requires'] = q['requires'] + [['worker_done_s3.log', 'DONE qwen38_sp_ablate#0'], ['worker_done_s3.log', 'DONE qwen38_sp_ablate_content#0']]
i = names.index('qwen38_abl4_content') + 1
new = tasks[:i] + [a, c] + tasks[i:]
snap = f'worker_tasks_s3_before_qwen_sp_ablate_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(new, open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
print([x['name'] for x in new][4:11], '| snapshot', snap)
