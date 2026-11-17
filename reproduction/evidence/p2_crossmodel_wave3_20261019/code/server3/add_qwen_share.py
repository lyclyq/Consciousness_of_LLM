"""server3 queue, user 10-23: the epsilon check of the manipulated subspace for Qwen3.8-27B (crossmodel_subspace_share.py, rank 8)
on both intervention fits: the thinking fit (thinking rendering) and the single-pass fit (non-thinking rendering). Forward
passes only. Placed after the swap smokes, before q35_27b_think900's successors. Atomic rewrite, snapshot kept."""
import json
import os
import shutil
import time

T = 'worker_tasks_s3.json'
STABLE = '/root/recursive_state_moe_server3_20260711/venv/bin/python'
ENV = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONUNBUFFERED': '1',
       'OMP_NUM_THREADS': '32', 'MKL_NUM_THREADS': '32'}
Q = '/root/data/hf_models/Qwen/Qwen3.8-27B'
SPL = '/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl'
C = ['crossmodel_subspace_share.py', '--model', Q, '--lang', 'zh', '--bodies', 'bodies_zh_150_list.json', '--splits', SPL, '--k', '8',
     '--max-memory-gib', '44']
new = [('qwen38_share_think', C + ['--on-kwargs', '{}', '--fit-from', 'thinking_ablate4/qwen38_27b/fit.pt', '--out', 'subspace_share/qwen38_27b_think.json']),
       ('qwen38_share_sp', C + ['--on-kwargs', '{"enable_thinking": false}', '--fit-from', 'singlepass_ablate/qwen38_27b/fit.pt',
                                '--out', 'subspace_share/qwen38_27b_sp.json'])]
new = [{'name': n, 'kind': 'think', 'requires': [], 'args': {'subprocess_cmd': [STABLE, *c], 'subprocess_log': f'{n}.log', 'subprocess_env': ENV}}
       for n, c in new]
tasks = json.load(open(T, encoding='utf-8'))
names = [x['name'] for x in tasks]
assert not {t['name'] for t in new} & set(names)
os.makedirs('subspace_share', exist_ok=True)
i = names.index('qwen38_swap_sp_L39_smoke') + 1
out = tasks[:i] + new + tasks[i:]
snap = f'worker_tasks_s3_before_share_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(out, open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
n = [x['name'] for x in out]
print(n[n.index('q35_27b_think900'):n.index('q35_27b_think900') + 8], '| snapshot', snap)
