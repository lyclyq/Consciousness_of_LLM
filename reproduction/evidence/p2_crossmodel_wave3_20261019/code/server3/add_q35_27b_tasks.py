"""server3 queue, user 10-23: Qwen3.5-122B leaves the main line (its thinking 900 runs about 92 h: 97 min per batch of 16 with the
reference linear-attention kernels across 4 GPUs). Qwen3.5-27B (the architecture of Qwen3.8-27B, Qwen3_5ForConditionalGeneration,
revision fc05daec) takes its place on the main-line protocol of Qwen3.8-27B: smoke (3 bodies, G), G, thinking 900, M, and the
thinking-rendering geometry, gated by Q35_27B_GTHINK_GO (written only if thinking 900 shows thinking self-reference); thinking M as a
side check. Chinese bodies; enable_thinking false / template default. The full tasks wait for Q35_27B_GO (written after the smoke
has been read). Placed after the Qwen3.8-27B interventions. The side checks that waited for q35fp8_think900 now wait for
q35_27b_think900; q35fp8_think900 and q35fp8_thinkm get STOP_W_ flags. Atomic rewrite, snapshot kept."""
import json
import os
import shutil
import time

T = 'worker_tasks_s3.json'
B = '/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl'
P = '/root/data/p2_small_checks_20261014/reasoning_m/problems.json'
STABLE = '/root/recursive_state_moe_server3_20260711/venv/bin/python'
M = '/root/data/hf_models/Qwen/Qwen3.5-27B'
OFF, ON = '{"enable_thinking": false}', '{}'
DL = [['download_qwen35_27b.log', 'DONE Qwen3.5-27B']]
GO = [['worker_done_s3.log', 'DONE q35_27b_smoke#0'], ['Q35_27B_GO', '']] + DL
OLD, NEW = ['worker_done_s3.log', 'DONE q35fp8_think900#0'], ['worker_done_s3.log', 'DONE q35_27b_think900#0']
new = [
    {'name': 'q35_27b_smoke', 'kind': 'suite', 'requires': DL,
     'args': {'model': M, 'tmpl_kwargs': OFF, 'bodies': 'role_bodies_3.jsonl', 'problems': P, 'out': 'suite/qwen35_27b_smoke', 'phases': 'G', 'max_memory_gib': 44}},
    {'name': 'q35_27b_G', 'kind': 'suite', 'requires': GO,
     'args': {'model': M, 'tmpl_kwargs': OFF, 'bodies': B, 'problems': P, 'out': 'suite/qwen35_27b', 'phases': 'G', 'max_memory_gib': 44}},
    {'name': 'q35_27b_think900', 'kind': 'think', 'requires': GO,
     'args': {'model': M, 'off_kwargs': OFF, 'on_kwargs': ON, 'bodies_zh': 'bodies_zh_150_list.json', 'out': 'thinking/qwen35_27b',
              'batch': 16, 'max_new': 3072, 'max_memory_gib': 44}},
    {'name': 'q35_27b_M', 'kind': 'suite', 'requires': GO,
     'args': {'model': M, 'tmpl_kwargs': OFF, 'bodies': B, 'problems': P, 'out': 'suite/qwen35_27b_M', 'phases': 'M', 'max_memory_gib': 44}},
    {'name': 'q35_27b_gthink', 'kind': 'suite', 'requires': GO + [['Q35_27B_GTHINK_GO', '']],
     'args': {'model': M, 'tmpl_kwargs': ON, 'bodies': B, 'problems': P, 'out': 'suite_think/qwen35_27b', 'phases': 'G', 'geometry_only': True,
              'max_memory_gib': 44}},
]
thinkm = {'name': 'q35_27b_thinkm', 'kind': 'think', 'requires': [['worker_done_s3.log', 'DONE thinkm_smoke#0'], ['THINKM_GO', ''], NEW],
          'args': {'subprocess_cmd': [STABLE, 'crossmodel_thinking_m.py', '--model', M, '--on-kwargs', ON, '--problems', P, '--lang', 'zh',
                                      '--out', 'thinking_m/qwen35_27b', '--batch', '16', '--max-new', '3072', '--max-memory-gib', '44', '--save-gen'],
                   'subprocess_log': 'q35_27b_thinkm.log',
                   'subprocess_env': {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                                      'PYTHONUNBUFFERED': '1'}}}
tasks = json.load(open(T, encoding='utf-8'))
names = [x['name'] for x in tasks]
assert not {t['name'] for t in new + [thinkm]} & set(names), 'already queued'
moved = []
for t in tasks:
    if OLD in t.get('requires', []):
        t['requires'] = [NEW if r == OLD else r for r in t['requires']]
        moved.append(t['name'])
for n in ('q35fp8_think900', 'q35fp8_thinkm'):
    open(f'STOP_W_{n}', 'w').close()
i = names.index('qwen38_replace_L39_sp') + 1
out = tasks[:i] + new + tasks[i:] + [thinkm]
snap = f'worker_tasks_s3_before_q35_27b_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(out, open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
n = [x['name'] for x in out]
print('requires moved to q35_27b_think900:', moved)
print(n[n.index('q35fp8_think900'):], '| snapshot', snap)
