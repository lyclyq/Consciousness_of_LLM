"""server3 queue, user 10-23: the full runs of the per-position replacement for Qwen3.8-27B (crossmodel_role_swap.py 2b032831), both
modes, rank 8, all 75 held-out bodies x 2 orders, aligned with gpt-oss-20b: all layers (ten conditions) with the content
control; without thinking also L39 (a prompt-only edit is meaningful there: no generated tokens; the paper's sufficiency test is
at L39) with its content control; the three regions (body / end of the message / question) at all layers, both modes.
Thinking single-layer replacement stays a smoke (prompt only; gpt-oss L12 null). Every task requires QWEN_SWAP_FULL_GO, written
only after the Qwen smokes have been read; each output folder's .GO is written here so the script does not wait again.
Placed after the region smokes, before q35_27b_M. Atomic rewrite, snapshot kept."""
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
ALL = ('base_A:assistant:-:-:0,base_U:user:-:-:0,base_S:system:-:-:0,idA8_A:assistant:assistant:V:8,'
       'swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,'
       'swA8_U:user:assistant:V:8,rswA8_U:user:assistant:R:8,swA8_S:system:assistant:V:8')
CON = ('base_A:assistant:-:-:0,swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,'
       'base_U:user:-:-:0,swA8_U:user:assistant:V:8')
REG = 'base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8'
COMMON = ['crossmodel_role_swap.py', '--model', Q, '--lang', 'zh', '--bodies', 'bodies_zh_150_list.json', '--splits', SPL, '--max-memory-gib', '44']
TH = COMMON + ['--mode', 'thinking', '--on-kwargs', '{}', '--fit-from', 'thinking_ablate4/qwen38_27b/fit.pt', '--batch', '16',
               '--max-new', '3072', '--save-gen']
SP = COMMON + ['--mode', 'single', '--on-kwargs', '{"enable_thinking": false}', '--fit-from', 'singlepass_ablate/qwen38_27b/fit.pt']
specs = [
    ('qwen38_swap_think_all_full', TH + ['--band', '0-63', '--conds', ALL], 'thinking_swap/qwen38_27b_all'),
    ('qwen38_swap_think_all_content', TH + ['--band', '0-63', '--conds', CON, '--question', 'content'], 'thinking_swap/qwen38_27b_all_content'),
    ('qwen38_swap_sp_all_full', SP + ['--band', '0-63', '--conds', ALL], 'singlepass_swap/qwen38_27b_all'),
    ('qwen38_swap_sp_all_content', SP + ['--band', '0-63', '--conds', CON, '--question', 'content'], 'singlepass_swap/qwen38_27b_all_content'),
    ('qwen38_swap_sp_L39_full', SP + ['--band', '39-39', '--conds', ALL], 'singlepass_swap/qwen38_27b_L39'),
    ('qwen38_swap_sp_L39_content', SP + ['--band', '39-39', '--conds', CON, '--question', 'content'], 'singlepass_swap/qwen38_27b_L39_content'),
]
for reg in ('body', 'end', 'question'):
    specs.append((f'qwen38_region_think_{reg}_full', TH + ['--band', '0-63', '--conds', REG, '--region', reg], f'thinking_swap/qwen38_27b_region_{reg}'))
    specs.append((f'qwen38_region_sp_{reg}_full', SP + ['--band', '0-63', '--conds', REG, '--region', reg], f'singlepass_swap/qwen38_27b_region_{reg}'))
new = [{'name': n, 'kind': 'think', 'requires': [['QWEN_SWAP_FULL_GO', '']],
        'args': {'subprocess_cmd': [STABLE, *c, '--out', o], 'subprocess_log': f'{n}.log', 'subprocess_env': ENV}} for n, c, o in specs]
tasks = json.load(open(T, encoding='utf-8'))
names = [x['name'] for x in tasks]
assert not {t['name'] for t in new} & set(names)
for _, _, o in specs:
    os.makedirs(os.path.dirname(o), exist_ok=True)
    open(o + '.GO', 'w').close()
i = names.index('qwen38_region_sp_question') + 1
out = tasks[:i] + new + tasks[i:]
snap = f'worker_tasks_s3_before_swap_full_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(out, open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
n = [x['name'] for x in out]
print(n[n.index('qwen38_region_sp_question'):n.index('qwen38_region_sp_question') + 14], '| snapshot', snap)
