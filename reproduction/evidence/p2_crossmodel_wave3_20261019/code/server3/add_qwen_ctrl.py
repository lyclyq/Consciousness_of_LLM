"""server3 queue, user 10-24: specificity controls for the two replacements kept in the main text (all layers; the last third,
L43-63), Qwen3.8-27B, both modes, full (75 held-out bodies, one question each): the content question (the last number of the
message; all layers already run) and an unrelated two-digit multiplication asked after the edited message (crossmodel_role_swap.py
--question arith). Conditions as the all-layer content control. Appended after the third-band smokes. Atomic rewrite, snapshot kept."""
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
CON = ('base_A:assistant:-:-:0,swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,'
       'base_U:user:-:-:0,swA8_U:user:assistant:V:8')
COMMON = ['crossmodel_role_swap.py', '--model', Q, '--lang', 'zh', '--bodies', 'bodies_zh_150_list.json', '--splits', SPL,
          '--max-memory-gib', '44', '--conds', CON]
TH = COMMON + ['--mode', 'thinking', '--on-kwargs', '{}', '--fit-from', 'thinking_ablate4/qwen38_27b/fit.pt', '--batch', '16',
               '--max-new', '3072', '--save-gen']
SP = COMMON + ['--mode', 'single', '--on-kwargs', '{"enable_thinking": false}', '--fit-from', 'singlepass_ablate/qwen38_27b/fit.pt']
specs = []
for mode, base, root in (('think', TH, 'thinking_swap'), ('sp', SP, 'singlepass_swap')):
    specs.append((f'qwen38_ctrl_content_late_{mode}', base + ['--band', '43-63', '--question', 'content'], f'{root}/qwen38_27b_third_late_content'))
    specs.append((f'qwen38_ctrl_arith_all_{mode}', base + ['--band', '0-63', '--question', 'arith'], f'{root}/qwen38_27b_all_arith'))
    specs.append((f'qwen38_ctrl_arith_late_{mode}', base + ['--band', '43-63', '--question', 'arith'], f'{root}/qwen38_27b_third_late_arith'))
new = [{'name': n, 'kind': 'think', 'requires': [],
        'args': {'subprocess_cmd': [STABLE, *c, '--out', o], 'subprocess_log': f'{n}.log', 'subprocess_env': ENV}} for n, c, o in specs]
tasks = json.load(open(T, encoding='utf-8'))
names = [x['name'] for x in tasks]
assert not {t['name'] for t in new} & set(names)
for _, _, o in specs:
    os.makedirs(os.path.dirname(o), exist_ok=True)
    open(o + '.GO', 'w').close()
out = tasks + new
snap = f'worker_tasks_s3_before_ctrl_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(out, open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
print([x['name'] for x in out][-12:], '| snapshot', snap)
