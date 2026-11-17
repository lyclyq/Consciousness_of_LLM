"""server3 queue, user 10-25: a depth curve of the replacement effect in Qwen3.8-27B without thinking. The eight coordinates are
replaced at every position but only in a window of 8 consecutive layers, the window moving by 4 layers from L0-L7 to L56-L63
(15 windows); conditions as the third-band runs (unedited own and User messages, own <- User, own <- User in a random subspace,
User <- own). Same script, subspace and settings as qwen38_third_sp_*_full. Every task requires QWEN_CURVE_GO (written once
server3 has its GPUs back). Appended at the end of the queue. Atomic rewrite, snapshot kept."""
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
REG = 'base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8'
SP = ['crossmodel_role_swap.py', '--model', Q, '--lang', 'zh', '--bodies', 'bodies_zh_150_list.json', '--splits', SPL, '--max-memory-gib', '44',
      '--conds', REG, '--mode', 'single', '--on-kwargs', '{"enable_thinking": false}', '--fit-from', 'singlepass_ablate/qwen38_27b/fit.pt']
specs = [(f'qwen38_window_sp_L{s:02d}_{s + 7:02d}', SP + ['--band', f'{s}-{s + 7}'], f'singlepass_swap/qwen38_27b_window_L{s:02d}_{s + 7:02d}')
         for s in range(0, 57, 4)]
new = [{'name': n, 'kind': 'think', 'requires': [['QWEN_CURVE_GO', '']],
        'args': {'subprocess_cmd': [STABLE, *c, '--out', o], 'subprocess_log': f'{n}.log', 'subprocess_env': ENV}} for n, c, o in specs]
tasks = json.load(open(T, encoding='utf-8'))
names = [x['name'] for x in tasks]
assert not {t['name'] for t in new} & set(names)
for _, _, o in specs:
    open(o + '.GO', 'w').close()  # crossmodel_role_swap.py waits for <out>.GO; the queue gate is QWEN_CURVE_GO
out = tasks + new
snap = f'worker_tasks_s3_before_window_curve_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(out, open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
print(len(new), 'window tasks:', [s[0] for s in specs][:2], '...', [s[0] for s in specs][-1], '| snapshot', snap)
