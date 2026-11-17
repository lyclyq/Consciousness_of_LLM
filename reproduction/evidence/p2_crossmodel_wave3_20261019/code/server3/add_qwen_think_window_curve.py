"""server3 queue, user 10-26: the layer-window map of the replacement for Qwen3.8-27B with thinking (the thirds are dropped from the
paper; the windows replace them for every readout). The eight coordinates are replaced at every position but only in a window of 8
consecutive layers, moving by 4 from L0-L7 to L56-L63 (15 windows), as for the single-pass maps (add_qwen_window_curve.py);
conditions and settings as the thinking third runs (qwen38_third_think_*_full: thinking rendering, thinking fit, batch 16, 3072 new
tokens). The unedited own and User items are copied from thinking_swap/qwen38_27b_all (identical prompts; the runner skips items on
disk), so each window runs own <- User, own <- User random, User <- own. Appended at the end of the queue. Atomic rewrite, snapshot kept."""
import glob
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
TH = ['crossmodel_role_swap.py', '--model', Q, '--lang', 'zh', '--bodies', 'bodies_zh_150_list.json', '--splits', SPL, '--max-memory-gib', '44',
      '--conds', REG, '--mode', 'thinking', '--on-kwargs', '{}', '--fit-from', 'thinking_ablate4/qwen38_27b/fit.pt', '--batch', '16',
      '--max-new', '3072', '--save-gen']
specs = [(f'qwen38_window_th_L{s:02d}_{s + 7:02d}', TH + ['--band', f'{s}-{s + 7}'], f'thinking_swap/qwen38_27b_window_L{s:02d}_{s + 7:02d}')
         for s in range(0, 57, 4)]
base = sorted(glob.glob('thinking_swap/qwen38_27b_all/items/base_[AU]__*.json'))
assert len(base) == 300, len(base)
for _, _, o in specs:
    os.makedirs(o + '/items', exist_ok=True)
    for f in base:
        dst = os.path.join(o, 'items', os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)
    open(o + '.GO', 'w').close()  # crossmodel_role_swap.py waits for <out>.GO
new = [{'name': n, 'kind': 'think', 'requires': [],
        'args': {'subprocess_cmd': [STABLE, *c, '--out', o], 'subprocess_log': f'{n}.log', 'subprocess_env': ENV}} for n, c, o in specs]
tasks = json.load(open(T, encoding='utf-8'))
names = [x['name'] for x in tasks]
assert not {t['name'] for t in new} & set(names)
out = tasks + new
snap = f'worker_tasks_s3_before_think_window_curve_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(out, open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
print(len(new), 'window tasks:', [s[0] for s in specs][:2], '...', [s[0] for s in specs][-1], f'| {len(base)} unedited items copied per window',
      '| snapshot', snap)
