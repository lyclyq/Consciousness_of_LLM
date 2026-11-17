"""server3 queue, user 10-25: (1) the layer-window curve also for Gemma-4-31B without thinking (natural order; the readouts with a
single-pass effect are mapped, Qwen3.8-27B already queued): 8 consecutive layers, the window moving by 4 from L0-L7 to L52-L59
(14 windows of 60 layers), conditions and settings as gemma_nat_sp_third_*; (2) the 64-dimension diagnostic completed for every
readout, as gptoss_k64_diag_pro.sh for gpt-oss-20b: every layer, the leading 64 dimensions of the same fit, own <- User,
User <- own and their random-subspace controls; the attribution question in Qwen3.8-27B without and with thinking and in
Gemma-4-31B without thinking, and the games (secret, score; main question) in Qwen3.8-27B with thinking. The unedited conditions are
copied from the 8-dimension runs (identical prompts; the scripts skip items present), so only the 64-dimension conditions run.
Appended at the end of the queue, the windows first. Atomic rewrite, snapshot kept."""
import json
import os
import shutil
import time
from pathlib import Path

T = 'worker_tasks_s3.json'
STABLE = '/root/recursive_state_moe_server3_20260711/venv/bin/python'
ENV = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONUNBUFFERED': '1',
       'OMP_NUM_THREADS': '32', 'MKL_NUM_THREADS': '32'}
Q = '/root/data/hf_models/Qwen/Qwen3.8-27B'
G = '/root/data/hf_models/google/gemma-4-31B-it'
SPL = '/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl'
NOTH = '{"enable_thinking": false}'
REG = 'base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8'
K64 = ('base_A:assistant:-:-:0,base_U:user:-:-:0,swU64_A:assistant:user:V:64,swA64_U:user:assistant:V:64,'
       'rswU64_A:assistant:user:R:64,rswA64_U:user:assistant:R:64')
GSP = ['crossmodel_role_swap_nat.py', '--mode', 'single', '--model', G, '--on-kwargs', NOTH, '--lang', 'en', '--bodies', 'bodies_en_150_list.json',
       '--splits', SPL, '--max-memory-gib', '44', '--fit-from', 'singlepass_ablate/gemma4_31b_nat/fit.pt']
QSP = ['crossmodel_role_swap.py', '--model', Q, '--lang', 'zh', '--bodies', 'bodies_zh_150_list.json', '--splits', SPL, '--max-memory-gib', '44',
       '--mode', 'single', '--on-kwargs', NOTH, '--fit-from', 'singlepass_ablate/qwen38_27b/fit.pt']
QTH = ['crossmodel_role_swap.py', '--model', Q, '--lang', 'zh', '--bodies', 'bodies_zh_150_list.json', '--splits', SPL, '--max-memory-gib', '44',
       '--mode', 'thinking', '--on-kwargs', '{}', '--fit-from', 'thinking_ablate4/qwen38_27b/fit.pt', '--batch', '16', '--max-new', '3072',
       '--save-gen']
QGA = ['crossmodel_games.py', '--mode', 'thinking', '--model', Q, '--on-kwargs', '{}', '--items', 'games/games_zh.json', '--lang', 'zh',
       '--fit-from', 'thinking_ablate4/qwen38_27b/fit.pt', '--batch', '16', '--max-new', '3072', '--max-memory-gib', '44', '--save-gen',
       '--games', 'secret,score', '--qkind', 'main']
specs = [(f'gemma_window_sp_L{s:02d}_{s + 7:02d}', GSP + ['--band', f'{s}-{s + 7}', '--conds', REG],
          f'singlepass_swap/gemma4_31b_nat_window_L{s:02d}_{s + 7:02d}', None) for s in range(0, 53, 4)]
specs += [('qwen38_k64_sp_attr', QSP + ['--band', '0-63', '--conds', K64], 'singlepass_swap/qwen38_27b_all_k64', 'singlepass_swap/qwen38_27b_all'),
          ('gemma_k64_sp_attr', GSP + ['--band', '0-59', '--conds', K64], 'singlepass_swap/gemma4_31b_nat_all_k64',
           'singlepass_swap/gemma4_31b_nat_all'),
          ('qwen38_k64_th_attr', QTH + ['--band', '0-63', '--conds', K64], 'thinking_swap/qwen38_27b_all_k64', 'thinking_swap/qwen38_27b_all'),
          ('qwen38_k64_th_games', QGA + ['--band', '0-63', '--conds', K64], 'games_swap/qwen38_27b_main_k64', 'games_swap/qwen38_27b_main')]
new = [{'name': n, 'kind': 'think', 'requires': [],
        'args': {'subprocess_cmd': [STABLE, *c, '--out', o], 'subprocess_log': f'{n}.log', 'subprocess_env': ENV}} for n, c, o, _ in specs]
tasks = json.load(open(T, encoding='utf-8'))
names = [x['name'] for x in tasks]
assert not {t['name'] for t in new} & set(names)
for n, _, o, src in specs:
    if src:  # copy the unedited conditions of the 8-dimension run
        (Path(o) / 'items').mkdir(parents=True, exist_ok=True)
        got = [f for f in sorted((Path(src) / 'items').glob('base_[AU]__*.json'))]
        assert len(got) >= 150, (src, len(got))
        for f in got:
            if not (Path(o) / 'items' / f.name).exists():
                shutil.copy2(f, Path(o) / 'items' / f.name)
        print(n, len(got), 'unedited items copied from', src)
    open(o + '.GO', 'w').close()  # the scripts wait for <out>.GO
out = tasks + new
snap = f'worker_tasks_s3_before_gemma_window_k64_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(out, open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
print(len(new), 'tasks appended:', [t['name'] for t in new][:2], '...', [t['name'] for t in new][-4:], '| snapshot', snap)
