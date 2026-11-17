"""server3 queue, user 10-21 afternoon ("统一 调整顺序"): one protocol for every model, and the long Qwen3.5-122B thinking run last.
  * add seed_oss_M (phase M, as every model with a single-pass mode) and seed_oss_think900 (150 bodies x 3 roles x 2 orders; the
    180 items of seed_oss_think180 are reused by file name: same arguments, pilot bodies a subset of the 150 with identical text);
  * move q35fp8_think900 to the end of the queue and make it wait for every Granite task and the two Seed tasks, so that it
    cannot start while the Granite tasks still wait for their download or for GRANITE_GO.
The worker starts the first pending task in file order. Atomic rewrite, snapshot kept. usage (crossmodel folder): python reorder_s3_unify.py"""
import json
import os
import shutil
import time

T = 'worker_tasks_s3.json'
SEED = '/root/data/hf_models/ByteDance-Seed/Seed-OSS-36B-Instruct'
B = '/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl'
P = '/root/data/p2_small_checks_20261014/reasoning_m/problems.json'

pilot = json.load(open('pilot_bodies_zh.json', encoding='utf-8'))
full = {b['id']: b['body'] for b in json.load(open('bodies_zh_150_list.json', encoding='utf-8'))}
assert len(pilot) == 30 and all(full.get(b['id']) == b['body'] for b in pilot), 'pilot bodies are not a subset of the 150'

tasks = json.load(open(T, encoding='utf-8'))
by = {t['name']: t for t in tasks}
assert 'seed_oss_M' not in by and 'seed_oss_think900' not in by, 'already queued'
think180 = by['seed_oss_think180']['args']
seed_think = dict(think180, bodies_zh='bodies_zh_150_list.json')
assert {k: v for k, v in seed_think.items() if k != 'bodies_zh'} == {k: v for k, v in think180.items() if k != 'bodies_zh'}
new = [
    {'name': 'seed_oss_M', 'kind': 'suite', 'requires': [],
     'args': {'model': SEED, 'tmpl_kwargs': '{"thinking_budget": 0}', 'bodies': B, 'problems': P, 'out': 'suite/seed_oss_36b_M',
              'phases': 'M', 'max_memory_gib': 44}},
    {'name': 'seed_oss_think900', 'kind': 'think', 'requires': [], 'args': seed_think},
]
q = by['q35fp8_think900']
gate = ['granite41_G', 'granite41_M', 'nat3_granite41', 'granite42_G', 'granite42_gthink', 'granite42_M', 'granite42_think900',
        'seed_oss_M', 'seed_oss_think900']
assert all(n in by for n in gate[:7])
q['requires'] = q.get('requires', []) + [['worker_done_s3.log', f'DONE {n}#0'] for n in gate]
rest = [t for t in tasks if t['name'] != 'q35fp8_think900']
snap = f'worker_tasks_s3_before_unify_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(rest + new + [q], open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
print('queued seed_oss_M, seed_oss_think900; q35fp8_think900 moved last, gated on', gate, '| snapshot', snap)
