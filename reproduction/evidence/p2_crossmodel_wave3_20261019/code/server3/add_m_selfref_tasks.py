"""server3 queue, user 10-21: own-step trust only in the modes with self-reference, and every follow-up on the weights still on
disk before they are deleted. Added after seed_oss_think900 (before q35fp8_think900, which now also waits for them):
  dsv41_m_smoke -> dsv41_m (DSV41M_GO)          V4.1 without thinking (natural-order self-reference in a single pass)
  thinkm_smoke (Qwen3.8-27B, 16 items) -> qwen38_thinkm, exaone_thinkm (THINKM_GO)   own-step trust with thinking
Atomic rewrite, snapshot kept. usage (crossmodel folder): python add_m_selfref_tasks.py"""
import json
import os
import shutil
import time

T = 'worker_tasks_s3.json'
STABLE = '/root/recursive_state_moe_server3_20260711/venv/bin/python'
P_ZH = '/root/data/p2_small_checks_20261014/reasoning_m/problems.json'
Q38 = '/root/data/hf_models/Qwen/Qwen3.8-27B'
EXA = '/root/data/hf_models/LGAI-EXAONE/EXAONE-4.0.1-32B'
ENV = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONUNBUFFERED': '1'}
DENV = {'CUDA_VISIBLE_DEVICES': '0', **ENV}  # as nat3_dsv41; the shell script sets the GPU per shard


def thinkm(name, model, kw, problems, lang, out, extra=(), req=()):
    return {'name': name, 'kind': 'think', 'requires': list(req),
            'args': {'subprocess_cmd': [STABLE, 'crossmodel_thinking_m.py', '--model', model, '--on-kwargs', kw, '--problems', problems,
                                        '--lang', lang, '--out', out, '--batch', '16', '--max-new', '3072', '--max-memory-gib', '44',
                                        '--save-gen', *extra],
                     'subprocess_log': f'{name}.log', 'subprocess_env': ENV}}


new = [
    {'name': 'dsv41_m_smoke', 'kind': 'think', 'requires': [],
     'args': {'subprocess_cmd': ['/bin/bash', 'dsv41/run_dsv41_m.sh', 'suite_m/dsv41_flash_1017_smoke', '4'],
              'subprocess_log': 'dsv41_m_smoke.log', 'subprocess_env': DENV}},
    {'name': 'dsv41_m', 'kind': 'think', 'requires': [['worker_done_s3.log', 'DONE dsv41_m_smoke#0'], ['DSV41M_GO', '']],
     'args': {'subprocess_cmd': ['/bin/bash', 'dsv41/run_dsv41_m.sh', 'suite_m/dsv41_flash_1017'],
              'subprocess_log': 'dsv41_m.log', 'subprocess_env': DENV}},
    thinkm('thinkm_smoke', Q38, '{}', P_ZH, 'zh', 'thinking_m/qwen38_27b_smoke', extra=('--limit', '16')),
    thinkm('qwen38_thinkm', Q38, '{}', P_ZH, 'zh', 'thinking_m/qwen38_27b',
           req=(['worker_done_s3.log', 'DONE thinkm_smoke#0'], ['THINKM_GO', ''])),
    thinkm('exaone_thinkm', EXA, '{"enable_thinking": true}', 'problems_en_120.json', 'en', 'thinking_m/exaone4_32b_en',
           req=(['worker_done_s3.log', 'DONE thinkm_smoke#0'], ['THINKM_GO', ''])),
]
tasks = json.load(open(T, encoding='utf-8'))
names = [t['name'] for t in tasks]
assert not set(names) & {t['name'] for t in new}, 'already queued'
for f in ('dsv41/dsv41_m.py', 'dsv41/run_dsv41_m.sh', 'crossmodel_thinking_m.py', 'problems_en_120.json', P_ZH):
    assert os.path.exists(f), f
q = tasks[names.index('q35fp8_think900')]
q['requires'] = q['requires'] + [['worker_done_s3.log', f'DONE {n}#0'] for n in ('dsv41_m', 'qwen38_thinkm', 'exaone_thinkm')]
i = names.index('seed_oss_think900') + 1
snap = f'worker_tasks_s3_before_m_selfref_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(tasks[:i] + new + tasks[i:], open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
print([t['name'] for t in tasks[:i] + new + tasks[i:]][names.index('q35fp8_G'):], '| snapshot', snap)
