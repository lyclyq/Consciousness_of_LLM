"""10-20 (user-approved): natural-order three-role calibration (crossmodel_natural3.py) on the six reference models whose
templates keep every message in place: a smoke test on GLM-4.6V-Flash first (3 bodies), then the 150-body runs with each
model's main-probe settings (template kwargs, language, environment), GLM-5.3-Flash last (geometry only, as its main run).
Every run is a subprocess of the worker, one after another. Backed up and replaced atomically.
usage (in the crossmodel directory): python add_nat3_tasks.py"""
import json
import os
import shutil
import time
from pathlib import Path

ST = '/root/recursive_state_moe_server3_20260711/venv/bin/python'
T26 = '/root/data/venv_t26/bin/python'
M = '/root/data/hf_models'
B = '/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl'
OFF = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}


def task(name, py, model, kw, out, extra=(), env=None, requires=()):
    cmd = [py, 'crossmodel_natural3.py', '--model', f'{M}/{model}', '--tmpl-kwargs', json.dumps(kw), '--bodies', B, '--out', out, *extra]
    return {'name': name, 'kind': 'think', 'requires': [list(r) for r in requires],
            'args': {'subprocess_cmd': cmd, 'subprocess_log': f'{name}.log', 'subprocess_env': dict(env or OFF)}}


smoke_done = [('worker_done_s3.log', 'DONE nat3_smoke#0'), ('NAT3_GO', '')]   # NAT3_GO: created by hand after the smoke output is checked
new = [
    task('nat3_smoke', ST, 'zai-org/GLM-4.6V-Flash', {'enable_thinking': False}, 'suite_nat3/glm46v_smoke',
         ['--max-memory-gib', '44']),
    task('nat3_glm46v', ST, 'zai-org/GLM-4.6V-Flash', {'enable_thinking': False}, 'suite_nat3/glm46v_flash',
         ['--max-memory-gib', '44'], requires=smoke_done),
    task('nat3_glm47', ST, 'zai-org/GLM-4.7-Flash', {'enable_thinking': False}, 'suite_nat3/glm47_flash',
         ['--max-memory-gib', '44'], requires=smoke_done),
    task('nat3_exaone', ST, 'LGAI-EXAONE/EXAONE-4.0.1-32B', {'enable_thinking': False}, 'suite_nat3/exaone4_32b_en',
         ['--max-memory-gib', '44', '--lang', 'en', '--bodies-en', 'bodies_en_150.json'], requires=smoke_done),
    task('nat3_gemma', ST, 'google/gemma-4-31B-it', {}, 'suite_nat3/gemma4_31b_en',
         ['--max-memory-gib', '44', '--lang', 'en', '--bodies-en', 'bodies_en_150.json'], requires=smoke_done),
    task('nat3_seed', ST, 'ByteDance-Seed/Seed-OSS-36B-Instruct', {'thinking_budget': 0}, 'suite_nat3/seed_oss_36b',
         ['--max-memory-gib', '44'], requires=smoke_done),
    task('nat3_glm53f', T26, 'zai-org/GLM-5.3-Flash', {}, 'suite_nat3/glm53_flash',
         ['--geometry-only', '--max-memory-gib', '40', '--cpu-offload-gib', '300'],
         env={**OFF, 'P2_FP8_FORCE_FALLBACK': '1'}, requires=smoke_done),
]
smoke = new[0]
smoke['args']['subprocess_cmd'][smoke['args']['subprocess_cmd'].index('--bodies') + 1] = 'role_bodies_3.jsonl'
p = Path('worker_tasks_s3.json')
shutil.copy(p, f'worker_tasks_s3_before_nat3_{time.strftime("%H%M%S")}.json')
tasks = json.loads(p.read_text(encoding='utf-8'))
names = {t['name'] for t in tasks}
for t in new:
    assert t['name'] not in names, t['name']
    tasks.append(t)
    print(t['name'], ' '.join(t['args']['subprocess_cmd'][1:]))
tmp = p.with_suffix('.json.tmp')
tmp.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding='utf-8')
os.replace(tmp, p)
