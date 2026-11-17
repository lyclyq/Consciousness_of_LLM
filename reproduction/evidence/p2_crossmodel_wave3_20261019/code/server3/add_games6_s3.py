"""server3 queue, user 10-27: the secret game v6 (make_games_secret6.py: v5 plus one System sentence that lets a question which is
not a move be answered; the gpt-oss models took the v5 control for a move, their v6 smokes pass on PRO) for the three server3
readouts, appended after the Flash-Next tasks. Everything else as in the v5 queue script (same runners, fits, bands, batch,
memory, eight dimensions): smokes (8 games, main and control), check_games6_s3.py (the release rule fixed in advance; .GO or
.SKIP), full runs (100 games, main then control). Atomic rewrite, snapshot kept."""
import json
import os
import shutil
import time

T = 'worker_tasks_s3.json'
STABLE = '/root/recursive_state_moe_server3_20260711/venv/bin/python'
ENV = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONUNBUFFERED': '1',
       'OMP_NUM_THREADS': '32', 'MKL_NUM_THREADS': '32'}
C8 = ('base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,'
      'rswA8_U:user:assistant:R:8')
NOTHINK = '{"enable_thinking": false}'
READ = [  # key, output tag, runner, mode, model, template kwargs, items, lang, fit, band
    ('qsp', 'qwen38_27b_sp', 'crossmodel_games.py', 'single', '/root/data/hf_models/Qwen/Qwen3.8-27B', NOTHINK,
     'games/games_zh_secret6.json', 'zh', 'singlepass_ablate/qwen38_27b/fit.pt', '0-63'),
    ('gsp', 'gemma4_31b_nat_sp', 'crossmodel_games_nat.py', 'single', '/root/data/hf_models/google/gemma-4-31B-it', NOTHINK,
     'games/games_en_secret6.json', 'en', 'singlepass_ablate/gemma4_31b_nat/fit.pt', '0-59'),
    ('qth', 'qwen38_27b', 'crossmodel_games.py', 'thinking', '/root/data/hf_models/Qwen/Qwen3.8-27B', '{}',
     'games/games_zh_secret6.json', 'zh', 'thinking_ablate4/qwen38_27b/fit.pt', '0-63'),
]


def cmd(r, q, out, smoke):
    key, tag, script, mode, model, kw, items, lang, fit, band = r
    c = [STABLE, script, '--mode', mode, '--model', model, '--on-kwargs', kw, '--items', items, '--games', 'secret', '--lang', lang,
         '--fit-from', fit, '--band', band, '--conds', C8, '--qkind', q, '--batch', '16', '--max-new', '3072', '--max-memory-gib', '44',
         '--save-gen', '--out', out]
    return c + (['--limit-bodies', '8'] if smoke else [])


def task(name, c):
    return {'name': name, 'kind': 'think', 'requires': [], 'args': {'subprocess_cmd': c, 'subprocess_log': f'{name}.log', 'subprocess_env': ENV}}


new = []
for r in READ:
    for q in ('main', 'control'):
        new.append(task(f'g6_{r[0]}_smoke_{q}', cmd(r, q, f'games_swap/{r[1]}_s6_smoke_{q}', True)))
new.append(task('g6_check', [STABLE, 'check_games6_s3.py']))
for r in READ:
    for q in ('main', 'control'):
        new.append(task(f'g6_{r[0]}_{q}', cmd(r, q, f'games_swap/{r[1]}_s6_{q}', False)))
tasks = json.load(open(T, encoding='utf-8'))
names = [x['name'] for x in tasks]
assert not {t['name'] for t in new} & set(names)
i = len(tasks)
out = tasks[:i] + new + tasks[i:]
snap = f'worker_tasks_s3_before_games6_{time.strftime("%Y%m%d_%H%M%S")}.json'
shutil.copyfile(T, snap)
json.dump(out, open(T + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
os.replace(T + '.tmp', T)
print([x['name'] for x in out][i - 1:i + len(new) + 1], '| snapshot', snap)
