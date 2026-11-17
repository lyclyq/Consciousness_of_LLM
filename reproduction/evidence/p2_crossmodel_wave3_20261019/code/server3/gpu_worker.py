"""Long-lived GPU worker for server3.

On server3 a broken container NVIDIA state leaves running processes working while newly started ones see no GPU (NVML error).
So all GPU work runs inside ONE Python process that initialises CUDA once and then takes tasks from a JSON task file:
crossmodel_suite.run (phases G/P/M, --geometry-only) or crossmodel_thinking_b.run (thinking attribution). Tasks run one after
another, each loading its own model and releasing it afterwards. The task file is re-read after every task, so tasks can be
appended while the worker runs; when nothing is pending the worker waits (keeping its CUDA context) instead of exiting.
A task is done when its name is in the done log; a failing task is logged and not retried until its 'retry' field changes.
Stop the worker with STOP_WORKER, skip a task with STOP_W_<name>. Both runners skip items finished on disk, so a rerun resumes.
usage: gpu_worker.py TASKS.json DONE.log"""
from __future__ import annotations
import argparse
import gc
import importlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import crossmodel_suite as cs  # noqa: E402
import crossmodel_thinking_b as ct  # noqa: E402

MIN_FREE_GIB = 40          # results must always fit: no task starts below this much free disk
SUITE = dict(model=None, tmpl_kwargs='{}', bodies=None, problems=None, out=None, phases='G', patch_bodies=75, max_memory_gib=44,
             summarize_only=0, lang='zh', bodies_en=None, cpu_offload_gib=0, geometry_only=False, disk_offload_dir=None)
THINK = dict(model=None, off_kwargs='{"enable_thinking": false}', on_kwargs='{}', bodies_zh=None, out=None, max_new=3072,
             max_memory_gib=44, save_gen=False, lang='zh', batch=1, cpu_offload_gib=0, disk_offload_dir=None)


def log(path, msg):
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(f'{time.strftime("%F %T")} {msg}\n')


def done_keys(path):
    if not Path(path).exists():
        return set()
    keys = set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2] in ('DONE', 'FAILED'):
            keys.add(parts[3])
    return keys


def requirements_met(task):
    """A task may list 'requires': [[path, substring], ...]; it is pending only when every file exists and contains its substring."""
    for path, sub in task.get('requires', []):
        p = Path(path)
        if not p.exists() or (sub and sub not in p.read_text(encoding='utf-8', errors='replace')):
            return False
    return True


def main():
    tasks_path, done_path = sys.argv[1], sys.argv[2]
    log(done_path, f'WORKER start cuda={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}')
    if not torch.cuda.is_available():
        log(done_path, 'WORKER exit: CUDA unavailable')
        sys.exit(1)
    while not Path('STOP_WORKER').exists():
        tasks = json.loads(Path(tasks_path).read_text(encoding='utf-8'))
        seen = done_keys(done_path)
        pending = [t for t in tasks if f'{t["name"]}#{t.get("retry", 0)}' not in seen and not Path(f'STOP_W_{t["name"]}').exists()
                   and requirements_met(t)]
        if not pending:
            time.sleep(300)
            continue
        free_gib = shutil.disk_usage('.').free / 2 ** 30
        if free_gib < MIN_FREE_GIB:
            log(done_path, f'WAIT disk free {free_gib:.0f} GiB < {MIN_FREE_GIB} GiB, no new task started')
            time.sleep(600)
            continue
        t = pending[0]
        key = f'{t["name"]}#{t.get("retry", 0)}'
        a = dict(SUITE if t['kind'] == 'suite' else THINK)
        a.update(t['args'])
        importlib.reload(cs)                           # code fixes on disk apply to the next task without a restart
        importlib.reload(ct)
        log(done_path, f'START {key} {json.dumps(t["args"], ensure_ascii=False)}')
        t0 = time.time()
        try:
            (cs.run if t['kind'] == 'suite' else ct.run)(argparse.Namespace(**a))
            log(done_path, f'DONE {key} {time.time() - t0:.0f}s')
        except BaseException as e:                     # SystemExit included (e.g. CUDA lost for a new allocation)
            traceback.print_exc()
            sys.stdout.flush()
            log(done_path, f'FAILED {key} {time.time() - t0:.0f}s {type(e).__name__}: {str(e)[:200]}')
            if isinstance(e, KeyboardInterrupt):
                raise
        gc.collect()
        torch.cuda.empty_cache()
        log(done_path, f'freed {key} gpu_alloc_gib={torch.cuda.memory_allocated() / 2 ** 30:.2f}')
    log(done_path, 'WORKER stop flag')


if __name__ == '__main__':
    main()
