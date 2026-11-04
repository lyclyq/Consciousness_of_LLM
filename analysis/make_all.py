"""Run every analysis script in order: results package (and feature package) in, every table, figure and summary of the paper
out. Each script runs in its own Python process (the figure scripts set their own matplotlib styles); its printout goes to
OUT/logs/<script>.txt. Outputs keep the relative paths of the archived ones (see README.md).

usage: make_all.py --results RES --out OUT [--features FEAT] [--item-cache DIR] [--paper-source DIR] [--defgeo DIR]
                   [--reasoning-dir DIR] [--only NAME[,NAME...]] [--skip NAME[,NAME...]]
  --features       the feature package (two_level_geometry and eq1_tests need it; skipped without it)
  --item-cache     cache of per-item records (speeds up the causal-test scripts on Windows)
  --paper-source   the paper's LaTeX source folder, for the phrase and table checks of hand_checks
  --defgeo         the per-layer features of the Qwen3.8-27B message-content run (not in the packages): message_geometry and
                   the Qwen part of eq1_tests
  --reasoning-dir  the per-problem records of the two-level reasoning run (not in the packages): ownstep's reasoning summary
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# (script, extra arguments it accepts: 'features', 'cache', 'paper', 'defgeo', 'reasoning')
STEPS = [('crossmodel_classification', ()), ('thresholds_orders', ()), ('model_templates', ()), ('ownstep', ('reasoning',)),
         ('causal_tests', ('cache',)), ('causal_controls', ('cache',)), ('hand_checks', ('paper',)), ('two_level_thinking', ()),
         ('fig_crossmodel', ()), ('natural_order', ()), ('benchmarks', ()), ('causal_tables', ()), ('english_ablation', ()), ('fig_depth', ()),
         ('fig_inducer', ()), ('fig_fd_layers', ()), ('two_level_geometry', ('features',)), ('fig_geometry', ()),
         ('fig_selfref', ()), ('fig_thinking', ()), ('fig_embedding', ()), ('eq1_tests', ('features', 'defgeo')),
         ('message_geometry', ('defgeo_only',)), ('appendix_tex', ()), ('fig_overview', ()), ('fig_readout', ()),
         ('fig_design', ()), ('fig_tasks', ())]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--features')
    ap.add_argument('--item-cache')
    ap.add_argument('--paper-source')
    ap.add_argument('--defgeo')
    ap.add_argument('--reasoning-dir')
    ap.add_argument('--only')
    ap.add_argument('--skip')
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    skip = set(a.skip.split(',')) if a.skip else set()
    logs = Path(a.out) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    failed = []
    for name, extra in STEPS:
        if (only is not None and name not in only) or name in skip:
            continue
        if 'defgeo_only' in extra:
            if not a.defgeo:
                print(f'{name}: skipped (needs --defgeo: inputs not in the packages)')
                continue
            cmd = [sys.executable, str(HERE / f'{name}.py'), '--defgeo', a.defgeo, '--out', a.out]
        else:
            cmd = [sys.executable, str(HERE / f'{name}.py'), '--results', a.results, '--out', a.out]
            if 'features' in extra:
                if not a.features:
                    print(f'{name}: skipped (needs --features)')
                    continue
                cmd += ['--features', a.features]
            if 'cache' in extra and a.item_cache:
                cmd += ['--item-cache', a.item_cache]
            if 'paper' in extra and a.paper_source:
                cmd += ['--paper-source', a.paper_source]
            if 'defgeo' in extra and a.defgeo:
                cmd += ['--defgeo', a.defgeo]
            if 'reasoning' in extra and a.reasoning_dir:
                cmd += ['--reasoning-dir', a.reasoning_dir]
        t0 = time.time()
        with open(logs / f'{name}.txt', 'w', encoding='utf-8') as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        print(f'{name}: {"ok" if rc == 0 else f"FAILED (exit {rc})"} in {time.time() - t0:.0f} s', flush=True)
        if rc:
            failed.append(name)
    print('all steps ok' if not failed else 'failed: ' + ', '.join(failed))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
