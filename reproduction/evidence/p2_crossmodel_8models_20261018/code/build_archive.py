"""Build the P2 cross-model archive (8 models, wave 1) from the raw server pulls, re-runnable.

Layout under ARCHIVE:
  _raw/*.tgz                         server pulls (no npz vectors; vectors stay on the servers, paths in README)
  code/                              exact scripts + SHA256 manifest
  items/                             item packages (bodies zh/en, pilot bodies, problems zh/en, question wording)
  models/<model>/<mode>/<measure>/   per-item JSON as written by the runs, plus that run's identity/summary
       mode = non_thinking | thinking;  measure = geo_attr (geometry + attribution) | patch | reason | think_attr | geo | natural | ...
       model / measure short names: see README_zh.md
  stats/                             tables computed here (CSV + JSON)
usage: build_archive.py ARCHIVE SCRATCH_CROSSMODEL_DIR
"""
from __future__ import annotations
import csv
import glob
import hashlib
import json
import os
import shutil
import sys
import tarfile
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

A = Path(sys.argv[1])
SRC = Path(sys.argv[2])
X = A / '_raw' / 'extract'

# (model, mode, measure) -> (server tag, path inside that server's crossmodel dir)
MAP = [
    ('qwen38_27b', 'non_thinking', 'geo_attr', 'pro', 'qwen38_27b/G'),
    ('qwen38_27b', 'non_thinking', 'patch', 'pro', 'qwen38_27b/P'),
    ('qwen38_27b', 'non_thinking', 'reason', 'pro', 'qwen38_27b/M'),
    ('qwen38_27b', 'non_thinking', 'geo_attr_s3', 's3', 'suite/qwen38_27b_bf16_s3/G'),
    ('qwen38_27b', 'non_thinking', 'geo_attr_en', 'pro', 'qwen38_27b_en/G'),
    ('qwen38_27b', 'non_thinking', 'patch_en', 's3', 'suite/qwen38_27b_en_P/P'),
    ('qwen38_27b', 'thinking', 'geo', 's3', 'suite_think/qwen38_27b/G'),
    ('qwen38_27b', 'non_thinking', 'geo_attr_simfp8', 's3', 'suite/qwen38_27b_simfp8_s3/G'),
    ('qwen35_122b', 'non_thinking', 'geo_attr', 's3', 'suite/qwen35_122b/G'),
    ('glm46v_flash', 'non_thinking', 'geo_attr', 's3', 'suite/glm46v_flash/G'),
    ('glm46v_flash', 'non_thinking', 'patch', 's3', 'suite/glm46v_flash/P'),
    ('glm46v_flash', 'non_thinking', 'reason', 's3', 'suite/glm46v_flash/M'),
    ('glm46v_flash', 'thinking', 'think_attr', 's3', 'thinking/glm46v_flash/items'),
    ('glm46v_flash', 'thinking', 'think_traces', 's3', 'thinking/glm46v_flash_traces/items'),
    ('glm46v_flash', 'thinking', 'geo', 's3', 'suite_think/glm46v_flash/G'),
    ('glm47_flash', 'non_thinking', 'geo_attr', 's3', 'suite/glm47_flash/G'),
    ('glm47_flash', 'non_thinking', 'patch', 's3', 'suite/glm47_flash/P'),
    ('glm47_flash', 'non_thinking', 'reason', 's3', 'suite/glm47_flash/M'),
    ('glm47_flash', 'thinking', 'think_attr', 's3', 'thinking/glm47_flash_b16/items'),
    ('glm47_flash', 'thinking', 'think_attr_b1', 's3', 'thinking/glm47_flash/items'),
    ('glm47_flash', 'thinking', 'geo', 's3', 'suite_think/glm47_flash/G'),
    ('qwen36_35b', 'non_thinking', 'geo_attr', 's3', 'suite/qwen36_35b_a3b/G'),
    ('qwen36_35b', 'non_thinking', 'patch', 's3', 'suite/qwen36_35b_a3b/P'),
    ('qwen36_35b', 'non_thinking', 'reason', 's3', 'suite/qwen36_35b_a3b/M'),
    ('qwen36_35b', 'thinking', 'think_attr', 's3', 'thinking/qwen36_35b_a3b/items'),
    ('qwen36_35b', 'thinking', 'geo', 's3', 'suite_think/qwen36_35b_a3b/G'),
    ('qwen36_27b', 'non_thinking', 'geo_attr', 'pro', 'qwen36_27b/G'),
    ('qwen36_27b', 'non_thinking', 'patch', 'pro', 'qwen36_27b/P'),
    ('qwen36_27b', 'non_thinking', 'reason', 'pro', 'qwen36_27b/M'),
    ('qwen36_27b', 'thinking', 'think_attr', 'pro', 'thinking/qwen36_27b/items'),
    ('qwen36_27b', 'thinking', 'geo', 'pro', 'suite_think/qwen36_27b/G'),
    ('gemma4_31b', 'non_thinking', 'geo_attr', 's3', 'suite/gemma4_31b_en/G'),
    ('gemma4_31b', 'non_thinking', 'patch', 's3', 'suite/gemma4_31b_en/P'),
    ('gemma4_31b', 'non_thinking', 'reason', 's3', 'suite/gemma4_31b_en/M'),
    ('gemma4_31b', 'thinking', 'think_attr', 's3', 'thinking/gemma4_31b_en/items'),
    ('gemma4_31b', 'thinking', 'geo', 's3', 'suite_think/gemma4_31b_en/G'),
    ('qwen38_flashnext', 'non_thinking', 'geo_attr', 'pro', 'qwen38_flashnext/G'),
    ('qwen38_flashnext', 'non_thinking', 'patch', 'pro', 'qwen38_flashnext/P'),
    ('qwen38_flashnext', 'non_thinking', 'reason', 'pro', 'qwen38_flashnext/M'),
    ('qwen38_flashnext', 'non_thinking', 'natural', 'pro', 'natural/qwen38_flashnext'),
    ('qwen38_flashnext', 'thinking', 'think_attr', 'pro', 'thinking/flashnext/items'),
    ('qwen38_flashnext', 'thinking', 'geo', 'pro', 'suite_think/qwen38_flashnext/G'),
    ('r1_llama70b', 'thinking', 'geo', 's3', 'suite/r1_distill_llama70b_v2/G'),
    ('r1_llama70b', 'thinking', 'think_attr', 's3', 'thinking/r1_distill_llama70b_v2/items'),
    ('r1_llama70b', 'thinking', 'think_attr_b1chk', 's3', 'thinking/r1_distill_llama70b_check_b1_v2/items'),
]
SIDECARS = ('identity.json', 'summary.json', 'status.json')
EXTRA = [  # single files: (server tag, path) -> archive path under models/
    ('pro', 'qwen38_flashnext_stream_geometry.json', 'qwen38_flashnext/non_thinking/stream_geo/stream_geometry.json'),
    ('pro', 'suite_think/qwen38_flashnext_stream_geometry.json', 'qwen38_flashnext/thinking/stream_geo/stream_geometry.json'),
    ('pro', 'flashnext_smoke.json', 'qwen38_flashnext/non_thinking/smoke/flashnext_smoke.json'),
]


def extract():
    for tag, pat in (('s3', 'server3_*.tgz'), ('pro', 'pro_*.tgz')):
        for f in sorted((A / '_raw').glob(pat)):
            with tarfile.open(f) as t:
                t.extractall(X / tag, filter='data')


def copy_measures():
    rows = []
    if (A / 'models').exists():
        shutil.rmtree(A / 'models')          # generated directory: rebuilt from _raw on every run
    for model, mode, measure, tag, rel in MAP:
        src = X / tag / rel
        dst = A / 'models' / model / mode / measure
        if not src.exists():
            rows.append((model, mode, measure, 0, 'MISSING'))
            continue
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        n = 0
        for f in src.glob('*.json'):
            shutil.copy2(f, dst / f.name)
            n += 1
        run_dir = src.parent if src.name in ('G', 'P', 'M', 'items') else src
        for s in SIDECARS:
            if (run_dir / s).exists():
                shutil.copy2(run_dir / s, dst / f'_run_{s}')
        rows.append((model, mode, measure, n, f'{tag}:{rel}'))
    for tag, rel, out in EXTRA:
        src = X / tag / rel
        if src.exists():
            (A / 'models' / out).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, A / 'models' / out)
            rows.append(tuple(out.split('/')[:3]) + (1, f'{tag}:{rel}'))
        else:
            rows.append(tuple(out.split('/')[:3]) + (0, 'MISSING'))
    with open(A / 'stats' / 'archive_inventory.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'mode', 'measure', 'n_files', 'source'])
        w.writerows(rows)
    return rows


def sign_p(k, n):
    t = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * t)


def attribution_stats():
    out = []
    for d in sorted(glob.glob(str(A / 'models' / '*' / 'non_thinking' / 'geo_attr'))):
        model = Path(d).parts[-3]
        recs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(d + '/role150_*.json'))]
        if not recs or 'mean' not in recs[0]['roles']['assistant']:
            continue
        for label, idx in (('3_orders', [0, 1, 2]), ('2_orders_as_paper', [0, 1])):
            m = lambda r, role, key: float(np.mean([r['roles'][role]['by_order'][i][key] for i in idx]))
            pa = np.array([m(r, 'assistant', 'self') for r in recs]); pu = np.array([m(r, 'user', 'self') for r in recs])
            ps = np.array([m(r, 'system', 'self') for r in recs])
            dl = pa - np.maximum(pu, ps); k = int((dl > 0).sum())
            out.append(dict(model=model, orders=label, n_bodies=len(recs), p_you_own=round(pa.mean(), 4), p_you_user=round(pu.mean(), 4),
                            p_you_system=round(ps.mean(), 4), own_minus_max_other=round(dl.mean(), 4), bodies_positive=k,
                            sign_p=float(f'{sign_p(k, len(dl)):.3g}'),
                            p_system_as_user=round(float(np.mean([m(r, 'system', 'user') for r in recs])), 4)))
    write_csv('attribution_non_thinking_body_level.csv', out)
    return out


def thinking_stats():
    out = []
    true = {'assistant': 'self', 'user': 'user', 'system': 'system'}
    for d in sorted(glob.glob(str(A / 'models' / '*' / 'thinking' / 'think_attr'))):
        model = Path(d).parts[-3]
        recs = [json.load(open(f, encoding='utf-8')) for f in glob.glob(d + '/role150_*.json')]
        for r in ('assistant', 'user', 'system'):
            rs = [x for x in recs if x['role'] == r]
            fin = [x for x in rs if x.get('finished')]
            c = Counter(x['on_answer'] for x in fin)
            out.append(dict(model=model, role=r, n_items=len(rs), n_finished=len(fin), answered_you=c.get('self', 0),
                            answered_user=c.get('user', 0), answered_system=c.get('system', 0), unparsed=c.get(None, 0),
                            correct_share=round(sum(x['on_answer'] == true[r] for x in fin) / max(1, len(fin)), 4),
                            you_share=round(c.get('self', 0) / max(1, len(fin)), 4)))
    write_csv('thinking_attribution_counts.csv', out)
    return out


def geometry_stats():
    out = []
    for f in sorted(glob.glob(str(A / 'models' / '*' / '*' / 'geo*' / '_run_summary.json'))):
        p = Path(f)
        model, mode, measure = p.parts[-4], p.parts[-3], p.parts[-2]
        s = json.load(open(f, encoding='utf-8'))
        L = s.get('n_layers')
        for kind in ('content', 'onset'):
            if kind not in s:
                continue
            k = s[kind]
            out.append(dict(model=model, mode=mode, measure=measure, position=kind, n_layers=L,
                            layers_A_apart_all_heldout=k['layers_A_lowest_all_bodies'], layers_A_apart_80pct=k['layers_A_lowest_80'],
                            median_R_A=round(float(np.median(k['mean_R'][0])), 4),
                            median_direction_cos=round(float(np.median(k['direction_cos_mean'])), 4)))
    write_csv('geometry_summary.csv', out)
    return out


def write_csv(name, rows):
    if not rows:
        return
    with open(A / 'stats' / name, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def code_and_items():
    (A / 'code').mkdir(exist_ok=True)
    (A / 'items').mkdir(exist_ok=True)
    code = ['crossmodel_suite.py', 'crossmodel_suite_5463.py', 'fp8_fallback_efb8.py', 'crossmodel_suite_v1.py', 'crossmodel_suite_d1f7.py', 'crossmodel_suite_b60a.py',
            'crossmodel_suite_8f0a.py', 'crossmodel_suite_012c.py', 'crossmodel_suite_5078.py', 'crossmodel_suite_d461.py',
            'crossmodel_suite_0eb7.py', 'crossmodel_suite_c49f.py', 'render_check.py', 'render_jinja.py',
            'glm_ondemand.py', 'glm53_compat.py', 'test_glm_ondemand_layer.py', 'test_glm_ondemand_e2e.py', 'test_glm53_compat.py',
            'crossmodel_thinking.py', 'crossmodel_thinking_v1.py', 'crossmodel_thinking_v2.py',
            'crossmodel_thinking_b.py', 'crossmodel_pilot.py', 'crossmodel_natural.py', 'gpu_worker.py', 'flashnext_patch.py',
            'fp8_fallback.py', 'fp8_fakequant.py', 'fp8_verify_pro.py', 'stream_geometry.py', 'tokenizer_check.py',
            'body_level_selfref.py', 'make_bodies_en150.py', 'make_problems_en120.py', 'build_archive.py', 'make_fig_crossmodel.py',
            'patch_suite_prefix.py', 'patch_suite_lang.py', 'patch_suite_lang_m.py', 'patch_suite_offload.py',
            'patch_suite_geoonly.py', 'patch_suite_span.py', 'patch_suite_tokfix.py']
    items = ['role_bodies.jsonl', 'bodies_en_150.json', 'bodies_zh_150_list.json', 'bodies_en_150_list.json', 'pilot_bodies_zh.json',
             'pilot_bodies_en.json', 'problems_zh_120.json', 'problems_en_120.json', 'worker_tasks_s3.json',
             'template_renderings.txt']
    man = []
    for group, names in (('code', code), ('items', items)):
        for n in names:
            f = SRC / n
            if f.exists():
                shutil.copy2(f, A / group / n)
                man.append((group, n, hashlib.sha256(f.read_bytes()).hexdigest()))
    with open(A / 'code' / 'SHA256SUMS.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['dir', 'file', 'sha256'])
        w.writerows(man)
    ids = []
    for f in sorted(glob.glob(str(A / 'models' / '*' / '*' / '*' / '_run_identity.json'))):
        p = Path(f)
        d = json.load(open(f, encoding='utf-8'))
        ids.append(dict(model=p.parts[-4], mode=p.parts[-3], measure=p.parts[-2], script_sha256=d.get('script_sha256'),
                        transformers=d.get('transformers'), torch=d.get('torch'), host=d.get('host'), tmpl_kwargs=json.dumps(d.get('tmpl_kwargs')),
                        lang=d.get('lang', 'zh'), fp8_backend=d.get('fp8_backend'), cpu_offload_gib=d.get('cpu_offload_gib'),
                        tokenizer_rebuilt_from=d.get('tokenizer_rebuilt_from')))
    write_csv('run_identities.csv', ids)


if __name__ == '__main__':
    (A / 'stats').mkdir(parents=True, exist_ok=True)
    extract()
    inv = copy_measures()
    code_and_items()
    a = attribution_stats()
    t = thinking_stats()
    g = geometry_stats()
    print(f'measures {len(inv)} (missing {sum(r[4] == "MISSING" for r in inv)}), attribution rows {len(a)}, thinking rows {len(t)}, geometry rows {len(g)}')
    for r in inv:
        if r[4] == 'MISSING' or r[3] == 0:
            print('  not yet:', r[:3])
