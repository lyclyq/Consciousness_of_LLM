"""Collects the per-run analysis JSONs of the persistence experiment (pulled from server3 and PRO into results/analysis_*/) into
one table: results/summary_persistence.md (+ .json). Verdicts as registered (PROTOCOL_zh.md section 5, main-table groups), and
for the natural order also the reading of protocol revision 6 / PREREG_phase3 section 5: group = index status at the full body
in the phase-2 natural-order run (>= 1 index layer beyond the first third), templates with only A and U keep the main group.
usage: collect_persistence_20261107.py RESULTS_DIR"""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
runs = {}
for f in sorted(R.glob('analysis_*/*.json')):
    j = json.loads(f.read_text(encoding='utf-8'))
    phase = '3' if f.name.startswith('p3') else '1-2'
    runs[(j['tag'], phase, j['order'])] = j


def m1s(j):
    m = j['M1']
    return ' '.join(str(m[k]['index_layers_beyond_first_third']) if m.get(k) and m[k].get('defined') else '-' for k in ('1', '2', '3', '4', '5', 'full'))


nat_label = {}
for (tag, ph, order), j in runs.items():
    if ph == '1-2' and order == 'natural':
        f = j['M1'].get('full')
        nat_label[tag] = ('index' if f['index_layers_beyond_first_third'] >= 1 else 'no_index') if f and f.get('defined') else j['group']


def verdict(j, group):
    au = j['roles'].get('A-U', {})
    m1 = [j['M1'].get(k) for k in ('1', '2', '3', '4', '5')] + [j['M1'].get('full')]
    ev = all(x and x.get('defined') for x in m1)
    if group == 'index':
        ok_m2 = bool(au.get('defined') and au['C1_reliable'] and au['M2_size_kept'] >= 0.8)
        ok_m1 = all(x['index_layers_beyond_first_third'] >= 1 for x in m1) if ev else None
        return 'holds' if ok_m2 and ok_m1 is not False else 'FAILS'
    if group == 'no_index':
        if not ev:
            return 'n/a'
        return 'holds' if all(x['index_layers_beyond_first_third'] == 0 for x in m1) else 'FAILS'
    return 'n/a'


rows, out = [], []
for (tag, ph, order), j in sorted(runs.items(), key=lambda x: (x[0][1], x[0][2], x[1]['group'] or '', x[0][0])):
    au = j['roles'].get('A-U', {})
    reg = verdict(j, j['group'])
    post = verdict(j, nat_label.get(tag, j['group'])) if order == 'natural' else ''
    ci = au.get('M2_ci95') or [None, None]
    rows.append(f"| {ph} | {order} | {tag} | {j['group']} | {j['n_held_out']} | {m1s(j)} | "
                f"{au.get('M2_size_kept', float('nan')):.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {au.get('M3_direction_cos', float('nan')):.3f} | {reg} | {post} |")
    out.append({'tag': tag, 'phase': ph, 'order': order, 'group': j['group'], 'natural_group': nat_label.get(tag), 'held_out': j['n_held_out'],
                'M1': {k: (v['index_layers_beyond_first_third'] if v and v.get('defined') else None) for k, v in j['M1'].items()},
                'M2': au.get('M2_size_kept'), 'M2_ci95': ci, 'M3': au.get('M3_direction_cos'), 'verdict_registered': reg,
                'verdict_natural_regrouped': post or None})
md = ['# 保持率实验汇总', '', 'M1 = 第一个三分之一之后的索引层数，依次为 k = 1、2、3、4、5、完整；M2 = A–U 共同部分大小保持 [95% 区间]；M3 = 方向余弦。',
      '"按原文" = 协议第 5 节（主表分组）；"自然顺序重分组" = 协议修订第 6 条与第三步预注册第 5 节的读法。', '',
      '| 步 | 顺序 | 设置 | 主表分组 | 留出 | M1 | M2 | M3 | 按原文 | 自然顺序重分组 |', '|---|---|---|---|---|---|---|---|---|---|'] + rows
(R / 'summary_persistence.md').write_bytes(('\n'.join(md) + '\n').encode('utf-8'))
(R / 'summary_persistence.json').write_bytes(json.dumps(out, ensure_ascii=False, indent=1).encode('utf-8'))
print('\n'.join(md))
