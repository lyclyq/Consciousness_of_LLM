"""User 10-26: Section 5.3, gpt-oss-20b numbers on one basis (native MXFP4, runs/analysis/PROVENANCE_gptoss_mxfp4_20261026.txt).
The Eq. 1 shares (81%, 51--93%, 94%, 90--98%, at most 7%) are unchanged: the MXFP4 re-run gives the same values. Changed:
  - the lowest onset share over held-out bodies is 0.699 (L23), so "above 70%" becomes "above 69%";
  - the median direction cosines, from the BF16 run (0.88--0.97 / 0.92--0.99) to MXFP4 (0.8861--0.9743 / 0.9593--0.9917).
The script recomputes each number from the JSON files and stops unless every replaced phrase occurs exactly once.
usage: fix_paper_gptoss_mxfp4_20261026.py PAPER_SOURCE_DIR   (run from evidence/p2_crossmodel_wave3_20261019)"""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
ds = json.load(open('runs/analysis/direction_stability_g.json', encoding='utf-8'))['suite_think/gpt_oss_20b_mxfp4_en']
eq = {k: json.load(open(f'runs/analysis/gptoss20b_eq1_{k}.json', encoding='utf-8')) for k in ('content', 'onset')}


def pct(x):
    return round(100 * x)


rc, ro = ds['content'], ds['onset']
cos_c = f"{rc['min_median'][0]:.2f}--{rc['max_median'][0]:.2f}"
cos_o = f"{ro['min_median'][0]:.2f}--{ro['max_median'][0]:.2f}"
assert min(min(rc['positive_share_by_layer']), min(ro['positive_share_by_layer'])) == 1.0
floor = int(100 * min(eq['onset']['share_min_over_bodies_by_layer']))  # 0.699 -> 69: every body is above this percentage
checks = {'content median': (pct(eq['content']['share_median_over_layers']), 81),
          'content range': ((pct(eq['content']['share_range'][0]), pct(eq['content']['share_range'][1])), (51, 93)),
          'onset median': (pct(eq['onset']['share_median_over_layers']), 94),
          'onset range': ((pct(eq['onset']['share_range'][0]), pct(eq['onset']['share_range'][1])), (90, 98)),
          'onset apart after, max': (pct(max(eq['onset']['apart_after_by_layer'])), 7)}
for k, (got, paper) in checks.items():
    print(f'{k:24s} paper {paper}  recomputed {got}')
    assert got == paper, k
print('cosines', cos_c, cos_o, '; onset floor over bodies', floor)

REP = {
    'self_en_body.tex': [
        ('where every held-out body is above 70\\% at every layer', f'where every held-out body is above {floor}\\% at every layer'),
        ('(median cosine 0.88--0.97 on content and 0.92--0.99 at the onset, positive in every body at every layer)',
         f'(median cosine {cos_c} on content and {cos_o} at the onset, positive in every body at every layer)'),
    ],
    'self_zh_body.tex': [
        ('且在回应起点上每份留出正文在每一层都高于 70\\%', f'且在回应起点上每份留出正文在每一层都高于 {floor}\\%'),
        ('余弦中位数在内容上为 0.88--0.97，在回应起点上为 0.92--0.99', f'余弦中位数在内容上为 {cos_c}，在回应起点上为 {cos_o}'),
    ],
}
for name, reps in REP.items():
    p = D / name
    t = p.read_text(encoding='utf-8')
    for a, b in reps:
        assert t.count(a) == 1, (name, a[:60], t.count(a))
        t = t.replace(a, b)
    p.write_text(t, encoding='utf-8')
    print(name, len(reps), 'replaced')
