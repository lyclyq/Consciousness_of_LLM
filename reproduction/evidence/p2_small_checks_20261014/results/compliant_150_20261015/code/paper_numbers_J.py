"""Print the check-J numbers quoted in the paper (from analyze_native_geometry's summary JSON)."""
import json
import sys

J = json.load(open(sys.argv[1]))
print('bodies', J['n_bodies'])
C = J['curves']


def row(key, pos, layers):
    out = []
    for l in layers:
        c = C[key][pos][l]
        r = min('AUS', key=lambda k: c[f'R_{k}'])
        out.append(f"L{l}: odd {r} R {c[f'R_{r}']:.2f} AU {c['closest']['AU']:.2f} AS {c['closest']['AS']:.2f} US {c['closest']['US']:.2f}")
    return out


LAY = [0, 1, 2, 3, 5, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
for key in ('preserve_false__reply', 'default__reply', 'preserve_false__attr', 'default__attr'):
    for pos in ('decision', 'turn_newline', 'body_mean', 'body_im_end', 'header_newline'):
        print(f'\n## {key} / {pos}')
        for line in row(key, pos, LAY):
            print('  ', line)
print('\n## decision JS (mean over bodies) at L47/L55/L63')
for k, v in J['decision_js'].items():
    print(f'  {k:32s}', ' '.join(f'{v[l]:.3f}' for l in (47, 55, 63)), '| max over L0-L47 %.3f' % max(v[:48]))
print('\n## attribution baselines (both option orders)')
for k, v in J['attr_baseline'].items():
    print(f'  {k:28s} self {v[0]:.3f} user {v[1]:.3f} system {v[2]:.3f}')
