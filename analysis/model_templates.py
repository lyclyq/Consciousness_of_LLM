"""How each chat template marks the model's own message, next to the model's content geometry and self-attribution (the data of
Table tab:self_models and of the format-ablation section), computed from archived records only.
  header      the text before the body of a history message under the Assistant and under the User role, after removing the part
              that both renderings share (system preamble), from the archived template renderings
  marker      'role word only' when the two headers are equal after the role word is replaced by ROLE; otherwise the Assistant-only
              format text
  geometry    content positions: layers at which the own message is apart in every held-out body, of all layers, and the median
              R_A = D_US / mean(D_AU, D_AS) (geometry_summary.csv of both archives, v4_geometry_summary.csv); non-thinking
              rendering where measured, else thinking
  self-ref    without thinking: mean P(you) for the own / the User message (attribution_non_thinking_body_level.csv,
              v4_attribution_non_thinking.csv; first row per model); with thinking: share answered "you" (thinking_attribution_counts,
              v4_thinking_counts; the 300-item rows); Qwen3.8-27B with thinking from the run thinking_swap/qwen38_27b_all
Inputs: the template renderings in the items folders of the results package and the stats tables of
crossmodel_classification.py (in OUT). Outputs: stats/template_markers.csv and .md.

usage: model_templates.py --results RES --out OUT
"""
import ast
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import M8, W3, Roots, arguments, run_path  # noqa: E402

# archive key -> name in the rendering records
NAMES = {'qwen38_27b': 'Qwen3.8-27B', 'qwen38_flashnext': 'Qwen3.8-Flash-Next-FP8', 'qwen35_27b': 'Qwen/Qwen3.5-27B',
         'qwen35_122b': 'Qwen3.5-122B-A10B', 'qwen36_27b': 'Qwen3.6-35B-A3B', 'qwen36_35b': 'Qwen3.6-35B-A3B',
         'gemma4_31b': 'gemma-4-31B-it', 'glm46v_flash': 'GLM-4.6V-Flash', 'glm47_flash': 'GLM-4.7-Flash', 'glm53_flash': 'zai-org/GLM-5.3-Flash',
         'r1_llama70b': 'DeepSeek-R1-Distill-Llama-70B', 'seed_oss_36b': 'ByteDance-Seed/Seed-OSS-36B-Instruct',
         'exaone4_32b': 'LGAI-EXAONE/EXAONE-4.0.1-32B', 'gpt_oss_20b': 'openai/gpt-oss-20b', 'gpt_oss_120b': 'openai/gpt-oss-120b',
         'granite41_30b': 'ibm-granite/granite-4.1-30b', 'granite42_30b': 'ibm-granite/granite-4.2-30b'}
# Qwen3.6-27B shares the Qwen3.6 template recorded for Qwen3.6-35B-A3B (note in template_renderings.txt)


def run(RM8, RW3, SM8, SW3):
    """RM8, RW3: the two archives in the results package; SM8, SW3: their stats folders in OUT."""
    def headers():
        """name -> {role: text before the body}, non-thinking kwargs preferred."""
        H = {}

        def put(name, kw, role, text):
            k = text.find('BODY')
            if k < 0:
                k = min([i for i in (text.find('账页青梧'), text.find('Ledger page')) if i >= 0], default=-1)
            if k < 0:
                return
            nonthink = 'false' in kw or 'thinking_budget' in kw
            slot = H.setdefault(name, {})
            if role not in slot or (nonthink and not slot[role][1]):
                slot[role] = (text[:k], nonthink)
        name = None
        for l in open(RM8 / 'items' / 'template_renderings.txt', encoding='utf-8'):
            if l.startswith('====='):
                name = l.strip('= \n')
                continue
            m = re.match(r"^(\{.*?\}) (assistant|user|system) (.*)$", l.rstrip('\n'))
            if m and name:
                put(name, m.group(1), m.group(2), ast.literal_eval(m.group(3)))
        for f in (RW3 / 'items' / 'template_renderings_wave3.txt', RW3 / 'items' / 'template_renderings_gptoss_20261026.txt',
                  RW3 / 'items' / 'template_renderings_table2_20261026.txt'):
            lines = open(f, encoding='utf-8').read().splitlines()
            name = None
            for i, l in enumerate(lines):
                if l.startswith('#####'):
                    name = l.strip('# ')
                    continue
                m = re.match(r'^--- kwargs (\{.*?\}) role (assistant|user)$', l) or \
                    re.match(r'^--- mode \S+ kwargs (\{.*?\}) question attribution conversation \[(assistant|user): body\]', l)
                if m and name and i + 1 < len(lines):
                    try:
                        put(name, m.group(1), m.group(2), ast.literal_eval(lines[i + 1]))
                    except (ValueError, SyntaxError):
                        pass
        return {n: {r: t for r, (t, _) in v.items()} for n, v in H.items()}

    def marker(h):
        a, u = h.get('assistant'), h.get('user')
        if a is None or u is None:
            return None, None, None
        p = 0
        while p < min(len(a), len(u)) and a[p] == u[p]:
            p += 1
        p = max(a.rfind('<', 0, p + 1), a.rfind('[', 0, p + 1), 0) if p < min(len(a), len(u)) else p
        ah, uh = a[p:], u[p:]
        na = re.sub(r'assistant|Assistant|model', 'ROLE', ah, count=1)
        nu = re.sub(r'user|User', 'ROLE', uh, count=1)
        if na == nu:
            return ah, uh, 'role word only'
        i = 0
        while i < min(len(na), len(nu)) and na[i] == nu[i]:
            i += 1
        if 0 < i and na[i - 1] not in '>\n]':  # do not split a special token such as <|channel|>
            i = na.rfind('<', 0, i)
        j = 0
        while j < min(len(na), len(nu)) - i and na[-1 - j] == nu[-1 - j]:
            j += 1
        return ah, uh, repr(na[i:len(na) - j])  # the Assistant-only format text

    def rows(path):
        return list(csv.DictReader(open(path, encoding='utf-8'))) if path.exists() else []

    def geometry():
        G = {}
        for r in rows(SM8 / 'geometry_summary.csv') + rows(SW3 / 'geometry_summary.csv') + rows(SW3 / 'v4_geometry_summary.csv'):
            if r.get('position') != 'content':
                continue
            G.setdefault(r['model'], {}).setdefault(r['mode'], []).append(
                (int(r['layers_A_apart_all_heldout']), int(r['n_layers']), float(r['median_R_A'])))
        out = {}
        for m, by in G.items():
            mode = 'non_thinking' if 'non_thinking' in by else 'thinking'
            v = by[mode]
            ap = sorted({x[0] for x in v})
            ra = sorted({x[2] for x in v})
            out[m] = (mode, f"{ap[0]}" + (f"-{ap[-1]}" if len(ap) > 1 else '') + f" / {v[0][1]}",
                      f"{ra[0]:.2f}" + (f"-{ra[-1]:.2f}" if len(ra) > 1 else ''))
        return out

    def selfref():
        S, T = {}, {}
        for r in rows(SM8 / 'attribution_non_thinking_body_level.csv') + rows(SW3 / 'attribution_non_thinking_body_level.csv') + \
                rows(SW3 / 'v4_attribution_non_thinking.csv'):
            S.setdefault(r['model'], f"{float(r['p_you_own']):.2f} / {float(r['p_you_user']):.2f}")
        tc = {}
        for r in rows(SM8 / 'thinking_attribution_counts.csv') + rows(SW3 / 'thinking_attribution_counts.csv') + \
                rows(SW3 / 'v4_thinking_counts.csv'):
            if r['n_items'] == '300':
                tc.setdefault(r['model'], {})[r['role']] = float(r['you_share'])
        s = json.load(open(run_path(RW3, 'thinking_swap/qwen38_27b_all') / 'summary.json', encoding='utf-8'))
        tc.setdefault('qwen38_27b', {'assistant': s['base_A']['you_share'], 'user': s['base_U']['you_share']})
        for m, d in tc.items():
            if 'assistant' in d and 'user' in d:
                T[m] = f"{d['assistant']:.2f} / {d['user']:.2f}"
        return S, T

    H = headers()
    G = geometry()
    S, T = selfref()
    out = []
    for key, name in NAMES.items():
        ah, uh, mk = marker(H.get(name, {}))
        g = G.get(key, ('-', '-', '-'))
        out.append({'model': key, 'assistant_header': ah, 'user_header': uh, 'own_message_marker': mk, 'geometry_mode': g[0],
                    'layers_apart': g[1], 'median_R_A': g[2], 'p_you_own_user_nonthinking': S.get(key, '-'),
                    'you_share_own_user_thinking': T.get(key, '-')})
    out.sort(key=lambda r: (r['own_message_marker'] != 'role word only', r['model']))
    with open(SW3 / 'template_markers.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    L = ['# Own-message markers, content geometry and self-attribution (make_template_markers.py)', '',
         '| model | Assistant header | User header | own-message marker | geometry (mode): layers apart / all, median R_A | P(you) own / User, no thinking | "you" own / User, thinking |',
         '|---|---|---|---|---|---|---|']
    for r in out:
        L.append(f"| {r['model']} | `{r['assistant_header']!r}` | `{r['user_header']!r}` | {r['own_message_marker']} | ({r['geometry_mode']}) {r['layers_apart']}, "
                 f"{r['median_R_A']} | {r['p_you_own_user_nonthinking']} | {r['you_share_own_user_thinking']} |")
    open(SW3 / 'template_markers.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main(argv=None):
    args = arguments(__doc__).parse_args(argv)
    r = Roots(args)
    run(r.res(M8), r.res(W3), r.out(M8 + '/stats'), r.out(W3 + '/stats'))


if __name__ == '__main__':
    main()
