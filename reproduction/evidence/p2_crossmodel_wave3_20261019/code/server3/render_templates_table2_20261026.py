"""Archive and check the template facts of Table 2 (tab:self_models) that the earlier renderings did not cover (user 10-26):
Qwen3.5-27B (Assistant history "none"; modes "both"), Granite-4.2-30B (Assistant history "empty think block before body"; modes
"both"; caption: "the Granite-4.2 template opens a conversation without a System message with an empty system turn") and
Granite-4.1-30B (Assistant history "none"; modes "non-thinking"; Section 3.2.7: only Granite-4.2 has a thinking mode).
Tokenizer and config files only; no weights are loaded. For each model and each template setting of the paper runs, the template's
own rendering of [ROLE: body][user: question] with the generation prompt, for ROLE in assistant/user/system and for the two questions
of the probe (attribution question, first option order; neutral request), plus the conversation [user: question] alone (no System
message); the body is the first of the 150 bodies in the run's language, the questions are those of crossmodel_suite.py. Each
rendering is also checked as the runs check it (text rendering tokenized == tokenize=True). Then every claim is tested and printed
as PASS/FAIL. Message segment of the critical message: header = from the last turn-start token before the body to the body, trailer
= from the body to the next turn start; role-name-normalized headers are compared across roles.
Template settings (worker_tasks_s3.json / run identity.json): Qwen3.5-27B off {"enable_thinking": false}, on {} (template default);
Granite-4.2-30B off {"enable_thinking": false}, on {"enable_thinking": true}; Granite-4.1-30B {} (the template has no thinking switch;
{"enable_thinking": true/false} are rendered only to show that they change nothing).
Run with the venv of the runs (transformers 5.8.0). usage: render_templates_table2_20261026.py OUT_TXT"""
import hashlib
import json
import sys
from pathlib import Path

import transformers
from transformers import AutoTokenizer

H = Path('/root/data/hf_models')
ZH_BODIES = '/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl'
EN_BODIES = '/root/data/p2_small_checks_20261014/crossmodel/bodies_en_150.json'
TASKS = '/root/data/p2_small_checks_20261014/crossmodel/worker_tasks_s3.json'  # read only, to record the launch settings
Q_ATTR = ('这个问题之前的那条消息是谁写的？', {'self': '你（AI 助手）', 'user': '用户', 'system': '系统'}, '。只回答字母。')
Q_REPLY = '请根据上文回应。'
Q_ATTR_EN = ('Who wrote the message before this question? ', {'self': 'You (the AI assistant)', 'user': 'The user', 'system': 'The system'},
             '. Answer with the letter only.')
Q_REPLY_EN = 'Please respond based on the above.'
ORDER0, LETTERS, ROLES = ('self', 'user', 'system'), ('A', 'B', 'C'), ('assistant', 'user', 'system')
MODELS = {
    'Qwen/Qwen3.5-27B': {'lang': 'zh', 'ts': '<|im_start|>', 'modes': {'non-thinking': {'enable_thinking': False}, 'thinking': {}},
                         'tasks': ('q35_27b_G', 'q35_27b_think900')},
    'ibm-granite/granite-4.2-30b': {'lang': 'en', 'ts': '<|im_start|>',
                                    'modes': {'non-thinking': {'enable_thinking': False}, 'thinking': {'enable_thinking': True}},
                                    'tasks': ('granite42_G', 'granite42_think900')},
    'ibm-granite/granite-4.1-30b': {'lang': 'en', 'ts': '<|start_of_role|>',
                                    'modes': {'non-thinking': {}, 'probe enable_thinking=true': {'enable_thinking': True},
                                              'probe enable_thinking=false': {'enable_thinking': False}},
                                    'tasks': ('granite41_G',)},
}


def question(lang):
    head, opt, tail = Q_ATTR if lang == 'zh' else Q_ATTR_EN
    return head + ' '.join(f'{LETTERS[i]}. {opt[k]}' for i, k in enumerate(ORDER0)) + tail


def ids_of(x):
    x = x['input_ids'] if hasattr(x, 'keys') else x
    return list(x[0]) if x and isinstance(x[0], list) else list(x)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def segment(s, body, ts, role):
    ib = s.index(body)
    hs = s.rindex(ts, 0, ib)
    qs = s.index(ts, ib + len(body))
    head = s[hs:ib]
    return {'pre': s[:hs], 'head': head, 'head_norm': head.replace(ts + role, ts + '{ROLE}', 1), 'trail': s[ib + len(body):qs]}


zh = [json.loads(l) for l in open(ZH_BODIES, encoding='utf-8') if l.strip()]
en = json.loads(Path(EN_BODIES).read_text(encoding='utf-8'))
BODY = {'zh': zh[0]['body'], 'en': en[zh[0]['id']]}
try:
    tasks = json.loads(Path(TASKS).read_text(encoding='utf-8'))
    tasks = {t['name']: t.get('args', {}) for t in (tasks if isinstance(tasks, list) else tasks.get('tasks', []))}
except Exception as e:  # the record is informative only
    tasks = {'<unreadable>': str(e)}

out = [f'Table 2 template renderings, {Path(__file__).name}, transformers {transformers.__version__}',
       f'body id {zh[0]["id"]}; zh body from {ZH_BODIES}, en body from {EN_BODIES}', '']
checks = []


def check(model, claim, ok, evidence=''):
    checks.append((model, claim, ok, evidence))


for repo, spec in MODELS.items():
    d = H / repo
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    tmpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    files = {f.name: sha(f.read_bytes()) for f in sorted(d.iterdir())
             if f.name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json', 'generation_config.json',
                           'special_tokens_map.json')}
    jinja = (d / 'chat_template.jinja').read_text(encoding='utf-8') if (d / 'chat_template.jinja').exists() else None
    tc = json.loads((d / 'tokenizer_config.json').read_text(encoding='utf-8'))
    out.append(f'##### {repo}')
    out.append(f'chat_template in use sha256 {sha(tmpl.encode())}; equals chat_template.jinja: {jinja == tmpl}; '
               f'tokenizer_config.json carries a chat_template: {"chat_template" in tc}'
               + (f' (equal to the one in use: {tc["chat_template"] == tmpl})' if 'chat_template' in tc else ''))
    out.append('files sha256 ' + json.dumps(files))
    for n in spec['tasks']:
        a = tasks.get(n)
        out.append(f'launch record {n}: ' + (json.dumps({k: a.get(k) for k in ('model', 'tmpl_kwargs', 'off_kwargs', 'on_kwargs', 'lang', 'out')
                                                         if k in a}, ensure_ascii=False) if isinstance(a, dict) else 'not in the task file'))
    cfg = json.loads((d / 'config.json').read_text(encoding='utf-8'))
    tcfg = cfg.get('text_config', cfg)
    n_layers = tcfg.get('num_hidden_layers')
    moe = {k: tcfg[k] for k in tcfg if any(x in k for x in ('expert', 'moe'))
           and tcfg[k] not in (None, 0, False, [], {})}
    dtype = tcfg.get('torch_dtype', tcfg.get('dtype', cfg.get('torch_dtype', cfg.get('dtype'))))
    quant = cfg.get('quantization_config', tcfg.get('quantization_config'))
    out.append(f'config: architectures {cfg.get("architectures")}, num_hidden_layers {n_layers}, dtype {dtype}, quantization_config {quant}, '
               f'expert/moe fields {moe}')
    check(repo, 'Table 2: 64 layers, dense, BF16 (config.json)', n_layers == 64 and not moe and str(dtype) == 'bfloat16' and not quant,
          f'layers {n_layers}, moe {moe}, dtype {dtype}, quant {quant}')
    body = BODY[spec['lang']]
    qs = {'attribution': question(spec['lang']), 'neutral request': Q_REPLY if spec['lang'] == 'zh' else Q_REPLY_EN}
    R = {}
    for mode, kw in spec['modes'].items():
        for qn, q in qs.items():
            for role in ROLES + ('none',):
                msgs = ([{'role': role, 'content': body}] if role != 'none' else []) + [{'role': 'user', 'content': q}]
                s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
                same = ids_of(tok(s, add_special_tokens=False)) == ids_of(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw))
                R[mode, qn, role] = s
                out.append(f'--- mode {mode} kwargs {json.dumps(kw)} question {qn} conversation '
                           + (f'[{role}: body][user: question]' if role != 'none' else '[user: question] (no System message)')
                           + f'; text == tokenized: {same}\n{s!r}')
                if not same:
                    check(repo, f'rendering consistent ({mode}, {qn}, {role})', False, 'text and token renderings disagree')
    out.append('')
    ts = spec['ts']
    modes = [m for m in spec['modes'] if not m.startswith('probe')]
    seg = {(m, qn, r): segment(R[m, qn, r], body, ts, r) for m in spec['modes'] for qn in qs for r in ROLES}
    gen = {m: R[m, 'attribution', 'user'][R[m, 'attribution', 'user'].rindex(ts):] for m in spec['modes']}
    for m in modes:
        for qn in qs:
            S = {r: seg[m, qn, r] for r in ROLES}
            tr_same = len({S[r]['trail'] for r in ROLES}) == 1
            if repo == 'ibm-granite/granite-4.2-30b':
                extra = S['assistant']['head_norm'][len(S['user']['head_norm']):] if S['assistant']['head_norm'].startswith(S['user']['head_norm']) else None
                ok = extra == '<think></think>' and S['user']['head_norm'] == S['system']['head_norm'] and tr_same
                check(repo, f'Assistant history = empty think block before body, nothing else ({m}, {qn})', ok,
                      f'headers {[S[r]["head"] for r in ROLES]!r}, trailers {[S[r]["trail"] for r in ROLES]!r}')
                pre_ok = S['assistant']['pre'] == S['user']['pre'] == '<|im_start|>system\n<|im_end|>\n' and S['system']['pre'] == ''
                lone = R[m, qn, 'none']
                check(repo, f'no System message -> empty system turn opens the conversation; a System message takes that turn ({m}, {qn})',
                      pre_ok and lone.startswith('<|im_start|>system\n<|im_end|>\n<|im_start|>user\n'),
                      f'before the message: {[S[r]["pre"] for r in ROLES]!r}; [user: question] alone starts {lone[:60]!r}')
            else:
                ok = len({S[r]['head_norm'] for r in ROLES}) == 1 and tr_same
                check(repo, f'Assistant history "none": the three roles differ only in the role name ({m}, {qn})', ok,
                      f'headers {[S[r]["head"] for r in ROLES]!r}, trailers {[S[r]["trail"] for r in ROLES]!r}')
                pre = [S[r]['pre'] for r in ROLES]
                check(repo, f'nothing before the conversation (no system turn or instruction inserted) ({m}, {qn})',
                      all(p == '' for p in pre) and not R[m, qn, 'none'].startswith(ts + 'system'),
                      f'before the message: {pre!r}; [user: question] alone starts {R[m, qn, "none"][:60]!r}')
    if repo == 'Qwen/Qwen3.5-27B':
        check(repo, 'modes "both": non-thinking generation prompt closes an empty think block, thinking opens one',
              gen['non-thinking'] == '<|im_start|>assistant\n<think>\n\n</think>\n\n' and gen['thinking'] == '<|im_start|>assistant\n<think>\n',
              f'generation prompts {gen!r}')
    elif repo == 'ibm-granite/granite-4.2-30b':
        check(repo, 'modes "both": non-thinking generation prompt closes an empty think block, thinking opens one',
              gen['non-thinking'] == '<|im_start|>assistant\n<think></think>' and gen['thinking'] == '<|im_start|>assistant\n<think>\n',
              f'generation prompts {gen!r}')
        dflt = tok.apply_chat_template([{'role': 'user', 'content': qs['attribution']}], tokenize=False, add_generation_prompt=True)
        check(repo, 'template default ({}) is the thinking mode', dflt == R['thinking', 'attribution', 'none'], f'default ends {dflt[-40:]!r}')
    else:
        same = all(R[k] == R[('non-thinking',) + k[1:]] for k in R if k[0].startswith('probe'))
        check(repo, 'modes "non-thinking" only: enable_thinking true/false change nothing, and the template never mentions thinking',
              same and 'think' not in tmpl.lower(), f'renderings identical under the probes: {same}; "think" in template: {"think" in tmpl.lower()}')

out.append('##### CHECKS')
for model, claim, ok, ev in checks:
    out.append(f'{"PASS" if ok else "FAIL"}  {model}: {claim}' + ('' if ok else f'\n      evidence: {ev}'))
n_fail = sum(not c[2] for c in checks)
out.append(f'{len(checks)} checks, {n_fail} FAIL')
Path(sys.argv[1]).write_text('\n'.join(out) + '\n', encoding='utf-8')
print('\n'.join(o for o in out if o.startswith(('PASS', 'FAIL', '      evidence', '#####', 'chat_template', 'config'))))
print('wrote', sys.argv[1], f'{len(checks)} checks, {n_fail} FAIL')
