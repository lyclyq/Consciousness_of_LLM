"""Appendix A (Rendered Items) of the P2 paper from the archived renderings (user 10-29): for every model, its own template's
rendering of one item of the main probe and of the natural conversation order, in each template setting of the runs.
Inputs (items/appendix_a_20261029/): appendix_a_render_{s3,pro,pro_tokonly}.json (render_appendix_a_20261029.py) and
appendix_a_render_dsv41.json (render_appendix_a_dsv41_20261029.py), with items/appendix_a_bodies_20261029.json. The body, question
and request are replaced by placeholders, a line feed is shown as a symbol, the date that the gpt-oss template inserts by a
placeholder, and the characters U+FF5C and U+2581 of DeepSeek token names by | and _; every other character is the template's
output. Models with identical renderings in every setting share one block. Output: the LaTeX block for the EN or ZH appendix
(between the markers %<<app_templates and %>>app_templates) and a plain-text copy of all renderings.
usage: make_appendix_a_tex.py ITEMS_DIR OUT_DIR   (writes appendix_a_{en,zh}.tex and appendix_a_renderings.txt)"""
import json
import re
import sys
from pathlib import Path

ITEMS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
D = ITEMS / 'appendix_a_20261029'
M = {}
for f in ('appendix_a_render_s3.json', 'appendix_a_render_pro.json', 'appendix_a_render_pro_tokonly.json'):
    for k, v in json.loads((D / f).read_text(encoding='utf-8'))['models'].items():
        if k in M:
            for part in ('main', 'natural'):
                assert M[k].get(part) == v.get(part), f'{k}: renderings differ between hosts'
        M.setdefault(k, v)
M['deepseek-ai/DeepSeek-V4.1-Flash'] = json.loads((D / 'appendix_a_render_dsv41.json').read_text(encoding='utf-8'))
assert M['deepseek-ai/DeepSeek-V4.1-Flash']['same_as_runs']
for k in M:
    M[k]['_repo'] = k
B = json.loads((ITEMS / 'appendix_a_bodies_20261029.json').read_text(encoding='utf-8'))
Q = {'zh': '这个问题之前的那条消息是谁写的？A. 你（AI 助手） B. 用户 C. 系统。只回答字母。',
     'en': 'Who wrote the message before this question? A. You (the AI assistant) B. The user C. The system. Answer with the letter only.'}
REQ = {'zh': '请写一段简短的描述。', 'en': 'Please write a short description.'}
NAME = {'Qwen/Qwen3.8-27B': 'Qwen3.8-27B', 'Qwen/Qwen3.8-Flash-Next-FP8': 'Qwen3.8-Flash-Next', 'Qwen/Qwen3.6-27B': 'Qwen3.6-27B',
        'Qwen/Qwen3.6-35B-A3B': 'Qwen3.6-35B-A3B', 'Qwen/Qwen3.5-27B': 'Qwen3.5-27B', 'zai-org/GLM-4.6V-Flash': 'GLM-4.6V-Flash',
        'zai-org/GLM-4.7-Flash': 'GLM-4.7-Flash', 'zai-org/GLM-5.3-Flash': 'GLM-5.3-Flash',
        'deepseek-ai/DeepSeek-R1-Distill-Llama-70B': 'DeepSeek-R1-Distill-Llama-70B', 'openai/gpt-oss-20b': 'gpt-oss-20b',
        'openai/gpt-oss-120b': 'gpt-oss-120b', 'google/gemma-4-31B-it': 'Gemma-4-31B', 'ByteDance-Seed/Seed-OSS-36B-Instruct': 'Seed-OSS-36B',
        'LGAI-EXAONE/EXAONE-4.0.1-32B': 'EXAONE-4.0.1-32B', 'ibm-granite/granite-4.2-30b': 'Granite-4.2-30B',
        'ibm-granite/granite-4.1-30b': 'Granite-4.1-30B', 'allenai/OLMo-2-1124-13B-Instruct': 'OLMo-2-13B',
        'allenai/OLMo-2-1124-7B-Instruct': 'OLMo-2-7B', 'deepseek-ai/DeepSeek-V4.1-Flash': 'DeepSeek-V4.1-Flash',
        'mistralai/Ministral-3-14B-Instruct-2512-BF16': 'Ministral-3-14B-Instruct', 'mistralai/Ministral-3-14B-Reasoning-2512': 'Ministral-3-14B-Reasoning'}
MAIN_ORDER = ['Qwen/Qwen3.8-27B', 'Qwen/Qwen3.8-Flash-Next-FP8', 'Qwen/Qwen3.6-27B', 'Qwen/Qwen3.6-35B-A3B', 'Qwen/Qwen3.5-27B',
              'zai-org/GLM-4.6V-Flash', 'zai-org/GLM-4.7-Flash', 'zai-org/GLM-5.3-Flash', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
              'openai/gpt-oss-20b', 'openai/gpt-oss-120b', 'google/gemma-4-31B-it', 'ByteDance-Seed/Seed-OSS-36B-Instruct',
              'LGAI-EXAONE/EXAONE-4.0.1-32B', 'ibm-granite/granite-4.2-30b', 'ibm-granite/granite-4.1-30b',
              'allenai/OLMo-2-1124-13B-Instruct', 'allenai/OLMo-2-1124-7B-Instruct']
NAT_GROUPS = [('three', ['ByteDance-Seed/Seed-OSS-36B-Instruct', 'google/gemma-4-31B-it', 'LGAI-EXAONE/EXAONE-4.0.1-32B',
                         'zai-org/GLM-4.6V-Flash', 'ibm-granite/granite-4.1-30b', 'allenai/OLMo-2-1124-7B-Instruct']),
              ('two', ['Qwen/Qwen3.8-27B', 'openai/gpt-oss-20b', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B']),
              ('merged', ['deepseek-ai/DeepSeek-V4.1-Flash', 'mistralai/Ministral-3-14B-Instruct-2512-BF16',
                          'mistralai/Ministral-3-14B-Reasoning-2512'])]
ROLES = ('assistant', 'user', 'system')
LABEL = {'assistant': 'Assistant', 'user': 'User', 'system': 'System'}
TXT = {'en': {'body': 'body', 'question': 'question', 'request': 'request', 'date': 'date', 'nt': 'Without thinking', 'th': 'With thinking',
              'zh': 'Chinese items', 'enl': 'English items', 'error': 'template error: ', 'absent': 'the template leaves the message out:',
              'moved': 'the template moves the message to the start:', 'not_run': ' (not run; shown for comparison)', 'start': 'start'},
       'zh': {'body': '正文', 'question': '问题', 'request': '请求', 'date': '日期', 'nt': '不思考', 'th': '思考',
              'zh': '中文题目', 'enl': '英文题目', 'error': '模板报错：', 'absent': '模板略去这条消息：',
              'moved': '模板把这条消息移到开头：', 'not_run': '（未运行，仅作对照）', 'start': '开头'}}


def tex_text(s, lang, L):
    """rendered text -> Verbatim body with commandchars \\{}"""
    s = s.replace('\\', '\x00BS').replace('{', '\x00LB').replace('}', '\x00RB')
    s = s.replace(B[lang], '\x00PH:body').replace(Q[lang], '\x00PH:question').replace(REQ[lang], '\x00PH:request')
    s = re.sub(r'Current date: \d{4}-\d{2}-\d{2}', 'Current date: \x00PH:date', s)
    s = s.replace('\uff5c', '|').replace('\u2581', '_').replace('\n', '\x00NL')
    assert all(ord(c) < 128 for c in s), [c for c in s if ord(c) >= 128]
    s = s.replace('\x00BS', '\\BS{}').replace('\x00LB', '\\LB{}').replace('\x00RB', '\\RB{}').replace('\x00NL', '\\NL{}')
    for k in ('body', 'question', 'request', 'date'):
        s = s.replace('\x00PH:' + k, '\\PH{' + TXT[L][k] + '}')
    return s


def plain(s, lang):
    s = s.replace(B[lang], '<body>').replace(Q[lang], '<question>').replace(REQ[lang], '<request>')
    return re.sub(r'Current date: \d{4}-\d{2}-\d{2}', 'Current date: <date>', s)


def shared_start(texts):
    """longest common start of the three renderings, cut where one token ends ('>' or ']') and the next begins ('<' or '[')"""
    n = 0
    while all(n < len(t) for t in texts) and len({t[n] for t in texts}) == 1:
        n += 1
    cut = max((c for c in range(1, n + 1) if texts[0][c - 1] in '>]' and all(c < len(t) and t[c] in '<[' for t in texts)), default=0)
    return cut if cut >= 200 else 0


START = {}  # (repo, mode) -> the Start text of the model's natural-order block


def block(rec, part, mode, L, notes):
    v = rec[part][mode]
    lines = []
    texts = [v[r]['text'] for r in ROLES if 'text' in v[r]]
    cut = shared_start(texts) if len(texts) == 3 else 0
    ext = ''
    if cut:
        START[rec['_repo'], mode] = texts[0][:cut]
        lines.append('Start'.ljust(11) + tex_text(texts[0][:cut], rec['lang'], L))
    elif part == 'main' and (rec['_repo'], mode) in START:
        ext = START[rec['_repo'], mode]  # Ministral main probe (not run): the Start of its natural-order block above
    for r in ROLES:
        x = v[r]
        lab = LABEL[r].ljust(11)
        if 'error' in x:
            lines.append(lab + '\\Note{' + TXT[L]['error'] + x['error'].split(': ', 1)[-1] + '}')
            continue
        t = x['text']
        pre = ''
        if B[rec['lang']] not in t:
            pre = '\\Note{' + TXT[L]['absent'] + '} '
        elif part == 'natural' and t.index(B[rec['lang']]) < t.index(REQ[rec['lang']]):
            pre = '\\Note{' + TXT[L]['moved'] + '} '
        if cut:
            pre += '\\PH{' + TXT[L]['start'] + '}'
            t = t[cut:]
        elif ext and t.startswith(ext):
            pre += '\\PH{' + TXT[L]['start'] + '}'
            t = t[len(ext):]
        lines.append(lab + pre + tex_text(t, rec['lang'], L))
    return '\\begin{Rendering}\n' + '\n'.join(lines) + '\n\\end{Rendering}\n'


def mode_name(m, L):
    return TXT[L]['nt'] if m == 'non-thinking' else TXT[L]['th']


def key(rec, part):
    return json.dumps({m: {r: v[r] for r in ROLES} for m, v in rec.get(part, {}).items()}, sort_keys=True, ensure_ascii=False)


def groups(order, part):
    out = []
    for repo in order:
        for g in out:
            if key(M[g[0]], part) == key(M[repo], part) and M[g[0]]['lang'] == M[repo]['lang']:
                g.append(repo)
                break
        else:
            out.append([repo])
    return out


def heading(g, L):
    names = [NAME[r] for r in g]
    lang = TXT[L]['zh'] if M[g[0]]['lang'] == 'zh' else TXT[L]['enl']
    sep = '、' if L == 'zh' else ', '
    return '\\paragraph{' + sep.join(names) + '} ' + ('（' + lang + '）' if L == 'zh' else f'({lang}.)') + '\n'


def section(part, order, L, not_run=False):
    s = ''
    for g in groups(order, part):
        rec = M[g[0]]
        s += heading(g, L)
        for m in rec[part]:
            s += ('\\noindent ' + mode_name(m, L) + (TXT[L]['not_run'] if not_run else '') + ('：' if L == 'zh' else ':') + '\n'
                  + block(rec, part, m, L, None))
    return s


def build(L):
    if L == 'en':
        s = ('This appendix gives the input that each model receives for one item: the first of the 150 bodies, with the attribution '
             'question in its first option order, rendered by the model\'s own chat template in each template setting of the runs. '
             'Every rendering was produced from the model\'s tokenizer and template files at the revision used in its runs '
             '(Appendix~\\ref{app:config}) and agrees with the template\'s tokenized output, as checked in the runs. '
             '\\PH{body}, \\PH{question}, and \\PH{request} stand for the body, the question, and the request that opens the natural '
             'order (``Please write a short description.\'\'), in the language in which the model is tested (Table~\\ref{tab:self_items} '
             'gives the English texts; the Chinese ones are in the data package); \\NL{} marks a line feed, '
             'and every other character is the template\'s output. The gpt-oss template inserts the day of rendering (\\PH{date}). '
             'The token names of the DeepSeek models contain the characters U+FF5C and U+2581, printed here as | and \\_. '
             'Models whose templates render identically in every setting share one block. Where the three renderings of a block '
             'begin with the same long text, it is printed once, on the line Start, and stands as \\PH{start} in each rendering. '
             'A line break without \\NL{} only wraps the line.\n\n'
             '\\subsection{Main Probe}\n\\label{app:templates_main}\n\n'
             'The conversation is [\\textit{role}: body] [user: question], followed by the generation prompt.\n\n')
        s += section('main', MAIN_ORDER, L)
        s += ('\n\\subsection{Natural Conversation Order}\n\\label{app:templates_natural}\n\n'
              'The conversation is [user: request] [\\textit{role}: body] [user: question], followed by the generation prompt, for the '
              'models of Appendix~\\ref{app:natural}. In the first six models, the template keeps a System message in place after a user '
              'turn.\n\n')
        s += section('natural', NAT_GROUPS[0][1], L)
        s += ('\nIn the next three, the template does not keep it in place, so these models are compared on two roles: the Qwen3.8 '
              'template raises an error, the gpt-oss template leaves the message out, and the DeepSeek-R1-Distill-Llama-70B template '
              'moves its text to the start of the input.\n\n')
        s += section('natural', NAT_GROUPS[1][1], L)
        s += ('\nThe last three templates do not accept the main probe and merge the User message under test with the request and '
              'the question into one user turn. DeepSeek-V4.1-Flash has no chat template; its inputs are produced by the prompt encoder '
              'of the release (\\texttt{encoding/encoding.py}), in chat mode and in thinking mode with reasoning effort ``high\'\'.\n\n')
        s += section('natural', NAT_GROUPS[2][1], L)
        s += ('\nUnder the main probe, the DeepSeek-V4.1-Flash encoder gives an opening Assistant message no role token, the Ministral '
              'templates reject it, and all three merge an opening User message with the question. In the Ministral blocks, '
              '\\PH{start} is the text on the line Start of the natural-order block above.\n\n')
        s += section('main', NAT_GROUPS[2][1], L, not_run=True)
    else:
        s = ('本附录给出每个模型对一道题实际收到的输入：150 份正文中的第一份，加上第一种选项顺序的归属问题，由模型自己的对话模板在运行用到的每种模板设置下渲染。'
             '每条渲染都由模型运行时所用修订版本的 tokenizer 与模板文件生成（附录~\\ref{app:config}），并与模板的 token 化输出一致（运行中做过同样的检查）。'
             '\\PH{正文}、\\PH{问题}、\\PH{请求}分别代表正文、问题，以及自然顺序开头的请求（“请写一段简短的描述。”），语言与该模型测试所用的语言相同（英文文本见表~\\ref{tab:self_items}，中文文本在数据包中）；'
             '\\NL{}表示换行，其余每个字符都是模板的原样输出。gpt-oss 模板会插入渲染当天的日期（\\PH{日期}）。'
             'DeepSeek 模型的 token 名称含 U+FF5C 与 U+2581 两个字符，这里印作 | 与 \\_。各种设置下渲染完全相同的模型合为一块。'
             '同一块中三条渲染开头有一大段相同的文本时，这段文本只在 Start 一行印一次，在各条渲染中记作\\PH{开头}。没有\\NL{}的断行只是排版折行。\n\n'
             '\\subsection{主探针}\n\\label{app:templates_main}\n\n'
             '对话为 [\\textit{角色}：正文] [user：问题]，其后是生成提示。\n\n')
        s += section('main', MAIN_ORDER, L)
        s += ('\n\\subsection{自然对话顺序}\n\\label{app:templates_natural}\n\n'
              '对话为 [user：请求] [\\textit{角色}：正文] [user：问题]，其后是生成提示，模型见附录~\\ref{app:natural}。前六个模型的模板在 user 轮之后保留 System 消息的位置。\n\n')
        s += section('natural', NAT_GROUPS[0][1], L)
        s += ('\n接下来三个模型的模板不在原位保留 System 消息，因此只比较两种角色：Qwen3.8 模板报错，gpt-oss 模板略去这条消息，'
              'DeepSeek-R1-Distill-Llama-70B 模板把它的文本移到输入开头。\n\n')
        s += section('natural', NAT_GROUPS[1][1], L)
        s += ('\n最后三个模板不接受主探针，并把被测的 User 消息与请求和问题并入同一个 user 轮。DeepSeek-V4.1-Flash 没有对话模板，'
              '它的输入由发布附带的提示编码器（\\texttt{encoding/\\allowbreak encoding.py}）生成，分为 chat 模式与推理强度为“high”的思考模式。\n\n')
        s += section('natural', NAT_GROUPS[2][1], L)
        s += ('\n在主探针下，DeepSeek-V4.1-Flash 的编码器不给开头的 Assistant 消息任何角色 token，Ministral 模板直接拒绝这样的对话，'
              '三者都把开头的 User 消息与问题合并。Ministral 各块中的\\PH{开头}即上面自然顺序块中 Start 一行的文本。\n\n')
        s += section('main', NAT_GROUPS[2][1], L, not_run=True)
    return s


OUT.mkdir(parents=True, exist_ok=True)
for L in ('en', 'zh'):
    (OUT / f'appendix_a_{L}.tex').write_text(build(L), encoding='utf-8')
txt = []
for repo, rec in M.items():
    for part in ('main', 'natural'):
        for m, v in rec.get(part, {}).items():
            for r in ROLES:
                x = v[r]
                txt.append(f'{NAME[repo]} | {part} | {m} {json.dumps(v["kwargs"])} | {r}\n  '
                           + (repr(plain(x['text'], rec['lang'])) if 'text' in x else 'ERROR ' + x['error']))
(OUT / 'appendix_a_renderings.txt').write_text('\n'.join(txt) + '\n', encoding='utf-8')
print('main groups:', [[NAME[r] for r in g] for g in groups(MAIN_ORDER, 'main')])
print('natural groups:', [[NAME[r] for r in g] for gs in NAT_GROUPS for g in groups(gs[1], 'natural')])
