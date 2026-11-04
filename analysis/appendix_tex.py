"""Appendices app:templates (Rendered Items) and app:config (Configurations): their LaTeX blocks, generated from the archived
renderings so that no rendering or revision is typed by hand.

appendix_a (stats/appendix_a_20261029/appendix_a_{en,zh}.tex and appendix_a_renderings.txt): for every model, its own template's
  rendering of one item of the main probe and of the natural conversation order, in each template setting of the runs. Inputs
  (items/appendix_a_20261029/ of the wave-3 archive): the render records appendix_a_render_*.json (template renderings from each
  model's tokenizer and template files) and appendix_a_render_dsv41.json (DeepSeek-V4.1-Flash, prompt encoder of its release),
  with items/appendix_a_bodies_20261029.json. The body, question and request are replaced by placeholders, a line feed is shown as
  a symbol, the date that the gpt-oss template inserts by a placeholder, and the characters U+FF5C and U+2581 of DeepSeek token
  names by | and _; every other character is the template's output. Models with identical renderings in every setting share one
  block. Output: the LaTeX block for the EN or ZH appendix (between the markers %<<app_templates and %>>app_templates) and a
  plain-text copy of all renderings.
appendix_b (stats/appendix_b_20261029/appendix_b_{en,zh}.tex): the table of checkpoints (tab:self_config), built from the same
  render records (download commit of the template and tokenizer files, which is the commit of every file of the checkpoint, and
  the template arguments of each mode); DeepSeek-V4.1-Flash-1017 comes from ModelScope (revision 9f9818127fdf). The text states
  the software, decoding, API and statistics settings recorded in the run identities and the analysis code (bootstrap:
  two-level message geometry 1,000 family resamples, the causal-test statistics 4,000, the replacement runs 5,000, the thinking
  and reasoning summaries 10,000, all seed 0; random control subspace: seed 20261022 + layer). The block between %<<app_config and
  %>>app_config of the paper source was edited after it was generated (the paragraph on software and hardware).

usage: appendix_tex.py --results RES --out OUT
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import W3, Roots, arguments  # noqa: E402

def renderings(D, *more):
    """the rendering records of every model; the render files are read in order of the number of models they hold (the most
    complete first), and a model in several files must render identically in each. more: further folders of render files (the
    models added on 11-03/04 are in items/appendix_a_20261104)"""
    files = [f for d in (D,) + more for f in d.glob('appendix_a_render_*.json') if f.name != 'appendix_a_render_dsv41.json']
    recs = [json.loads(f.read_text(encoding='utf-8'))['models'] for f in sorted(files, key=lambda f: f.name)]
    return sorted(recs, key=lambda m: -len(m))


def appendix_a(ITEMS, OUT):
    D = ITEMS / 'appendix_a_20261029'
    M = {}
    for models in renderings(D, ITEMS / 'appendix_a_20261104'):
        for k, v in models.items():
            if k in M:
                for part in ('main', 'natural'):
                    assert M[k].get(part) == v.get(part), f'{k}: renderings differ between render files'
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
            'mistralai/Ministral-3-14B-Instruct-2512-BF16': 'Ministral-3-14B-Instruct', 'mistralai/Ministral-3-14B-Reasoning-2512': 'Ministral-3-14B-Reasoning',
            'baidu/ERNIE-4.5-21B-A3B-Thinking': 'ERNIE-4.5-21B-A3B-Thinking', 'XiaomiMiMo/MiMo-7B-RL-0530': 'MiMo-7B-RL-0530',
            'XiaomiMiMo/MiMo-VL-7B-RL-2508': 'MiMo-VL-7B-RL-2508', 'XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B': 'MiMo-V2.6-Distill-Qwen-9B',
            'moonshotai/Moonlight-16B-A3B-Instruct': 'Moonlight-16B-A3B-Instruct', 'swiss-ai/Apertus-8B-Instruct-2509': 'Apertus-8B-Instruct-2509',
            'microsoft/phi-4': 'phi-4'}
    MAIN_ORDER = ['Qwen/Qwen3.8-27B', 'Qwen/Qwen3.8-Flash-Next-FP8', 'Qwen/Qwen3.6-27B', 'Qwen/Qwen3.6-35B-A3B', 'Qwen/Qwen3.5-27B',
                  'zai-org/GLM-4.6V-Flash', 'zai-org/GLM-4.7-Flash', 'zai-org/GLM-5.3-Flash', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
                  'openai/gpt-oss-20b', 'openai/gpt-oss-120b', 'google/gemma-4-31B-it', 'ByteDance-Seed/Seed-OSS-36B-Instruct',
                  'LGAI-EXAONE/EXAONE-4.0.1-32B', 'ibm-granite/granite-4.2-30b', 'ibm-granite/granite-4.1-30b',
                  'allenai/OLMo-2-1124-13B-Instruct', 'allenai/OLMo-2-1124-7B-Instruct', 'baidu/ERNIE-4.5-21B-A3B-Thinking', 'XiaomiMiMo/MiMo-7B-RL-0530', 'XiaomiMiMo/MiMo-VL-7B-RL-2508', 'XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B', 'moonshotai/Moonlight-16B-A3B-Instruct', 'swiss-ai/Apertus-8B-Instruct-2509', 'microsoft/phi-4']
    NAT_GROUPS = [('three', ['ByteDance-Seed/Seed-OSS-36B-Instruct', 'google/gemma-4-31B-it', 'LGAI-EXAONE/EXAONE-4.0.1-32B',
                             'zai-org/GLM-4.6V-Flash', 'ibm-granite/granite-4.1-30b', 'allenai/OLMo-2-1124-7B-Instruct',
                             'baidu/ERNIE-4.5-21B-A3B-Thinking', 'XiaomiMiMo/MiMo-7B-RL-0530', 'moonshotai/Moonlight-16B-A3B-Instruct',
                             'microsoft/phi-4']),
                  ('two', ['Qwen/Qwen3.8-27B', 'openai/gpt-oss-20b', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
                           'swiss-ai/Apertus-8B-Instruct-2509']),
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
                  'models of Appendix~\\ref{app:natural}. In the first ten models, the template keeps a System message in place after a user '
                  'turn.\n\n')
            s += section('natural', NAT_GROUPS[0][1], L)
            s += ('\nIn the next four, the template does not keep it in place, so these models are compared on two roles: the Qwen3.8 '
                  'and Apertus templates raise an error, the gpt-oss template leaves the message out, and the DeepSeek-R1-Distill-Llama-70B template '
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
                  '对话为 [user：请求] [\\textit{角色}：正文] [user：问题]，其后是生成提示，模型见附录~\\ref{app:natural}。前十个模型的模板在 user 轮之后保留 System 消息的位置。\n\n')
            s += section('natural', NAT_GROUPS[0][1], L)
            s += ('\n接下来四个模型的模板不在原位保留 System 消息，因此只比较两种角色：Qwen3.8 与 Apertus 模板报错，gpt-oss 模板略去这条消息，'
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


def appendix_b(ITEMS, OUT):
    D = ITEMS / 'appendix_a_20261029'
    M = {}
    for models in renderings(D, ITEMS / 'appendix_a_20261104'):
        for k, v in models.items():
            M.setdefault(k, v)
    ORDER = ['Qwen/Qwen3.8-27B', 'Qwen/Qwen3.8-Flash-Next-FP8', 'Qwen/Qwen3.6-27B', 'Qwen/Qwen3.6-35B-A3B', 'Qwen/Qwen3.5-27B',
             'zai-org/GLM-4.6V-Flash', 'zai-org/GLM-4.7-Flash', 'zai-org/GLM-5.3-Flash', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
             'openai/gpt-oss-20b', 'openai/gpt-oss-120b', 'google/gemma-4-31B-it', 'ByteDance-Seed/Seed-OSS-36B-Instruct',
             'LGAI-EXAONE/EXAONE-4.0.1-32B', 'ibm-granite/granite-4.2-30b', 'ibm-granite/granite-4.1-30b',
             'allenai/OLMo-2-1124-13B-Instruct', 'allenai/OLMo-2-1124-7B-Instruct', 'baidu/ERNIE-4.5-21B-A3B-Thinking', 'XiaomiMiMo/MiMo-7B-RL-0530', 'XiaomiMiMo/MiMo-VL-7B-RL-2508', 'XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B', 'moonshotai/Moonlight-16B-A3B-Instruct', 'swiss-ai/Apertus-8B-Instruct-2509', 'microsoft/phi-4',
             'mistralai/Ministral-3-14B-Instruct-2512-BF16', 'mistralai/Ministral-3-14B-Reasoning-2512']
    LANG = {'en': {'zh': 'Chinese', 'en': 'English', 'both': 'Chinese, English', 'default': 'default'},
            'zh': {'zh': '中文', 'en': '英文', 'both': '中文、英文', 'default': '默认'}}


    def tt(s):
        return '\\texttt{' + s.replace('_', '\\_') + '}'


    def args(kw, L):
        if kw is None:
            return '--'
        if not kw:
            return LANG[L]['default']
        return tt(', '.join(f'{k}={v}' for k, v in kw.items()))


    def commit(rec):
        c = {v['commit'] for k, v in rec['files'].items() if k in ('chat_template.jinja', 'tokenizer_config.json') and v['commit']}
        assert len(c) == 1, c
        return c.pop()[:12]


    def rows(L):
        out = []
        for repo in ORDER:
            rec = M[repo]
            modes = {**rec.get('main', {}), **rec.get('natural', {})}
            nt = modes.get('non-thinking', {}).get('kwargs') if 'non-thinking' in modes else None
            th = modes.get('thinking', {}).get('kwargs') if 'thinking' in modes else None
            out.append(f'{tt(repo)} & {tt(commit(rec))} & {args(nt, L)} & {args(th, L)} \\\\')
            if repo == 'Qwen/Qwen3.8-27B':  # format ablation: the arguments recorded in the identity files of its runs
                tag = ' (format ablation)' if L == 'en' else '（格式消融）'
                nt_abl = '\\texttt{enable\\_thinking=\\allowbreak False, preserve\\_thinking=\\allowbreak False}'  # breakable in the narrow column
                out.append(f'{tt(repo)}{tag} & {tt(commit(rec))} & {nt_abl} & {tt("preserve_thinking=False")} \\\\')
            if repo == 'allenai/OLMo-2-1124-7B-Instruct':  # DeepSeek-V4.1-Flash before the two Ministral models, as in the natural-order appendix
                enc = ('\\texttt{thinking\\_mode=chat}', '\\texttt{thinking\\_mode=thinking, reasoning\\_effort=high}')
                out.append(f'{tt("deepseek-ai/DeepSeek-V4.1-Flash-1017")} & {tt("9f9818127fdf")}$^{{a}}$ & {enc[0]} & {enc[1]} \\\\')
        return '\n'.join(out)


    def build(L):
        if L == 'en':
            head = ('Table~\\ref{tab:self_config} identifies every checkpoint by its repository and the revision of its files; all files of '
                    'a checkpoint come from that one revision, and the weights are used in the format of the release '
                    '(Table~\\ref{tab:self_models}). The table also gives the template arguments of each mode, with which the items of '
                    'Appendix~\\ref{app:templates} are rendered (``default\'\': no argument).\n\n')
            cap = ('Checkpoints and template arguments. Revision: the first 12 characters of the commit. '
                   '$^{a}$~ModelScope revision; the files are identical to those on Hugging Face. '
                   'DeepSeek-V4.1-Flash has no chat template, and its arguments are those of the prompt encoder of the release.')
            hdr = 'Repository & Revision & Without thinking & With thinking \\\\'
            text = ('\\paragraph{Software.} Local runs use PyTorch (2.5.1 to 2.11.0) and Hugging Face Transformers (5.8.0 or 5.19.0); the '
                    'identity file of each run in the data package records the versions it used. DeepSeek-V4.1-Flash has no Transformers '
                    'implementation and runs on the model code of its release, with the custom kernels of the release replaced by PyTorch '
                    'implementations of the same arithmetic and the weights kept in the release format (FP8, with FP4 expert weights). '
                    'Both Ministral-3-14B models run in BF16.\n\n'
                    '\\paragraph{Decoding and APIs.} In thinking mode, local runs generate greedily, up to 4,096 tokens for Qwen3.8-27B in '
                    'Section~\\ref{sec:res_thinking} and up to 3,072 tokens elsewhere. The thinking-mode '
                    'self-reference of two models comes from the developer\'s API with the same inputs: GLM-5.3-Flash through the Z.ai API '
                    '(model \\texttt{glm-5.3-flash}, temperature 0, at most 3,072 tokens), and DeepSeek-V4.1-Flash through the DeepSeek API '
                    '(model \\texttt{deepseek-flash}, which the provider documents as served by DeepSeek-V4.1-Flash-1017; reasoning effort '
                    'high, temperature 1.0 and top-$p$ 1.0, the settings the developer recommends, since the thinking mode offers no greedy '
                    'decoding; at most 3,072 tokens). For every request, the number of prompt tokens reported by the API equals that of the '
                    'local rendering.\n\n'
                    '\\paragraph{Statistics.} All intervals are 95\\% intervals with a fixed seed: percentile bootstrap intervals over wording '
                    'families for the geometry (1,000 resamples), over bodies or items elsewhere (4,000 to 10,000 resamples), and over models '
                    'for the rank correlations of Appendix~\\ref{app:benchmarks} (10,000 resamples); the shares of finished items in '
                    'Figure~\\ref{fig:self_mechanism} carry Wilson intervals. Each subspace of the replacement is fitted on the discovery bodies '
                    'and stored with the random subspace of its control (the orthonormalized columns of a Gaussian matrix, drawn per layer with '
                    'a fixed seed); every test that uses it reads the stored coordinates.\n')
        else:
            head = ('表~\\ref{tab:self_config} 用仓库和修订版本标识每个检查点：一个检查点的全部文件都来自同一个修订版本，权重按发布格式使用'
                    '（表~\\ref{tab:self_models}）。表中还给出每种模式的模板参数，附录~\\ref{app:templates} 的题目就按这些参数渲染（“默认”：不传参数）。\n\n')
            cap = ('检查点与模板参数。修订版本：提交号的前 12 位。$^{a}$~ModelScope 修订版本，文件与 Hugging Face 上的相同。'
                   'DeepSeek-V4.1-Flash 没有对话模板，其参数是发布附带的提示编码器的参数。')
            hdr = '仓库 & 修订版本 & 不思考 & 思考 \\\\'
            text = ('\\paragraph{软件。} 本地运行使用 PyTorch（2.5.1 至 2.11.0）与 Hugging Face Transformers（5.8.0 或 5.19.0），'
                    '每次运行所用的版本记录在数据包中该运行的身份文件里。DeepSeek-V4.1-Flash 没有 Transformers 实现，运行于其发布附带的模型代码，'
                    '其中的自定义内核换成按相同算术规则实现的 PyTorch 版本，权重保持发布格式（FP8，专家权重为 FP4）。两个 Ministral-3-14B 模型以 BF16 运行。\n\n'
                    '\\paragraph{解码与 API。} 本地运行在思考模式下贪心生成：第~\\ref{sec:res_thinking} 节的 Qwen3.8-27B 最多 4,096 个 token，'
                    '其余最多 3,072 个。两个模型的思考模式自指来自开发方 API，输入相同：'
                    'GLM-5.3-Flash 用 Z.ai API（模型 \\texttt{glm-5.3-flash}，temperature 0，最多 3,072 个 token）；DeepSeek-V4.1-Flash 用 DeepSeek API'
                    '（模型 \\texttt{deepseek-flash}，服务方文档写明由 DeepSeek-V4.1-Flash-1017 提供；推理强度 high，temperature 1.0、top-$p$ 1.0，'
                    '即开发方推荐的设置，因为思考模式不提供贪心解码；最多 3,072 个 token）。每个请求中 API 报告的提示 token 数都与本地渲染相同。\n\n'
                    '\\paragraph{统计。} 所有区间都是固定随机种子的 95\\% 区间：几何为按措辞族重抽样的百分位 bootstrap 区间（1,000 次），其余按正文或题目重抽样'
                    '（4,000 至 10,000 次），附录~\\ref{app:benchmarks} 的秩相关按模型重抽样（10,000 次）；图~\\ref{fig:self_mechanism} 中已完成题目的比例用 Wilson 区间。'
                    '替换所用的每个子空间都在发现组正文上拟合，并与其对照用的随机子空间（高斯矩阵各列正交化，每层用固定种子抽取）一起保存；'
                    '用到它的每项检验都读取保存的坐标。\n')
        table = ('\\begin{table}[htbp]\n\\centering\n\\caption{' + cap + '}\n\\label{tab:self_config}\n\\scriptsize\n'
                 '\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{llp{3.2cm}p{3.5cm}}\n\\toprule\n' + hdr + '\n\\midrule\n'
                 + rows(L) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n')
        return head + table + text


    OUT.mkdir(parents=True, exist_ok=True)
    for L in ('en', 'zh'):
        (OUT / f'appendix_b_{L}.tex').write_text(build(L), encoding='utf-8')
    print(rows('en'))


def main(argv=None):
    args = arguments(__doc__).parse_args(argv)
    r = Roots(args)
    ITEMS = r.res(W3 + '/items')
    appendix_a(ITEMS, r.out(W3 + '/stats/appendix_a_20261029'))
    appendix_b(ITEMS, r.out(W3 + '/stats/appendix_b_20261029'))


if __name__ == '__main__':
    main()
