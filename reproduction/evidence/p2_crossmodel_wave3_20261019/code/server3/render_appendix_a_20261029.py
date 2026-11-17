"""Appendix A renderings (user 10-29): for every model of the paper, its own chat template's rendering of one item of the main
probe and of the natural conversation order, in each template setting of the runs (tmpl_kwargs in the runs' identity.json).
Tokenizer and config files only, no weights. Main probe: [ROLE: body][user: attribution question, first option order] plus the
generation prompt. Natural order: [user: request][ROLE: body][user: attribution question] plus the generation prompt. ROLE is
assistant / user / system; a template error is recorded in place of the rendering. Body: the first of the 150 bodies
(role150_numeric_records_d00) in the run's language; question and request as in crossmodel_suite.py and crossmodel_natural3.py.
The tokenizer is built as in the runs (AutoTokenizer with trust_remote_code, then crossmodel_suite.checked_tokenizer), and each
rendering is checked as the runs check it (text tokenized == tokenize=True). Also recorded: sha256 and download commit of the
tokenizer and template files. Models whose folder is absent under HF_ROOT are skipped.
usage: render_appendix_a_20261029.py HF_ROOT BODIES_JSON OUT_JSON   (next to crossmodel_suite.py, with the venv of the runs)"""
import hashlib
import json
import sys
from pathlib import Path

import transformers
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402

NT, TH = 'non-thinking', 'thinking'
OFF, ON = {'enable_thinking': False}, {'enable_thinking': True}
# repo: (language, main-probe settings, natural-order settings)
MODELS = {
    'Qwen/Qwen3.8-27B': ('zh', {NT: OFF, TH: ON}, {NT: OFF}),
    'Qwen/Qwen3.8-Flash-Next-FP8': ('zh', {NT: OFF, TH: {}}, {}),
    'Qwen/Qwen3.6-27B': ('zh', {NT: OFF, TH: {}}, {}),
    'Qwen/Qwen3.6-35B-A3B': ('zh', {NT: OFF, TH: {}}, {}),
    'Qwen/Qwen3.5-27B': ('zh', {NT: OFF, TH: {}}, {}),
    'zai-org/GLM-4.6V-Flash': ('zh', {NT: OFF, TH: {}}, {NT: OFF}),
    'zai-org/GLM-4.7-Flash': ('zh', {NT: OFF, TH: ON}, {}),
    'zai-org/GLM-5.3-Flash': ('zh', {TH: {}}, {}),
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B': ('zh', {TH: {}}, {TH: {}}),
    'google/gemma-4-31B-it': ('en', {NT: {}, TH: ON}, {NT: {}}),
    'ByteDance-Seed/Seed-OSS-36B-Instruct': ('zh', {NT: {'thinking_budget': 0}, TH: {}}, {NT: {'thinking_budget': 0}}),
    'LGAI-EXAONE/EXAONE-4.0.1-32B': ('en', {NT: OFF, TH: ON}, {NT: OFF}),
    'openai/gpt-oss-20b': ('en', {TH: {}}, {TH: {}}),
    'openai/gpt-oss-120b': ('en', {TH: {}}, {}),
    'ibm-granite/granite-4.1-30b': ('en', {NT: {}}, {NT: {}}),
    'ibm-granite/granite-4.2-30b': ('en', {NT: OFF, TH: ON}, {}),
    'allenai/OLMo-2-1124-7B-Instruct': ('en', {NT: {}}, {NT: {}}),
    'allenai/OLMo-2-1124-13B-Instruct': ('en', {NT: {}}, {}),
    # main probe of the two Ministral models: not run (the templates reject it); rendered for Appendix A only (added 10-29)
    'mistralai/Ministral-3-14B-Instruct-2512-BF16': ('en', {NT: {}}, {NT: {}}),
    'mistralai/Ministral-3-14B-Reasoning-2512': ('en', {TH: {}}, {TH: {}}),
}
REQ = {'zh': '请写一段简短的描述。', 'en': 'Please write a short description.'}
FILES = ('chat_template.jinja', 'chat_template.json', 'tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json',
         'generation_config.json', 'config.json')


def question(lang):
    head, opt, tail = cs.Q_ATTR if lang == 'zh' else cs.Q_ATTR_EN
    return head + ' '.join(f'{cs.LETTERS[i]}. {opt[k]}' for i, k in enumerate(cs.ORDERS[0])) + tail


def commit(d, name):
    m = d / '.cache' / 'huggingface' / 'download' / (name + '.metadata')
    return m.read_text().split('\n')[0].strip() if m.exists() else None


def render(tok, msgs, kw):
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        ref = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
    except Exception as e:
        return {'error': f'{type(e).__name__}: {str(e)[:300]}'}
    return {'text': s, 'tokenized_equal': cs.ids_of(tok(s, add_special_tokens=False)) == cs.ids_of(ref)}


root, out_path = Path(sys.argv[1]), Path(sys.argv[3])
bodies = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
out = {'script': Path(__file__).name, 'transformers': transformers.__version__, 'hf_root': str(root), 'body_id': bodies['id'],
       'models': {}}
for repo, (lang, main, natural) in MODELS.items():
    d = root / repo
    try:
        if not d.is_dir():
            continue
    except OSError as e:  # e.g. a stale NFS mount on PRO
        print(repo, 'skipped:', e, flush=True)
        continue
    rec = {'lang': lang, 'files': {f: {'sha256': hashlib.sha256((d / f).read_bytes()).hexdigest(), 'commit': commit(d, f)}
                                   for f in FILES if (d / f).exists()}}
    try:
        tok = cs.checked_tokenizer(AutoTokenizer.from_pretrained(d, trust_remote_code=True), str(d))
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {str(e)[:300]}'
        out['models'][repo] = rec
        print(repo, 'ERROR', rec['error'], flush=True)
        continue
    rec['tokenizer_class'] = type(tok).__name__
    rec['rebuilt_from'] = getattr(tok, '_p2_rebuilt_from', None)
    body, q, req = bodies[lang], question(lang), REQ[lang]
    rec['main'] = {m: dict(kwargs=kw, **{r: render(tok, [{'role': r, 'content': body}, {'role': 'user', 'content': q}], kw)
                                         for r in cs.ROLES}) for m, kw in main.items()}
    rec['natural'] = {m: dict(kwargs=kw, **{r: render(tok, [{'role': 'user', 'content': req}, {'role': r, 'content': body},
                                                            {'role': 'user', 'content': q}], kw) for r in cs.ROLES})
                      for m, kw in natural.items()}
    out['models'][repo] = rec
    print(repo, 'ok', flush=True)
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('wrote', out_path, len(out['models']), 'models')
