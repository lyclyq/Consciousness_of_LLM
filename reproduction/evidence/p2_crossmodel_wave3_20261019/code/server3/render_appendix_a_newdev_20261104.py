"""Appendix A renderings of the models added on 11-03/04 (render_appendix_a_20261029.py with this model list; Moonlight through
moonlight_compat_20261103.install, as its runs). Original docstring follows.
Appendix A renderings (user 10-29): for every model of the paper, its own chat template's rendering of one item of the main
probe and of the natural conversation order, in each template setting of the runs (tmpl_kwargs in the runs' identity.json).
Tokenizer and config files only, no weights. Main probe: [ROLE: body][user: attribution question, first option order] plus the
generation prompt. Natural order: [user: request][ROLE: body][user: attribution question] plus the generation prompt. ROLE is
assistant / user / system; a template error is recorded in place of the rendering. Body: the first of the 150 bodies
(role150_numeric_records_d00) in the run's language; question and request as in crossmodel_suite.py and crossmodel_natural3.py.
The tokenizer is built as in the runs (AutoTokenizer with trust_remote_code, then crossmodel_suite.checked_tokenizer), and each
rendering is checked as the runs check it (text tokenized == tokenize=True). Also recorded: sha256 and download commit of the
tokenizer and template files. Models whose folder is absent under HF_ROOT are skipped.
usage: render_appendix_a_newdev_20261104.py HF_ROOT BODIES_JSON OUT_JSON   (next to crossmodel_suite.py, with the venv of the runs)"""
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
    # the seven models added on 11-03/04 (settings as their runs: newdev_server3_20261103.sh, apertus_pro_20261103.sh,
    # newdev_pro_mimo2_20261103.sh, newdev_phi4_server3_20261103.sh); folders absent on this machine are skipped
    'baidu/ERNIE-4.5-21B-A3B-Thinking': ('zh', {TH: {}}, {TH: {}}),
    'XiaomiMiMo/MiMo-7B-RL-0530': ('zh', {NT: OFF, TH: {}}, {NT: OFF}),
    'moonshotai/Moonlight-16B-A3B-Instruct': ('zh', {NT: {}}, {NT: {}}),
    'microsoft/phi-4': ('en', {NT: {}}, {NT: {}}),
    'swiss-ai/Apertus-8B-Instruct-2509': ('en', {NT: {}}, {NT: {}}),
    'XiaomiMiMo/MiMo-VL-7B-RL-2508': ('zh', {TH: {}}, {}),
    'XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B': ('zh', {NT: OFF, TH: {}}, {}),
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
    if repo.startswith('moonshotai/'):
        import moonlight_compat_20261103 as mc
        mc.install(d)
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
