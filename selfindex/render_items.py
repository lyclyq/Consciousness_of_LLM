"""Renderings of the items under each model's own template (Appendix A) and checks of the templates.

Only tokenizer, template and config files are read, never weights. The checkpoints and template settings are those of Table 10:
CHECKPOINTS gives each repository with the revision of its files, the language of its items and the template arguments of each mode
of the main probe and of the natural conversation order.

Modes (--mode):
  fetch         download the tokenizer, template and config files of the checkpoints at their revisions into ROOT/<repository>.
  appendix      for every checkpoint found under ROOT, its rendering of one item of the main probe, [ROLE: body][user: attribution
                question, first option order], and of the natural conversation order, [user: request][ROLE: body][user: question],
                with the generation prompt, ROLE assistant / user / system, in each template setting of its runs. The body is the
                first of the 150 bodies in the language of the model. The tokenizer is built as in the runs (AutoTokenizer with
                trust_remote_code, then selfindex.load.checked_tokenizer), and each rendering is checked as the runs check it (the
                text rendering tokenized again equals the token rendering); a template error is recorded in place of the
                rendering. Also recorded: sha256 and download commit of the tokenizer and template files. --with-ids adds the token
                ids of each rendering.
  deepseek      the same two items for DeepSeek-V4.1-Flash, which has no chat template, through the prompt encoder of its release
                (encoding.py; its sha256 is compared with the one of the runs), in chat mode and in thinking mode with reasoning
                effort "high". --tokenizer (a folder with its tokenizer.json) and --with-ids add the token ids.
  placeholders  the template identity of each checkpoint: the sha256 of the chat template in use and of the tokenizer files, and
                the rendering of [ROLE: BODY][user: Q] with the generation prompt in each template setting (--extra-kwargs adds
                settings).
  facts         the template facts of Table 2 that the item renderings do not show directly, tested and printed as PASS/FAIL:
                Qwen3.5-27B, Granite-4.2-30B and Granite-4.1-30B (message header and trailer of each role, what precedes the
                conversation, the generation prompt of each mode, layers, experts and data type of config.json) and gpt-oss-20b /
                gpt-oss-120b (final-channel tag of an Assistant message, System as a developer message, MXFP4 MoE).
  check-inputs  compare the token ids that an earlier run fed the model (features/<body>__<role>.npz: input_token_ids, as in
                the run folders of the fragment inputs) with the template's own rendering of the same messages.
Templates that read the current date (gpt-oss: "Current date: ...") render it from --date (default: today).
usage: python -m selfindex.render_items --mode fetch --root DIR [--models REPO,REPO]
       python -m selfindex.render_items --mode appendix --root DIR --out OUT.json [--models REPO,REPO] [--date YYYY-MM-DD] [--with-ids]
       python -m selfindex.render_items --mode deepseek --out OUT.json [--encoding PY] [--tokenizer DIR --with-ids]
       python -m selfindex.render_items --mode placeholders --root DIR --out OUT.txt [--models ...] [--extra-kwargs JSON]
       python -m selfindex.render_items --mode facts --root DIR --out OUT.txt [--date YYYY-MM-DD]
       python -m selfindex.render_items --mode check-inputs --tokenizer DIR --runs RUN_DIR [RUN_DIR ...] [--kwargs JSON] [--out OUT.txt]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

from . import items as I
from .load import checked_tokenizer
from .render import ids_of

NT, TH = 'non-thinking', 'thinking'
OFF, ON = {'enable_thinking': False}, {'enable_thinking': True}
# Table 10. repository: (revision, language, main-probe settings, natural-order settings); the order is that of the renderings.
# The two Ministral models do not accept the main probe (their templates reject an opening Assistant message); their main-probe
# renderings are shown for Appendix A only.
CHECKPOINTS = {
    'Qwen/Qwen3.8-27B': ('1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0', 'zh', {NT: OFF, TH: ON}, {NT: OFF}),
    'Qwen/Qwen3.8-Flash-Next-FP8': ('236dfdf285828023ca3bcd3f37366c58a3469b13', 'zh', {NT: OFF, TH: {}}, {}),
    'Qwen/Qwen3.6-27B': ('6a9e13bd6fc8f0983b9b99948120bc37f49c13e9', 'zh', {NT: OFF, TH: {}}, {}),
    'Qwen/Qwen3.6-35B-A3B': ('995ad96eacd98c81ed38be0c5b274b04031597b0', 'zh', {NT: OFF, TH: {}}, {}),
    'Qwen/Qwen3.5-27B': ('fc05daec18b0a78c049392ed2e771dde82bdf654', 'zh', {NT: OFF, TH: {}}, {}),
    'zai-org/GLM-4.6V-Flash': ('411bb4d77144a3f03accbf4b780f5acb8b7cde4e', 'zh', {NT: OFF, TH: {}}, {NT: OFF}),
    'zai-org/GLM-4.7-Flash': ('7dd20894a642a0aa287e9827cb1a1f7f91386b67', 'zh', {NT: OFF, TH: ON}, {}),
    'zai-org/GLM-5.3-Flash': ('46a050bec13002ad29ee795c16e61cf4fa3c4a9c', 'zh', {TH: {}}, {}),
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B': ('b1c0b44b4369b597ad119a196caf79a9c40e141e', 'zh', {TH: {}}, {TH: {}}),
    'google/gemma-4-31B-it': ('842da3794eaa0b77d5f08bae87a17459d91ff475', 'en', {NT: {}, TH: ON}, {NT: {}}),
    'ByteDance-Seed/Seed-OSS-36B-Instruct': ('497f1dca95ebdec98e41d517b9f060ee753c902f', 'zh', {NT: {'thinking_budget': 0}, TH: {}},
                                             {NT: {'thinking_budget': 0}}),
    'LGAI-EXAONE/EXAONE-4.0.1-32B': ('a6b0bd4b0fb47cd3c644d60d12dba8ecdb9dc810', 'en', {NT: OFF, TH: ON}, {NT: OFF}),
    'openai/gpt-oss-20b': ('6cee5e81ee83917806bbde320786a8fb61efebee', 'en', {TH: {}}, {TH: {}}),
    'openai/gpt-oss-120b': ('b5c939de8f754692c1647ca79fbf85e8c1e70f8a', 'en', {TH: {}}, {}),
    'ibm-granite/granite-4.1-30b': ('4fae6278f7132abf5e971f9de49ebbad09c54cce', 'en', {NT: {}}, {NT: {}}),
    'ibm-granite/granite-4.2-30b': ('9e668ce1c538387ef24d3644e9b0606647762636', 'en', {NT: OFF, TH: ON}, {}),
    'allenai/OLMo-2-1124-7B-Instruct': ('470b1fba1ae01581f270116362ee4aa1b97f4c84', 'en', {NT: {}}, {NT: {}}),
    'allenai/OLMo-2-1124-13B-Instruct': ('3a5c85baefbb1896a54d56fe2e76c0395627ddf4', 'en', {NT: {}}, {}),
    'mistralai/Ministral-3-14B-Instruct-2512-BF16': ('3cea74c1ebaf5ce5f5a2553de470e2ceab825142', 'en', {NT: {}}, {NT: {}}),
    'mistralai/Ministral-3-14B-Reasoning-2512': ('51f9210f3cd20f3452a80d5819d15dc61cc50630', 'en', {TH: {}}, {TH: {}}),
}
# DeepSeek-V4.1-Flash-1017: no chat template; the encoder settings of its runs. Its runs used the ModelScope revision 9f9818127fdf;
# the Hugging Face revision below holds the same files (equal sha256 of encoding.py, tokenizer.json and the inference code).
DEEPSEEK = ('deepseek-ai/DeepSeek-V4.1-Flash-1017', '1bae0e1088993b9b82de08f59de5be5a34e42170')
DEEPSEEK_KW = {NT: {'thinking_mode': 'chat'}, TH: {'thinking_mode': 'thinking', 'reasoning_effort': 'high'}}
ENCODING_SHA256_OF_RUNS = '13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2'
ENCODING_PY = Path(__file__).resolve().parent / 'dsv41' / 'official' / 'encoding' / 'encoding.py'
FILES = ('chat_template.jinja', 'chat_template.json', 'tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json',
         'generation_config.json', 'config.json')
FETCH_PATTERNS = ['*.json', '*.jinja', '*.model', '*.txt', 'tokenizer*']
DEEPSEEK_PATTERNS = ['tokenizer*', 'config.json', 'encoding/encoding.py', 'inference/config.json']


# ----------------------------------------------------------------------------------------------- inputs
def appendix_body():
    """The body of the Appendix A item: the first of the 150 bodies, with its id and both languages."""
    zh, en = I.bodies('zh')[0], I.bodies('en')[0]
    assert zh['id'] == en['id']
    return {'id': zh['id'], 'zh': zh['body'], 'en': en['body']}


def question(lang):
    return I.attr_question(I.orders()[0], lang)


def main_msgs(role, body, lang):
    return [{'role': role, 'content': body}, {'role': 'user', 'content': question(lang)}]


def natural_msgs(role, body, lang):
    return [{'role': 'user', 'content': I.natural_order_request(lang)}, {'role': role, 'content': body},
            {'role': 'user', 'content': question(lang)}]


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def commit(d, name):
    """The commit a file was downloaded from (huggingface_hub writes it into the folder's download metadata)."""
    m = d / '.cache' / 'huggingface' / 'download' / (name + '.metadata')
    return m.read_text().split('\n')[0].strip() if m.exists() else None


@contextlib.contextmanager
def fixed_date(date):
    """Templates that read the current date (strftime_now) render `date` (YYYY-MM-DD) instead of today."""
    if not date:
        yield
        return
    import datetime as dt
    from transformers.utils import chat_template_utils as C
    day = dt.datetime.strptime(date, '%Y-%m-%d')

    class Fixed(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(day.year, day.month, day.day)
    old = C.datetime
    C.datetime = Fixed
    try:
        yield
    finally:
        C.datetime = old


def load_tokenizer(d):
    from transformers import AutoTokenizer
    return checked_tokenizer(AutoTokenizer.from_pretrained(d, trust_remote_code=True), str(d))


def render(tok, msgs, kw, with_ids=False):
    """The template's text rendering with the generation prompt, and whether it tokenizes to the token rendering."""
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        ref = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
    except Exception as e:
        return {'error': f'{type(e).__name__}: {str(e)[:300]}'}
    rec = {'text': s, 'tokenized_equal': ids_of(tok(s, add_special_tokens=False)) == ids_of(ref)}
    if with_ids:
        rec['ids'] = ids_of(ref)
    return rec


def selected(models):
    if not models:
        return list(CHECKPOINTS)
    want = models.split(',') if isinstance(models, str) else list(models)
    bad = [m for m in want if m not in CHECKPOINTS]
    assert not bad, f'not in Table 10: {bad}'
    return [m for m in CHECKPOINTS if m in want]


def present(root, repo):
    d = Path(root) / repo
    try:
        return d.is_dir()
    except OSError as e:  # e.g. a stale network mount
        print(repo, 'skipped:', e, flush=True)
        return False


# ----------------------------------------------------------------------------------------------- fetch
def fetch(root, models=None, deepseek=False):
    from huggingface_hub import snapshot_download
    for repo in selected(models):
        rev = CHECKPOINTS[repo][0]
        p = snapshot_download(repo, revision=rev, local_dir=os.path.join(root, repo), max_workers=4, allow_patterns=FETCH_PATTERNS)
        print(repo, rev[:12], '->', p, flush=True)
    if deepseek:
        repo, rev = DEEPSEEK
        p = snapshot_download(repo, revision=rev, local_dir=os.path.join(root, repo), max_workers=4, allow_patterns=DEEPSEEK_PATTERNS)
        print(repo, rev[:12], '->', p, flush=True)


# ----------------------------------------------------------------------------------------------- Appendix A
def appendix(root, models=None, with_ids=False, date=None):
    """The Appendix A renderings of the checkpoints present under root, in the layout of the archived renderings."""
    import transformers
    root = Path(root)
    body = appendix_body()
    out = {'script': Path(__file__).name, 'transformers': transformers.__version__, 'hf_root': str(root), 'body_id': body['id'],
           'models': {}}
    with fixed_date(date):
        for repo in selected(models):
            _, lang, main, natural = CHECKPOINTS[repo]
            if not present(root, repo):
                continue
            d = root / repo
            rec = {'lang': lang, 'files': {f: {'sha256': sha256((d / f).read_bytes()), 'commit': commit(d, f)}
                                           for f in FILES if (d / f).exists()}}
            try:
                tok = load_tokenizer(d)
            except Exception as e:
                rec['error'] = f'{type(e).__name__}: {str(e)[:300]}'
                out['models'][repo] = rec
                print(repo, 'ERROR', rec['error'], flush=True)
                continue
            rec['tokenizer_class'] = type(tok).__name__
            rec['rebuilt_from'] = getattr(tok, '_p2_rebuilt_from', None)
            b = body[lang]
            rec['main'] = {m: dict(kwargs=kw, **{r: render(tok, main_msgs(r, b, lang), kw, with_ids) for r in I.ROLES})
                           for m, kw in main.items()}
            rec['natural'] = {m: dict(kwargs=kw, **{r: render(tok, natural_msgs(r, b, lang), kw, with_ids) for r in I.ROLES})
                              for m, kw in natural.items()}
            out['models'][repo] = rec
            print(repo, 'ok', flush=True)
    return out


def load_encoder(path=None):
    spec = importlib.util.spec_from_file_location('ds_encoding', Path(path or ENCODING_PY))
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    return E


def appendix_deepseek(encoding=None, tokenizer=None, with_ids=False):
    """The Appendix A renderings of DeepSeek-V4.1-Flash through the prompt encoder of its release (the main probe was not run)."""
    enc = Path(encoding or ENCODING_PY)
    sha = sha256(enc.read_bytes())
    E = load_encoder(enc)
    tok = None
    if with_ids:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(tokenizer) / 'tokenizer.json'))
    body = appendix_body()['zh']
    out = {'encoding_py': str(enc), 'encoding_py_sha256': sha, 'same_as_runs': sha == ENCODING_SHA256_OF_RUNS, 'lang': 'zh',
           'main': {}, 'natural': {}}
    for m, kw in DEEPSEEK_KW.items():
        for part, msgs_of in (('main', main_msgs), ('natural', natural_msgs)):
            out[part][m] = {'kwargs': kw}
            for r in I.ROLES:
                try:
                    rec = {'text': E.encode_messages(msgs_of(r, body, 'zh'), **kw)}
                except Exception as e:
                    rec = {'error': f'{type(e).__name__}: {str(e)[:300]}'}
                if tok is not None and 'text' in rec:
                    rec['ids'] = tok.encode(rec['text'], add_special_tokens=False).ids
                out[part][m][r] = rec
    return out


# ----------------------------------------------------------------------------------------------- template identity
def placeholder_block(repo, d, modes):
    """Template identity of one checkpoint: template and file hashes, and [ROLE: BODY][user: Q] in each setting."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    tmpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    files = {f.name: sha256(f.read_bytes())[:16] for f in sorted(d.iterdir())
             if f.name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json', 'generation_config.json')}
    out = [f'##### {repo}\nchat_template sha256 {sha256(tmpl.encode())}\nfiles {json.dumps(files)}']
    for kw in modes:
        for role in I.ROLES:
            msgs = [{'role': role, 'content': 'BODY'}, {'role': 'user', 'content': 'Q'}]
            try:
                s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            except Exception as e:  # some templates reject a role in some positions
                s = f'<error: {type(e).__name__}: {e}>'
            out.append(f'--- kwargs {json.dumps(kw)} role {role}\n{s!r}')
    return out


def placeholders(root, models=None, extra=(), date=None):
    out = []
    with fixed_date(date):
        for repo in selected(models):
            if not present(root, repo):
                continue
            _, _, main, _ = CHECKPOINTS[repo]
            modes = []
            for kw in list(main.values()) + list(extra):
                if kw not in modes:
                    modes.append(kw)
            out += placeholder_block(repo, Path(root) / repo, modes)
    return out


# ----------------------------------------------------------------------------------------------- Table 2 facts
TABLE2 = {
    'Qwen/Qwen3.5-27B': {'ts': '<|im_start|>', 'modes': {NT: OFF, TH: {}}},
    'ibm-granite/granite-4.2-30b': {'ts': '<|im_start|>', 'modes': {NT: OFF, TH: ON}},
    'ibm-granite/granite-4.1-30b': {'ts': '<|start_of_role|>', 'modes': {NT: {}, 'probe enable_thinking=true': ON,
                                                                        'probe enable_thinking=false': OFF}},
}


def _segment(s, body, ts, role):
    """Message segment of the critical message: header = from the last turn start before the body to the body, trailer = from the
    body to the next turn start; the header with the role name replaced, for comparing roles."""
    ib = s.index(body)
    hs = s.rindex(ts, 0, ib)
    qs = s.index(ts, ib + len(body))
    head = s[hs:ib]
    return {'pre': s[:hs], 'head': head, 'head_norm': head.replace(ts + role, ts + '{ROLE}', 1), 'trail': s[ib + len(body):qs]}


def table2_facts(root, checks):
    """Renderings and checks of Qwen3.5-27B, Granite-4.2-30B and Granite-4.1-30B; appends (model, claim, ok, evidence) to checks."""
    root = Path(root)
    body = appendix_body()
    out = []

    def check(model, claim, ok, evidence=''):
        checks.append((model, claim, ok, evidence))
    for repo, spec in TABLE2.items():
        d = root / repo
        lang = CHECKPOINTS[repo][1]
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
        tmpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
        files = {f.name: sha256(f.read_bytes()) for f in sorted(d.iterdir())
                 if f.name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json', 'generation_config.json',
                               'special_tokens_map.json')}
        jinja = (d / 'chat_template.jinja').read_text(encoding='utf-8') if (d / 'chat_template.jinja').exists() else None
        tc = json.loads((d / 'tokenizer_config.json').read_text(encoding='utf-8'))
        out.append(f'##### {repo}')
        out.append(f'chat_template in use sha256 {sha256(tmpl.encode())}; equals chat_template.jinja: {jinja == tmpl}; '
                   f'tokenizer_config.json carries a chat_template: {"chat_template" in tc}'
                   + (f' (equal to the one in use: {tc["chat_template"] == tmpl})' if 'chat_template' in tc else ''))
        out.append('files sha256 ' + json.dumps(files))
        cfg = json.loads((d / 'config.json').read_text(encoding='utf-8'))
        tcfg = cfg.get('text_config', cfg)
        n_layers = tcfg.get('num_hidden_layers')
        moe = {k: tcfg[k] for k in tcfg if any(x in k for x in ('expert', 'moe')) and tcfg[k] not in (None, 0, False, [], {})}
        dtype = tcfg.get('torch_dtype', tcfg.get('dtype', cfg.get('torch_dtype', cfg.get('dtype'))))
        quant = cfg.get('quantization_config', tcfg.get('quantization_config'))
        out.append(f'config: architectures {cfg.get("architectures")}, num_hidden_layers {n_layers}, dtype {dtype}, quantization_config {quant}, '
                   f'expert/moe fields {moe}')
        check(repo, 'Table 2: 64 layers, dense, BF16 (config.json)', n_layers == 64 and not moe and str(dtype) == 'bfloat16' and not quant,
              f'layers {n_layers}, moe {moe}, dtype {dtype}, quant {quant}')
        b = body[lang]
        qs = {'attribution': question(lang), 'neutral request': I.reply_request(lang)}
        R = {}
        for mode, kw in spec['modes'].items():
            for qn, q in qs.items():
                for role in I.ROLES + ('none',):
                    msgs = ([{'role': role, 'content': b}] if role != 'none' else []) + [{'role': 'user', 'content': q}]
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
        seg = {(m, qn, r): _segment(R[m, qn, r], b, ts, r) for m in spec['modes'] for qn in qs for r in I.ROLES}
        gen = {m: R[m, 'attribution', 'user'][R[m, 'attribution', 'user'].rindex(ts):] for m in spec['modes']}
        for m in modes:
            for qn in qs:
                S = {r: seg[m, qn, r] for r in I.ROLES}
                tr_same = len({S[r]['trail'] for r in I.ROLES}) == 1
                if repo == 'ibm-granite/granite-4.2-30b':
                    extra = S['assistant']['head_norm'][len(S['user']['head_norm']):] if S['assistant']['head_norm'].startswith(S['user']['head_norm']) else None
                    ok = extra == '<think></think>' and S['user']['head_norm'] == S['system']['head_norm'] and tr_same
                    check(repo, f'Assistant history = empty think block before body, nothing else ({m}, {qn})', ok,
                          f'headers {[S[r]["head"] for r in I.ROLES]!r}, trailers {[S[r]["trail"] for r in I.ROLES]!r}')
                    pre_ok = S['assistant']['pre'] == S['user']['pre'] == '<|im_start|>system\n<|im_end|>\n' and S['system']['pre'] == ''
                    lone = R[m, qn, 'none']
                    check(repo, f'no System message -> empty system turn opens the conversation; a System message takes that turn ({m}, {qn})',
                          pre_ok and lone.startswith('<|im_start|>system\n<|im_end|>\n<|im_start|>user\n'),
                          f'before the message: {[S[r]["pre"] for r in I.ROLES]!r}; [user: question] alone starts {lone[:60]!r}')
                else:
                    ok = len({S[r]['head_norm'] for r in I.ROLES}) == 1 and tr_same
                    check(repo, f'Assistant history "none": the three roles differ only in the role name ({m}, {qn})', ok,
                          f'headers {[S[r]["head"] for r in I.ROLES]!r}, trailers {[S[r]["trail"] for r in I.ROLES]!r}')
                    pre = [S[r]['pre'] for r in I.ROLES]
                    check(repo, f'nothing before the conversation (no system turn or instruction inserted) ({m}, {qn})',
                          all(p == '' for p in pre) and not R[m, qn, 'none'].startswith(ts + 'system'),
                          f'before the message: {pre!r}; [user: question] alone starts {R[m, qn, "none"][:60]!r}')
        if repo == 'Qwen/Qwen3.5-27B':
            check(repo, 'modes "both": non-thinking generation prompt closes an empty think block, thinking opens one',
                  gen[NT] == '<|im_start|>assistant\n<think>\n\n</think>\n\n' and gen[TH] == '<|im_start|>assistant\n<think>\n',
                  f'generation prompts {gen!r}')
        elif repo == 'ibm-granite/granite-4.2-30b':
            check(repo, 'modes "both": non-thinking generation prompt closes an empty think block, thinking opens one',
                  gen[NT] == '<|im_start|>assistant\n<think></think>' and gen[TH] == '<|im_start|>assistant\n<think>\n',
                  f'generation prompts {gen!r}')
            dflt = tok.apply_chat_template([{'role': 'user', 'content': qs['attribution']}], tokenize=False, add_generation_prompt=True)
            check(repo, 'template default ({}) is the thinking mode', dflt == R[TH, 'attribution', 'none'], f'default ends {dflt[-40:]!r}')
        else:
            same = all(R[k] == R[(NT,) + k[1:]] for k in R if k[0].startswith('probe'))
            check(repo, 'modes "non-thinking" only: enable_thinking true/false change nothing, and the template never mentions thinking',
                  same and 'think' not in tmpl.lower(), f'renderings identical under the probes: {same}; "think" in template: {"think" in tmpl.lower()}')
    return out


def gptoss_facts(root, checks):
    """Renderings of [ROLE: BODY][user: Q] for gpt-oss-20b and gpt-oss-120b (template default, as in the runs) and their checks."""
    root = Path(root)
    out, rend = [], {}
    for name in ('gpt-oss-20b', 'gpt-oss-120b'):
        d = root / 'openai' / name
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(d)
        tmpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
        files = {f.name: sha256(f.read_bytes())[:16] for f in sorted(d.iterdir())
                 if f.name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json', 'generation_config.json')}
        out.append(f'##### openai/{name}\nchat_template sha256 {sha256(tmpl.encode())}\nfiles {json.dumps(files)}')
        cfg = json.loads((d / 'config.json').read_text())
        out.append(f"config num_hidden_layers {cfg.get('num_hidden_layers')} num_local_experts {cfg.get('num_local_experts')} "
                   f"quantization {json.dumps(cfg.get('quantization_config', {}).get('quant_method'))}")
        rend[name] = {}
        for role in I.ROLES:
            msgs = [{'role': role, 'content': 'BODY'}, {'role': 'user', 'content': 'Q'}]
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            assert ids_of(tok(s, add_special_tokens=False)) == ids_of(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)), (name, role)
            rend[name][role] = s
            out.append(f'--- kwargs {{}} role {role}\n{s!r}')
        r = rend[name]
        checks += [(name, 'assistant: final-channel tag before the body', '<|start|>assistant<|channel|>final<|message|>BODY<|end|>' in r['assistant'], ''),
                   (name, 'user: no tag before the body', '<|start|>user<|message|>BODY<|end|>' in r['user'], ''),
                   (name, 'system: rendered as a developer message', '<|start|>developer<|message|># Instructions\n\nBODY' in r['system'], ''),
                   (name, 'roles differ only in the message turn',
                    len({x.split('<|start|>user<|message|>Q')[1] for x in r.values()}) == 1 and len({x.split('BODY')[0].split('<|end|>')[0] for x in r.values()}) == 1, '')]
        if name == 'gpt-oss-120b':
            checks += [(name, '36 decoder blocks', cfg.get('num_hidden_layers') == 36, ''), (name, 'MoE', (cfg.get('num_local_experts') or 0) > 1, ''),
                       (name, 'MXFP4 weights', cfg.get('quantization_config', {}).get('quant_method') == 'mxfp4', '')]
    nodate = {n: {k: re.sub(r'Current date: [0-9-]+', 'Current date: D', v) for k, v in rr.items()} for n, rr in rend.items()}
    checks.append(('both', 'gpt-oss-120b renders as gpt-oss-20b (date line aside)', nodate['gpt-oss-20b'] == nodate['gpt-oss-120b'], ''))
    return out


def facts(root, date=None):
    import transformers
    out = [f'Table 2 template facts, {Path(__file__).name}, transformers {transformers.__version__}',
           f'body id {appendix_body()["id"]} (the first of the 150 bodies)', '']
    checks = []
    with fixed_date(date):
        out += table2_facts(root, checks)
        out += gptoss_facts(root, checks)
    out.append('##### CHECKS')
    for model, claim, ok, ev in checks:
        out.append(f'{"PASS" if ok else "FAIL"}  {model}: {claim}' + ('' if ok or not ev else f'\n      evidence: {ev}'))
    n_fail = sum(not c[2] for c in checks)
    out.append(f'{len(checks)} checks, {n_fail} FAIL')
    return out, n_fail


# ----------------------------------------------------------------------------------------------- inputs of earlier runs
def compare_ids(tok, msgs, ids, kw):
    """One line: whether the template's own token rendering of msgs (kw) equals ids, else where they first differ."""
    try:
        nat = ids_of(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw))
    except Exception as exc:
        return f'TEMPLATE ERROR {type(exc).__name__}: {str(exc)[:80]}'
    if nat == ids:
        return f'IDENTICAL ({len(nat)} tokens)'
    i = next((k for k, (a, b) in enumerate(zip(nat, ids)) if a != b), min(len(nat), len(ids)))
    return (f'DIFFERS at token {i} (native {len(nat)} vs ours {len(ids)})\n     native: {tok.convert_ids_to_tokens(nat[max(0, i - 3):i + 6])}'
            f'\n     ours  : {tok.convert_ids_to_tokens(ids[max(0, i - 3):i + 6])}')


def check_inputs(tok, runs, kwargs_list=None):
    """runs: (label, folder, thinking, user turn or None). Each folder holds identity.json (the body list), bodies/<id>.json and
    features/<id>__<role>.npz (input_token_ids). The first body under each role, [ROLE: body] plus [user: user turn] when the run
    had one, against the native token rendering in each template setting (default: the run's thinking flag with
    preserve_thinking false and true)."""
    import numpy as np
    out = []
    for s in ('<|im_start|>', '<|im_end|>', '<think>', '</think>', 'assistant', 'user', 'system'):
        out.append(f'{s!r:16} -> {tok.encode(s, add_special_tokens=False)}')
    for name, run_dir, thinking, user_turn in runs:
        run_dir = Path(run_dir)
        bid = json.loads((run_dir / 'identity.json').read_text(encoding='utf-8'))['bodies'][0]
        body = json.loads((run_dir / 'bodies' / f'{bid}.json').read_text(encoding='utf-8'))['body']
        out.append(f'\n===== {name}')
        for role in I.ROLES:
            ours = np.load(run_dir / 'features' / f'{bid}__{role}.npz')['input_token_ids'].tolist()
            msgs = [{'role': role, 'content': body}] + ([{'role': 'user', 'content': user_turn}] if user_turn else [])
            for kw in kwargs_list or ({'enable_thinking': thinking, 'preserve_thinking': False}, {'enable_thinking': thinking, 'preserve_thinking': True}):
                label = ' '.join(f'{k}={v}' for k, v in kw.items() if k != 'enable_thinking')
                out.append(f'  {role:9s} {label}: {compare_ids(tok, msgs, ours, kw)}')
    return out


# ----------------------------------------------------------------------------------------------- main
def run(args):
    if args.mode == 'fetch':
        fetch(args.root, args.models, deepseek=args.deepseek_files)
        return
    if args.mode == 'appendix':
        out = appendix(args.root, args.models, args.with_ids, args.date)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
        print('wrote', args.out, len(out['models']), 'models')
    elif args.mode == 'deepseek':
        out = appendix_deepseek(args.encoding, args.tokenizer, args.with_ids)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
        print('encoding sha256', out['encoding_py_sha256'][:16], 'same as the runs:', out['same_as_runs'])
    elif args.mode == 'placeholders':
        extra = json.loads(args.extra_kwargs) if args.extra_kwargs else []
        out = placeholders(args.root, args.models, extra, args.date)
        Path(args.out).write_text('\n'.join(out) + '\n', encoding='utf-8')
        print('wrote', args.out, len(out), 'blocks')
    elif args.mode == 'facts':
        out, n_fail = facts(args.root, args.date)
        Path(args.out).write_text('\n'.join(out) + '\n', encoding='utf-8')
        print('\n'.join(o for o in out if o.startswith(('PASS', 'FAIL', '      evidence'))))
        print('wrote', args.out, f'{n_fail} FAIL')
        if n_fail:
            sys.exit(1)
    elif args.mode == 'check-inputs':
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        runs = []
        for r in args.runs:  # identity.json: mode reply / think, user_turn (the user message after the body, if the run had one)
            ident = json.loads((Path(r) / 'identity.json').read_text(encoding='utf-8'))
            ut = ident.get('user_turn')
            runs.append((f"{ident['mode']}_{'turn' if ut else 'noturn'}", r, ident['mode'] == 'think', ut))
        out = check_inputs(tok, runs, json.loads(args.kwargs) if args.kwargs else None)
        text = '\n'.join(out) + '\n'
        if args.out:
            Path(args.out).write_text(text, encoding='utf-8')
        print(text, end='')


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--mode', choices=('fetch', 'appendix', 'deepseek', 'placeholders', 'facts', 'check-inputs'), required=True)
    p.add_argument('--root', default=None, help='folder with one subfolder per repository (ROOT/<org>/<name>)')
    p.add_argument('--out', default=None)
    p.add_argument('--models', default=None, help='comma-separated repositories of Table 10 (default: all)')
    p.add_argument('--deepseek-files', action='store_true', help='fetch: also the tokenizer and encoder of DeepSeek-V4.1-Flash')
    p.add_argument('--date', default=None, help='YYYY-MM-DD for templates that print the current date (default: today)')
    p.add_argument('--with-ids', action='store_true', help='also record the token ids of each rendering')
    p.add_argument('--encoding', default=None, help="DeepSeek's encoding.py (default: the copy in selfindex/dsv41/official)")
    p.add_argument('--tokenizer', default=None, help='deepseek: folder with tokenizer.json; check-inputs: the tokenizer folder')
    p.add_argument('--extra-kwargs', default=None, help='placeholders: JSON list of further template settings to render')
    p.add_argument('--runs', nargs='*', default=[], help='check-inputs: run folders')
    p.add_argument('--kwargs', default=None, help='check-inputs: JSON list of template settings (default: thinking flag of the run x preserve_thinking)')
    return p


def main(argv=None):
    run(parser().parse_args(argv))


if __name__ == '__main__':
    main()
