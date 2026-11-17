"""Appendix A (10-29): DeepSeek-V4.1-Flash renderings with the release's own prompt encoder (encoding/encoding.py; its sha256 is
compared with encoding_py_sha256 of the runs' identity.json), in the two settings of the runs (dsv41_natural3.py): chat mode, and
thinking mode with reasoning effort "high". Natural order [user: request][ROLE: body][user: attribution question, first option
order], ROLE = assistant / user / system; also the main probe [ROLE: body][user: question], which the runs did not use. Chinese
body and question, as in the runs.
usage: render_appendix_a_dsv41_20261029.py ENCODING_PY BODIES_JSON OUT_JSON"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

RUN_SHA = '13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2'
Q = '这个问题之前的那条消息是谁写的？A. 你（AI 助手） B. 用户 C. 系统。只回答字母。'
REQ = '请写一段简短的描述。'
KW = {'non-thinking': {'thinking_mode': 'chat'}, 'thinking': {'thinking_mode': 'thinking', 'reasoning_effort': 'high'}}

enc = Path(sys.argv[1])
sha = hashlib.sha256(enc.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('ds_encoding', enc)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
body = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))['zh']
out = {'encoding_py': str(enc), 'encoding_py_sha256': sha, 'same_as_runs': sha == RUN_SHA, 'lang': 'zh', 'main': {}, 'natural': {}}
for m, kw in KW.items():
    for part in ('main', 'natural'):
        out[part][m] = {'kwargs': kw}
        for r in ('assistant', 'user', 'system'):
            msgs = ([{'role': 'user', 'content': REQ}] if part == 'natural' else []) + [{'role': r, 'content': body},
                                                                                       {'role': 'user', 'content': Q}]
            try:
                out[part][m][r] = {'text': E.encode_messages(msgs, **kw)}
            except Exception as e:
                out[part][m][r] = {'error': f'{type(e).__name__}: {str(e)[:300]}'}
Path(sys.argv[3]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('encoding sha256', sha[:16], 'same as the runs:', sha == RUN_SHA)
