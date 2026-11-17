"""Archive the template identity of the wave-3 models (10-19): for each model, the chat template's own rendering of
[ROLE: BODY][user: Q] with the generation prompt, for ROLE in assistant/user/system and in the template's modes, plus the
sha256 of the chat template text and the tokenizer files. Run with the torch 2.6 / transformers 5.19 venv (it knows every
architecture here). usage: render_templates_wave3.py OUT_TXT"""
import hashlib
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path('/root/data/hf_models')
MODELS = {'ByteDance-Seed/Seed-OSS-36B-Instruct': [{'thinking_budget': 0}, {}],
          'LGAI-EXAONE/EXAONE-4.0.1-32B': [{'enable_thinking': False}, {'enable_thinking': True}],
          'openai/gpt-oss-20b': [{}],
          'zai-org/GLM-5.3-Flash': [{'enable_thinking': False}, {}]}
out = []
for repo, modes in MODELS.items():
    d = ROOT / repo
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    tmpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in sorted(d.iterdir())
             if f.name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json', 'generation_config.json')}
    out.append(f'##### {repo}\nchat_template sha256 {hashlib.sha256(tmpl.encode()).hexdigest()}\nfiles {json.dumps(files)}')
    for kw in modes:
        for role in ('assistant', 'user', 'system'):
            msgs = [{'role': role, 'content': 'BODY'}, {'role': 'user', 'content': 'Q'}]
            try:
                s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            except Exception as e:  # some templates reject a role in some positions
                s = f'<error: {type(e).__name__}: {e}>'
            out.append(f'--- kwargs {json.dumps(kw)} role {role}\n{s!r}')
Path(sys.argv[1]).write_text('\n'.join(out) + '\n', encoding='utf-8')
print('wrote', sys.argv[1], len(out), 'blocks')
