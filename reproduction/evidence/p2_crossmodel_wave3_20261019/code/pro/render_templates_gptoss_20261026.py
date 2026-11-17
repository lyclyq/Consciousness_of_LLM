"""Archive the template identity of gpt-oss-120b (10-26: Table 2 states its template facts, but no rendering of it was archived;
gpt-oss-20b's is in items/template_renderings_wave3.txt, made on server3). On PRO, tokenizer files only, CPU: for gpt-oss-20b and
gpt-oss-120b, the chat template's rendering of [ROLE: BODY][user: Q] with the generation prompt for ROLE in assistant/user/system
(template default, as in the runs: tmpl_kwargs {}), the sha256 of the template and tokenizer files, and the checks of Table 2:
  - the earlier Assistant message carries the final-channel tag before the body (<|channel|>final<|message|>BODY), User none;
  - the System message is rendered as a developer message (<|start|>developer<|message|># Instructions ... BODY);
  - 36 decoder blocks, MoE, MXFP4 (config.json) for 120b; the two models render identically (date line aside).
usage: render_templates_gptoss_20261026.py OUT_TXT   (PRO, models under /root/hf_models/openai)"""
import hashlib
import json
import re
import sys
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path('/root/hf_models/openai')
out, rend, checks = [], {}, []
for name in ('gpt-oss-20b', 'gpt-oss-120b'):
    d = ROOT / name
    tok = AutoTokenizer.from_pretrained(d)
    tmpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in sorted(d.iterdir())
             if f.name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json', 'generation_config.json')}
    out.append(f'##### openai/{name}\nchat_template sha256 {hashlib.sha256(tmpl.encode()).hexdigest()}\nfiles {json.dumps(files)}')
    cfg = json.loads((d / 'config.json').read_text())
    out.append(f"config num_hidden_layers {cfg.get('num_hidden_layers')} num_local_experts {cfg.get('num_local_experts')} "
               f"quantization {json.dumps(cfg.get('quantization_config', {}).get('quant_method'))}")
    rend[name] = {}
    for role in ('assistant', 'user', 'system'):
        msgs = [{'role': role, 'content': 'BODY'}, {'role': 'user', 'content': 'Q'}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert tok(s, add_special_tokens=False)['input_ids'] == tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False), (name, role)
        rend[name][role] = s
        out.append(f'--- kwargs {{}} role {role}\n{s!r}')
    r = rend[name]
    checks += [(name, 'assistant: final-channel tag before the body', '<|start|>assistant<|channel|>final<|message|>BODY<|end|>' in r['assistant']),
               (name, 'user: no tag before the body', '<|start|>user<|message|>BODY<|end|>' in r['user']),
               (name, 'system: rendered as a developer message', '<|start|>developer<|message|># Instructions\n\nBODY' in r['system']),
               (name, 'roles differ only in the message turn',
                len({x.split('<|start|>user<|message|>Q')[1] for x in r.values()}) == 1 and len({x.split('BODY')[0].split('<|end|>')[0] for x in r.values()}) == 1)]
    if name == 'gpt-oss-120b':
        checks += [(name, '36 decoder blocks', cfg.get('num_hidden_layers') == 36), (name, 'MoE', (cfg.get('num_local_experts') or 0) > 1),
                   (name, 'MXFP4 weights', cfg.get('quantization_config', {}).get('quant_method') == 'mxfp4')]
nodate = {n: {k: re.sub(r'Current date: [0-9-]+', 'Current date: D', v) for k, v in rr.items()} for n, rr in rend.items()}
checks.append(('both', 'gpt-oss-120b renders as gpt-oss-20b (date line aside)', nodate['gpt-oss-20b'] == nodate['gpt-oss-120b']))
out.append('##### checks')
out += [f"{'PASS' if ok else 'FAIL'}  {n}: {c}" for n, c, ok in checks]
Path(sys.argv[1]).write_text('\n'.join(out) + '\n', encoding='utf-8')
print('\n'.join(out[-len(checks) - 1:]))
